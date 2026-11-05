"""Robot isolation, mapping reference vs reachable home, and bounded startup."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.show import prepare
from vivis.runtime.trajectory import load_targets
from vivis.motion.mapping import compile_trajectory,inputs

CONFIG=ROOT/'configs/robots/xiaoshuai/robot.json'
HELD={7,8,22,23,29,32,33,34}


def face_fixture():
    robot=Robot(CONFIG);work=robot.work('0001')
    ids,positions,fps=load_targets(work['targets'],robot.profile('0001'),work['audio'],robot.calibration)
    profile,ids,trajectory=prepare(robot.tracking(),robot.profile('0001'),ids,positions,
        telemetry_policy=robot.telemetry(),mode='face',mouth_command_mode='lips-direct')
    return robot,profile,ids,trajectory


class XiaoshuaiTests(unittest.TestCase):
    def test_own_mapping_and_shared_media_do_not_use_mojia_motor_targets(self):
        robot=Robot(CONFIG);mojia=Robot();work=robot.work('0001');old=mojia.work('0001')
        self.assertEqual(set(robot.axes),set(range(2,35)))
        self.assertEqual(work['audio'],old['audio']);self.assertEqual(work['source_clip'],old['source_clip'])
        self.assertNotEqual(work['targets'],old['targets']);self.assertNotEqual(robot.calibration_sha256,mojia.calibration_sha256)
        values,fps,order,shapes=inputs(work['source_clip'],robot.calibration,robot.resolve('channels'),set(robot.axes))
        expected_ids,expected,_=compile_trajectory(values,order,shapes,robot.motion_profile(),'face_eyes')
        ids,actual,rate=load_targets(work['targets'],robot.profile('0001'),work['audio'],robot.calibration)
        self.assertEqual(ids,expected_ids);np.testing.assert_array_equal(actual,expected)
        self.assertEqual((len(actual),rate),(6774,60))
        self.assertTrue({9,24}<=set(ids));self.assertFalse({29,35,36}&set(ids))

    def test_reference2047_is_kept_when_home_is_clipped_to2030(self):
        robot=Robot(CONFIG);p=robot.motion_profile();motors={m['id']:m for m in p['motors']}
        self.assertEqual(robot.axes[24]['neutral'],2047)
        self.assertEqual((motors[24]['mapping_neutral'],motors[24]['neutral']),(2047,2030))
        self.assertEqual(next(a['neutral'] for a in robot.home()['axes'] if a['id']==24),2030)
        _,_,order,shapes=inputs(robot.work('0001')['source_clip'],robot.calibration,robot.resolve('channels'))
        # A different valid cheek endpoint distinguishes a mapping-reference
        # calculation from silently moving the expression's zero to2030.
        shapes=copy.deepcopy(shapes);shapes['CheekSquintRight'][24]=1950
        values=np.zeros((1,61));values[0,order.index('CheekSquintRight')]=.5
        ids,positions,_=compile_trajectory(values,order,shapes,p,'face_eyes')
        self.assertEqual(positions[0,ids.index(24)],2047+.5*(1950-2047))

    def test_jaw_open_is_not_diluted_by_neutral_records_from_unrelated_shapes(self):
        robot=Robot(CONFIG);work=robot.work('0001');p=robot.motion_profile()
        values,fps,order,shapes=inputs(work['source_clip'],robot.calibration,robot.resolve('channels'),set(robot.axes))
        ids,targets,_=compile_trajectory(values,order,shapes,p,'face_eyes')
        jaw=ids.index(31);expected=2047+values[:,order.index('JawOpen')]*(1850-2047)
        np.testing.assert_allclose(targets[:,jaw],expected,rtol=0,atol=1e-10)
        legacy=copy.deepcopy(p);legacy['jaw_owner']='all_calibrated'
        old_ids,old,_=compile_trajectory(values,order,shapes,legacy,'face_eyes')
        self.assertEqual(ids,old_ids)
        np.testing.assert_array_equal(np.delete(targets,jaw,axis=1),np.delete(old,jaw,axis=1))
        self.assertLess(targets[:,jaw].min(),1920);self.assertGreater(old[:,jaw].min(),1990)
        self.assertEqual(sum(31 in points for points in shapes.values()),50)
        self.assertEqual(Robot().motion_profile()['jaw_owner'],'all_calibrated')
        self.assertTrue(all(robot.axes[sid]['max_speed_counts_s']==600 for sid in (5,6,20,21,31)))

    def test_round_lip_gain_does_not_reintroduce_cross_channel_jaw_dilution(self):
        from vivis.motion.channel_gain import round_lip_gain_targets
        robot=Robot(CONFIG);work=robot.work('0001')
        ids,positions,fps=load_targets(work['targets'],robot.profile('0001'),work['audio'],robot.calibration)
        adjusted,report=round_lip_gain_targets(robot,work,ids,positions,fps,1.2)
        np.testing.assert_array_equal(adjusted[:,ids.index(31)],positions[:,ids.index(31)])
        self.assertNotIn(31,report['changed_ids'])

    def test_session_and_home_exclude_every_held_and_unmapped_axis(self):
        robot,p,ids,trajectory=face_fixture()
        self.assertEqual(len(ids),25);self.assertFalse(set(ids)&HELD)
        self.assertEqual(set(ids),set(p['diagnostic_command_ids']))
        self.assertEqual(set(ids),{m['id'] for m in p['motors']})
        self.assertEqual(set(ids),{a['id'] for a in robot.home()['axes']})
        preset=robot.startup(active_ids=ids,inventory_ids=ids)
        self.assertEqual(preset['excluded_ids'],[])
        self.assertEqual({a['id'] for a in preset['axes']},set(ids))
        self.assertEqual(next(a['model_raw'] for a in preset['axes'] if a['id']==31),11272)
        self.assertTrue(all('pid_raw' not in a and 'deadband_raw' not in a for a in preset['axes']))
        self.assertTrue(np.all(trajectory[:,ids.index(24)]==2030))
        self.assertNotIn(29,p['mouth_gain_ids'])

    def test_wrong_topology_or_unconfigured_camera_fails_before_opening_devices(self):
        from vivis.runtime.developer import run_example
        with patch('vivis.hardware.feetech.sdk.OfficialSDKBus',side_effect=AssertionError('serial opened')):
            for options,reason in ((['--parallel-neck','on'],'Independent joints'),(['--camera','left'],'Unknown camera')):
                with self.assertRaisesRegex(ValueError,reason):
                    run_example('eyes-neck',['--robot',str(CONFIG),'--work','0001','--check',*options])

    def test_invalid_home_or_model_is_rejected_in_robot_loader(self):
        robot=Robot(CONFIG);definition=copy.deepcopy(robot.definition)
        for key in ('calibration','works','channels','homing','tracking','telemetry'):
            definition[key]=str(robot.resolve(key))
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);definition['hardware']=str(folder/'hardware.json')
            (folder/'robot.json').write_text(json.dumps(definition))
            for sid,field,value in ((24,'home_position',2047),(31,'model_raw_expected',65535)):
                hardware=copy.deepcopy(robot.hardware)
                next(a for a in hardware['axes'] if a['id']==sid)[field]=value
                (folder/'hardware.json').write_text(json.dumps(hardware))
                with self.assertRaisesRegex(ValueError,'envelope mismatch'):Robot(folder/'robot.json')

    def test_shared_timed_playback_supports_sm45bl_face_and_never_sends_held_ids(self):
        from test_runtime import PlaybackTests
        from vivis.runtime.playback import play
        robot,p,ids,_=face_fixture();p.update(control_hz=60,face_update_hz=60,changed_goals_only=False)
        _,bus,clock=PlaybackTests().fixture();clock.duration=.3
        motors={m['id']:m for m in p['motors']};positions={sid:m['neutral'] for sid,m in motors.items()}
        data=np.array([[positions[sid] for sid in ids]]*18,dtype=float)
        data[:,ids.index(31)]-=np.arange(18)*2
        def state(sid):
            self.assertNotIn(sid,HELD);m=motors[sid]
            return dict(position=positions[sid],temperature_c=25,load_raw=0,status_register=0,
                        voltage_raw=(m['min_voltage_raw']+m['max_voltage_raw'])//2)
        def snapshot(sid):
            m=motors[sid]
            return dict(id=sid,model_raw=m['model_raw_expected'],mode=0,torque_enable=1,
                configured_min=m['device_min'],configured_max=m['device_max'],position_raw=positions[sid],
                **{k:v for k,v in state(sid).items() if k!='position'})
        def read(sid,address,count):
            s=state(sid)
            return {56:s['position'].to_bytes(2,'little'),60:b'\0\0',
                    62:bytes([s['voltage_raw'],25]),65:b'\0'}[address]
        bus.snapshot.side_effect=snapshot;bus.feedback.side_effect=state;bus.read.side_effect=read
        bus.goals.side_effect=lambda goals,speeds:positions.update(goals)
        with tempfile.TemporaryDirectory() as folder, \
             patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus), \
             patch('vivis.runtime.playback.AudioClock',return_value=clock):
            result=play(p,ids,data,60,'fake',Path(folder)/'log.json',Path(folder)/'STOP')
        self.assertTrue(result['completed'])
        for event in result['events']:
            self.assertEqual(set(event['goals']),set(ids));self.assertEqual(event['goals'][24],2030)
            self.assertEqual(event['goals'][31],round(data[event['frame'],ids.index(31)]))

    def test_full_session_initializes_only_face_axes_with_bounded_ram_writes(self):
        import vivis.runtime.developer as launcher
        import vivis.runtime.show as show
        robot,profile,ids,trajectory=face_fixture();allowed=set(ids)
        config={};ram={};positions={};writes=[];goals=[]
        for sid in ids:
            a=robot.axes[sid];c=bytearray(40);r=bytearray(31)
            c[3:5]=a['model_raw_expected'].to_bytes(2,'little');c[18]=4
            c[9:11]=a['device_min'].to_bytes(2,'little');c[11:13]=a['device_max'].to_bytes(2,'little')
            c[13]=a['device_max_temperature_c'];c[14]=a['max_voltage_raw'];c[15]=a['min_voltage_raw']
            c[21:24]=bytes([32,32,0]);r[8:10]=(1000).to_bytes(2,'little')
            config[sid]=c;ram[sid]=r;positions[sid]=2045 if sid==24 else 2047
        def raw(sid):
            self.assertIn(sid,allowed)
            ram[sid][16:18]=positions[sid].to_bytes(2,'little')
            return config[sid],ram[sid]
        def state(sid):
            c,r=raw(sid);a=robot.axes[sid]
            return dict(position=positions[sid],temperature_c=25,voltage_raw=(a['min_voltage_raw']+a['max_voltage_raw'])//2,load_raw=0,status_register=0)
        def snapshot(sid):
            c,r=raw(sid);s=state(sid)
            return dict(id=sid,model_raw=robot.axes[sid]['model_raw_expected'],mode=0,
                torque_enable=r[0],goal_position_raw=int.from_bytes(r[2:4],'little'),position_raw=s['position'],
                configured_min=robot.axes[sid]['device_min'],configured_max=robot.axes[sid]['device_max'],
                configured_max_temperature=c[13],configured_min_voltage_raw=c[15],configured_max_voltage_raw=c[14],
                config_0_39_hex=c.hex(),state_40_70_hex=r.hex(),**{k:v for k,v in s.items() if k!='position'})
        def read(sid,address,count):
            c,r=raw(sid);block,offset=(c,address) if address<40 else (r,address-40)
            return bytes(block[offset:offset+count])
        def write(sid,address,value,width):
            self.assertIn(sid,allowed);self.assertIn(address,(40,41,46))
            writes.append((sid,address,value,width));ram[sid][address-40:address-40+width]=value.to_bytes(width,'little');return 0,0
        def move(targets,speeds):
            self.assertFalse(set(targets)&HELD)
            for sid,value in targets.items():
                self.assertIn(sid,allowed);self.assertTrue(robot.axes[sid]['device_min']<=value<=robot.axes[sid]['device_max'])
                ram[sid][2:4]=value.to_bytes(2,'little');ram[sid][6:8]=speeds[sid].to_bytes(2,'little')
                ram[sid][0]=1;positions[sid]=value
            goals.append(dict(targets))
        bus=Mock();bus.snapshot.side_effect=snapshot;bus.feedback.side_effect=state;bus.read.side_effect=read;bus.goals.side_effect=move
        bus.sdk.write1ByteTxRx.side_effect=lambda sid,a,v:write(sid,a,v,1)
        bus.sdk.write2ByteTxRx.side_effect=lambda sid,a,v:write(sid,a,v,2)
        clock=SimpleNamespace(now=100.)
        def sleep(delay):clock.now+=max(delay,.000001)
        def played(*args,**kwargs):
            self.assertEqual(set(args[1]),allowed);self.assertEqual(args[2].shape,(6774,25))
            self.assertIsNone(kwargs['gaze_override'])
            return dict(completed=True,full_clip_requested=True,audio_seconds_at_stop=112.9,sent_frames=6774,skipped_source_frames=0)
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()), \
             patch.object(launcher,'ROOT',Path(folder)),patch.object(show,'ROOT',Path(folder)), \
             patch.object(launcher,'check_competitors'), \
             patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus) as constructor, \
             patch('vivis.runtime.homing.time.monotonic',side_effect=lambda:clock.now), \
             patch('vivis.runtime.homing.time.sleep',side_effect=sleep), \
             patch.object(show,'play',side_effect=played),patch.object(show,'unload_tongue') as unload, \
             patch.object(show,'TargetReceiver',side_effect=AssertionError('camera requested')):
            output=Path(folder)/'run'
            launcher.run_example('face',['--robot',str(CONFIG),'--work','0001','--repeat','1',
                '--mouth-command-mode','lips-direct','--output',str(output)])
            summary=json.loads((output/'performance/summary.json').read_text())
            prep=json.loads((output/'performance/round_1/preparation.json').read_text())
            self.assertTrue(summary['completed']);self.assertTrue(prep['homing']['completed'])
            self.assertEqual(prep['homing']['neutral_targets']['24'],2030)
            self.assertTrue(any(r['id']==24 and r['confirmed'] for r in prep['homing']['disabled_position_recoveries']))
            self.assertEqual(constructor.call_args.kwargs['feedback_mode'],'short_registers')
            self.assertEqual(constructor.call_args.kwargs['post_goal_quiet_ms'],1)
            unload.assert_not_called()
        self.assertTrue(writes);self.assertTrue(goals)
        self.assertEqual(positions[24],2030)
        self.assertTrue(all(ram[sid][0]==1 for sid in allowed))


if __name__=='__main__':unittest.main()
