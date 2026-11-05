"""Fixed brow camera, independent neck ownership and bounded three-axis pursuit."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.trajectory import load_targets
from vivis.runtime.show import prepare
from vivis.motion.ownership import MojiaPlaybackGaze
from vivis.vision.policy import VERSION
from vivis.vision.independent_follow import validate

CONFIG=ROOT/'configs/robots/xiaoshuai/robot.json'


class Receiver:
    def __init__(self):self.target=dict(x=.6,y=.6,eye_line_roll_rad=.4)
    def poll(self,now):
        return dict(schema_version=VERSION,sequence=int(now*1000),timestamp_monotonic=now,target=self.target)


def fixture(mode='eyes-neck'):
    robot=Robot(CONFIG);work=robot.work('0001');config=robot.tracking()
    ids,positions,fps=load_targets(work['targets'],robot.profile('0001'),work['audio'],robot.calibration)
    profile,ids,data=prepare(config,robot.profile('0001'),ids,positions,telemetry_policy=robot.telemetry(),
        mode=mode,mouth_command_mode='lips-direct')
    return robot,config,profile,ids,data


def snapshots(profile,positions):
    rows=[]
    for m in profile['motors']:
        raw=bytearray(40);raw[18]=4;ram=bytearray(31);ram[1]=0
        rows.append(dict(id=m['id'],model_raw=m['model_raw_expected'],mode=0,torque_enable=1,
            configured_min=m['device_min'],configured_max=m['device_max'],position_raw=int(positions[m['id']]),
            temperature_c=30,load_raw=0,status_register=0,voltage_raw=235 if m['model_raw_expected']==11272 else 60,
            config_0_39_hex=raw.hex(),state_40_70_hex=ram.hex()))
    return rows


def controller():
    robot,c,p,ids,data=fixture();receiver=Receiver();g=MojiaPlaybackGaze(c,p,ids,receiver,neck=True)
    initial={m['id']:m['neutral'] for m in p['motors']};last=np.array([initial[sid] for sid in ids],float)
    g.prepare(data,last,snapshots(p,initial),10.);g.set_audio_time(0.,1000.)
    return robot,c,p,ids,data,receiver,g,last


class IndependentTrackingTests(unittest.TestCase):
    def test_all_modes_check_without_devices_and_use_distinct_home_inventories(self):
        from vivis.runtime.developer import run_example
        robot=Robot(CONFIG)
        self.assertTrue(all('独立轴' in robot.axes[sid]['role'] for sid in (32,33,34)))
        with patch('vivis.hardware.feetech.sdk.OfficialSDKBus',side_effect=AssertionError('serial opened')), \
             patch('vivis.runtime.developer.subprocess.Popen',side_effect=AssertionError('camera opened')):
            for mode,count,owned in (('face',25,set()),('eyes',29,{7,8,22,23}),('eyes-neck',32,{7,8,22,23,32,33,34})):
                with redirect_stdout(io.StringIO()) as output:
                    run_example(mode,['--robot',str(CONFIG),'--work','0001','--check'])
                result=json.loads(output.getvalue())
                self.assertEqual(len(result['home_ids']),count)
                self.assertEqual(set(result['home_ids']),set(result['commanded']))
                self.assertEqual(set(result['gaze_owner_ids']),owned)
                self.assertEqual(result['independent_neck'],mode=='eyes-neck');self.assertFalse(result['parallel_neck'])
                self.assertNotIn(29,result['commanded'])
        face=fixture('face');full=fixture()
        for sid in face[3]:np.testing.assert_array_equal(face[4][:,face[3].index(sid)],full[4][:,full[3].index(sid)])

    def test_confirmed_joint_and_calibrated_eye_directions_follow_both_image_sides(self):
        for sign in (-1,1):
            robot,c,p,ids,data,receiver,g,last=controller()
            receiver.target=dict(x=sign*.6,y=sign*.6,eye_line_roll_rad=sign*.4)
            for tick in range(180):
                now=10+(tick+1)/60
                for sid in g.owned:g.observe(sid,dict(position=last[ids.index(sid)]),now)
                old=last.copy();last=g.apply(data[tick],last,1/60,now)
                for sid in set(ids)-set(g.owned):self.assertEqual(last[ids.index(sid)],data[tick,ids.index(sid)])
            self.assertGreater(sign*(last[ids.index(8)]-2047),0)
            self.assertGreater(sign*(last[ids.index(23)]-2047),0)
            self.assertLess(sign*(last[ids.index(7)]-2047),0)
            self.assertGreater(sign*(last[ids.index(22)]-2047),0)
            self.assertGreater(sign*(last[ids.index(32)]-2047),0)  # Onsite image-to-yaw direction correction.
            self.assertGreater(sign*(last[ids.index(33)]-2047),0)  # Increasing pitch looks down.
            self.assertGreater(sign*(last[ids.index(34)]-2047),0)  # Increasing roll tilts robot-right.
            self.assertIsNone(c['control']['camera_eye']);self.assertEqual(c['control']['camera_mount'],'fixed')
            self.assertIn(22,g.feedback_priority_ids)

    def test_lost_face_returns_all_owned_axes_smoothly_to_neutral(self):
        robot,c,p,ids,data,receiver,g,last=controller()
        for tick in range(180):
            now=10+(tick+1)/60
            for sid in g.owned:g.observe(sid,dict(position=last[ids.index(sid)]),now)
            last=g.apply(data[tick],last,1/60,now)
        receiver.target=None;maximum_step={sid:0 for sid in g.owned}
        for tick in range(600):
            now=13+(tick+1)/60
            for sid in g.owned:g.observe(sid,dict(position=last[ids.index(sid)]),now)
            old=last.copy();last=g.apply(data[0],last,1/60,now)
            for sid in g.owned:maximum_step[sid]=max(maximum_step[sid],abs(last[ids.index(sid)]-old[ids.index(sid)]))
        for sid in g.owned:
            self.assertAlmostEqual(last[ids.index(sid)],2047,delta=1.)
            speed=next(m['max_speed_counts_s'] for m in p['motors'] if m['id']==sid)
            self.assertLessEqual(maximum_step[sid],speed/60+1.)

    def test_stale_encoder_feedback_brakes_pitch_roll_without_invented_head_pose(self):
        robot,c,p,ids,data,receiver,g,last=controller()
        for tick in range(120):
            now=10+(tick+1)/60
            for sid in g.owned:g.observe(sid,dict(position=last[ids.index(sid)]),now)
            last=g.apply(data[0],last,1/60,now)
        for tick in range(120):last=g.apply(data[0],last,1/60,12+(tick+1)/60)
        self.assertEqual(g.platform.diagnostic['reason'],'feedback_stale')
        self.assertEqual(g.platform.diagnostic['requested_velocity'],{33:0.,34:0.})
        self.assertAlmostEqual(g.platform.reference.planners[33].velocity,0.,delta=1e-6)

    def test_fractional_eye_observations_accumulate_before_motion_threshold(self):
        robot,c,p,ids,data,receiver,g,last=controller();f=g.platform
        for tick in range(1,5):f.observe(22,2047+tick*.25,10+tick*.02)
        self.assertEqual(f.eye_motion,1.);self.assertAlmostEqual(f.eye_motion_at,10.08)

    def test_home_only_and_yaw_planner_off_remain_available(self):
        from vivis.runtime.developer import run_example
        for options in (['--home-only'],['--upstream-trajectory','off']):
            with redirect_stdout(io.StringIO()) as output:
                run_example('eyes-neck',['--robot',str(CONFIG),'--work','0001','--check',*options])
            result=json.loads(output.getvalue());self.assertFalse(result['upstream_trajectory']['enabled'])
            if '--home-only' in options:self.assertFalse(result['camera_required'])
            else:self.assertTrue(result['independent_neck'])

    def test_reversal_respects_independent_position_velocity_acceleration_and_jerk_limits(self):
        robot,c,p,ids,data,receiver,g,last=controller()
        for tick in range(900):
            sign=1 if tick%180<90 else -1;receiver.target=dict(x=sign*.8,y=sign*.8,eye_line_roll_rad=sign*.5)
            now=10+(tick+1)/60
            for sid in g.owned:g.observe(sid,dict(position=last[ids.index(sid)]),now)
            acceleration={sid:plan.acceleration for sid,plan in g.platform.reference.planners.items()}
            last=g.apply(data[0],last,1/60,now)
            for sid,plan in g.platform.reference.planners.items():
                self.assertTrue(plan.low-1e-7<=plan.position<=plan.high+1e-7)
                self.assertLessEqual(abs(plan.velocity),plan.maximum_speed+1e-6)
                self.assertLessEqual(abs(plan.acceleration),plan.maximum_acceleration+1e-6)
                self.assertLessEqual(abs(plan.acceleration-acceleration[sid])*60,plan.maximum_jerk+1e-4)

    def test_wrong_model_topology_direction_or_camera_is_rejected(self):
        robot,c,p,ids,data=fixture()
        for kind in ('model','direction','parallel','camera'):
            cc=copy.deepcopy(c);pp=copy.deepcopy(p)
            if kind=='model':next(m for m in pp['motors'] if m['id']==33)['model_raw_expected']=10760
            if kind=='direction':cc['independent_neck']['pitch']['sign']=0
            if kind=='parallel':cc['parallel_neck']={'enabled':True}
            if kind=='camera':cc['control']['camera_mount']='eye'
            with self.assertRaises(ValueError):validate(cc,pp)

    def test_tracking_span_is_not_the_device_feedback_boundary(self):
        from vivis.runtime.playback import check_feedback
        robot,c,p,ids,data=fixture();motors={m['id']:m for m in p['motors']}
        for sid in p['gaze_owner_ids']:
            m=motors[sid]
            self.assertEqual((m['feedback_min'],m['feedback_max']),(m['device_min'],m['device_max']))
            state=dict(position=m['feedback_max'],temperature_c=30,load_raw=0,voltage_raw=235 if sid>=32 else 60,status_register=0)
            check_feedback(sid,state,m)
            for outside in (m['feedback_min']-1,m['feedback_max']+1):
                with self.assertRaisesRegex(RuntimeError,'outside safe range'):check_feedback(sid,{**state,'position':outside},m)
        self.assertEqual((motors[33]['safe_min'],motors[33]['safe_max']),(1987,2107))
        self.assertNotIn('feedback_min',motors[31])

    def test_actual_timed_player_supports_all_independent_axes_with_one_bus_owner(self):
        self.run_timed_player()

    def test_sample2113_continues_without_expanding_command_span(self):
        result=self.run_timed_player(feedback33=2113)
        samples=result['feedback_outside_command_range_samples']
        self.assertTrue(any(s['id']==33 and s['position']==2113 for s in samples))
        self.assertTrue(all(1987<=e['goals'][33]<=2107 for e in result['events']))

    def test_sample_beyond_device_range_still_stops_the_player(self):
        with self.assertRaisesRegex(RuntimeError,'outside safe range'):self.run_timed_player(feedback33=2301)

    def run_timed_player(self,feedback33=None):
        import test_runtime
        from vivis.runtime.playback import play
        robot,c,p,ids,data,receiver,g,last=controller()
        p['changed_goals_only']=False  # Sparse SDK packet accounting has its own transport tests.
        _,bus,clock=test_runtime.PlaybackTests().fixture();clock.duration=.5
        positions={sid:int(round(last[j])) for j,sid in enumerate(ids)}
        rows={s['id']:s for s in snapshots(p,positions)}
        playing=[False];start=clock.start
        def begin():playing[0]=True;start()
        clock.start=begin
        def state(sid):return dict(position=positions[sid],temperature_c=30,load_raw=0,status_register=0,voltage_raw=rows[sid]['voltage_raw'])
        bus.snapshot.side_effect=lambda sid:{**rows[sid],'position_raw':positions[sid]}
        bus.feedback.side_effect=state
        def read(sid,address,count):
            self.assertNotEqual(sid,29)
            position=feedback33 if sid==33 and playing[0] and feedback33 is not None else positions[sid]
            return {56:position.to_bytes(2,'little'),60:b'\0\0',62:bytes([rows[sid]['voltage_raw'],30]),65:b'\0'}[address]
        bus.read.side_effect=read;bus.goals.side_effect=lambda goals,speeds:positions.update(goals)
        with tempfile.TemporaryDirectory() as folder, \
             patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus), \
             patch('vivis.runtime.playback.AudioClock',return_value=clock):
            result=play(p,ids,data[:30],60,'fake',Path(folder)/'log.json',Path(folder)/'STOP',gaze_override=g)
        self.assertTrue(result['completed']);self.assertEqual(set(result['command_ids']),set(ids))
        self.assertEqual(set(result['runtime_feedback_schedule']['priority_ids']),{22,23,32,33,34})
        self.assertTrue(all(set(event['goals'])==set(ids) for event in result['events']))
        self.assertTrue(any(event['person_gaze']['independent_neck']['enabled'] for event in result['events']))
        return result


if __name__=='__main__':unittest.main()
