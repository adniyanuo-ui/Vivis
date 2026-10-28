"""Task speed preset must reach profile, startup and bus without other tuning."""
import copy,io,json,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
import test_mouth_commands as helpers
from vivis.runtime.robot import Robot
from vivis.runtime.startup import startup_plan
from vivis.runtime.playback import play


class MouthSpeedTests(unittest.TestCase):
    def test_override_changes_only_task_mouth_speed_and_preserves_canonical_file(self):
        robot=Robot();path=robot.resolve('hardware');raw=path.read_bytes();before=copy.deepcopy(robot.axes)
        tracking=robot.tracking();home=robot.home();report=robot.set_mouth_speed(1500)
        mouth=set(report['axes']);self.assertEqual(len(mouth),14)
        for sid,axis in robot.axes.items():
            expected=copy.deepcopy(before[sid])
            if sid in mouth:expected['max_speed_counts_s']=1500
            self.assertEqual(axis,expected)
        self.assertEqual(path.read_bytes(),raw);self.assertEqual(robot.home(),home);self.assertEqual(robot.tracking(),tracking)
        self.assertEqual({m['id']:m['max_speed_counts_s'] for m in robot.motion_profile()['motors'] if m['id'] in mouth},dict.fromkeys(mouth,1500))
        self.assertEqual(robot.motion_profile()['mouth_speed_override'],report)
        self.assertTrue(all(Robot().axes[sid]['max_speed_counts_s']==1200 for sid in mouth))
        for speed in (0,-1,32768,True,1500.5,float('nan')):
            with self.assertRaises(ValueError):Robot().set_mouth_speed(speed)

    def test_startup_restoration_and_homing_map_use_same_task_speeds(self):
        robot=Robot();robot.set_mouth_speed(1500);preset=robot.startup();samples=[]
        for sid,a in robot.axes.items():
            config=bytearray(40);config[18]=4
            if 'pid_raw' in a:config[21:24]=bytes(a['pid_raw'])
            if 'deadband_raw' in a:config[26:28]=bytes(a['deadband_raw'])
            ram=bytearray(31);ram[1]=a.get('acceleration_raw',0);ram[6:8]=(1200).to_bytes(2,'little')
            samples.append(dict(id=sid,model_raw=a['model_raw_expected'],mode=0,status_register=0,
                configured_min=a.get('device_min',0),configured_max=a.get('device_max',4095),
                config_0_39_hex=config.hex(),state_40_70_hex=ram.hex()))
        speeds,writes=startup_plan(preset,samples,60)
        mouth=robot.motion_profile()['mouth_motion_ids']
        self.assertTrue(all(speeds[sid]==1500 for sid in mouth))
        self.assertTrue(all(w['address']==46 and w['target']==60 for w in writes))
        self.assertEqual(next(a for a in preset['axes'] if a['id']==32)['speed_raw'],600)
        self.assertTrue(all(next(a for a in preset['axes'] if a['id']==sid)['acceleration_raw']==0 for sid in mouth))

    def test_cli_checks_propagate_speed_and_keep_old_defaults(self):
        from vivis.runtime.developer import run_example
        with patch('vivis.hardware.feetech.sdk.OfficialSDKBus',side_effect=AssertionError('serial opened')), \
             patch('vivis.runtime.developer.subprocess.Popen',side_effect=AssertionError('camera started')):
            for mode in ('face','eyes','eyes-neck'):
                with redirect_stdout(io.StringIO()) as output:
                    run_example(mode,['--check','--mouth-command-mode','lips-direct','--mouth-speed','1500'])
                report=json.loads(output.getvalue());self.assertEqual(report['mouth_speed_override']['speed_counts_s'],1500)
                self.assertEqual(report['mouth_gain_report']['gain'],1.)
            with redirect_stdout(io.StringIO()) as output:run_example('face',['--check'])
            self.assertIsNone(json.loads(output.getvalue())['mouth_speed_override'])

    def test_music_bus_receives_1500_and_other_axis_speed_is_unchanged(self):
        profile,bus,clock=helpers.MouthCommandTests().fixture(True)
        next(m for m in profile['motors'] if m['id']==12)['max_speed_counts_s']=1500
        trajectory=np.full((9,2),2200.);trajectory[0]=2047.
        with tempfile.TemporaryDirectory() as folder,patch('vivis.runtime.playback.Bus',return_value=bus), \
             patch('vivis.runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2,12],trajectory,60,'fake',Path(folder)/'log.json',Path(folder)/'STOP')
        self.assertTrue(result['completed'])
        for call in bus.goals.call_args_list:self.assertEqual(call.args[1],{2:300,12:1500})
