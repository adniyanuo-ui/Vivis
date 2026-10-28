"""Optional song-only direct mouth goals; hardware speed and ownership stay fixed."""
import copy,io,json,tempfile,time,unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
from vivis.runtime.robot import Robot
from vivis.runtime.show import prepare
from vivis.runtime.playback import play
import test_runtime


class MouthCommandTests(unittest.TestCase):
    def test_mode_changes_delivery_only_and_derives_all_mouth_axes_from_calibration(self):
        robot=Robot();config=robot.tracking();profile=robot.profile('wz2');before=copy.deepcopy(profile)
        ids=profile['diagnostic_command_ids'];positions=np.full((3,len(ids)),2047.)
        positions[:,ids.index(31)]=1900.
        limited,monitored,a=prepare(config,profile,ids,positions,True,1.,robot.telemetry(),'face')
        direct,other,b=prepare(config,profile,ids,positions,True,1.,robot.telemetry(),'face','lips-direct')
        self.assertEqual(monitored,other);np.testing.assert_array_equal(a,b)
        self.assertEqual(profile,before)
        self.assertEqual(limited['direct_position_ids'],[])
        self.assertEqual(direct['direct_position_ids'],[10,11,12,13,15,25,26,27,28,30])
        self.assertEqual(direct['motors'],limited['motors'])
        self.assertTrue(all(m['max_speed_counts_s']==1200 for m in direct['motors'] if m['id'] in direct['direct_position_ids']))
        with self.assertRaises(ValueError):prepare(config,profile,ids,positions,mode='face',mouth_command_mode='invalid')

    def test_cli_offline_checks_propagate_mode_in_all_three_examples(self):
        from vivis.runtime.developer import run_example
        with patch('vivis.hardware.feetech.sdk.OfficialSDKBus',side_effect=AssertionError('serial opened')), \
             patch('vivis.runtime.developer.subprocess.Popen',side_effect=AssertionError('camera started')):
            for mode in ('face','eyes','eyes-neck'):
                with redirect_stdout(io.StringIO()) as output:
                    run_example(mode,['--check','--work','wz2','--mouth-command-mode','lips-direct'])
                report=json.loads(output.getvalue())
                self.assertEqual(report['mouth_command_mode'],'lips-direct')
                self.assertEqual(len(report['direct_position_ids']),10)
                self.assertFalse(set(report['direct_position_ids'])&set(report['gaze_owner_ids']))

    def fixture(self,direct=False):
        profile,bus,clock=test_runtime.PlaybackTests().fixture()
        mouth=copy.deepcopy(profile['motors'][0]);mouth.update(id=12,max_speed_counts_s=1200)
        profile['motors'].append(mouth)
        profile.update(mouth_motion_ids=[12],round_lip_motion_ids=[12],direct_position_ids=[12] if direct else [],
                       mouth_command_mode='lips-direct' if direct else 'limited')
        positions={2:2047,12:2047}
        bus.snapshot.side_effect=lambda sid:dict(model_raw=123,mode=0,torque_enable=1,configured_min=0,
            configured_max=4095,position_raw=positions[sid],temperature_c=25,load_raw=0,status_register=0)
        bus.feedback.side_effect=lambda sid:dict(position=positions[sid],temperature_c=25,load_raw=0,status_register=0)
        bus.goals.side_effect=lambda goals,speeds:positions.update(goals)
        clock.duration=.15
        return profile,bus,clock

    def test_direct_song_goal_is_not_preclipped_and_hardware_speed_stays_1200(self):
        outputs={}
        for direct in (False,True):
            profile,bus,clock=self.fixture(direct)
            trajectory=np.full((9,2),2250.);trajectory[0]=2047.
            with tempfile.TemporaryDirectory() as directory,patch('vivis.runtime.playback.Bus',return_value=bus), \
                 patch('vivis.runtime.playback.AudioClock',return_value=clock):
                result=play(profile,[2,12],trajectory,60,'fake',Path(directory)/'log.json',Path(directory)/'STOP')
            self.assertTrue(result['completed'])
            first=next(e for e in result['events'] if e['frame']>0)
            outputs[direct]=first['goals'][12]
            for e in result['events']:
                if direct:self.assertEqual(e['goals'][12],round(trajectory[e['frame'],1]))
            for a,b in zip(result['events'],result['events'][1:]):
                self.assertLessEqual(abs(b['goals'][2]-a['goals'][2]),300*(b['sent_monotonic']-a['sent_monotonic'])+1.)
            for call in bus.goals.call_args_list:self.assertEqual(call.args[1][12],1200)
        self.assertEqual(outputs[True],2250)
        self.assertLess(outputs[False],2250)

    def test_direct_mode_keeps_initial_approach_and_rejects_bad_ownership_or_limits(self):
        profile,bus,clock=self.fixture(True);trajectory=np.full((9,2),2250.);trajectory[0]=2100.
        with tempfile.TemporaryDirectory() as directory,patch('vivis.runtime.playback.Bus',return_value=bus), \
             patch('vivis.runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2,12],trajectory,60,'fake',Path(directory)/'log.json',Path(directory)/'STOP')
        self.assertGreater(result['initial_approach_commands'],1)
        calls=bus.goals.call_args_list[:result['initial_approach_commands']]
        self.assertTrue(all(call.args[0][12]<=2100 for call in calls))
        for bad in ('ownership','limit','tracked','mode','group'):
            profile,_,_=self.fixture(True);trajectory=np.full((3,2),2047.)
            if bad=='ownership':profile['direct_position_ids']=[2]
            elif bad=='tracked':profile['gaze_owner_ids']=[12]
            elif bad=='mode':profile['mouth_command_mode']='limited'
            elif bad=='group':profile['round_lip_motion_ids']=[2]
            else:trajectory[1,1]=2301.
            with tempfile.TemporaryDirectory() as directory,patch('vivis.runtime.playback.Bus') as hardware, \
                 patch('vivis.runtime.playback.AudioClock') as audio:
                with self.assertRaises(ValueError):play(profile,[2,12],trajectory,60,'fake',Path(directory)/'log.json',Path(directory)/'STOP')
                hardware.assert_not_called();audio.assert_not_called()
