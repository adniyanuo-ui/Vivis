"""Mojia calibration regressions without serial IO or song media."""
import unittest
import copy
import tempfile
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from prepare_mojia import MOJIA_IDS, proposal, compare_calibrations, hardware_profile
from mojia_player import startup_target, arm
from xiaobai_runtime.mapping import ROOT, inputs, compile_trajectory, live_preflight
from xiaobai_runtime.playback import check_feedback
from inspect_mojia_dynamics import decode


class MojiaMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip=ROOT/'tests/fixtures/blendshapes61.json'
        cls.header=ROOT/'xiaobai/blendshape_channels_61.csv'
        cls.calibration=ROOT/'mojia/墨甲9.23.json'
        cls.values,cls.fps,cls.order,cls.shapes=inputs(
            cls.clip,cls.calibration,cls.header,allowed_motor_ids=MOJIA_IDS)

    def test_xiaobai_reader_still_rejects_new_ids_by_default(self):
        with self.assertRaisesRegex(ValueError,'Bad calibration endpoint'):
            inputs(self.clip,self.calibration,self.header)

    def compile(self,weights,policy='all_calibrated'):
        return compile_trajectory(weights,self.order,self.shapes,
            proposal(self.shapes,2047,policy),'face_eyes',draft=True)

    def test_new_axes_and_tongue_not_dropped_and_cheeks_not_commanded(self):
        values=np.zeros((4,61))
        for row,name in enumerate(['JawForward','JawLeft','TongueOut'],1):
            values[row,self.order.index(name)]=1
        ids,positions,_=self.compile(values)
        self.assertTrue({29,35,36}<=set(ids))
        self.assertTrue(set(ids).isdisjoint({9,24,32,33,34}))
        np.testing.assert_array_equal(positions[0],np.full(len(ids),2047))
        self.assertEqual(positions[1,ids.index(35)],2247)
        self.assertEqual(positions[2,ids.index(36)],2089)
        self.assertEqual(positions[3,ids.index(29)],1447)
        self.assertEqual(positions[1,ids.index(31)],1547)

    def test_jaw_policy_explicitly_controls_shared_motor(self):
        values=np.zeros((1,61));values[0,self.order.index('JawForward')]=1
        ids,all_shapes,_=self.compile(values)
        _,jaw_open_only,_=self.compile(values,'JawOpen')
        self.assertEqual(all_shapes[0,ids.index(31)],1547)
        self.assertEqual(jaw_open_only[0,ids.index(31)],2047)
        self.assertEqual(jaw_open_only[0,ids.index(35)],2247)

    def test_reversed_blink_direction_and_bounded_combination(self):
        values=np.zeros((1,61));values[0,self.order.index('EyeBlinkLeft')]=1
        ids,positions,_=self.compile(values)
        self.assertEqual(positions[0,ids.index(5)],2320)
        ids,positions,_=self.compile(np.ones((1,61)))
        motors={m['id']:m for m in proposal(self.shapes,2047,'all_calibrated')['motors']}
        for sid,position in zip(ids,positions[0]):
            low,high=motors[sid]['endpoint_envelope_not_safe_limits']
            self.assertTrue(low<=position<=high)

    def test_comparison_identifies_removed_and_new_mappings(self):
        *_,old=inputs(self.clip,ROOT/'xiaobai/xiaobai标定文件.json',self.header)
        diff=compare_calibrations(old,self.shapes)
        self.assertEqual(diff['newly_mapped_ids'],[29,35,36])
        self.assertEqual(diff['no_longer_mapped_ids'],[9,24])
        self.assertEqual(len(diff['jaw31_shapes']),7)


class MojiaStartupTests(unittest.TestCase):
    def test_performance_tongue_residual_36_uses_only_startup_tolerance(self):
        from mojia_performance import recover_tongue
        from unittest.mock import patch
        motor={**self.motors[29],'initial_settle_tolerance_counts':40,
               'feedback_min':0,'feedback_max':4095}
        bus=Mock();bus.snapshot.return_value={**self.states[29],'position_raw':2207}
        bus.read.return_value=b'\x01'
        bus.feedback.return_value=dict(position=2083,temperature_c=44,voltage_raw=60,load_raw=320,status_register=0)
        events=[]
        with patch('mojia_performance.time.sleep'):
            recover_tongue(bus,motor,lambda:None,events)
        self.assertEqual(len(events),2)
        for call in bus.goals.call_args_list:self.assertEqual(call.args,({29:2047},{29:900}))
        bus.feedback.return_value={**bus.feedback.return_value,'position':4096}
        with self.assertRaises(RuntimeError):recover_tongue(bus,motor,lambda:None,[])

    def setUp(self):
        *_,shapes=inputs(ROOT/'tests/fixtures/blendshapes61.json',ROOT/'mojia/墨甲9.23.json',
                        ROOT/'xiaobai/blendshape_channels_61.csv',allowed_motor_ids=MOJIA_IDS)
        self.states={sid:dict(id=sid,model_raw=521 if sid<=30 or sid==36 else 10760,
            mode=0,torque_enable=0,position_raw=2047,configured_min=0,configured_max=4095,
            configured_max_temperature=80,configured_min_voltage_raw=40,configured_max_voltage_raw=254,
            temperature_c=30,voltage_raw=60,load_raw=0,status_register=0) for sid in range(2,37)}
        self.states[35].update(configured_min=2040,configured_max=2247)
        self.scan=dict(port='mock',servos=list(self.states.values()))
        self.profile=hardware_profile(proposal(shapes,2047,'all_calibrated'),self.scan)
        self.motors={m['id']:m for m in self.profile['motors']}

    def test_model_groups_include_35_as_sm45_and_36_as_sts(self):
        self.assertEqual(self.motors[35]['model_raw_expected'],10760)
        self.assertEqual(self.motors[36]['model_raw_expected'],521)
        self.assertEqual(self.motors[35]['safe_min'],2047)
        self.assertEqual(self.motors[35]['device_min'],2040)

    def test_incomplete_scan_or_wrong_model_cannot_make_profile(self):
        for mutate in [lambda s:s['servos'].pop(),lambda s:s['servos'][-1].update(model_raw=10760)]:
            scan=copy.deepcopy(self.scan);mutate(scan)
            with self.assertRaises(ValueError):hardware_profile(dict(motors=[]),scan)

    def test_no_write_when_any_axis_is_outside_device_limits(self):
        self.states[35]['position_raw']=1969
        bus=Mock();bus.snapshot.side_effect=lambda sid:self.states[sid]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'ID35'):
                arm(bus,self.profile,Path(directory),lambda:None)
        bus.goals.assert_not_called();bus.sdk.write1ByteTxRx.assert_not_called()

    def test_face_recovery_cannot_exceed_ten_counts(self):
        state={**self.states[29],'position_raw':2060}
        with self.assertRaisesRegex(ValueError,'bounded policy'):
            startup_target(state,self.motors[29])

    def test_neck_can_only_prime_inside_device_limits(self):
        state={**self.states[33],'position_raw':2141}
        self.assertEqual(startup_target(state,self.motors[33],True),2141)
        state['position_raw']=4096
        with self.assertRaises(ValueError):startup_target(state,self.motors[33],True)

    def test_changed_limits_require_new_profile(self):
        state={**self.states[35],'configured_max':2250}
        with self.assertRaisesRegex(ValueError,'limits changed'):
            startup_target(state,self.motors[35])

    def test_explicit_disabled_recovery_keeps_target_in_limits(self):
        state={**self.states[35],'position_raw':1969}
        target=startup_target(state,self.motors[35],recover_disabled=True)
        self.assertEqual(target,2047)
        self.assertTrue(self.motors[35]['device_min']<=target<=self.motors[35]['device_max'])

    def test_recovery_cannot_ignore_enabled_axes_or_other_ids(self):
        for sid,changes in [(35,dict(torque_enable=1)),(31,{}),(35,dict(position_raw=5000))]:
            with self.subTest(sid=sid,changes=changes),self.assertRaises(ValueError):
                startup_target({**self.states[sid],**changes},self.motors[sid],recover_disabled=True)

    def test_enabled_axis_can_keep_existing_feedback_tolerance(self):
        state={**self.states[35],'position_raw':2042,'torque_enable':1}
        self.assertEqual(startup_target(state,self.motors[35]),2047)
        state['position_raw']=2039
        with self.assertRaises(ValueError):startup_target(state,self.motors[35])

    def test_feedback_limits_are_separate_from_calibrated_target_envelope(self):
        motor={**self.motors[29],'device_max':2147,'feedback_max':2147}
        state=dict(position=2053,load_raw=0,temperature_c=30,voltage_raw=60,status_register=0)
        self.assertEqual(motor['safe_max'],2047)
        check_feedback(29,state,motor)
        with self.assertRaises(RuntimeError):check_feedback(29,{**state,'position':2148},motor)

    def test_feedback_range_cannot_exceed_device_limits(self):
        self.motors[29]['feedback_max']=4096
        with self.assertRaisesRegex(ValueError,'feedback_limits'):
            live_preflight(self.profile,self.profile['diagnostic_command_ids'])

    def test_initial_settle_tolerance_is_bounded(self):
        self.assertEqual(self.motors[31]['initial_settle_tolerance_counts'],32)
        live_preflight(self.profile,self.profile['diagnostic_command_ids'])
        self.motors[31]['initial_settle_tolerance_counts']=41
        with self.assertRaisesRegex(ValueError,'initial_settle_tolerance'):
            live_preflight(self.profile,self.profile['diagnostic_command_ids'])


class DynamicsRegisterTests(unittest.TestCase):
    def state(self,phase=44,acc=0,speed=300):
        config=bytearray(40);config[0:2]=bytes([2,54]);config[18]=phase
        ram=bytearray(31);ram[1]=acc;ram[6:8]=speed.to_bytes(2,'little')
        return dict(id=31,model_raw=10760,torque_enable=1,position_raw=2047,
                    config_0_39_hex=config.hex(),state_40_70_hex=ram.hex())

    def test_speed_units_follow_phase_bit(self):
        self.assertEqual(decode(self.state(phase=44))['goal_speed_limit_counts_s_nominal'],300)
        self.assertEqual(decode(self.state(phase=40))['goal_speed_limit_counts_s_nominal'],15000)

    def test_zero_acceleration_is_not_zero_physical_acceleration(self):
        row=decode(self.state(),bytes([0,0,0,0,150,100,2]))
        self.assertTrue(row['acceleration_zero_special'])
        self.assertIsNone(row['acceleration_counts_s2_nominal'])
        self.assertEqual(row['acceleration_multiplier_raw_86'],2)
        self.assertEqual(row['factory_max_speed_counts_s_nominal'],7500)

    def test_positive_acceleration_and_missing_factory_data_stay_distinct(self):
        row=decode(self.state(acc=10))
        self.assertEqual(row['acceleration_counts_s2_nominal'],1000)
        self.assertIsNone(row['factory_acceleration_limit_raw_85'])

    def test_zero_speed_uses_phase_semantics(self):
        row=decode(self.state(phase=44,speed=0))
        self.assertEqual(row['goal_speed_zero_mode'],'factory_maximum')
        self.assertIsNone(row['goal_speed_limit_counts_s_nominal'])
        row=decode(self.state(phase=36,speed=0))
        self.assertEqual(row['goal_speed_zero_mode'],'stop')
        self.assertEqual(row['goal_speed_limit_counts_s_nominal'],0)


if __name__=='__main__':unittest.main()
