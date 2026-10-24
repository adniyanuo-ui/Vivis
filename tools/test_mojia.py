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


class MojiaWorkTests(unittest.TestCase):
    def test_expression_gain_scales_only_brows_and_nose_with_existing_limits(self):
        import json
        from prepare_mojia_work import scale_expression_targets
        profile=json.loads((ROOT/'mojia/0001_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1200_PROFILE.json').read_text())
        *_,shapes=inputs(ROOT/'tests/fixtures/blendshapes61.json',ROOT/'mojia/墨甲9.23.json',
                       ROOT/'xiaobai/blendshape_channels_61.csv',allowed_motor_ids=MOJIA_IDS)
        ids=[m['id'] for m in profile['motors']];motors={m['id']:m for m in profile['motors']}
        selected={2,3,4,14,16,17,18,19};positions=np.full((3,len(ids)),2047.)
        for sid in selected:
            positions[1,ids.index(sid)]+=20 if motors[sid]['safe_max']>=2067 else -20
        positions[2,ids.index(2)]=motors[2]['safe_max']
        positions[:,ids.index(12)]=1900.;positions[:,ids.index(5)]=2100.;before=positions.copy()
        result,report=scale_expression_targets(ids,positions,profile,shapes,1.5)
        self.assertEqual({a['id'] for a in report['axes']},selected)
        for j,sid in enumerate(ids):
            if sid in selected:
                self.assertEqual(result[1,j],2047+1.5*(positions[1,j]-2047))
                self.assertTrue(motors[sid]['safe_min']<=result[:,j].min()<=result[:,j].max()<=motors[sid]['safe_max'])
            else:np.testing.assert_array_equal(result[:,j],positions[:,j])
        self.assertEqual(result[2,ids.index(2)],motors[2]['safe_max'])
        np.testing.assert_array_equal(positions,before)
        self.assertIn('CheekPuff',report['unmapped_cheek_shapes'])

    def test_expression_gain_rejects_invalid_gain_and_shared_motor_roles(self):
        from prepare_mojia_work import scale_expression_targets
        ids=[2];positions=np.array([[2047.]])
        profile=dict(motors=[dict(id=2,neutral=2047,safe_min=1900,safe_max=2300,device_min=1800,device_max=2400)])
        shapes=dict(BrowDownLeft={2:2300})
        for gain in (0,-1,True,float('inf'),float('nan')):
            with self.assertRaises(ValueError):scale_expression_targets(ids,positions,profile,shapes,gain)
        shapes['MouthSmileLeft']={2:2100}
        with self.assertRaisesRegex(ValueError,'shared'):
            scale_expression_targets(ids,positions,profile,shapes,1.5)

    def test_native_30_fps_times_preserved_at_60_hz(self):
        from prepare_mojia_work import hold_at_60
        values=np.zeros((3,61));values[:,17]=[0.,.2,.4]
        held=hold_at_60(values,30)
        self.assertEqual(held.shape,(6,61))
        for i in range(3):
            np.testing.assert_array_equal(held[i*2],values[i])
            np.testing.assert_array_equal(held[i*2+1],values[i])
        np.testing.assert_array_equal(hold_at_60(values,60),values)
        with self.assertRaises(ValueError):hold_at_60(values,25)

    def test_audio_alignment_preserves_pcm_and_refuses_trim_or_large_padding(self):
        import wave
        from prepare_mojia_work import align_audio
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'decoded.wav';out=Path(directory)/'aligned.wav'
            pcm=np.arange(1000,dtype='<i2').tobytes()
            with wave.open(str(source),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(pcm)
            report=align_audio(source,out,4)
            with wave.open(str(out)) as wav:result=wav.readframes(wav.getnframes())
            self.assertEqual(result[:len(pcm)],pcm)
            self.assertFalse(any(result[len(pcm):]))
            self.assertAlmostEqual(report['duration_seconds'],4/60,delta=1/16000)
            with self.assertRaises(ValueError):align_audio(source,Path(directory)/'short.wav',1)
            with self.assertRaises(ValueError):align_audio(source,Path(directory)/'long.wav',60)

    def test_work_export_binds_its_own_media_and_keeps_reference_profile(self):
        import io,json,sys,wave
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from prepare_mojia_work import main
        from mojia_player import load_targets
        reference=ROOT/'mojia/0001_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1200_PROFILE.json'
        before=reference.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.json';video=root/'video.mp4';audio=root/'decoded.wav';prefix=root/'second'
            rows=np.zeros((2,61));rows[1,17]=.3
            source.write_text(json.dumps(dict(blendShapes=rows.tolist(),fps=dict(num=30,den=1),name='test')))
            video.write_bytes(b'fixture-provenance')
            with wave.open(str(audio),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(bytes(1000*2))
            argv=['prepare_mojia_work','--clip',str(source),'--video',str(video),'--decoded-audio',str(audio),'--reference-profile',str(reference),'--prefix',str(prefix)]
            with patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):main()
            p=json.loads((root/'second_MOJIA_MOTOR_TARGETS_60fps_PROFILE.json').read_text())
            ids,positions,fps=load_targets(root/'second_MOJIA_MOTOR_TARGETS_60fps.json',p,root/'second.wav',ROOT/'mojia/墨甲9.23.json')
            self.assertEqual(fps,60)
            self.assertEqual(len(positions),4)
            np.testing.assert_array_equal(positions[0],positions[1])
            np.testing.assert_array_equal(positions[2],positions[3])
            self.assertNotIn('blink_trial',p)
            self.assertEqual(p['work_provenance']['original_frames'],2)
            self.assertEqual(reference.read_bytes(),before)


class MojiaBlinkTests(unittest.TestCase):
    def fixture(self):
        ids = [5, 6, 20, 21, 8, 31]
        positions = np.full((360, len(ids)), 2047.)
        positions[:, -1] = 2000 + np.arange(360) % 20
        payload = dict(motor_ids=ids, positions=positions.tolist(), frames=360, fps=dict(num=60, den=1))
        endpoints = {5:2320, 6:2150, 20:1803, 21:1943}
        profile = dict(motors=[dict(id=sid, neutral=2047, safe_min=min(2047, end),
            safe_max=max(2047, end), device_min=min(2047, end), device_max=max(2047, end))
            for sid, end in endpoints.items()])
        shapes = dict(EyeBlinkLeft={5:2320, 6:2150}, EyeBlinkRight={20:1803, 21:1943})
        weights = np.zeros((360, 2)); weights[60:64] = 1.
        return payload, profile, weights, ['EyeBlinkLeft', 'EyeBlinkRight'], shapes

    def test_blink_variant_preserves_other_axes_and_reaches_original_endpoints(self):
        from adjust_mojia_blinks import adjust_blinks
        payload, profile, weights, order, shapes = self.fixture()
        before = copy.deepcopy(payload)
        output, report = adjust_blinks(payload, profile, weights, order, shapes)
        self.assertEqual(payload, before)
        x = np.asarray(output['positions']); source = np.asarray(payload['positions'])
        np.testing.assert_array_equal(x[:, 4:], source[:, 4:])
        self.assertEqual(len(report['events']), 2)
        for event in report['events']:
            for sid, endpoint in event['endpoints'].items():
                j = payload['motor_ids'].index(int(sid))
                self.assertEqual(x[event['closed_start_frame'], j], endpoint)
                np.testing.assert_array_equal(x[event['closed_start_frame']:event['closed_end_frame'], j], endpoint)
                m = next(m for m in profile['motors'] if m['id'] == int(sid))
                self.assertTrue(m['safe_min'] <= x[:, j].min() <= x[:, j].max() <= m['safe_max'])
        self.assertFalse(report['fixed_endpoint_dwell'])
        self.assertTrue(all(e['duration_s'] < .6 for e in report['events']))
        for e in report['events']:
            self.assertGreater(e['opening_s'], e['closing_s'])
            self.assertLessEqual(e['command_plateau_s'], 1/60)

    def test_reduced_schedule_removes_original_blinks_from_json(self):
        from adjust_mojia_blinks import adjust_blinks, runs
        payload, profile, weights, order, shapes = self.fixture()
        weights[:] = 0.
        source = np.asarray(payload['positions']); endpoints = [2320, 2150, 1803, 1943]
        for peak in range(30, 275, 35):
            weights[peak:peak+4] = 1.
            source[peak:peak+4, :4] = endpoints
        payload['positions'] = source.tolist()
        output, report = adjust_blinks(payload, profile, weights, order, shapes,
                                      audio_energy=np.full(360, .5), keep_fraction=.7)
        self.assertEqual(report['frequency']['source_events'], 7)
        self.assertEqual(report['frequency']['selected_events'], 5)
        result = np.asarray(output['positions'])
        self.assertEqual(len(runs(result[:, 0] >= 2319)), 5)
        np.testing.assert_array_equal(result[:, 4:], source[:, 4:])
        removed = [c for c in report['frequency']['candidates'] if not c['selected']]
        for c in removed:
            np.testing.assert_array_equal(result[c['begin']:c['end'], :4], 2047.)

    def test_audio_context_changes_duration_without_random_timing_or_moving_peak(self):
        from adjust_mojia_blinks import adjust_blinks
        payload, profile, weights, order, shapes = self.fixture()
        weights[180:184] = 1.
        energy = np.ones(360); energy[150:215] = 0.
        output, report = adjust_blinks(payload, profile, weights, order, shapes, audio_energy=energy)
        events = [e for e in report['events'] if e['side'] == 'Left']
        self.assertEqual([e['source_peak_frame'] for e in events], [60, 180])
        self.assertLess(events[0]['duration_s'], events[1]['duration_s'])
        self.assertLess(events[0]['opening_s'], events[1]['opening_s'])
        repeat, again = adjust_blinks(payload, profile, weights, order, shapes, audio_energy=energy)
        self.assertEqual(output, repeat)
        self.assertEqual(report, again)
        with self.assertRaisesRegex(ValueError, 'audio energy'):
            adjust_blinks(payload, profile, weights, order, shapes, audio_energy=energy[:-1])

    def test_expression_changes_release_with_the_same_audio(self):
        from adjust_mojia_blinks import adjust_blinks
        payload, profile, weights, order, shapes = self.fixture()
        weights[180:184] = 1.
        order = order + ['BrowInnerUp', 'MouthFrownLeft', 'MouthFrownRight']
        weights = np.column_stack([weights, np.zeros((360, 3))])
        weights[160:210, 2:] = .8
        _, report = adjust_blinks(payload, profile, weights, order, shapes, audio_energy=np.full(360, .5))
        events = [e for e in report['events'] if e['side'] == 'Left']
        self.assertLess(events[0]['opening_s'], events[1]['opening_s'])

    def test_audio_envelope_preserves_opposite_phase_stereo_energy(self):
        import wave
        from adjust_mojia_blinks import audio_energy_from_wav
        rate = 16000; signal = np.sin(np.arange(rate)*2*np.pi*220/rate)
        signal[:rate//2] *= .01; signal[rate//2:] *= .5
        pcm = (np.column_stack([signal, -signal])*32767).astype('<i2')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stereo.wav'
            with wave.open(str(path), 'wb') as wav:
                wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(rate); wav.writeframes(pcm.tobytes())
            energy = audio_energy_from_wav(path, 60, 60)
        self.assertGreater(float(energy[40:50].mean()), float(energy[10:20].mean())+.5)

    def test_two_blinks_with_sustained_partial_squint_are_not_merged(self):
        from adjust_mojia_blinks import adjust_blinks
        payload, profile, weights, order, shapes = self.fixture()
        weights[:] = .3; weights[60:64] = 1.; weights[180:184] = 1.
        _, report = adjust_blinks(payload, profile, weights, order, shapes)
        self.assertEqual(len(report['events']), 4)
        self.assertTrue(all(e['command_plateau_s'] < .5 for e in report['events']))

    def test_no_blinks_unchanged_and_unsafe_endpoint_rejected(self):
        from adjust_mojia_blinks import adjust_blinks
        payload, profile, weights, order, shapes = self.fixture()
        output, report = adjust_blinks(payload, profile, weights*0, order, shapes)
        self.assertEqual(output['positions'], payload['positions'])
        self.assertFalse(report['events'])
        shapes['EyeBlinkLeft'][5] += 1
        with self.assertRaisesRegex(ValueError, 'endpoint outside'):
            adjust_blinks(payload, profile, weights, order, shapes)


class MojiaStartupTests(unittest.TestCase):
    def test_mouth_gain_scales_only_mouth_displacement_without_mutating_input(self):
        from mojia_performance import mouth_gain_targets,MOUTH_IDS
        ids=list(self.motors);positions=np.full((3,len(ids)),2047.)
        for sid in MOUTH_IDS:positions[:,ids.index(sid)]=[2047,2007,1967]
        positions[:,ids.index(8)]=[2007,2017,2027];before=positions.copy()
        amplified,report=mouth_gain_targets(self.profile,ids,positions,1.5)
        for sid in MOUTH_IDS:np.testing.assert_array_equal(amplified[:,ids.index(sid)],[2047,1987,1927])
        for j,sid in enumerate(ids):
            if sid not in MOUTH_IDS:np.testing.assert_array_equal(amplified[:,j],positions[:,j])
        np.testing.assert_array_equal(positions,before)
        self.assertEqual(report['gain'],1.5)
        self.assertTrue(all(a['clipped_frames']==0 for a in report['axes']))

    def test_mouth_gain_clips_at_existing_limits_and_records_saturation(self):
        from mojia_performance import mouth_gain_targets,MOUTH_IDS
        ids=list(self.motors);positions=np.full((3,len(ids)),2047.)
        for sid in MOUTH_IDS:
            m=self.motors[sid];positions[:,ids.index(sid)]=[m['safe_min'],m['safe_max'],m['neutral']]
        amplified,report=mouth_gain_targets(self.profile,ids,positions,1.5)
        for sid in MOUTH_IDS:
            m=self.motors[sid];np.testing.assert_array_equal(amplified[:,ids.index(sid)],[m['safe_min'],m['safe_max'],m['neutral']])
        self.assertGreater(sum(a['clipped_frames'] for a in report['axes']),0)
        for gain in (0,-1,float('nan'),float('inf'),True,'1.5'):
            with self.assertRaises(ValueError):mouth_gain_targets(self.profile,ids,positions,gain)
        same,_=mouth_gain_targets(self.profile,ids,positions,1.)
        np.testing.assert_array_equal(same,positions)

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
