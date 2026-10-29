"""Debounce physical sensor alarms without blocking goals or losing raw bits."""
import copy,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import test_runtime
from vivis.runtime.robot import Robot
from vivis.runtime.telemetry import SensorFaultMonitor,ConfirmedSensorFaultError,validate_policy
from vivis.runtime.playback import check_feedback,play
from vivis.runtime.feedback import BudgetedFeedback
from vivis.hardware.feetech.sdk import OfficialSDKBus,COMM_SUCCESS,COMM_RX_CORRUPT,COMM_RX_TIMEOUT
from vivis.hardware.feetech.protocol import ServoStatusError,CorruptResponse,ReadTimeout


class SensorFaultTests(unittest.TestCase):
    def fixture(self):
        robot=Robot();policy=robot.telemetry();records=[]
        motor=next(m for m in robot.motion_profile()['motors'] if m['id']==20)
        healthy=dict(position=2047,temperature_c=41,load_raw=0,voltage_raw=60,status_register=0)
        return policy,records,motor,healthy,SensorFaultMonitor(policy,records)

    def test_each_fault_requires_four_same_axis_observations_and_records_raw_data(self):
        for changes,kind in [({'packet_status_bits':1},'voltage'),({'voltage_raw':38},'voltage'),
            ({'status_register':4},'temperature'),({'temperature_c':101},'temperature'),
            ({'status_register':8},'current'),({'status_register':32},'overload')]:
            with self.subTest(kind=kind,changes=changes):
                policy,records,motor,healthy,monitor=self.fixture();bad={**healthy,**changes};before=dict(bad)
                for _ in range(3):
                    check_feedback(20,bad,motor,defer_sensor_faults=True);monitor.observe(20,bad,motor)
                self.assertEqual(monitor.pending_ids(),[20])
                with self.assertRaisesRegex(ConfirmedSensorFaultError,kind+':4'):monitor.observe(20,bad,motor)
                self.assertEqual(records[0]['consecutive_fault_samples'],4);self.assertEqual(len(records[0]['samples']),4)
                self.assertEqual(bad,before);self.assertFalse(records[0]['recovered'])

    def test_normal_read_resets_streak_and_different_axes_or_kinds_do_not_combine(self):
        policy,records,motor,healthy,monitor=self.fixture()
        for _ in range(3):monitor.observe(20,{**healthy,'packet_status_bits':1},motor)
        monitor.observe(20,healthy,motor);self.assertTrue(records[0]['recovered']);self.assertEqual(monitor.pending_ids(),[])
        for _ in range(3):
            monitor.observe(20,{**healthy,'packet_status_bits':1},motor)
            monitor.observe(27,{**healthy,'status_register':8},motor)
        self.assertEqual(monitor.active[(20,'voltage')]['consecutive_fault_samples'],3)
        self.assertEqual(monitor.active[(27,'current')]['consecutive_fault_samples'],3)
        monitor.observe(20,healthy,motor)
        for _ in range(5):
            monitor.observe(20,{**healthy,'status_register':1},motor)
            monitor.observe(20,{**healthy,'status_register':8},motor)
        self.assertNotIn((20,'voltage'),monitor.active)

    def test_boundaries_and_warning_only_load_are_not_stop_faults(self):
        policy,records,motor,healthy,monitor=self.fixture()
        for voltage in (40,90):
            for _ in range(5):
                state={**healthy,'voltage_raw':voltage,'temperature_c':100,'load_raw':900}
                warnings=check_feedback(20,state,motor,defer_sensor_faults=True)
                self.assertEqual({w['kind'] for w in warnings},{'load','temperature'})
                monitor.observe(20,state,motor)
        self.assertEqual(records,[])

    def test_position_and_encoder_or_unknown_alarm_still_stop_immediately(self):
        policy,records,motor,healthy,monitor=self.fixture()
        for changes in ({'position':4096},{'position':1700},{'status_register':2},
                        {'packet_status_bits':16},{'status_register':3}):
            with self.subTest(changes=changes),self.assertRaises(RuntimeError):
                check_feedback(20,{**healthy,**changes},motor,defer_sensor_faults=True)
        with self.assertRaises(ServoStatusError):check_feedback(20,{**healthy,'status_register':1},motor)

    def test_policy_is_explicit_and_rejects_bad_confirmation_counts(self):
        policy,*_=self.fixture()
        for count in (1,6,True,2.5):
            with self.assertRaises(ValueError):validate_policy({**policy,'voltage_stop_consecutive_samples':count})
        for period in (0,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):validate_policy({**policy,'sensor_fault_confirmation_period_s':period})
        for count in range(2,6):validate_policy({**policy,'voltage_stop_consecutive_samples':count})

    def sdk_bus(self):
        bus=OfficialSDKBus.__new__(OfficialSDKBus);bus.sdk=Mock();bus.feedback_mode='short_registers'
        bus.servo_status_policy='defer_sensor_faults';bus.servo_status_notifications=[]
        bus.read_requests=0;bus.read_failures=[];bus.corrupt_responses=[];bus.retry_reads=False
        payload={56:b'\xff\x07',60:b'\x00\x00',62:b'\x33\x29',65:b'\x00'}
        bus.sdk.readTxRx.side_effect=lambda sid,a,n:(list(payload[a]),COMM_SUCCESS,1)
        return bus

    def test_sdk_retains_valid_alarm_payload_but_rejects_bad_packets_and_other_bits(self):
        bus=self.sdk_bus();state=bus.feedback(20)
        self.assertEqual(state['voltage_raw'],51);self.assertEqual(state['status_register'],0)
        self.assertEqual(state['packet_status_bits'],1);self.assertEqual(len(bus.servo_status_notifications),4)
        for bits in (2,16,3):
            bus.sdk.readTxRx.side_effect=None;bus.sdk.readTxRx.return_value=([255,7],COMM_SUCCESS,bits)
            with self.assertRaises(ServoStatusError):bus._read_once(20,56,2)
        for code,exc in ((COMM_RX_CORRUPT,CorruptResponse),(COMM_RX_TIMEOUT,ReadTimeout)):
            bus.sdk.readTxRx.return_value=([],code,1)
            with self.assertRaises(exc):bus._read_once(20,56,2)

    def test_four_alarm_packets_form_one_observation_not_four_stop_votes(self):
        policy,records,motor,healthy,monitor=self.fixture();bus=self.sdk_bus();now=[1.]
        original=bus.sdk.readTxRx.side_effect
        def reply(*args):now[0]+=.0002;return original(*args)
        bus.sdk.readTxRx.side_effect=reply
        reader=BudgetedFeedback(bus,[20],priority_ids=(),max_requests_per_step=4);clocks={20:.9}
        def sample(sid,state,started):monitor.observe(sid,state,motor);clocks[sid]=now[0]
        with patch('vivis.runtime.feedback.time',SimpleNamespace(monotonic=lambda:now[0])):
            for count in range(1,4):
                reader.step(now[0]+.01,clocks,sample,lambda *args:self.fail('communication error'))
                self.assertEqual(records[0]['consecutive_fault_samples'],count)
                now[0]+=.03
            with self.assertRaises(ConfirmedSensorFaultError):
                reader.step(now[0]+.01,clocks,sample,lambda *args:None)
        self.assertEqual(bus.read_requests,16)

    def test_corrupt_partial_sample_never_increments_fault_counter(self):
        policy,records,motor,healthy,monitor=self.fixture();bus=self.sdk_bus();now=[1.];errors=[]
        def reply(sid,a,n):
            now[0]+=.0002
            return ([255,7],COMM_SUCCESS,1) if a==56 else ([],COMM_RX_CORRUPT,0)
        bus.sdk.readTxRx.side_effect=reply;reader=BudgetedFeedback(bus,[20],priority_ids=(),max_requests_per_step=4)
        with patch('vivis.runtime.feedback.time',SimpleNamespace(monotonic=lambda:now[0])):
            reader.step(now[0]+.01,{20:.9},lambda sid,s,t:monitor.observe(sid,s,motor),lambda *a:errors.append(a))
        self.assertEqual(records,[]);self.assertEqual(len(errors),1);self.assertEqual(reader.packet_bits,0)

    def test_confirmation_priority_respects_freshness_urgency_and_request_budget(self):
        now=[1.03];samples=[];bus=Mock()
        def read(sid,address,count):
            now[0]+=.0002
            return {56:b'\xff\x07',60:b'\x00\x00',62:b'\x3c\x29',65:b'\x00'}[address]
        bus.read.side_effect=read
        for stale,expected in ((.9,20),(0.,2)):
            reader=BudgetedFeedback(bus,[2,20,23],priority_ids=(23,),max_requests_per_step=4)
            clocks={2:stale,20:1.,23:1.};samples.clear()
            with patch('vivis.runtime.feedback.time',SimpleNamespace(monotonic=lambda:now[0])):
                result=reader.step(now[0]+.01,clocks,lambda sid,s,t:samples.append(sid),lambda *a:None,confirmation_ids=[20])
            self.assertEqual(samples[0],expected);self.assertEqual(result['reads'],4)

    def test_mock_playback_continues_after_isolated_fault_and_stops_after_four(self):
        from vivis.runtime.telemetry import apply_policy
        for persistent in (False,True):
            profile,bus,clock=test_runtime.PlaybackTests().fixture();clock.duration=.15
            apply_policy(profile,Robot().telemetry());started=[False];original_start=clock.start.side_effect
            def start():started[0]=True;original_start()
            clock.start.side_effect=start;count=[0]
            def feedback(sid):
                state=dict(position=2047,temperature_c=41,load_raw=0,status_register=0)
                if started[0]:
                    count[0]+=1
                    if persistent or count[0]==1:state['packet_status_bits']=1
                return state
            bus.feedback.side_effect=feedback
            with tempfile.TemporaryDirectory() as folder,patch('vivis.runtime.playback.Bus',return_value=bus),\
                 patch('vivis.runtime.playback.AudioClock',return_value=clock):
                if persistent:
                    with self.assertRaises(ConfirmedSensorFaultError):
                        play(profile,[2],np.full((9,1),2047.),60,'fake',Path(folder)/'log.json',Path(folder)/'STOP')
                    self.assertEqual(count[0],4)
                else:
                    result=play(profile,[2],np.full((9,1),2047.),60,'fake',Path(folder)/'log.json',Path(folder)/'STOP')
                    self.assertTrue(result['completed']);self.assertTrue(result['sensor_fault_confirmations'][0]['recovered'])
            bus.close.assert_called_once();clock.close.assert_called_once()

    def test_scheduled_sdk_path_defers_only_after_audio_start_and_confirms_real_alarm_fields(self):
        from vivis.runtime.show import prepare
        for persistent in (False,True):
            robot=Robot();p=robot.profile('wz2');ids=p['diagnostic_command_ids']
            profile,ids,trajectory=prepare(robot.tracking(),p,ids,np.full((30,len(ids)),2047.),
                True,1.,robot.telemetry(),'face','lips-direct')
            motors={m['id']:m for m in profile['motors']};bus=self.sdk_bus();bus.servo_status_policy='raise'
            bus.handler=SimpleNamespace();bus.close=Mock();bus.last_transmitted_ids=[]
            bus.goal_axes_sent=0;bus.goal_axes_suppressed=0;bus.goal_packets_sent=0
            def snapshot(sid):
                m=motors[sid]
                return dict(id=sid,model_raw=m['model_raw_expected'],mode=0,torque_enable=1,
                    configured_min=m['device_min'],configured_max=m['device_max'],position_raw=2047,
                    temperature_c=41,voltage_raw=238 if 31<=sid<=35 else 60,load_raw=0,status_register=0)
            bus.snapshot=Mock(side_effect=snapshot)
            def goals(targets,speeds):
                bus.last_transmitted_ids=list(targets);bus.goal_axes_sent+=len(targets);bus.goal_packets_sent+=1
            bus.goals=Mock(side_effect=goals)
            _,_,clock=test_runtime.PlaybackTests().fixture();clock.duration=.5;started=[False];count=[0]
            original_start=clock.start.side_effect
            def start():started[0]=True;original_start()
            clock.start.side_effect=start
            def reply(sid,address,width):
                if not started[0]:self.assertEqual(bus.servo_status_policy,'raise')
                else:self.assertEqual(bus.servo_status_policy,'defer_sensor_faults')
                alarm=0
                if started[0] and sid==20 and address==65:
                    count[0]+=1
                    if persistent or count[0]==1:alarm=1
                payload={56:b'\xff\x07',60:b'\x00\x00',62:bytes([238 if 31<=sid<=35 else 60,41]),65:bytes([alarm])}
                return list(payload[address]),COMM_SUCCESS,alarm
            bus.sdk.readTxRx.side_effect=reply
            with tempfile.TemporaryDirectory() as folder,patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus),\
                 patch('vivis.runtime.playback.AudioClock',return_value=clock):
                if persistent:
                    with self.assertRaises(ConfirmedSensorFaultError):
                        play(profile,ids,trajectory,60,'fake',Path(folder)/'log.json',Path(folder)/'STOP')
                    self.assertEqual(count[0],4)
                else:
                    result=play(profile,ids,trajectory,60,'fake',Path(folder)/'log.json',Path(folder)/'STOP')
                    self.assertTrue(result['completed']);self.assertTrue(result['sensor_fault_confirmations'][0]['recovered'])
                    self.assertTrue(any(s.get('packet_status_bits')==1 for e in result['events'] for s in e.get('feedback_samples',[])))
            bus.close.assert_called_once();clock.close.assert_called_once()
