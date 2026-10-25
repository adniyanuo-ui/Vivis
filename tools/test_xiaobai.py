"""Hardware-independent boundary and PTY serial tests for Xiaobai playback."""
import copy
import json
import os
import pty
import struct
import tempfile
import threading
import unittest
from pathlib import Path
import numpy as np
from xiaobai_runtime.protocol import Bus,packet,decode_status,goal_packet,ProtocolError,CorruptResponse,ReadTimeout
from xiaobai_runtime.mapping import ROOT,inputs,audit,draft_profile,compile_trajectory,limit_step,live_preflight

class ProtocolTests(unittest.TestCase):
    def test_sdk_reports_missing_or_inaccessible_serial_device(self):
        from unittest.mock import patch
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        with patch('xiaobai_runtime.feetech_sdk.os.path.exists',return_value=False):
            with self.assertRaisesRegex(ProtocolError,'Serial device is missing'):
                OfficialSDKBus('/dev/vivis-test-serial')
        with patch('xiaobai_runtime.feetech_sdk.os.path.exists',return_value=True),patch('xiaobai_runtime.feetech_sdk.os.access',return_value=False):
            with self.assertRaisesRegex(ProtocolError,'sudo setfacl'):
                OfficialSDKBus('/dev/vivis-test-serial')

    def test_deferred_read_failure_has_no_inline_sleep_or_retry(self):
        from unittest.mock import Mock,patch
        bus=Bus.__new__(Bus);bus.read_requests=0;bus.corrupt_responses=[];bus.read_failures=[]
        bus.retry_reads=False;bus._read_once=Mock(side_effect=ReadTimeout('missing'))
        with patch('xiaobai_runtime.protocol.time.sleep') as sleep:
            with self.assertRaises(ReadTimeout):bus.read(2,56,2)
        sleep.assert_not_called();self.assertEqual(bus._read_once.call_count,1)
        self.assertEqual(len(bus.read_failures),1)
        bus.read_requests=99;bus.read_failures=[{}]*5
        with self.assertRaisesRegex(ProtocolError,'exceeds 5%'):bus.read(2,56,2)

    def test_sdk_timeout_is_instance_local_and_timeouts_still_fail(self):
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        from scservo_sdk.port_handler import LATENCY_TIMER
        master,slave=pty.openpty()
        bus=OfficialSDKBus(os.ttyname(slave),transaction_timeout_ms=8)
        try:
            bus.handler.setPacketTimeout(21)
            self.assertEqual(bus.handler.packet_timeout,8)
            self.assertEqual(LATENCY_TIMER,50)
            with self.assertRaises(ReadTimeout):bus._read_once(2,56,15)
        finally:bus.close();os.close(master);os.close(slave)
        for value in [0,1,51,True,float('nan')]:
            with self.assertRaises(ValueError):OfficialSDKBus('unused',transaction_timeout_ms=value)

    def test_raw_trace_preserves_io_and_bounds_capture(self):
        from unittest.mock import Mock
        from xiaobai_runtime.serial_trace import SerialTrace
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.json';port=Mock();port.in_waiting=2
            port.read.side_effect=[b'',b'\xff\xff'];port.write.return_value=3
            trace=SerialTrace(port,path,max_events=2)
            self.assertEqual(trace.read(2),b'')
            self.assertEqual(trace.read(2),b'\xff\xff')
            self.assertEqual(trace.write(b'abc'),3)
            trace.reset_input_buffer();trace.close()
            port.reset_input_buffer.assert_called_once();port.close.assert_called_once()
            result=json.loads(path.read_text())
            self.assertEqual([e['direction'] for e in result['events']],['rx','tx'])
            self.assertEqual(result['events'][0]['hex'],'ffff')
            self.assertEqual(result['dropped_events'],1)

    def test_official_write_example(self):
        self.assertEqual(packet(1,3,bytes.fromhex('2a 00 08 00 00 e8 03')).hex(),'ffff0109032a00080000e803d5')
    def test_read_checksum_status_and_wrong_id(self):
        valid=bytes.fromhex('ff ff 01 04 00 00 08 f2')
        self.assertEqual(decode_status(valid,1,2),b'\x00\x08')
        for bad in [valid[:-1]+b'\x00',bytes.fromhex('ff ff 01 02 20 dc')]:
            with self.assertRaises(ProtocolError):decode_status(bad,1,2)
        with self.assertRaises(ProtocolError):decode_status(valid,2,2)
    def test_sync_goal_only_writes_42_through_47(self):
        raw=goal_packet({2:2047,31:1850},{2:300,31:200})
        self.assertEqual(raw[:7],bytes.fromhex('ff ff fe 12 83 2a 06'))
        self.assertEqual(raw[7:14],bytes([2])+struct.pack('<HHH',2047,0,300))
        self.assertEqual(sum(raw[2:])&255,255)
    def test_reject_unsafe_position_or_unbounded_speed(self):
        for position,speed in [(-1,300),(4096,300),(2047,0),(2047,32768)]:
            with self.assertRaises(ValueError):goal_packet({2:position},{2:speed})
    def test_actual_pyserial_over_pty_read_and_echo(self):
        master,slave=pty.openpty();bus=Bus(os.ttyname(slave),timeout=.15)
        seen=[]
        def firmware():
            request=os.read(master,1024);seen.append(request)
            body=bytes([2,4,0,0xff,7]);reply=b'\xff\xff'+body+bytes([(~sum(body))&255])
            os.write(master,request+reply[:3]);os.write(master,reply[3:])
        thread=threading.Thread(target=firmware);thread.start()
        try:self.assertEqual(bus.read(2,56,2),bytes([255,7]))
        finally:thread.join(1);bus.close();os.close(master);os.close(slave)
        self.assertEqual(seen[0][4],2)
    def test_single_corrupt_read_recovers_but_repeated_corruption_and_fault_stop(self):
        from unittest.mock import Mock
        bus=Bus.__new__(Bus);bus.read_requests=0;bus.corrupt_responses=[];bus.read_failures=[]
        bus._read_once=Mock(side_effect=[CorruptResponse('damaged'),b'OK'])
        self.assertEqual(bus.read(2,56,2),b'OK')
        self.assertEqual(len(bus.corrupt_responses),1)
        bus._read_once=Mock(side_effect=CorruptResponse('damaged'))
        with self.assertRaises(CorruptResponse):bus.read(2,56,2)
        self.assertEqual(bus._read_once.call_count,2)
        bus._read_once=Mock(side_effect=ProtocolError('servo status fault'))
        with self.assertRaises(ProtocolError):bus.read(2,56,2)
        self.assertEqual(bus._read_once.call_count,1)
        bus._read_once=Mock(side_effect=[ReadTimeout('missing'),b'OK'])
        self.assertEqual(bus.read(2,56,2),b'OK')
        bus._read_once=Mock(side_effect=ReadTimeout('missing'))
        with self.assertRaises(ReadTimeout):bus.read(2,56,2)
        self.assertEqual(bus._read_once.call_count,2)
    def test_grouped_goals_cover_all_ids_and_validate_before_any_write(self):
        from unittest.mock import Mock,patch
        bus=Bus.__new__(Bus);bus.port=Mock();bus.port.baudrate=1000000
        bus.port.write.side_effect=lambda raw:len(raw)
        goals={i:2047 for i in range(2,32)};speeds={i:300 for i in goals}
        with patch('xiaobai_runtime.protocol.time.sleep'):
            bus.goals(goals,speeds)
        decoded={}
        for call in bus.port.write.call_args_list:
            raw=call.args[0];self.assertEqual(sum(raw[2:])&255,255)
            self.assertEqual(raw[5:7],bytes([42,6]))
            for offset in range(7,len(raw)-1,7):
                sid=raw[offset];self.assertNotIn(sid,decoded)
                decoded[sid]=struct.unpack('<HHH',raw[offset+1:offset+7])
        self.assertEqual(decoded,{i:(2047,0,300) for i in goals})
        bus.port.write.reset_mock();goals[31]=5000
        with self.assertRaises(ValueError):bus.goals(goals,speeds)
        bus.port.write.assert_not_called()

class OfficialSDKTests(unittest.TestCase):
    def test_split_sync_packets_preserve_targets_and_validate_whole_frame(self):
        from unittest.mock import Mock,patch
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        bus=OfficialSDKBus.__new__(OfficialSDKBus)
        bus.acceleration={2:10,3:10,4:10};bus.sync_group_size=1;bus.sync_group_gap_ms=.2
        bus.baud=1000000;bus.write_mode='position_speed';bus.port=Mock();bus._send_goals=Mock()
        with patch('xiaobai_runtime.feetech_sdk.time.sleep') as sleep:
            bus.goals({4:2050,2:2047,3:2048},{2:300,3:300,4:300})
        self.assertEqual([c.args[0] for c in bus._send_goals.call_args_list],[{2:2047},{3:2048},{4:2050}])
        self.assertEqual(bus.last_transmitted_ids,[2,3,4]);self.assertEqual(sleep.call_count,3)
        self.assertAlmostEqual(sleep.call_args.args[0],.00036)
        bus._send_goals.reset_mock()
        with self.assertRaises(ValueError):bus.goals({2:2047,4:5000},{2:300,4:300})
        bus._send_goals.assert_not_called()

    def test_quiet_interval_follows_successful_write_only(self):
        from unittest.mock import Mock, patch
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        bus=OfficialSDKBus.__new__(OfficialSDKBus)
        bus.acceleration={2:10};bus.changed_goals_only=True;bus.goal_axes_suppressed=0
        bus.post_goal_quiet_ms=2;bus.baud=1000000;bus.write_mode='position_speed'
        order=[];bus._send_goals=Mock(side_effect=lambda *a:order.append('write'))
        bus.port=Mock();bus.port.flush.side_effect=lambda:order.append('flush')
        with patch('xiaobai_runtime.feetech_sdk.time.sleep',side_effect=lambda t:order.append(t)):
            bus.goals({2:2047},{2:300})
            bus.goals({2:2047},{2:300})
        self.assertEqual(order[:2],['write','flush'])
        self.assertAlmostEqual(order[2],0.00216)
        self.assertEqual(len(order),3)
        bus._send_goals.side_effect=ProtocolError('failed')
        with patch('xiaobai_runtime.feetech_sdk.time.sleep') as sleep:
            with self.assertRaises(ProtocolError):bus.goals({2:2048},{2:300})
            sleep.assert_not_called()

    def test_sparse_writes_keep_changed_axes_and_speed_changes(self):
        from unittest.mock import Mock
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        bus=OfficialSDKBus.__new__(OfficialSDKBus)
        bus.acceleration={2:10,31:10};bus.changed_goals_only=True
        bus.goal_axes_suppressed=0;bus._send_goals=Mock()
        bus.goals({2:2047,31:2040},{2:300,31:300})
        self.assertEqual(bus.last_transmitted_ids,[2,31])
        bus.goals({2:2047,31:2040},{2:300,31:300})
        self.assertEqual(bus.last_transmitted_ids,[])
        self.assertEqual(bus._send_goals.call_count,1)
        bus.goals({2:2047,31:2039},{2:300,31:300})
        self.assertEqual(bus.last_transmitted_ids,[31])
        bus.goals({2:2047,31:2039},{2:200,31:300})
        self.assertEqual(bus.last_transmitted_ids,[2])
        self.assertEqual(bus.goal_axes_sent,4)
        self.assertEqual(bus.goal_axes_suppressed,4)
        with self.assertRaises(ValueError):bus.goals({2:5000},{2:300})
        self.assertEqual(bus._send_goals.call_count,3)

    def test_sparse_failed_write_does_not_commit_command_cache(self):
        from unittest.mock import Mock
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        bus=OfficialSDKBus.__new__(OfficialSDKBus)
        bus.acceleration={2:10};bus.changed_goals_only=True;bus.goal_axes_suppressed=0
        bus._send_goals=Mock(side_effect=ProtocolError('write failed'))
        with self.assertRaises(ProtocolError):bus.goals({2:2047},{2:300})
        self.assertEqual(getattr(bus,'last_sent_commands',{}),{})
        bus._send_goals=Mock();bus.goals({2:2047},{2:300})
        bus._send_goals.assert_called_once()
        self.assertEqual(bus.last_transmitted_ids,[2])

    def test_position_only_preserves_device_speed_and_writes_only_two_bytes(self):
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus,sms_sts,GroupSyncWrite
        class Port:
            is_using=False
            def clearPort(self):pass
            def writePort(self,raw):self.raw=bytes(raw);return len(raw)
        port=Port();bus=OfficialSDKBus.__new__(OfficialSDKBus);bus.sdk=sms_sts(port)
        bus.position_writer=GroupSyncWrite(bus.sdk,42,2);bus.write_mode='position_only'
        bus.acceleration={2:10,31:10};bus.configured_speeds={2:300,31:300};bus.goal_packets_sent=0
        bus.goals({2:2047,31:2040},{2:300,31:300})
        self.assertEqual(port.raw,packet(254,0x83,bytes([42,2,2])+struct.pack('<H',2047)+bytes([31])+struct.pack('<H',2040)))
        with self.assertRaisesRegex(ProtocolError,'bounded device speeds'):
            bus.goals({2:2047},{2:200})
        self.assertEqual(bus.goal_packets_sent,1)
    def test_short_register_reads_keep_all_feedback_guards(self):
        from unittest.mock import Mock
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        bus=OfficialSDKBus.__new__(OfficialSDKBus);bus.feedback_mode='short_registers'
        bus.read=Mock(side_effect=[b'\xff\x07',b'\x2a\x04',b'\x4b\x38',b'\x00'])
        self.assertEqual(bus.feedback(4),{'position':2047,'load_raw':1066,'voltage_raw':75,'temperature_c':56,'status_register':0})
        self.assertEqual([c.args for c in bus.read.call_args_list],[(4,56,2),(4,60,2),(4,62,2),(4,65,1)])
        bus.read=Mock(side_effect=ProtocolError('servo status'))
        with self.assertRaises(ProtocolError):bus.feedback(4)
    def test_sdk_packet_matches_existing_42_to_47_encoder(self):
        from xiaobai_runtime.feetech_sdk import sms_sts
        class Port:
            is_using=False
            def clearPort(self):pass
            def writePort(self,raw):self.raw=bytes(raw);return len(raw)
        port=Port();sdk=sms_sts(port)
        goals={2:2047,31:1999};speeds={2:300,31:60}
        param=[]
        for sid in goals:param.extend(bytes([sid])+struct.pack('<HHH',goals[sid],0,speeds[sid]))
        self.assertEqual(sdk.syncWriteTxOnly(42,6,param,len(param)),0)
        self.assertEqual(port.raw,goal_packet(goals,speeds))
    def test_sdk_acceleration_is_preserved_and_unread_motors_are_rejected(self):
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus,sms_sts
        class Port:
            is_using=False
            def clearPort(self):pass
            def writePort(self,raw):self.raw=bytes(raw);return len(raw)
        port=Port();bus=OfficialSDKBus.__new__(OfficialSDKBus)
        bus.sdk=sms_sts(port);bus.acceleration={2:10,31:7};bus.goal_packets_sent=0
        bus.goals({2:2047,31:1999},{2:300,31:60})
        expected=packet(254,0x83,bytes([41,7,2,10])+struct.pack('<HHH',2047,0,300)+bytes([31,7])+struct.pack('<HHH',1999,0,60))
        self.assertEqual(port.raw,expected)
        with self.assertRaisesRegex(ProtocolError,'snapshot'):bus.goals({3:2047},{3:300})
        self.assertEqual(bus.goal_packets_sent,1)
    def test_sdk_never_ignores_voltage_fault_or_corrupt_packet(self):
        from unittest.mock import Mock
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus,COMM_RX_CORRUPT
        bus=OfficialSDKBus.__new__(OfficialSDKBus);bus.sdk=Mock()
        bus.sdk.readTxRx.return_value=([0]*15,0,1)
        with self.assertRaisesRegex(ProtocolError,'status 0x01'):bus._read_once(4,56,15)
        self.assertEqual(bus.last_sdk_response['servo_status'],1)
        bus.sdk.readTxRx.return_value=([],COMM_RX_CORRUPT,0)
        with self.assertRaises(CorruptResponse):bus._read_once(4,56,15)

class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip=ROOT/'tests/fixtures/blendshapes61.json'
        cls.calibration=ROOT/'tests/fixtures/legacy_calibration.json';cls.header=ROOT/'xiaobai/blendshape_channels_61.csv'
        cls.values,cls.fps,cls.order,cls.shapes=inputs(cls.clip,cls.calibration,cls.header)
        cls.profile=draft_profile(audit(cls.clip,cls.calibration,cls.header))
    def test_unconfirmed_profile_cannot_drive(self):
        with self.assertRaises(ValueError):live_preflight(self.profile,[2,31])
        with self.assertRaisesRegex(ValueError,'eye angle'):compile_trajectory(self.values,self.order,self.shapes,self.profile)
    def test_zero_and_one_hot_endpoints_with_shared_jaw_owner(self):
        v=np.zeros((2,61));v[1,self.order.index('JawOpen')]=1
        v[1,self.order.index('BrowDownLeft')]=1
        ids,p,_=compile_trajectory(v,self.order,self.shapes,self.profile,draft=True)
        self.assertTrue(np.all(p[0]==2047))
        self.assertEqual(p[1,ids.index(31)],1850)
        self.assertEqual(p[1,ids.index(2)],2347)
    def test_combined_endpoint_envelope_and_missing_axes_are_reported(self):
        ids,p,r=compile_trajectory(self.values,self.order,self.shapes,self.profile,draft=True)
        self.assertNotIn(29,ids);self.assertNotIn(32,ids)
        self.assertTrue(r['missing_eyes']);self.assertTrue(r['missing_neck'])
        self.assertGreater(sum(r['clipped_samples_by_id'].values()),0)
        for i,sid in enumerate(ids):
            m=next(m for m in self.profile['motors'] if m['id']==sid)
            self.assertGreaterEqual(min(p[:,i]),m['endpoint_envelope_not_safe_limits'][0])
            self.assertLessEqual(max(p[:,i]),m['endpoint_envelope_not_safe_limits'][1])
    def test_radians_map_to_conjugate_eye_motor_endpoints(self):
        profile=copy.deepcopy(self.profile)
        profile['eye_angle_endpoints_rad']={n:.2 for n in self.order if n.startswith('EyeLook')}
        v=np.zeros((1,61));v[0,self.order.index('LeftEyeYaw')]=.2;v[0,self.order.index('RightEyeYaw')]=.2
        ids,p,_=compile_trajectory(v,self.order,self.shapes,profile,draft=True)
        self.assertEqual(p[0,ids.index(8)],1750)  # subject-left eye outward
        self.assertEqual(p[0,ids.index(23)],1700) # subject-right eye inward
    def test_lid_gaze_not_double_applied(self):
        profile=copy.deepcopy(self.profile)
        profile['eye_angle_endpoints_rad']={n:.2 for n in self.order if n.startswith('EyeLook')}
        v=np.zeros((1,61));v[0,self.order.index('LeftEyePitch')]=.2
        ids,p,_=compile_trajectory(v,self.order,self.shapes,profile,draft=True)
        self.assertEqual(p[0,ids.index(5)],2047);self.assertEqual(p[0,ids.index(7)],1900)
    def test_rate_limiter_and_clock_stall(self):
        result=limit_step(np.array([2047.,2047.]),np.array([2500.,1700.]),np.array([300.,120.]),1/60)
        np.testing.assert_allclose(result,[2052,2045])
        with self.assertRaises(ValueError):limit_step(result,result,np.ones(2),.11)
        with self.assertRaises(ValueError):limit_step(result,np.array([float('nan'),0]),np.ones(2),.01)

class FinalCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip=ROOT/'tests/fixtures/blendshapes61.json'
        cls.cal=ROOT/'xiaobai/xiaobai标定文件.json';cls.header=ROOT/'xiaobai/blendshape_channels_61.csv'
        cls.values,cls.fps,cls.order,cls.shapes=inputs(cls.clip,cls.cal,cls.header)
        cls.profile=draft_profile(audit(cls.clip,cls.cal,cls.header))
        cls.profile['mixing']='bounded_weighted_deltas'
    def test_final_file_has_no_old_jaw_placeholders(self):
        self.assertEqual(self.shapes['JawOpen'],{31:1950})
        self.assertEqual(sum(31 in p for p in self.shapes.values()),1)
    def test_one_shape_formula_and_half_weight(self):
        v=np.zeros((3,61));v[:,self.order.index('JawOpen')]=[0,.5,1]
        ids,p,_=compile_trajectory(v,self.order,self.shapes,self.profile,draft=True)
        np.testing.assert_allclose(p[:,ids.index(31)],[2047,1998.5,1950])
    def test_blink_has_priority_even_when_eye_wide(self):
        v=np.zeros((1,61))
        for n in ['EyeBlinkLeft','EyeWideLeft','EyeSquintLeft','CheekSquintLeft']:v[0,self.order.index(n)]=1
        ids,p,_=compile_trajectory(v,self.order,self.shapes,self.profile,draft=True)
        self.assertEqual(p[0,ids.index(5)],1700);self.assertEqual(p[0,ids.index(6)],2200)
    def test_full_strength_combinations_do_not_expand_endpoint_envelope(self):
        ids,p,r=compile_trajectory(np.ones((1,61)),self.order,self.shapes,self.profile,draft=True)
        self.assertEqual(sum(r['clipped_samples_by_id'].values()),0)
    def test_measured_voltage_is_checked_not_only_confirmation_flag(self):
        from xiaobai_runtime.playback import check_feedback
        profile=json.loads((ROOT/'xiaobai/playback_profile.json').read_text())
        motor=next(m for m in profile['motors'] if m['id']==8)
        motor=copy.deepcopy(motor);motor.update(min_voltage_raw=40,max_voltage_raw=74)
        state={'position':2047,'temperature_c':55,'load_raw':0,'status_register':0,'voltage_raw':85}
        with self.assertRaisesRegex(RuntimeError,'voltage'):check_feedback(8,state,motor)
        state['voltage_raw']=70
        check_feedback(8,state,motor)
    def test_trial_gain_and_real_device_limits(self):
        profile=json.loads((ROOT/'xiaobai/playback_profile.json').read_text())
        ids,p,r=compile_trajectory(self.values,self.order,self.shapes,profile,scope='face_eyes')
        self.assertNotIn(32,ids);self.assertIn(8,ids)
        self.assertEqual(r['motion_gain'],.25)
        for j,sid in enumerate(ids):
            m=next(x for x in profile['motors'] if x['id']==sid)
            self.assertGreaterEqual(p[:,j].min(),m['device_min'])
            self.assertLessEqual(p[:,j].max(),m['device_max'])
        profile=copy.deepcopy(profile);profile['power_checked']=False
        with self.assertRaisesRegex(ValueError,'power_checked'):live_preflight(profile,ids)

class AudioClockTests(unittest.TestCase):
    def test_audio_trim_scales_pcm_without_changing_duration(self):
        import sys,wave
        from unittest.mock import Mock,patch
        from xiaobai_runtime.playback import AudioClock
        sd=Mock();sd.query_devices.return_value={'default_samplerate':48000}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audio.wav'
            with wave.open(str(path),'wb') as wav:
                wav.setparams((1,2,48000,0,'NONE','not compressed'))
                wav.writeframes(np.array([-32768,32767]*2400,dtype='<i2').tobytes())
            with patch.dict(sys.modules,{'sounddevice':sd}):
                clock=AudioClock(path,'USB Audio Device',gain=.8956)
                self.assertAlmostEqual(clock.duration,.1)
                self.assertLessEqual(float(np.max(abs(clock.data))),.895601)
                clock.close()
        for gain in [0,True,float('nan'),1.01]:
            with self.assertRaises(ValueError):AudioClock('unused','unused',gain=gain)

    def test_buffered_output_keeps_dac_clock_and_reports_underflow(self):
        import sys
        import wave
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from xiaobai_runtime.playback import AudioClock
        sd=Mock();sd.query_devices.return_value={'default_samplerate':48000}
        sd.CallbackStop=type('CallbackStop',(Exception,),{})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audio.wav'
            with wave.open(str(path),'wb') as wav:
                wav.setparams((1,2,48000,0,'NONE','not compressed'))
                wav.writeframes(np.zeros(48000,dtype='<i2').tobytes())
            with patch.dict(sys.modules,{'sounddevice':sd}):
                clock=AudioClock(path,'pulse',latency=.2)
                self.assertEqual(sd.OutputStream.call_args.kwargs['latency'],.2)
                clock.callback(np.empty((480,1),np.float32),480,
                               SimpleNamespace(outputBufferDacTime=10.2),False)
                clock.stream.time=10.1
                self.assertAlmostEqual(clock.time(),-.1)
                clock.stream.time=10.5
                self.assertAlmostEqual(clock.time(),.3)
                clock.callback(np.empty((480,1),np.float32),480,
                               SimpleNamespace(outputBufferDacTime=10.21),'output underflow')
                self.assertEqual(clock.fault,'output underflow')
                self.assertAlmostEqual(clock.first_dac,10.2)
                clock.close()

class PlaybackTests(unittest.TestCase):
    def test_playback_rechecks_isolated_high_temperature_and_continues(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        from xiaobai_runtime.telemetry_policy import apply_policy
        profile,bus,clock=self.fixture()
        policy=json.loads((ROOT/'mojia/telemetry_policy.json').read_text());apply_policy(profile,policy)
        cold=dict(position=2047,temperature_c=43,load_raw=0,status_register=0)
        samples=iter([{**cold,'temperature_c':165},cold])
        bus.feedback.side_effect=lambda sid:next(samples,cold)
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),\
                patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2],np.full((3,1),2047.),60,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
        self.assertTrue(result['completed']);record=result['sensor_confirmations'][0]
        self.assertEqual(record['samples'][0]['state']['temperature_c'],165);self.assertTrue(record['recovered'])

    def test_playback_stops_on_four_consecutive_hot_samples(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        from xiaobai_runtime.telemetry_policy import apply_policy,CriticalTemperatureError
        profile,bus,clock=self.fixture()
        apply_policy(profile,json.loads((ROOT/'mojia/telemetry_policy.json').read_text()))
        bus.feedback.side_effect=None
        bus.feedback.return_value=dict(position=2047,temperature_c=165,load_raw=0,status_register=0)
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),\
                patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            with self.assertRaisesRegex(CriticalTemperatureError,'4 consecutive'):
                play(profile,[2],np.full((3,1),2047.),60,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
            result=json.loads((Path(directory)/'log.json').read_text())
        self.assertEqual(bus.feedback.call_count,4);self.assertEqual(result['sensor_confirmations'][0]['consecutive_high_samples'],4)

    def test_explicit_load_and_temperature_warnings_allow_playback(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        from xiaobai_runtime.telemetry_policy import apply_policy
        profile,bus,clock=self.fixture()
        apply_policy(profile,dict(schema_version='mojia-telemetry-policy-1.0',load_warning_raw=800,
                     temperature_warning_c=80,temperature_stop_c=100,hardware_protection='preserve'))
        bus.feedback.side_effect=None
        bus.feedback.return_value=dict(position=2047,temperature_c=90,load_raw=900,status_register=0)
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),\
                patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2],np.full((3,1),2047.),60,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
        self.assertTrue(result['completed'])
        self.assertEqual({r['kind'] for r in result['telemetry_warnings']},{'load','temperature'})
        self.assertTrue(all(r['count']>=1 for r in result['telemetry_warnings']))

    def test_explicit_critical_temperature_stops_without_confirmation_or_record_only_bypass(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        from xiaobai_runtime.telemetry_policy import apply_policy,CriticalTemperatureError
        for mode in ('stop_on_fault','record_only'):
            profile,bus,clock=self.fixture();profile['feedback_policy']=mode
            profile['confirm_sensor_transients']=True
            apply_policy(profile,dict(schema_version='mojia-telemetry-policy-1.0',load_warning_raw=800,
                         temperature_warning_c=80,temperature_stop_c=100,hardware_protection='preserve'))
            bus.feedback.side_effect=None
            bus.feedback.return_value=dict(position=2047,temperature_c=101,load_raw=0,status_register=0)
            with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),\
                    patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
                with self.assertRaises(CriticalTemperatureError):
                    play(profile,[2],np.full((3,1),2047.),60,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
            self.assertEqual(bus.feedback.call_count,1)

    def test_software_acceleration_is_applied_to_final_playback_commands(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        profile,bus,clock=self.fixture();profile['motors'][0]['max_acceleration_counts_s2']=128
        trajectory=np.full((24,1),2200.);trajectory[0]=2047.;clock.duration=.4
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2],trajectory,60,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
        self.assertTrue(result['completed'])
        self.assertEqual(result['software_acceleration_limits'],{2:128})
        self.assertTrue(all(2 in e['software_motion'] for e in result['events']))
        self.assertGreater(result['sent_frames'],15)
        self.assertLessEqual(max(e['goals'][2] for e in result['events']),2059)
        self.assertGreater(max(e['goals'][2] for e in result['events']),2047)

    def test_mojia_rejects_xiaobai_gaze_before_opening_devices(self):
        from unittest.mock import Mock,patch
        from xiaobai_runtime.playback import play
        profile,_,_=self.fixture();profile['schema_version']='mojia-playback-0.1'
        with patch('xiaobai_runtime.playback.AudioClock') as audio,patch('xiaobai_runtime.playback.Bus') as bus:
            with self.assertRaisesRegex(ValueError,'Mojia'):
                play(profile,[2],np.full((3,1),2047.),60,'unused','unused','unused',gaze_override=Mock())
            audio.assert_not_called();bus.assert_not_called()

    def test_merged_audio_latency_fields_and_cli_precedence(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        cases=[({},'low'),({'audio_latency_s':.2},.2),
               ({'audio_latency_seconds':.3},.3),
               ({'audio_latency_s':.2,'audio_latency_seconds':.3},.2)]
        for settings,expected in cases:
            profile,bus,clock=self.fixture();profile.update(settings)
            with self.subTest(settings=settings),tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),patch('xiaobai_runtime.playback.AudioClock',return_value=clock) as audio:
                result=play(profile,[2],np.full((3,1),2047.),60,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
                self.assertTrue(result['completed'])
                self.assertEqual(audio.call_args.kwargs['latency'],expected)
                self.assertEqual(result['audio_requested_latency'],expected)

    def test_invalid_trajectory_is_rejected_before_opening_devices_in_both_policies(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        profile,_,_=self.fixture()
        bad=[np.empty((0,1)),np.zeros((3,2)),np.array([[float('nan')]]),
             np.array([[float('inf')]]),np.array([[1799.]]),np.array([[2301.]])]
        with patch('xiaobai_runtime.playback.Bus') as bus,patch('xiaobai_runtime.playback.AudioClock') as audio:
            for policy in ['stop_on_fault','record_only']:
                profile['feedback_policy']=policy
                for trajectory in bad:
                    with self.subTest(policy=policy,trajectory=trajectory.tolist()):
                        with self.assertRaises(ValueError):
                            play(profile,[2],trajectory,60,'unused','unused','unused')
            for fps in [0,-1,float('nan'),float('inf'),True,121]:
                with self.assertRaisesRegex(ValueError,'FPS'):
                    play(profile,[2],np.full((3,1),2047.),fps,'unused','unused','unused')
            for ids in [[],[2,2],[True],[254]]:
                with self.assertRaisesRegex(ValueError,'IDs'):
                    play(profile,ids,np.full((3,len(ids)),2047.),60,'unused','unused','unused')
            bus.assert_not_called();audio.assert_not_called()

    def test_diagnostic_selection_keeps_monitoring_other_axes(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        profile,bus,clock=self.fixture()
        profile['motors'].append({**profile['motors'][0],'id':3})
        profile['diagnostic_command_ids']=[2]
        trajectory=np.array([[2050.,2200.],[2051.,2250.],[2052.,2100.]])
        original=trajectory.copy()
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2,3],trajectory,60,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
        self.assertTrue(result['completed']);self.assertEqual(result['command_ids'],[2])
        self.assertEqual(result['monitor_ids'],[2,3])
        self.assertTrue(any(c.args==(3,) for c in bus.feedback.call_args_list))
        for call in bus.goals.call_args_list:
            self.assertEqual(set(call.args[0]),{2});self.assertEqual(set(call.args[1]),{2})
        np.testing.assert_array_equal(trajectory,original)

    def test_diagnostic_selection_cannot_add_neck_or_invalid_axes(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        profile,_,_=self.fixture()
        with patch('xiaobai_runtime.playback.Bus') as factory,patch('xiaobai_runtime.playback.AudioClock') as audio:
            for selection in [[],[31],[2,2],[True],None]:
                profile['diagnostic_command_ids']=selection
                with self.assertRaisesRegex(ValueError,'Diagnostic command IDs'):
                    play(profile,[2],np.full((3,1),2047.),60,'fake.wav','unused','unused')
            factory.assert_not_called();audio.assert_not_called()

    def test_sensor_confirmation_requires_two_clean_reads_and_no_writes(self):
        from unittest.mock import Mock,patch
        from xiaobai_runtime.playback import verified_feedback
        from xiaobai_runtime.protocol import ServoStatusError
        profile,_,_=self.fixture();motor=profile['motors'][0]
        healthy={'position':2047,'temperature_c':50,'load_raw':0,'status_register':0}
        hot={**healthy,'temperature_c':109};bus=Mock();records=[]
        bus.feedback.side_effect=[hot,healthy,healthy]
        with patch('xiaobai_runtime.playback.time.sleep'):
            self.assertEqual(verified_feedback(bus,2,motor,True,records),healthy)
        bus.goals.assert_not_called();self.assertTrue(records[0]['recovered'])
        for samples in [[hot]*4,[hot,healthy,hot,healthy],[ServoStatusError(2,1)]*4]:
            bus.feedback=Mock(side_effect=samples)
            with patch('xiaobai_runtime.playback.time.sleep'),self.assertRaises(RuntimeError):
                verified_feedback(bus,2,motor,True,[])
        for fault in [ServoStatusError(2,4),RuntimeError('overload')]:
            bus.feedback=Mock(side_effect=fault)
            with self.assertRaises(RuntimeError):verified_feedback(bus,2,motor,True,[])
            self.assertEqual(bus.feedback.call_count,1)
        bus.feedback=Mock(side_effect=[hot,healthy,healthy])
        with self.assertRaises(RuntimeError):verified_feedback(bus,2,motor,False,[])
        self.assertEqual(bus.feedback.call_count,1)
    def test_one_count_edge_requires_recovery_and_preserves_hard_guards(self):
        from unittest.mock import Mock,patch
        from xiaobai_runtime.playback import verified_feedback
        profile,_,_=self.fixture();motor=profile['motors'][0]
        motor.update(safe_min=2047,safe_max=2108,device_min=1850,device_max=2108,
                     confirm_position_edge_counts=1)
        healthy=dict(position=2047,temperature_c=40,load_raw=0,status_register=0)
        edge={**healthy,'position':2046};bus=Mock();records=[]
        bus.feedback.side_effect=[edge,healthy,healthy]
        with patch('xiaobai_runtime.playback.time.sleep'):
            self.assertEqual(verified_feedback(bus,9,motor,True,records),healthy)
        self.assertTrue(records[0]['recovered']);bus.goals.assert_not_called()
        for samples in [[edge]*4,[edge,healthy,edge,healthy]]:
            bus.feedback=Mock(side_effect=samples)
            with patch('xiaobai_runtime.playback.time.sleep'),self.assertRaises(RuntimeError):
                verified_feedback(bus,9,motor,True,[])
        for bad in [{**edge,'position':2045},{**edge,'position':2109},
                    {**edge,'load_raw':motor['max_load_raw']},{**edge,'status_register':4}]:
            bus.feedback=Mock(return_value=bad)
            with self.assertRaises(RuntimeError):verified_feedback(bus,9,motor,True,[])
            self.assertEqual(bus.feedback.call_count,1)
        bus.feedback=Mock(return_value=edge)
        with self.assertRaises(RuntimeError):verified_feedback(bus,9,{**motor,'confirm_position_edge_counts':0},True,[])
        self.assertEqual(bus.feedback.call_count,1)
        for value in [2,-1,True]:
            motor['confirm_position_edge_counts']=value
            with self.assertRaises(ValueError):live_preflight(profile,[2])

    def test_five_count_feedback_tolerance_keeps_targets_and_other_guards(self):
        from xiaobai_runtime.playback import check_feedback
        profile,_,_=self.fixture();motor=profile['motors'][0]
        motor.update(safe_min=2000,safe_max=2047,position_feedback_tolerance_counts=5)
        state=dict(position=2047,temperature_c=40,load_raw=0,status_register=0)
        for position in [1995,1999,2048,2052]:check_feedback(2,{**state,'position':position},motor)
        for position in [1994,2053,-1,4096]:
            with self.assertRaises(RuntimeError):check_feedback(2,{**state,'position':position},motor)
        for fault in [dict(temperature_c=55),dict(load_raw=900),dict(status_register=2)]:
            with self.assertRaises(RuntimeError):check_feedback(2,{**state,'position':2048,**fault},motor)
        self.assertEqual((motor['safe_min'],motor['safe_max']),(2000,2047))
        for value in [6,-1,True,1.5]:
            motor['position_feedback_tolerance_counts']=value
            with self.assertRaises(ValueError):live_preflight(profile,[2])

    def test_device_temperature_limit_replaces_trial_limit_without_disabling_guard(self):
        from xiaobai_runtime.playback import check_feedback
        profile,_,_=self.fixture();motor=profile['motors'][0]
        motor.update(device_max_temperature_c=80,max_temperature_c=80)
        live_preflight(profile,[2])
        state={'position':2047,'temperature_c':60,'load_raw':0,'status_register':0}
        check_feedback(2,state,motor)
        state['temperature_c']=80
        with self.assertRaisesRegex(RuntimeError,'over temperature'):check_feedback(2,state,motor)
        motor['max_temperature_c']=81
        with self.assertRaisesRegex(ValueError,'temperature'):live_preflight(profile,[2])
    def fixture(self):
        from unittest.mock import MagicMock
        profile={'mixing_confirmed':True,'jaw_owner_confirmed':True,'physical_audio_output_confirmed':True,
          'hardware_stop_description':'test fixture only','audio_device':0,'actuator_latency_ms':0,
          'port':'FAKE','baud':1000000,'stop_method':'stop transmission',
          'motors':[{'id':2,'neutral':2047,'neutral_confirmed':True,'register_layout_confirmed':True,
           'safe_min':1800,'safe_max':2300,'max_speed_counts_s':300,'max_temperature_c':55,
           'max_load_raw':900,'model_raw_expected':123}]}
        bus=MagicMock();position=[2047]
        bus.snapshot.return_value={'model_raw':123,'mode':0,'torque_enable':1,'configured_min':0,
          'configured_max':4095,'position_raw':2047,'temperature_c':25,'load_raw':0,'status_register':0}
        bus.feedback.side_effect=lambda sid:{'position':position[0],'temperature_c':25,'load_raw':0,'status_register':0}
        bus.goals.side_effect=lambda goals,speeds:position.__setitem__(0,goals[2])
        import time
        clock=MagicMock();clock.duration=.05;clock.fault=None;epoch=[None]
        clock.start.side_effect=lambda:epoch.__setitem__(0,time.monotonic())
        clock.time.side_effect=lambda:time.monotonic()-epoch[0]
        return profile,bus,clock
    def test_audio_clock_loop_with_mock_hardware(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        profile,bus,clock=self.fixture()
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2],np.array([[2047.],[2050.],[2055.]]),60,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
        self.assertTrue(result['completed']);self.assertGreaterEqual(result['sent_frames'],2)
        bus.close.assert_called_once();clock.close.assert_called_once()
        for call in bus.goals.call_args_list:self.assertLessEqual(abs(call.args[0][2]-2047),8)
    def test_30fps_keeps_both_feedback_samples_and_checks_second_axis_fault(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        profile,bus,clock=self.fixture();profile['feedback_polls_per_frame']=2
        profile['motors'].append({**profile['motors'][0],'id':3})
        clock.duration=2/30
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2,3],np.full((2,2),2047.),30,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
        self.assertTrue(result['completed'])
        for event in result['events']:
            self.assertEqual({s['id'] for s in event['feedback_samples']},{2,3})
        healthy=dict(position=2047,temperature_c=25,load_raw=0,status_register=0)
        bus.feedback.side_effect=[healthy,healthy,healthy,{**healthy,'status_register':2}]
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            with self.assertRaisesRegex(RuntimeError,'status 0x02'):
                play(profile,[2,3],np.full((2,2),2047.),30,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
    def test_fault_stops_before_motion(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        profile,bus,clock=self.fixture();clock.fault='simulated underrun'
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            with self.assertRaisesRegex(RuntimeError,'Audio callback fault'):
                play(profile,[2],np.full((3,1),2047.),60,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
        bus.goals.assert_not_called();bus.close.assert_called_once()
    def test_short_commissioning_run_preserves_audio_clock(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        profile,bus,clock=self.fixture()
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2],np.full((3,1),2047.),60,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP',max_seconds=.01)
        self.assertTrue(result['completed']);self.assertFalse(result['full_clip_requested'])
        self.assertEqual(result['requested_duration_seconds'],.01)
        self.assertEqual(result['sent_frames'],1)
    def test_bad_runtime_feedback_is_discarded_only_with_bounded_freshness(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        profile,bus,clock=self.fixture();profile['feedback_max_age_s']=.025
        healthy={'position':2047,'temperature_c':25,'load_raw':0,'status_register':0}
        bus.feedback.side_effect=[healthy,CorruptResponse('bad packet'),CorruptResponse('bad packet')]
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            log=Path(directory)/'log.json'
            with self.assertRaisesRegex(RuntimeError,'Feedback watchdog'):
                play(profile,[2],np.full((3,1),2047.),60,'fake.wav',log,Path(directory)/'STOP')
            saved=json.loads(log.read_text())
        self.assertTrue(all('feedback_error' in e and 'feedback' not in e for e in saved['events']))
        self.assertLessEqual(bus.goals.call_count,2)
    def test_explicit_record_only_keeps_playing_and_never_invents_feedback(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        from xiaobai_runtime.protocol import ServoStatusError
        profile,bus,clock=self.fixture();profile['feedback_policy']='record_only'
        bus.last_sdk_response=None
        profile['feedback_max_age_s']=.001
        healthy=dict(position=2047,temperature_c=25,load_raw=0,status_register=0)
        bus.feedback.side_effect=[healthy,ServoStatusError(2,1),CorruptResponse('bad packet'),
                                  {**healthy,'temperature_c':90,'status_register':32}]
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2],np.full((3,1),2047.),60,'fake.wav',Path(directory)/'log.json',Path(directory)/'STOP')
        self.assertTrue(result['completed'])
        self.assertEqual(result['feedback_policy'],'record_only')
        self.assertEqual(len(result['feedback_notifications']),3)
        self.assertFalse(result['feedback_notifications'][0]['valid_payload'])
        self.assertNotIn('feedback',result['events'][0]);self.assertNotIn('feedback',result['events'][1])
        for call in bus.goals.call_args_list:self.assertEqual(call.args[0],{2:2047})
    def test_record_only_still_rejects_targets_outside_limits_before_hardware(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        profile,_,_=self.fixture();profile['feedback_policy']='record_only'
        with patch('xiaobai_runtime.playback.Bus') as factory,patch('xiaobai_runtime.playback.AudioClock') as audio:
            with self.assertRaisesRegex(ValueError,'targets exceed command limits'):
                play(profile,[2],np.full((3,1),2400.),60,'fake.wav','unused','unused')
            factory.assert_not_called();audio.assert_not_called()
    def test_stop_file_blocks_audio_and_serial(self):
        from unittest.mock import patch
        from xiaobai_runtime.playback import play
        profile,_,_=self.fixture()
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus') as factory,patch('xiaobai_runtime.playback.AudioClock') as audio:
            stop=Path(directory)/'STOP';stop.touch()
            with self.assertRaisesRegex(RuntimeError,'Stop file'):
                play(profile,[2],np.full((3,1),2047.),60,'fake.wav',Path(directory)/'log.json',stop)
            factory.assert_not_called();audio.assert_not_called()

class BranchIsolationTests(unittest.TestCase):
    def test_restart_never_enables_before_goal_readback(self):
        from unittest.mock import Mock,patch
        import struct
        import xiaobai_rearm_ttl
        profile=json.loads((ROOT/'xiaobai/playback_profile_raw_trace.json').read_text())
        motors={m['id']:m for m in profile['motors']}
        for verified,auto_enable in [(True,False),(False,False),(True,True)]:
            with self.subTest(verified=verified,auto_enable=auto_enable),tempfile.TemporaryDirectory() as temp:
                bus=Mock();order=[];enabled=[False]
                def snapshot(sid):
                    m=motors[sid]
                    return dict(model_raw=521,mode=0,torque_enable=int(sid!=2 or enabled[0]),
                                position_raw=2047,configured_min=m['device_min'],configured_max=m['device_max'],
                                configured_max_temperature=m['device_max_temperature_c'],
                                temperature_c=40,load_raw=0,status_register=0,voltage_raw=70)
                def read(sid,address,count):
                    order.append(('read',address))
                    if address==42:return struct.pack('<HHH',2047,0,60) if verified else bytes(6)
                    return bytes([enabled[0]])
                def enable(sid,address,value):
                    self.assertEqual((sid,address,value),(2,40,1))
                    self.assertIn(('read',42),order);self.assertIn(('read',40),order)
                    order.append(('enable',40));enabled[0]=True;return 0,0
                bus.snapshot.side_effect=snapshot;bus.read.side_effect=read
                def prime(*args):
                    order.append(('prime',42))
                    if auto_enable:enabled[0]=True
                bus.goals.side_effect=prime
                bus.sdk.write1ByteTxRx.side_effect=enable
                bus.feedback.return_value=dict(position=2047,temperature_c=40,load_raw=0,status_register=0,voltage_raw=70)
                # Keep a real operator STOP in the workspace independent of this mock-device test.
                with patch('xiaobai_rearm_ttl.ROOT',Path(temp)),patch('xiaobai_rearm_ttl.read_json',return_value=profile), \
                        patch('sys.argv',['rearm','--apply','--output',str(Path(temp)/'run')]),patch('xiaobai_branch_trial.probe_branch'),patch('xiaobai_runtime.feetech_sdk.OfficialSDKBus',return_value=bus),patch('xiaobai_rearm_ttl.signal.signal'):
                    if verified:xiaobai_rearm_ttl.main()
                    else:
                        with self.assertRaisesRegex(RuntimeError,'priming verification failed'):xiaobai_rearm_ttl.main()
                self.assertEqual(enabled[0],verified);bus.close.assert_called_once()
                if verified and not auto_enable:self.assertLess(order.index(('prime',42)),order.index(('enable',40)))
                else:bus.sdk.write1ByteTxRx.assert_not_called()

    def test_restart_targets_are_nearby_and_never_reposition_enabled_axes(self):
        from xiaobai_rearm_ttl import rearm_target
        state=dict(model_raw=521,mode=0,torque_enable=0,position_raw=2039,configured_min=1850,configured_max=2108)
        motor=dict(safe_min=2047,safe_max=2108)
        self.assertEqual(rearm_target(state,motor),2049)
        self.assertEqual(rearm_target({**state,'position_raw':2050},motor),2050)
        for changes in [{'position_raw':2038},{'torque_enable':1},{'model_raw':11272},{'mode':1},{'torque_enable':2}]:
            with self.assertRaises(ValueError):rearm_target({**state,**changes},motor)

    def rows(self,online):
        return [dict(id=sid,model_raw=521 if sid<=30 else 11272,result='responded')
                if sid in online else dict(id=sid,result='no_response_twice')
                for sid in range(2,35)]

    def test_mixed_restart_accepts_verified_jaw_but_excludes_neck(self):
        from xiaobai_rearm_ttl import rearm_target
        from xiaobai_branch_trial import validate_branch_probe
        state=dict(id=31,model_raw=11272,mode=0,torque_enable=0,
                   position_raw=2047,configured_min=1900,configured_max=2047)
        motor=dict(safe_min=1950,safe_max=2047)
        self.assertEqual(rearm_target(state,motor,include_jaw=True),2047)
        with self.assertRaises(ValueError):rearm_target(state,motor)
        for sid in [29,32,33,34]:
            with self.assertRaises(ValueError):rearm_target({**state,'id':sid},motor,include_jaw=True)
        validate_branch_probe(self.rows(range(2,35)),'both')
        with self.assertRaises(ValueError):validate_branch_probe(self.rows(range(2,31)),'both')
        with self.assertRaises(ValueError):validate_branch_probe(self.rows(range(2,35)),'unknown')

    def test_whole_branches_pass_but_partial_disconnection_fails(self):
        from xiaobai_branch_trial import validate_branch_probe
        validate_branch_probe(self.rows(range(2,31)),'ttl')
        validate_branch_probe(self.rows(range(31,35)),'rs485-jaw')
        partial=[14,17,18,19,21,24,25,26,27,28,31,32,33,34]
        for branch in ['ttl','rs485-jaw']:
            with self.assertRaisesRegex(ValueError,'isolation not established'):
                validate_branch_probe(self.rows(partial),branch)

    def test_incomplete_scan_unknown_failure_and_wrong_model_cannot_pass(self):
        from xiaobai_branch_trial import validate_branch_probe
        rows=self.rows(range(31,35))
        with self.assertRaises(ValueError):validate_branch_probe(rows[1:],'rs485-jaw')
        rows[0]['result']='inconclusive'
        with self.assertRaises(ValueError):validate_branch_probe(rows,'rs485-jaw')
        rows=self.rows(range(31,35));rows[-1]['model_raw']=521
        with self.assertRaises(ValueError):validate_branch_probe(rows,'rs485-jaw')




class AudioBufferTests(unittest.TestCase):
    def test_invalid_latency_rejected_before_opening_resources(self):
        from xiaobai_runtime.playback import AudioClock
        for latency in [True, 0, -1, 1.1, float('nan'), float('inf'), 'invalid', None]:
            with self.subTest(latency=latency), self.assertRaises(ValueError):
                AudioClock('unused.wav', 'unused', latency=latency)

    def test_buffer_override_preserves_dac_clock(self):
        import wave
        from unittest.mock import patch
        from types import SimpleNamespace
        from xiaobai_runtime.playback import AudioClock
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audio.wav'
            with wave.open(str(path),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
                wav.writeframes(bytes(320))
            with patch('sounddevice.query_devices',return_value={'default_samplerate':48000}), patch('sounddevice.OutputStream') as stream:
                clock=AudioClock(path,'USB Audio Device',latency=.2)
                try:
                    self.assertEqual(stream.call_args.kwargs['latency'],.2)
                    self.assertIsNone(clock.time())
                    clock.callback(np.zeros((48,1)),48,SimpleNamespace(outputBufferDacTime=10.),False)
                    clock.stream.time=10.125
                    self.assertAlmostEqual(clock.time(),.125)
                    self.assertAlmostEqual(clock.duration,.01)
                finally:clock.close()

class StartupFeedbackTests(unittest.TestCase):
    def test_startup_requires_two_consecutive_complete_valid_samples(self):
        import time
        from unittest.mock import Mock
        from xiaobai_runtime.playback import startup_feedback
        healthy=dict(position=2047,load_raw=0,temperature_c=40,status_register=0)
        reader=Mock(side_effect=[CorruptResponse('two bad packets in bus.read'),healthy,
                                 ReadTimeout('lost fresh sample'),healthy,healthy])
        records=[];wait=Mock()
        result=startup_feedback(reader,4,time.monotonic()+1,records,wait=wait)
        self.assertEqual(result,healthy);self.assertEqual(reader.call_count,5)
        self.assertTrue(records[0]['recovered']);self.assertEqual(wait.call_count,4)

    def test_startup_recovery_is_bounded_and_never_retries_actual_alarm(self):
        import time
        from unittest.mock import Mock
        from xiaobai_runtime.playback import startup_feedback,SensorReadingError
        from xiaobai_runtime.protocol import ServoStatusError
        records=[];reader=Mock(side_effect=CorruptResponse('persistent bad packet'))
        with self.assertRaisesRegex(RuntimeError,'startup communication recovery failed'):
            startup_feedback(reader,4,time.monotonic()+1,records,wait=Mock())
        self.assertEqual(reader.call_count,5);self.assertFalse(records[0]['recovered'])
        for fault in (SensorReadingError('hot'),ServoStatusError(4,4),ProtocolError('5% gate')):
            reader=Mock(side_effect=fault)
            with self.assertRaises(type(fault)):
                startup_feedback(reader,4,time.monotonic()+1,[],wait=Mock())
            self.assertEqual(reader.call_count,1)
        reader=Mock(side_effect=ReadTimeout('missing'))
        with self.assertRaises(ReadTimeout):
            startup_feedback(reader,4,time.monotonic()+1,[],wait=Mock(),can_recover=lambda:False)
        self.assertEqual(reader.call_count,1)

    def test_startup_recovery_honors_deadline_and_operator_stop(self):
        import time
        from unittest.mock import Mock
        from xiaobai_runtime.playback import startup_feedback
        reader=Mock(side_effect=ReadTimeout('missing'))
        with self.assertRaisesRegex(RuntimeError,'startup communication recovery failed'):
            startup_feedback(reader,4,time.monotonic(),[],wait=Mock())
        self.assertEqual(reader.call_count,1)
        reader.reset_mock()
        with self.assertRaisesRegex(RuntimeError,'Operator stop'):
            startup_feedback(reader,4,time.monotonic()+1,[],should_stop=lambda:True,wait=Mock())
        reader.assert_not_called()


if __name__=='__main__':unittest.main()
