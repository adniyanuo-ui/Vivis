"""Catch deployment-env mismatches and expose non-admission without hardware."""
import asyncio
import builtins
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from vivis.interaction.sound_attention import configuration,SideSpeechGate
from vivis.paths import ROOT
from vivis.speech.sound_direction import DirectionObserver,configuration as direction_configuration,runtime_dependencies


class RuntimeDependencyTests(unittest.TestCase):
    def test_selected_environment_not_supervisor_determines_module_presence(self):
        with tempfile.TemporaryDirectory() as folder:
            python=Path(folder)/'venv/bin/python';python.parent.mkdir(parents=True);python.touch()
            site=python.parent.parent/'lib/python3.12/site-packages';site.mkdir(parents=True)
            (site/'webrtcvad.py').write_text("raise AssertionError('No module execution in offline check')")
            result=runtime_dependencies(python)
            self.assertFalse(result['available']);self.assertEqual(result['missing_modules'],['serial'])
            (site/'serial').mkdir();(site/'serial/__init__.py').write_text("raise AssertionError('No serial opened')")
            result=runtime_dependencies(python)
            self.assertTrue(result['available'])
            self.assertEqual(result['check'],'module_presence_only_no_device_open')

    def test_speech_install_manifest_includes_serial_not_just_motion_environment(self):
        self.assertIn('pyserial==3.5',(ROOT/'tools/requirements/speech.txt').read_text().splitlines())

    def test_unknown_environment_is_unknown_not_a_success(self):
        self.assertIsNone(runtime_dependencies('/not-an-installed-env/bin/python')['available'])

    def test_missing_serial_import_is_reported_without_opening_another_device(self):
        observer=DirectionObserver(direction_configuration(),'/dev/ttyCH343USB0')
        original=builtins.__import__
        def missing(name,*args,**kwargs):
            if name=='serial.tools.list_ports':
                observer.stop.set();raise ModuleNotFoundError("No module named 'serial'")
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=missing), \
                patch('vivis.speech.sound_direction.open_receiver',side_effect=AssertionError('opened hardware')):
            observer.run()
        state=observer.status(10.)
        self.assertEqual(state['status'],'error');self.assertIn("No module named 'serial'",state['error'])
        self.assertFalse(state['serial_open']);self.assertIsNone(state['sector'])

    def test_observer_receives_zero_in_actual_import_path_and_keeps_device_ownership(self):
        observer=DirectionObserver(direction_configuration(),'/dev/ttyCH343USB0');states=[]
        receiver=Mock(in_waiting=5);calls=[]
        def read(_):
            calls.append(1)
            if len(calls)==1:return bytes.fromhex('55 AA 00 2C 7E')
            states.append(observer.status());observer.stop.set();return b''
        receiver.read.side_effect=read
        port=SimpleNamespace(device='/dev/ttyUSB0',vid=0x1a86,pid=0x7523)
        with patch('serial.tools.list_ports.comports',return_value=[port]), \
                patch('vivis.speech.sound_direction.open_receiver',return_value=receiver) as opened:
            observer.run()
        self.assertEqual(opened.call_args.args[0],'/dev/ttyUSB0')
        self.assertEqual(states[0]['status'],'fresh');self.assertEqual(states[0]['sector'],0)
        self.assertEqual(states[0]['statistics']['valid_frames'],1)
        self.assertFalse(states[0]['checksum_verified'])
        receiver.close.assert_called_once();receiver.write.assert_not_called()


class AdmissionDiagnosticsTests(unittest.TestCase):
    def test_stateful_vad_still_observes_eligible_same_direction_frames(self):
        vad=Mock(return_value=True);g=SideSpeechGate(configuration(),vad)
        result=g.observe(10.,b'x',dict(clean_rms=.01,reference_rms=0.),
            dict(updated_monotonic=10.,frame_valid=True,sector=8),
            dict(updated_monotonic=10.,phase='idle',single_face=True,settled=True,main_bearing_deg=0.),
            available=True,now=10.)
        self.assertEqual(result,(False,None));self.assertEqual(g.diagnostic['reason'],'same_direction')
        vad.assert_called_once_with(b'x')

    def test_unsettled_gaze_does_not_discard_semantics_but_echo_is_still_denied(self):
        c=configuration();g=SideSpeechGate(c,lambda _:True)
        feedback=dict(updated_monotonic=10.,phase='idle',single_face=True,settled=False,main_bearing_deg=0.)
        direction=dict(updated_monotonic=10.,frame_valid=True,sector=0)
        metrics=dict(clean_rms=.01,reference_rms=0.)
        self.assertEqual(g.observe(10.,b'x',metrics,direction,feedback,available=True,now=10.),(True,None))
        self.assertEqual(g.diagnostic['reason'],'collecting_speech')
        feedback['settled']=True;metrics['reference_rms']=.1
        self.assertEqual(g.observe(10.02,b'x',metrics,direction,feedback,available=True,now=10.02),(False,None))
        self.assertEqual(g.diagnostic['reason'],'robot_playback_overlap')
        metrics['reference_rms']=0.
        self.assertEqual(g.observe(10.04,b'x',metrics,None,feedback,available=True,now=10.04),(False,None))
        self.assertEqual(g.diagnostic['reason'],'direction_unavailable')


class RuntimeStatusTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from test_cloud_control import CloudBackendTests
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.s=CloudBackendTests().service(self.tmp.name,[])
        self.addCleanup(self.s.log.close);self.addCleanup(self.s.recorders.close)
        self.addCleanup(self.s.conversation_animation.close)
        self.s.sound_gate=SideSpeechGate(configuration(),lambda _:True)
        self.s.direction_observer=DirectionObserver(direction_configuration(),'/dev/ttyCH343USB0')

    async def test_missing_module_is_visible_once_and_main_chat_is_not_disabled(self):
        s=self.s;s.direction_observer.error="ModuleNotFoundError: No module named 'serial'"
        s.log=Mock();events=[];s.emit=events.append
        enabled=s.enabled
        for now in (10.,10.3,10.6):s.poll_sound_diagnostics(now,dict(clean_rms=.01,reference_rms=0.))
        self.assertEqual(len(events),1);self.assertIn("No module named 'serial'",events[0]['message'])
        self.assertIn('普通聊天继续',events[0]['message']);self.assertEqual(s.enabled,enabled)
        self.assertFalse(s.sound_status()['current_speaker_location_confirmed'])

    async def test_fresh_frame_logs_sector_and_no_fabricated_position_or_identity(self):
        from vivis.hardware.haokai import DirectionParser,DirectionState
        s=self.s;observer=s.direction_observer;observer.opened=True;observer.port='/dev/ttyUSB0';observer.generation=1
        state=DirectionState(observer.c['coordinates'],.75,False)
        state.observe(DirectionParser(False).feed(bytes.fromhex('55 AA 00 2C 7E'))[0],10.)
        observer.latest=state.snapshot(10.1);events=[];s.emit=events.append
        old=(s.memory_session,s.input_generation)
        s.poll_sound_diagnostics(10.1,dict(clean_rms=.01,reference_rms=0.))
        self.assertTrue(any('坐标00' in e['message'] for e in events))
        self.assertEqual(s.sound_status(now=10.2),s.sound_status(now=10.3))  # No hot timestamps in cloud instructions.
        status=s.sound_status(details=True,now=10.3)
        self.assertEqual(status['observer_status'],'fresh');self.assertFalse(status['current_speaker_location_confirmed'])
        self.assertTrue(status['head_yaw_supported']);self.assertFalse(status['manual_turn_tool_available'])
        self.assertFalse(status['direction_reference_calibrated'])
        self.assertEqual((s.memory_session,s.input_generation),old)
        self.assertEqual(s.sound_status(now=11.)['observer_status'],'held')
        self.assertFalse(s.sound_status(now=11.)['direction_event_fresh'])
        self.assertEqual(s.sound_status(now=12.)['observer_status'],'unresponsive')

    async def test_status_tool_returns_real_capability_and_direction_error_for_anonymous_user(self):
        from vivis.speech.realtime import VoiceSession
        s=self.s;s.context.update(mode='interaction',target='interaction',ready=True,revision=1)
        s.identified_person=lambda:None;s.presence.visible=lambda:True;s.cloud_state=VoiceSession(s.player,s.log)
        s.input_generation=1;s.begin_user_turn('where');s.response_revisions['r']=1;s.response_inputs['r']=1
        s.direction_observer.error="ModuleNotFoundError: No module named 'serial'"
        sent=[]
        async def send(value):sent.append(json.loads(value))
        await s.tool(SimpleNamespace(send=send),dict(name='get_robot_status',arguments='{}',call_id='where',response_id='r'))
        output=json.loads(next(e['item']['output'] for e in sent if e.get('item',{}).get('type')=='function_call_output'))
        self.assertTrue(output['accepted'])
        self.assertEqual(output['sound_attention']['observer_status'],'error')
        self.assertFalse(output['sound_attention']['current_speaker_location_confirmed'])
        self.assertTrue(output['sound_attention']['head_yaw_supported']);self.assertIsNone(output['current_person'])

    async def test_voice_runtime_can_start_diagnostic_writer_without_motion_dependencies(self):
        import time
        s=self.s;s.sound_started=False
        s.context.update(mode='standby',ready=False)
        s.sound_mailbox=SimpleNamespace(start=Mock(),close=Mock())
        with patch.object(s.direction_observer,'start',return_value=s.direction_observer):
            self.assertFalse(s.acoustic_microphone(time.monotonic(),bytes(640),dict(clean_rms=0.,reference_rms=0.)))
        await asyncio.to_thread(s.close_sound_attention)
        data=json.loads((Path(self.tmp.name)/'sound_direction.json').read_text())
        self.assertEqual(data['observer_status'],'stopped')


if __name__=='__main__':unittest.main()
