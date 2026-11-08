"""Standalone realtime speech: no hardware/network in offline tests."""
import asyncio
import base64
from contextlib import redirect_stdout,redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.paths import ROOT
from vivis.speech.realtime import VoiceSession,Journal,connect_session,configuration,handshake_diagnostic,key_status,conversation,main
from vivis.speech.realtime_audio import VoiceBuffer


def arguments(**extra):
    return SimpleNamespace(**dict(dict(config=ROOT/'configs/perception/realtime_voice.json',region=None,
        workspace_id=None,robot=None,voice=None,model=None,seconds=10,speaker=None,mic=None),**extra))


class RealtimeVoiceTests(unittest.TestCase):
    def test_resampling_is_continuous_across_network_packet_seams(self):
        pcm=(np.sin(np.arange(2400)*.13)*10000).astype('<i2')
        a=VoiceBuffer();b=VoiceBuffer();a.add('a',pcm.tobytes())
        for piece in np.array_split(pcm,11):b.add('b',piece.tobytes())
        np.testing.assert_array_equal(a.render(4800,10.),b.render(4800,10.))

    def test_cancel_bounds_tail_and_rejects_late_old_audio_without_stopping_new_audio(self):
        player=VoiceBuffer(fade_ms=40);pcm=np.full(24000,2000,dtype='<i2').tobytes()
        player.add('old',pcm);player.render(960,10.);result=player.cancel('old')
        self.assertEqual(result['fade_samples'],1920);self.assertEqual(player.count,1920)
        self.assertFalse(player.add('old',pcm))
        player.add('new',pcm);player.cancel('old')
        self.assertNotIn('new',player.cancelled)
        self.assertTrue(np.any(player.render(1920+48000,10.)[3000:]))
        unseen=VoiceBuffer();unseen.add('unheard',pcm);unseen.cancel('unheard')
        self.assertEqual(unseen.count,0)

    def test_response_done_does_not_discard_unplayed_audio_and_stale_cancellation_is_scoped(self):
        player=VoiceBuffer();log=Mock();state=VoiceSession(player,log)
        def add(rid):
            state.event(dict(type='response.created',response=dict(id=rid)))
            state.event(dict(type='response.audio.delta',response_id=rid,delta=base64.b64encode(b'\x10\x10'*2400).decode()))
        add('a');state.event(dict(type='response.done',response=dict(id='a',status='completed')))
        self.assertEqual(player.count,4800)
        add('b');state.event(dict(type='response.done',response=dict(id='a',status='cancelled')))
        self.assertEqual(state.generating,'b');self.assertNotIn('b',state.blocked)
        self.assertFalse(player.add('a',b'\x00\x00'))

    def test_barge_in_rejects_delayed_packets_and_logs_no_base64_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events.jsonl';log=Journal(path,secret='test-key-not-real')
            player=VoiceBuffer();state=VoiceSession(player,log)
            state.event(dict(type='response.created',response=dict(id='one')))
            encoded=base64.b64encode(b'\x10\x11'*2400).decode()
            state.event(dict(type='response.audio.delta',response_id='one',delta=encoded))
            state.event(dict(type='input_audio_buffer.speech_started',item_id='user'))
            state.event(dict(type='response.audio.delta',response_id='one',delta=encoded))
            self.assertEqual(state.late_packets,1);self.assertLessEqual(player.count,1920)
            log.write('test',message='test-key-not-real');log.close();text=path.read_text()
            self.assertNotIn(encoded,text);self.assertNotIn('test-key-not-real',text);self.assertIn('[redacted]',text)

    def test_session_requires_server_confirmation_of_male_voice_and_smart_turn(self):
        _,_,_,_,session=configuration(arguments())
        class WS:
            def __init__(self,accepted):self.accepted=accepted;self.sent=[]
            async def send(self,value):self.sent.append(json.loads(value))
            async def recv(self):return json.dumps(dict(type='session.updated',session=self.accepted))
        ws=WS(session);asyncio.run(connect_session(ws,session,Mock()))
        self.assertEqual(ws.sent[0]['session']['voice'],'longanyang_v3.1')
        self.assertEqual(ws.sent[0]['session']['tools'],[])
        self.assertEqual(ws.sent[0]['session']['turn_detection']['type'],'smart_turn_v2')
        with self.assertRaisesRegex(RuntimeError,'confirm'):
            asyncio.run(connect_session(WS({**session,'voice':'other'}),session,Mock()))

    def test_offline_check_and_missing_key_cannot_open_audio_network_or_actuators(self):
        with patch('vivis.speech.realtime.AudioIO') as audio,patch('asyncio.run') as network:
            with redirect_stdout(io.StringIO()) as out:self.assertEqual(main(['--check']),0)
            self.assertFalse(json.loads(out.getvalue())['hardware_opened'])
            with patch.dict('os.environ',{},clear=True),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(['--probe'])
            audio.assert_not_called();network.assert_not_called()

    def test_region_is_explicit_and_handshake_diagnostic_never_dumps_headers_or_key(self):
        _,_,_,url,_=configuration(arguments(region='singapore',workspace_id='example'))
        self.assertIn('example.ap-southeast-1.maas.aliyuncs.com',url)
        error=SimpleNamespace(response=SimpleNamespace(status_code=401,
            body=json.dumps(dict(code='InvalidApiKey',message='bad test-secret',request_id='r1')).encode(),
            headers={'Authorization':'Bearer test-secret','set-cookie':'private-cookie'}))
        result=handshake_diagnostic(error,'test-secret')
        self.assertEqual(result['http_status'],401)
        self.assertNotIn('test-secret',str(result));self.assertNotIn('private-cookie',str(result))
        self.assertIn('[redacted]',result['message'])
        self.assertTrue(key_status('sk-not-a-real-key')['starts_with_sk'])
        self.assertNotIn('sk-not-a-real-key',str(key_status('sk-not-a-real-key')))

    def test_probe_can_confirm_a_session_without_constructing_audio_io(self):
        cfg,speech,speaker,url,session=configuration(arguments())
        class Connection:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):pass
            async def send(self,data):pass
            async def recv(self):return json.dumps(dict(type='session.updated',session=session))
        client=ModuleType('websockets.asyncio.client');client.connect=Mock(return_value=Connection())
        with patch.dict('sys.modules',{'websockets':ModuleType('websockets'),
                'websockets.asyncio':ModuleType('websockets.asyncio'),'websockets.asyncio.client':client}), \
             patch('vivis.speech.realtime.AudioIO') as audio,redirect_stdout(io.StringIO()):
            result=asyncio.run(conversation(SimpleNamespace(probe=True),cfg,speech,speaker,url,session,
                                           'unit-test-placeholder',Path('/unused'),Mock(),{}))
        audio.assert_not_called();self.assertTrue(result['probe']);self.assertFalse(result['audio_opened'])

    def test_authentication_failure_is_redacted_and_saved_without_touching_devices(self):
        class Rejected(Exception):
            def __init__(self):
                super().__init__('rejected fake-private-key')
                self.response=SimpleNamespace(status_code=401,body=b'{"code":"InvalidApiKey","message":"bad fake-private-key"}',headers={})
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{'DASHSCOPE_API_KEY':'fake-private-key'}), \
             patch('vivis.speech.realtime.importlib.util.find_spec',return_value=object()), \
             patch('vivis.speech.realtime.AudioIO') as audio, \
             patch('asyncio.run',side_effect=lambda co:(co.close(),(_ for _ in ()).throw(Rejected()))[-1]), \
             redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()) as err:
            folder=Path(tmp)/'run';self.assertEqual(main(['--probe','--output',str(folder)]),1)
            summary=json.loads((folder/'summary.json').read_text())
            self.assertEqual(summary['handshake']['http_status'],401)
            self.assertNotIn('fake-private-key',err.getvalue())
            self.assertNotIn('fake-private-key',(folder/'events.jsonl').read_text())
            audio.assert_not_called()


if __name__=='__main__':unittest.main()
