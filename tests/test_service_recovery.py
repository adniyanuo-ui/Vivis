"""Transient provider failures must not reboot devices or replay robot actions."""
import asyncio
import json
import unittest
from unittest.mock import Mock,patch

from vivis.paths import ROOT
from vivis.speech.service_recovery import ServiceRecovery,TransientServiceError,transient_error
from vivis.speech.realtime import connect_session


def policy():return json.loads((ROOT/'configs/perception/realtime_voice.json').read_text())['service_recovery']


class RecoveryPolicyTests(unittest.TestCase):
    def test_only_explicit_retryable_codes_are_classified(self):
        for code in ('InternalError','ServiceUnavailable'):
            event=dict(type='error',event_id='provider-event',error=dict(code=code,message='temporary'))
            error=transient_error(event)
            self.assertIsInstance(error,TransientServiceError);self.assertEqual(error.event_id,'provider-event')
            self.assertIsInstance(transient_error(dict(type='response.done',response=dict(status='failed',status_details=dict(error=event['error'])))),TransientServiceError)
        for event in (dict(type='error',error=dict(code='InvalidApiKey')),dict(type='error',error=dict(code='invalid_value')),
                      dict(type='error',error=dict(code='InternalError.InvalidInput')),
                      dict(type='response.done',response=dict(status='failed',status_details=None))):
            self.assertIsNone(transient_error(event))

    def test_three_recoveries_per_window_even_if_connections_succeeded_between_errors(self):
        retry=ServiceRecovery(policy())
        for stamp,delay in ((100,.5),(101,1),(102,2)):
            self.assertEqual(retry.next(stamp)['delay_s'],delay)
        with self.assertRaisesRegex(RuntimeError,'exhausted'):retry.next(103)
        self.assertEqual(retry.next(161)['attempt'],2)  #100/101 expire;102 is still in window.
        self.assertEqual(retry.next(222)['attempt'],1)

    def test_invalid_retry_policies_are_rejected(self):
        for change in (dict(max_reconnects=True),dict(window_s=float('nan')),dict(backoff_s=[0,1,2]),dict(backoff_s=[.5])):
            with self.assertRaises(ValueError):ServiceRecovery({**policy(),**change})


class RecoveryRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import tempfile
        from test_cloud_control import CloudBackendTests
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.events=[];self.service=CloudBackendTests().service(self.tmp.name,self.events)
        self.addCleanup(self.service.log.close)

    async def test_recovery_retains_context_and_ready_state_but_does_not_replay_old_intent(self):
        service=self.service;original=dict(revision=4,mode='interaction',target='interaction',ready=True)
        service.context.update(original);service.enabled=True
        service.resume_history.append(dict(role='user',text='合成历史背景。'))
        service.continue_response=True;service.announcement_revision=4
        calls=0
        async def connection():
            nonlocal calls
            calls+=1
            if calls==1:raise TransientServiceError('InternalError','temporary','event1')
            self.assertEqual(service.context,original);self.assertTrue(service.enabled)
            self.assertEqual(len(service.resume_history),1)
            self.assertTrue(service.awaiting_recovery_user)
            self.assertIsNone(service.announcement_revision);self.assertFalse(service.continue_response)
            service.shutdown.set()
        async def no_delay(_):pass
        with patch.object(service,'_connection',side_effect=connection),patch('vivis.speech.cloud_managed.asyncio.sleep',side_effect=no_delay):
            await service.cloud(0)
        self.assertEqual(calls,2)
        self.assertTrue(any(e['kind']=='voice_warning' for e in self.events))
        self.assertFalse(any(e['kind'] in ('cloud_error','cloud_action') for e in self.events))

    async def test_persistent_service_failure_is_bounded_and_fails_closed(self):
        service=self.service;count=0
        async def connection():
            nonlocal count
            count+=1;raise TransientServiceError('InternalError','persistent')
        async def no_delay(_):pass
        with patch.object(service,'_connection',side_effect=connection),patch('vivis.speech.cloud_managed.asyncio.sleep',side_effect=no_delay):
            await service.cloud(0)
        self.assertEqual(count,4);self.assertTrue(service.shutdown.is_set());self.assertFalse(service.enabled)
        self.assertEqual(sum(e['kind']=='voice_warning' for e in self.events),3)
        self.assertEqual(sum(e['kind']=='cloud_error' for e in self.events),1)

    async def test_handshake_internal_error_is_typed_but_bad_key_is_not_retryable(self):
        class WS:
            def __init__(self,code):self.code=code
            async def send(self,_):pass
            async def recv(self):return json.dumps(dict(type='error',error=dict(code=self.code,message='test')))
        with self.assertRaises(TransientServiceError):await connect_session(WS('InternalError'),{},Mock())
        with self.assertRaises(RuntimeError) as error:await connect_session(WS('InvalidApiKey'),{},Mock())
        self.assertNotIsInstance(error.exception,TransientServiceError)

    async def test_real_receive_path_keeps_speaker_lease_and_recovers_until_fresh_speech(self):
        from test_managed_voice import FakeAudio
        service=self.service;service.audio=FakeAudio({},'',service.player)
        service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
        service.enabled=True;service.presence.wake();service.reception.c=dict(service.reception.c,enabled=False)
        service.resume_history.append(dict(role='user',text='合成历史保留。'))
        connections=[];sessions=[]
        class WS:
            def __init__(self):self.messages=asyncio.Queue();self.sent=[]
            async def __aenter__(self):connections.append(self);return self
            async def __aexit__(self,*_):return False
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await self.messages.get())
            async def send(self,text):self.sent.append(json.loads(text))
        async def connected(ws,session,log):sessions.append(session);return session
        async def until(predicate):
            for _ in range(300):
                if task.done():task.result()
                if predicate():return
                await asyncio.sleep(.01)
            raise AssertionError('Recovery did not finish')
        with patch('vivis.speech.cloud_managed.websocket_connection',side_effect=lambda *_:WS()), \
             patch('vivis.speech.cloud_managed.connect_session',side_effect=connected):
            task=asyncio.create_task(service.cloud(0))
            try:
                await until(lambda:len(sessions)==1 and service.audio.output_stream is not None)
                stream=service.audio.output_stream
                await connections[0].messages.put(dict(type='response.created',response=dict(id='failed')))
                await connections[0].messages.put(dict(type='response.audio_transcript.done',response_id='failed',transcript=''))
                await connections[0].messages.put(dict(type='response.done',response=dict(id='failed',status='failed')))
                await connections[0].messages.put(dict(type='error',event_id='injected-event',error=dict(code='InternalError',message='injected')))
                await until(lambda:len(sessions)==2 and service.ws is not None)
                self.assertIs(service.audio.output_stream,stream)
                self.assertEqual(service.audio.opens,0);self.assertEqual(service.context['revision'],1)
                self.assertIn('合成历史保留',json.dumps(connections[1].sent,ensure_ascii=False))
                self.assertTrue(service.awaiting_recovery_user)
                self.assertTrue(any(e['kind']=='voice_warning' and '已恢复' in e['message'] for e in self.events))
                self.assertFalse(any(p['type']=='response.create' for p in connections[1].sent))
                self.assertFalse(any(e['kind']=='cloud_transcript' and not e['text'].strip() for e in self.events))
                await connections[1].messages.put(dict(type='response.created',response=dict(id='unrequested')))
                await connections[1].messages.put(dict(type='response.function_call_arguments.done',response_id='unrequested',
                    call_id='replay',name='robot_program',arguments=json.dumps(dict(action='play',program='baigufuren'))))
                await until(lambda:any(p.get('item',{}).get('call_id')=='replay' for p in connections[1].sent))
                self.assertFalse(any(e['kind']=='cloud_action' for e in self.events))
                await connections[1].messages.put(dict(type='input_audio_buffer.speech_started',item_id='fresh-user'))
                await until(lambda:not service.awaiting_recovery_user)
                self.assertFalse(any(e['kind']=='cloud_error' for e in self.events))
            finally:
                task.cancel();await asyncio.gather(task,return_exceptions=True)


if __name__=='__main__':unittest.main()
