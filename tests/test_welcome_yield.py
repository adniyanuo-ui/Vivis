"""User evidence preceding speaker output must cancel only proactive welcome."""
import asyncio
import base64
import json
import tempfile
import time
import unittest
from unittest.mock import Mock,patch
import numpy as np
from vivis.speech.welcome import WelcomeTurnGate
from vivis.speech.realtime_audio import VoiceBuffer
import test_cloud_control
from test_managed_voice import FakeAudio


class WelcomeGateTests(unittest.TestCase):
    def setUp(self):self.g=WelcomeTurnGate();self.player=VoiceBuffer()
    def event(self,kind='conversation.item.input_audio_transcription.delta'):
        return dict(type=kind,item_id='user',text='')
    def test_early_empty_asr_prevents_welcome_before_request(self):
        self.assertFalse(self.g.user_event(self.event(),self.player));self.assertFalse(self.g.arm())
    def test_during_ack_or_before_created_late_response_is_cancelled(self):
        self.assertTrue(self.g.arm());self.g.sent()
        self.assertTrue(self.g.user_event(self.event(),self.player))
        self.assertTrue(self.g.created('welcome'));self.assertFalse(self.g.created('answer'))
    def test_buffered_welcome_has_not_started_and_is_yielded(self):
        self.g.arm();self.g.sent();self.g.created('welcome')
        self.player.add('welcome',b'\x20\x20'*2400)
        self.assertTrue(self.g.user_event(self.event(),self.player))
        self.assertEqual(self.g.responses,{'welcome'})
    def test_already_playing_welcome_uses_original_barge_in(self):
        self.g.arm();self.g.sent();self.g.created('welcome')
        self.player.add('welcome',b'\x20\x20'*2400);self.player.render(960,10.)
        self.assertFalse(self.g.user_event(self.event(),self.player));self.assertFalse(self.g.cancelled)
        self.assertGreater(self.player.count,0)
    def test_connection_reset_and_ordinary_reply_are_not_welcome(self):
        self.assertFalse(self.g.created('answer'))
        self.assertFalse(self.g.user_event(self.event(),self.player))
        self.assertFalse(WelcomeTurnGate().user_seen)
    def test_missing_source_and_robot_transcript_are_not_user_evidence(self):
        self.g.arm()
        self.assertFalse(self.g.user_event(dict(type='conversation.item.input_audio_transcription.delta'),self.player))
        self.assertFalse(self.g.user_event(dict(type='response.audio_transcript.delta',item_id='robot'),self.player))
        self.assertTrue(self.g.active)


class WelcomeReceiveTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,late_created=False,with_tool=False):
        queue=asyncio.Queue();sent=[]
        class WS:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await queue.get())
            async def send(self,raw):
                packet=json.loads(raw);sent.append(packet)
                if packet['type']=='session.update':
                    await queue.put(dict(type='session.updated',session=packet['session']))
                if packet['type']=='conversation.item.create' and 'id' in packet['item']:
                    await queue.put(dict(type='conversation.item.created',item=dict(id=packet['item']['id'])))
        async def connected(ws,session,log):return session
        with tempfile.TemporaryDirectory() as tmp,patch('vivis.speech.managed.AudioIO',FakeAudio), \
             patch('vivis.speech.cloud_managed.websocket_connection',return_value=WS()), \
             patch('vivis.speech.cloud_managed.connect_session',side_effect=connected):
            s=test_cloud_control.CloudBackendTests().service(tmp,[])
            s.context_snapshot=lambda:('synthetic',dict(recognized=False,person_id=None),True)
            s.verify_identity=Mock();s.schedule_task_notice=Mock();s.memory=None
            s.reception.c=dict(s.reception.c,enabled=False)
            s.context.update(revision=1,mode='interaction',target='interaction',ready=True)
            s.enabled=True;s.presence.wake();s.announcement_revision=1
            task=asyncio.create_task(s._connection())
            async def until(predicate):
                end=time.monotonic()+3
                while not predicate():
                    if task.done():task.result()
                    self.assertLess(time.monotonic(),end);await asyncio.sleep(.01)
            pcm=base64.b64encode(b'\x20\x20'*2400).decode()
            try:
                await until(lambda:any(p['type']=='response.create' for p in sent))
                if with_tool:
                    await queue.put(dict(type='response.created',response=dict(id='welcome-status')))
                    await queue.put(dict(type='response.function_call_arguments.done',response_id='welcome-status',name='get_robot_status',call_id='status',arguments='{}'))
                    await queue.put(dict(type='response.done',response=dict(id='welcome-status',status='completed')))
                    await until(lambda:sum(p['type']=='response.create' for p in sent)>=2)
                evidence=dict(type='conversation.item.input_audio_transcription.delta',item_id='u',text='')
                if late_created:await queue.put(evidence)
                await queue.put(dict(type='response.created',response=dict(id='welcome')))
                if not late_created:
                    await queue.put(dict(type='response.audio.delta',response_id='welcome',delta=pcm))
                    await until(lambda:s.player.count>0)
                    await queue.put(evidence)
                await until(lambda:'welcome' in s.blocked)
                await queue.put(dict(type='response.audio.delta',response_id='welcome',delta=pcm))
                await queue.put(dict(type='response.function_call_arguments.done',response_id='welcome',name='get_robot_status',call_id='old',arguments='{}'))
                await queue.put(dict(type='response.done',response=dict(id='welcome',status='cancelled')))
                await queue.put(dict(type='input_audio_buffer.speech_started',item_id='u'))
                await queue.put(dict(type='input_audio_buffer.speech_stopped',item_id='u'))
                await queue.put(dict(type='conversation.item.input_audio_transcription.completed',item_id='u',transcript='小帅，晚上好。'))
                await until(lambda:s.pending_user_text=='小帅，晚上好。')
                self.assertEqual(s.player.count,0)
                outputs=[p['item'] for p in sent if p.get('item',{}).get('call_id')=='old']
                self.assertEqual(len(outputs),1)
                self.assertEqual(json.loads(outputs[0]['output'])['status'],'welcome_cancelled')
                await queue.put(dict(type='response.created',response=dict(id='answer')))
                await queue.put(dict(type='response.audio.delta',response_id='answer',delta=pcm))
                await queue.put(dict(type='response.audio_transcript.done',response_id='answer',transcript='何工，晚上好！'))
                await queue.put(dict(type='response.done',response=dict(id='answer',status='completed')))
                await until(lambda:s.player.count>0)
                self.assertNotIn('answer',s.blocked)
                self.assertTrue(np.any(s.player.render(960,20.)))
                self.assertTrue(any(p['type']=='response.cancel' for p in sent))
            finally:
                task.cancel();await asyncio.gather(task,return_exceptions=True);s.log.close()
    async def test_user_asr_precedes_queued_welcome_output(self):await self.exercise()
    async def test_user_asr_precedes_welcome_response_created(self):await self.exercise(True)

    async def test_user_precedes_welcome_after_identity_tool_continuation(self):await self.exercise(with_tool=True)
