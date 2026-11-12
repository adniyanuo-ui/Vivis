import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch

from test_cloud_control import CloudBackendTests
from test_managed_voice import FakeAudio
from vivis.speech.explicit_wake import addressed_wake,AudioPermissionGate
from vivis.speech.realtime import VoiceSession
from vivis.speech.runtime_policy import policy,VoiceTransportError


class ExplicitWakeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.events=[]
        self.s=CloudBackendTests().service(tmp.name,self.events);self.addCleanup(self.s.log.close)
        self.s.enabled=True;self.s.context.update(mode='interaction',target='interaction',ready=True,revision=1)
        self.s.presence.visible=Mock(return_value=False);self.s.identified_person=Mock(return_value=None)
        self.state=VoiceSession(self.s.player,self.s.log)

    def final(self,text='小帅你是哪你是哪一家的模型？',invalid=False):
        s=self.s;s.input_generation+=1;s.begin_user_turn('user');s.question_started('user',self.state)
        import time
        s.turn_ended_at=time.monotonic()
        if invalid:
            s.question_invalidated(dict(item_id='user'));s.turn_ended_at=None
            s.memory_turns['user']['valid']=False;s.turn_item_id=None
        return dict(type='conversation.item.input_audio_transcription.completed',item_id='user',transcript=text)

    async def test_direct_address_not_third_person_report(self):
        for text in ('小帅','小帅你是哪你是哪一家的模型？','喂，小帅，唱首歌','你好小帅'):
            self.assertTrue(addressed_wake(text,['小帅']),text)
        for text in ('他说小帅你好','不要叫小帅','小帅说过这个','小帅的模型是什么','你好小王'):
            self.assertFalse(addressed_wake(text,['小帅']),text)

    async def test_no_face_wake_waits_for_audio_and_item_ack_without_identity_grant(self):
        event=self.final();s=self.s
        self.assertTrue(s.admit_explicit_wake(event,self.state));self.assertTrue(s.presence.allowed())
        self.assertFalse(s.can_speak());self.assertIsNone(s.identified_person())
        audio=SimpleNamespace(enable=AsyncMock());controller=SimpleNamespace(append=AsyncMock())
        await s.complete_explicit_wake(None,self.state,controller,audio,'instructions')
        audio.enable.assert_awaited_once();controller.append.assert_awaited_once()
        self.assertIn(event['transcript'],controller.append.call_args.args[-1])
        self.assertTrue(s.can_speak());self.assertTrue(s.continue_response)
        self.assertFalse(s.admit_explicit_wake(event,self.state))

    async def test_invalid_named_turn_can_wake_but_orphan_and_echo_cannot(self):
        event=self.final(invalid=True)
        self.assertTrue(self.s.admit_explicit_wake(event,self.state))
        self.s.explicit_wake_seen.clear();self.s.explicit_wake_pending=None
        event['item_id']='orphan';self.assertFalse(self.s.admit_explicit_wake(event,self.state))
        self.state.generating='robot';self.state.responses['robot']=dict(transcript='小帅你是哪你是哪一家的模型？')
        event=self.final();self.assertFalse(self.s.admit_explicit_wake(event,self.state))

    async def test_standby_transition_retains_question_not_welcome(self):
        s=self.s;s.context.update(mode='standby',target='standby');s.enabled=False
        event=self.final();self.assertTrue(s.admit_explicit_wake(event,self.state))
        actions=[e for e in self.events if e['kind']=='cloud_action'];self.assertEqual(len(actions),1)
        self.assertEqual(actions[0]['arguments']['action'],'wake')
        for revision,ready in ((2,False),(3,True)):
            s.request(dict(kind='voice_context',mode='interaction',target='interaction',revision=revision,ready=ready))
            self.assertEqual(s.explicit_wake_pending['text'],event['transcript'])
        self.assertIsNone(s.announcement_revision)
        s.enabled=True
        await s.complete_explicit_wake(None,self.state,SimpleNamespace(append=AsyncMock()),SimpleNamespace(enable=AsyncMock()),'ready')
        self.assertTrue(s.can_speak())

    async def test_normal_audible_turn_does_not_create_second_answer(self):
        s=self.s;s.presence.wake();s.cloud_audio_enabled=True
        event=self.final()
        self.assertFalse(s.admit_explicit_wake(event,self.state))
        self.assertIsNone(s.explicit_wake_pending);self.assertFalse(s.continue_response)

    async def test_tool_permission_alone_is_not_server_audio_confirmation(self):
        s=self.s;s.presence.wake();s.cloud_audio_enabled=False
        event=self.final()
        self.assertTrue(s.admit_explicit_wake(event,self.state));self.assertFalse(s.can_speak())

    async def test_changed_identity_connection_and_task_cannot_resume_old_wake(self):
        for change in ('identity','connection','task'):
            self.setUp();s=self.s;event=self.final();s.admit_explicit_wake(event,self.state)
            if change=='identity':s.identified_person.return_value='new-person'
            elif change=='connection':s.message_connection=object()
            else:s.request(dict(kind='voice_context',mode='task',target='baigufuren',revision=2,ready=True))
            gate=SimpleNamespace(enable=AsyncMock())
            self.assertFalse(await s.complete_explicit_wake(None,self.state,SimpleNamespace(append=AsyncMock()),gate,'test'))
            gate.enable.assert_not_awaited()

    async def test_new_turn_during_server_ack_does_not_answer_old_question(self):
        event=self.final();s=self.s;s.admit_explicit_wake(event,self.state)
        async def new_turn(*_):s.input_generation+=1;s.begin_user_turn('new')
        controller=SimpleNamespace(append=AsyncMock())
        self.assertFalse(await s.complete_explicit_wake(None,self.state,controller,SimpleNamespace(enable=AsyncMock(side_effect=new_turn)),'ready'))
        controller.append.assert_not_awaited();self.assertFalse(s.continue_response)

    async def test_session_ack_requires_audio_and_has_bounded_failure(self):
        gate=AudioPermissionGate();ws=SimpleNamespace(send=AsyncMock())
        task=asyncio.create_task(gate.enable(ws,'test'))
        await asyncio.sleep(.01)
        gate.observe(dict(type='session.updated',session=dict(modalities=['text'])))
        await asyncio.sleep(.01);self.assertFalse(task.done())
        gate.observe(dict(type='session.updated',session=dict(modalities=['text','audio'])))
        await task
        with patch('vivis.speech.explicit_wake.policy',return_value={**policy(),'controller_ack_s':.02}):
            with self.assertRaises(VoiceTransportError):await gate.enable(ws,'test')
        self.assertIsNone(gate.pending)

    async def test_actual_receive_sequence_replaces_text_only_reply_after_audio_ack(self):
        s=self.s;s.audio=FakeAudio();s.memory=None;s.reception.c=dict(s.reception.c,enabled=False)
        messages=asyncio.Queue();sent=[];pending_audio=[]
        class WS:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await messages.get())
            async def send(self,text):
                event=json.loads(text);sent.append(event)
                if event['type']=='session.update' and 'audio' in event['session'].get('modalities',[]):pending_audio.append(event)
                if event['type']=='conversation.item.create':
                    await messages.put(dict(type='conversation.item.created',item=event['item']))
        async def connected(ws,session,log):return session
        async def until(check):
            for _ in range(200):
                if task.done():task.result()
                if check():return
                await asyncio.sleep(.01)
            self.fail('Expected event did not arrive')
        with patch('vivis.speech.cloud_managed.websocket_connection',return_value=WS()),patch('vivis.speech.cloud_managed.connect_session',side_effect=connected):
            task=asyncio.create_task(s._connection())
            try:
                await until(lambda:s.cloud_state is not None)
                await messages.put(dict(type='input_audio_buffer.speech_started',item_id='question'))
                await messages.put(dict(type='input_audio_buffer.speech_stopped',item_id='question'))
                await messages.put(dict(type='response.created',response=dict(id='silent')))
                await messages.put(dict(type='response.done',response=dict(id='silent',status='completed')))
                final=dict(type='conversation.item.input_audio_transcription.completed',item_id='question',transcript='小帅你是哪你是哪一家的模型？')
                await messages.put(final);await until(lambda:bool(pending_audio))
                self.assertFalse(s.can_speak());self.assertFalse(any(e['type']=='response.create' for e in sent))
                await messages.put(dict(type='session.updated',session=dict(modalities=['text','audio'])))
                await until(lambda:any(e['type']=='response.create' for e in sent))
                self.assertTrue(s.can_speak());self.assertIn('silent',s.blocked)
                await messages.put(final);await asyncio.sleep(.1)
                self.assertEqual(sum(e['type']=='response.create' for e in sent),1)
                controller=[e for e in sent if e['type']=='conversation.item.create']
                self.assertEqual(len(controller),1);self.assertIn(final['transcript'],controller[0]['item']['content'][0]['text'])
            finally:task.cancel();await asyncio.gather(task,return_exceptions=True)


if __name__=='__main__':unittest.main()
