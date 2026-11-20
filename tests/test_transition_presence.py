"""Public hello/farewell acknowledgement; no real devices, cloud or DB writes."""
import asyncio
import base64
import io
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch

import numpy as np
import test_cloud_control as fixtures
from vivis.interaction.cloud_control import SpeechPresence,PresencePublisher
from vivis.interaction.voice_bridge import VoiceBridge
from vivis.speech.realtime import VoiceSession
from vivis.speech.message_receipts import ReceiptRejected


class TransitionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.events=[];self.s=fixtures.CloudBackendTests().service(folder.name,self.events)
        self.addCleanup(self.s.log.close)
        self.s.presence=SpeechPresence(self.s.presence.c,folder.name)
        self.pub=PresencePublisher(self.s.presence.c,folder.name)
        self.s.context.update(mode='standby',target='standby',ready=True,revision=1)
        self.s.identified_person=Mock(return_value=None)
        self.state=VoiceSession(self.s.player,self.s.log);self.s.cloud_state=self.state
        self.s.active_voice='longanyang_v3.1'
        self.addAsyncCleanup(self.cleanup_speech)

    async def cleanup_speech(self):
        self.s.cancel_farewell('test_cleanup')
        if self.s.farewell_task:await asyncio.gather(self.s.farewell_task,return_exceptions=True)

    def body(self,count=1,pending=False):
        now=time.monotonic();self.pub.last=-float('inf')
        self.pub.publish(0,now,now,person=dict(enabled=True,available=True,persons=count,
            observed_monotonic=None if pending else now))

    def final(self,text):
        s=self.s;s.input_generation+=1;s.begin_user_turn('user');s.question_started('user',self.state)
        s.memory_turns['user']['text']=text;s.turn_ended_at=time.monotonic()
        return dict(item_id='user',transcript=text)

    async def until(self,condition):
        deadline=time.monotonic()+1
        while not condition():
            if time.monotonic()>deadline:raise AssertionError('transition test timeout')
            await asyncio.sleep(.005)

    async def test_actual_hello_handoff_waits_for_first_async_body_result(self):
        s=self.s;self.body();self.assertTrue(s.admit_explicit_wake(self.final('Hello. '),self.state))
        s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=False,revision=2))
        self.body(count=0,pending=True)
        s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=True,revision=3))
        controller=SimpleNamespace(append=AsyncMock());audio=SimpleNamespace(enable=AsyncMock())
        # The actual manager publishes ready before speaker handoff finishes.
        self.assertFalse(await s.complete_explicit_wake(None,self.state,controller,audio,'instructions'))
        self.assertIsNotNone(s.explicit_wake_pending)
        s.enabled=True
        self.assertFalse(await s.complete_explicit_wake(None,self.state,controller,audio,'instructions'))
        self.assertIsNotNone(s.explicit_wake_pending);audio.enable.assert_not_awaited()
        self.body();self.assertTrue(await s.complete_explicit_wake(None,self.state,controller,audio,'instructions'))
        self.assertTrue(s.can_speak());controller.append.assert_awaited_once()

    async def test_final_hello_source_survives_receipt_reset_before_new_response(self):
        s=self.s
        for text in ('小帅。','你好。','Hi.'):
            with self.subTest(text=text):
                s.request(dict(kind='voice_context',mode='standby',target='standby',ready=True,revision=s.context['revision']+1))
                s.explicit_wake_seen.clear();self.body()
                event=self.final(text);s.message_receipts.filter(dict(type='conversation.item.input_audio_transcription.completed',**event))
                self.assertTrue(s.admit_explicit_wake(event,self.state))
                s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=False,revision=s.context['revision']+1))
                self.assertNotIn('user',s.message_receipts.sources)
                s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=True,revision=s.context['revision']+1));s.enabled=True
                async def response_during_controller_ack(*_):
                    # A controller-initiated continuation still belongs to the
                    # REAL final-ASR item, not the synthetic controller item.
                    s.message_receipts.bind('reply',s.input_generation,'user')
                    self.assertEqual(s.message_receipts.responses['reply']['kind'],'ordinary')
                self.assertTrue(await s.complete_explicit_wake(None,self.state,
                    SimpleNamespace(append=AsyncMock(side_effect=response_during_controller_ack)),
                    SimpleNamespace(enable=AsyncMock()),'instructions'))
                audio=dict(type='response.audio.delta',response_id='reply',delta=base64.b64encode(b'\x01\0'*240).decode())
                self.assertEqual(s.message_receipts.filter(audio),[audio])
                self.assertFalse(s.message_receipts.expired());self.assertIsNone(s.message_lane)

    async def test_restoring_named_wake_with_message_does_not_bypass_business_guard(self):
        s=self.s;event=self.final('小帅，给小王留言，让他明天来。');s.admit_explicit_wake(event,self.state)
        s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=True,revision=2));s.enabled=True
        await s.complete_explicit_wake(None,self.state,SimpleNamespace(append=AsyncMock()),SimpleNamespace(enable=AsyncMock()),'instructions')
        s.message_receipts.bind('reply',s.input_generation,'user')
        self.assertEqual(s.message_receipts.responses['reply']['kind'],'needs_tool')
        with self.assertRaises(ReceiptRejected):
            s.message_receipts.filter(dict(type='response.audio_transcript.done',response_id='reply',transcript='留言已经保存。'))

    async def test_unknown_response_source_is_not_a_database_failure(self):
        s=self.s;s.receipt_failure_notice=Mock()
        s.report_receipt_failure(dict(kind='await_source',turn=s.input_generation))
        s.receipt_failure_notice.assert_not_called()
        self.assertNotIn('账本',self.events[-1]['message']);self.assertNotIn('留言',self.events[-1]['message'])
        s.report_receipt_failure(dict(kind='courier_code',turn=s.input_generation))
        s.receipt_failure_notice.assert_called_once()
        self.events.clear();s.receipt_failure_notice.reset_mock()
        s.report_receipt_failure(dict(kind='receipt',turn=s.input_generation-1))
        self.assertFalse(self.events);s.receipt_failure_notice.assert_not_called()

    async def test_real_receive_handoff_and_receipt_gate_deliver_greeting_pcm(self,early=False,invalid=False):
        s=self.s;messages=asyncio.Queue();sent=[];self.body();s.reception.c={**s.reception.c,'enabled':False}
        s.audio=Mock(output_stream=None)
        class WS:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await messages.get())
            async def send(self,text):
                event=json.loads(text);sent.append(event)
                if event['type']=='session.update':
                    await messages.put(dict(type='session.updated',session=event['session']))
                elif event['type']=='conversation.item.create' and event.get('item',{}).get('id','').startswith('controller_'):
                    await messages.put(dict(type='conversation.item.created',item=event['item']))
                elif event['type']=='response.create':
                    for response in (
                        dict(type='response.created',response=dict(id='audible-hello')),
                        dict(type='response.audio.delta',response_id='audible-hello',delta=base64.b64encode(b'\x01\0'*1200).decode()),
                        dict(type='response.audio_transcript.done',response_id='audible-hello',transcript='您好，我在呢。'),
                        dict(type='response.done',response=dict(id='audible-hello',status='completed'))):
                        await messages.put(response)
        async def connect(ws,session,log):return session
        with patch('vivis.speech.cloud_managed.websocket_connection',return_value=WS()),\
             patch('vivis.speech.cloud_managed.connect_session',side_effect=connect):
            connection=asyncio.create_task(s._connection())
            try:
                await self.until(lambda:s.ws is not None)
                if invalid:s.cloud_state.current_user_turn=dict(item_id='old-user',confirmed_end_monotonic=time.monotonic()-10)
                for event in (dict(type='input_audio_buffer.speech_started',item_id='onsite-user'),
                              dict(type='input_audio_buffer.speech_stopped',item_id='onsite-user',**({'reason':'turn_invalid'} if invalid else {})),
                              dict(type='conversation.item.input_audio_transcription.completed',item_id='onsite-user',transcript='你好。')):
                    await messages.put(event)
                await self.until(lambda:s.explicit_wake_pending is not None)
                s.request(dict(kind='voice_context',revision=2,mode='interaction',target='interaction',ready=False))
                s.request(dict(kind='voice_context',revision=3,mode='interaction',target='interaction',ready=not early,
                    **({'speech_ready':True} if early else {})));s.enabled=True
                if early:self.pub.close()  # Actual old-camera release before the new detector is ready.
                await self.until(lambda:s.player.count>0)
                if early:
                    self.assertFalse(s.context['ready']);self.assertTrue(s.can_speak())
                    self.assertNotIn('audible-hello',s.cloud_state.blocked)
                    count=s.player.count
                    s.request(dict(kind='voice_context',revision=4,mode='interaction',target='interaction',ready=True))
                    self.assertEqual(s.player.count,count)
                    self.assertNotIn('audible-hello',s.cloud_state.blocked)
                    self.assertIsNone(s.announcement_revision)
                self.assertEqual(s.message_receipts.responses['audible-hello']['kind'],'ordinary')
                self.assertEqual(s.message_receipts.responses['audible-hello']['source_item'],'onsite-user')
                self.assertEqual(s.cloud_state.responses['audible-hello']['user_turn']['item_id'],'onsite-user')
                self.assertNotIn('audible-hello',s.blocked);self.assertIsNone(s.message_lane)
                s.player.render(s.player.count,time.monotonic()+.001);s.cloud_state.poll_playback()
                await self.until(lambda:any(e.get('text')=='您好，我在呢。' for e in self.events))
                self.assertFalse(any('回执' in e.get('message','') for e in self.events))
                self.assertEqual(len([e for e in sent if e['type']=='response.create']),1)
            finally:
                connection.cancel();await asyncio.gather(connection,return_exceptions=True)

    async def test_real_greeting_audio_precedes_motion_ready_without_second_welcome(self):
        await self.test_real_receive_handoff_and_receipt_gate_deliver_greeting_pcm(early=True)

    async def test_semantically_rejected_greeting_binds_new_reply_not_previous_item(self):
        await self.test_real_receive_handoff_and_receipt_gate_deliver_greeting_pcm(invalid=True)

    async def test_verified_body_greeting_uses_existing_grace_during_camera_handoff(self):
        s=self.s;self.body();s.admit_explicit_wake(self.final('Hello'),self.state)
        s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=False,speech_ready=True,revision=2))
        s.enabled=True;self.pub.close()
        gate=SimpleNamespace(enable=AsyncMock());controller=SimpleNamespace(append=AsyncMock())
        self.assertFalse(s.presence.greeting_visible())
        self.assertTrue(await s.complete_explicit_wake(None,self.state,controller,gate,'instructions'))
        self.assertFalse(s.context['ready']);self.assertTrue(s.can_speak())
        self.assertEqual(s.presence.wake_until,0)
        directive=controller.append.call_args.args[-1]
        self.assertNotIn('云端已确认开启语音',directive);self.assertIn('自然回应',directive)

    async def test_greeting_handoff_does_not_extend_grace_without_observations(self):
        s=self.s;self.body();s.admit_explicit_wake(self.final('Hello'),self.state)
        self.pub.close();s.presence.last_greeted_person=time.monotonic()-s.presence.c['face_loss_grace_s']-1
        original=s.presence.last_greeted_person;s.presence.greet()
        self.assertEqual(s.presence.last_greeted_person,original)
        s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=False,speech_ready=True,revision=2));s.enabled=True
        gate=SimpleNamespace(enable=AsyncMock())
        self.assertFalse(await s.complete_explicit_wake(None,self.state,SimpleNamespace(append=AsyncMock()),gate,'instructions'))
        gate.enable.assert_not_awaited();self.assertFalse(s.can_speak())

    async def test_hello_during_automatic_arrival_gap_preserves_original_question(self):
        s=self.s;self.body();now=time.monotonic()
        # Arrival was genuinely observed, then the camera owner closed.
        s.presence.last_face=now;self.pub.close()
        s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=False,revision=2))
        self.assertTrue(s.admit_explicit_wake(self.final('你好'),self.state))
        self.assertFalse(s.question_onset['visible_person']);self.assertTrue(s.question_onset['preparation_presence'])
        self.assertEqual(self.events[-1]['arguments']['wake_reason'],'greeting')
        s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=False,speech_ready=True,revision=3));s.enabled=True
        controller=SimpleNamespace(append=AsyncMock())
        self.assertTrue(await s.complete_explicit_wake(None,self.state,controller,SimpleNamespace(enable=AsyncMock()),'instructions'))
        self.assertIn('你好',controller.append.call_args.args[-1]);self.assertTrue(s.can_speak())

    async def test_human_call_during_automatic_wake_preparation_requests_early_audio(self):
        for text in ('小帅','Hello'):
            with self.subTest(text=text):
                s=self.s;s.explicit_wake_seen.clear();s.enabled=False;self.body()
                s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=False,revision=s.context['revision']+1))
                self.assertTrue(s.admit_explicit_wake(self.final(text),self.state))
                self.assertEqual(self.events[-1]['kind'],'cloud_action')
                self.assertEqual(self.events[-1]['arguments']['action'],'wake')
                s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=False,speech_ready=True,revision=s.context['revision']+1))
                s.enabled=True
                self.assertTrue(await s.complete_explicit_wake(None,self.state,SimpleNamespace(append=AsyncMock()),SimpleNamespace(enable=AsyncMock()),'instructions'))
                self.assertTrue(s.can_speak());self.assertFalse(s.context['ready'])

    async def test_preparation_speech_keeps_expression_disabled_and_cancellation_mutes(self):
        s=self.s;self.body();s.admit_explicit_wake(self.final('小帅'),self.state)
        s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=False,speech_ready=True,revision=2))
        s.enabled=True
        await s.complete_explicit_wake(None,self.state,SimpleNamespace(append=AsyncMock()),SimpleNamespace(enable=AsyncMock()),'instructions')
        self.assertTrue(s.can_speak());self.assertFalse(s.context['ready'])
        s.response_revisions['expression']=2;s.expression.publish=Mock()
        await s.tool(SimpleNamespace(send=AsyncMock()),dict(name='set_expression',call_id='too-early',response_id='expression',
            arguments='{"expression":"smile","intensity":0.5}'))
        s.expression.publish.assert_not_called()
        s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=False,revision=3))
        self.assertFalse(s.output_ready())

    def test_bridge_speech_permission_does_not_mark_motion_ready_or_enable_standby(self):
        stream=io.StringIO();bridge=VoiceBridge(stream)
        bridge.context('interaction','interaction',False,speech_ready=True)
        event=json.loads(stream.getvalue().splitlines()[-1]);self.assertFalse(event['ready']);self.assertTrue(event['speech_ready'])
        bridge.context('standby','standby',False,speech_ready=True)
        self.assertNotIn('speech_ready',json.loads(stream.getvalue().splitlines()[-1]))

    async def test_body_warmup_wait_is_bounded_and_does_not_grant_named_permission(self):
        s=self.s;self.body();s.admit_explicit_wake(self.final('你好'),self.state)
        s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=True,revision=2))
        s.enabled=True;self.body(pending=True)
        gate=SimpleNamespace(enable=AsyncMock());controller=SimpleNamespace(append=AsyncMock())
        s.explicit_wake_pending['presence_wait_started']=time.monotonic()-2
        self.assertFalse(await s.complete_explicit_wake(None,self.state,controller,gate,'instructions'))
        self.assertIsNone(s.explicit_wake_pending);gate.enable.assert_not_awaited()
        self.assertEqual(s.presence.wake_until,0)

    async def test_public_greeting_survives_identity_becoming_known_during_prepare(self):
        s=self.s;self.body();s.admit_explicit_wake(self.final('你好'),self.state)
        s.request(dict(kind='voice_context',mode='interaction',target='interaction',ready=True,revision=2));s.enabled=True
        s.identified_person.return_value='confirmed-by-separate-identity-layer'
        self.assertTrue(await s.complete_explicit_wake(None,self.state,
            SimpleNamespace(append=AsyncMock()),SimpleNamespace(enable=AsyncMock()),'instructions'))
        self.assertEqual(s.presence.wake_until,0)

    async def farewell(self,mode='interaction'):
        s=self.s;s.context.update(mode=mode,target='interaction' if mode=='interaction' else next(iter(s.program_ids)))
        self.final('你休息吧');s.response_revisions['r']=1
        await s.tool(SimpleNamespace(send=AsyncMock()),dict(call_id='rest',response_id='r',name='robot_program',arguments='{"action":"standby"}'))
        self.assertEqual(self.events[-1]['arguments']['action'],'standby')  # Motion is not delayed for speech.
        return self.events[-1]['transition_reply_id']

    def ready_standby(self,reply,ready=True):
        self.s.request(dict(kind='voice_context',revision=self.s.context['revision']+1,
            mode='standby',target='standby',ready=ready,transition_reply_id=reply))

    async def test_farewell_uses_current_voice_once_and_completes_only_after_dac(self):
        s=self.s;reply=await self.farewell();self.ready_standby(reply,False)
        engine=SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(samples=np.ones(1200)*.01,sample_rate=24000)))
        with patch('vivis.speech.trusted_reading.RealtimeReader',return_value=engine) as reader:
            s.schedule_farewell(self.state);self.assertIsNotNone(s.farewell_task)
            await asyncio.sleep(.01)
            engine.generate.assert_awaited_once();self.assertEqual(s.player.count,0)
            self.ready_standby(reply);s.enabled=True
            self.assertFalse(s.can_speak())  # No general standby chat permission.
            s.schedule_farewell(self.state);s.schedule_farewell(self.state)
            await self.until(lambda:s.player.count>0)
            self.assertFalse(any(e['kind']=='voice_reply_done' for e in self.events))
            s.player.render(s.player.count,time.monotonic()+.02)
            await asyncio.wait_for(s.farewell_task,1)
            self.assertEqual(reader.call_args.args[2]['voice'],'longanyang_v3.1')
            engine.generate.assert_awaited_once()
        done=[e for e in self.events if e['kind']=='voice_reply_done']
        self.assertEqual(len(done),1);self.assertEqual(done[0]['reason'],'played')
        self.assertTrue(any(e.get('source')=='transition_farewell' for e in self.events))
        s.schedule_farewell(self.state);self.assertFalse(s.can_speak())

    async def test_task_rest_still_requests_immediate_stop_before_farewell(self):
        await self.farewell('task')
        self.assertIsNone(self.s.farewell_task)

    async def test_new_user_input_cancels_queued_farewell_and_releases_ticket(self):
        reply=await self.farewell();self.ready_standby(reply);self.s.enabled=True
        self.s.input_generation+=1;self.s.schedule_farewell(self.state)
        self.assertIsNone(self.s.farewell)
        self.assertEqual(self.events[-1]['kind'],'voice_reply_done')
        self.assertIsNone(self.s.farewell_task)

    async def test_farewell_cloud_failure_is_local_and_keeps_standby(self):
        reply=await self.farewell();self.ready_standby(reply);s=self.s;s.enabled=True
        with patch('vivis.speech.trusted_reading.RealtimeReader',return_value=SimpleNamespace(generate=AsyncMock(side_effect=TimeoutError()))):
            s.schedule_farewell(self.state);await asyncio.wait_for(s.farewell_task,1)
        self.assertEqual(self.events[-1]['reason'],'speech_failed');self.assertFalse(s.shutdown.is_set())
        self.assertEqual(s.context['mode'],'standby');self.assertFalse(s.can_speak())

    async def test_auto_idle_and_initial_standby_do_not_speak_farewell(self):
        self.s.enabled=True;self.s.schedule_farewell(self.state)
        self.assertIsNone(self.s.farewell_task);self.assertFalse(self.events)


class FarewellBridgeTests(unittest.TestCase):
    def test_reply_ticket_is_sent_only_for_standby_and_old_completion_cannot_clear_new_one(self):
        stream=io.StringIO();bridge=VoiceBridge(stream)
        bridge.transition_reply_id='one';bridge.context('standby','standby',True)
        self.assertEqual(json.loads(stream.getvalue().splitlines()[-1])['transition_reply_id'],'one')
        bridge.transition_reply_id='two'
        self.assertFalse(bridge.accept(dict(kind='voice_reply_done',reply_id='one',epoch=0)))
        self.assertEqual(bridge.transition_reply_id,'two')
        self.assertTrue(bridge.accept(dict(kind='voice_reply_done',reply_id='two',epoch=0)))
        bridge.context('standby','standby',True)
        self.assertNotIn('transition_reply_id',json.loads(stream.getvalue().splitlines()[-1]))
        bridge.transition_reply_id='three';bridge.context('interaction','interaction',True)
        self.assertNotIn('transition_reply_id',json.loads(stream.getvalue().splitlines()[-1]))


if __name__=='__main__':unittest.main()
