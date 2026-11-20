"""Public hello/farewell acknowledgement; no real devices, cloud or DB writes."""
import asyncio
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
