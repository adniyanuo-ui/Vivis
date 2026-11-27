"""Delayed primary events must not kill a separately attributed side reply."""
import asyncio
import base64
from contextlib import asynccontextmanager
import json
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,Mock,patch

import numpy as np

from vivis.interaction.sound_attention import configuration,SideSpeechGate
from vivis.speech.side_dialogue import SideDialogue


class SideBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from test_cloud_control import CloudBackendTests
        from vivis.speech.realtime import VoiceSession
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        v=CloudBackendTests().service(self.tmp.name,[]);self.v=v
        self.addCleanup(v.log.close);self.addCleanup(v.recorders.close);self.addCleanup(v.conversation_animation.close)
        v.enabled=True;v.health_available=True;v.side_suppressed=False
        v.context=dict(revision=8,mode='interaction',ready=True);v.cloud_state=VoiceSession(v.player,v.log)
        v.sound_gate=SideSpeechGate(configuration(),lambda _:True);v.sound_gate.intervals.append([9.75,12.])
        self.request=dict(id='1'*32,owner='owner',person=None,session=v.memory_session,revision=8,
            connection=v.message_connection,generation=v.input_generation,guest_key='guest',guest=None,
            heard_monotonic=10.,created_monotonic=10.08,sector=5,bearing_deg=-90.,
            action='glance',mount_calibrated=False)
        self.lane=SideDialogue(v,self.request);v.side_dialogue=self.lane;v.sound_attention=self.request
        self.ws=SimpleNamespace(send=AsyncMock());self.timeline=SimpleNamespace(at=lambda _:9.4)
        self.main=(v.turn_person,v.memory_session,v.input_generation)

    async def onset(self,item='late'):
        return await self.v.filter_side_event(dict(type='input_audio_buffer.speech_started',
            item_id=item,audio_start_ms=100),self.timeline,self.ws)

    async def test_ambiguous_preceding_onset_is_quarantined_without_changing_actor_or_generation(self):
        self.assertTrue(await self.onset());self.assertTrue(self.lane.valid())
        self.assertIn('late',self.v.side_items)
        self.assertEqual(self.main,(self.v.turn_person,self.v.memory_session,self.v.input_generation))
        self.assertTrue(await self.v.filter_side_event(dict(type='conversation.item.input_audio_transcription.completed',
            item_id='late',transcript='读取何工的留言'),self.timeline,self.ws))
        self.assertEqual(json.loads(self.ws.send.await_args.args[0])['type'],'conversation.item.delete')

    async def test_known_preceding_main_onset_keeps_its_original_path(self):
        self.v.side_onsets['known']=9.4
        self.assertFalse(await self.onset('known'));self.assertNotIn('known',self.v.side_items)

    async def test_actual_new_main_input_still_revokes_old_side_tools(self):
        self.v.sound_gate.intervals[-1][1]=10.05;self.timeline.at=lambda _:10.3
        self.assertFalse(await self.onset())
        self.v.input_generation+=1  # The normal receive loop, unchanged.
        self.assertEqual(self.lane.invalid_reason(),'new_main_generation')
        with self.assertRaises(ValueError):await self.lane.tool('get_robot_status',{})

    async def test_unknown_old_or_changed_connection_cannot_extend_side_lease(self):
        for stamp in (None,7.9):
            self.timeline.at=lambda _,stamp=stamp:stamp
            self.assertFalse(await self.onset(str(stamp)))
        self.timeline.at=lambda _:9.4;self.v.message_connection=object()
        self.assertFalse(await self.onset('changed'));self.assertFalse(self.lane.valid())

    async def test_delayed_main_reply_waits_for_side_output_in_existing_buffer(self):
        self.lane.rids.add('current_side');self.v.player.begin_priority('current_side')
        event=dict(type='response.created',response=dict(id='future_main'))
        self.assertFalse(await self.v.filter_side_event(event,self.timeline,self.ws))
        self.assertIn('future_main',self.lane.paused)
        audio=np.full(2400,1000,dtype='<i2').tobytes()
        self.v.player.add('future_main',audio);self.v.player.finish('future_main')
        self.assertFalse(np.any(self.v.player.render(4800,time.monotonic())))
        self.assertNotIn('current_side',self.v.player.paused)
        self.v.player.add('current_side',audio);self.v.player.finish('current_side')
        self.assertTrue(np.any(self.v.player.render(4800,time.monotonic())))

    async def test_finished_motion_does_not_cancel_unfinished_side_semantics(self):
        now=time.monotonic();self.request.update(heard_monotonic=now-.1,created_monotonic=now)
        self.v.side_task=asyncio.get_running_loop().create_future()
        self.v.sound_mailbox=SimpleNamespace(submit=Mock(),latest=dict(updated_monotonic=now,
            owner='owner',phase='idle',result='empty',completed=dict(id=self.request['id'],elapsed_s=5.9)))
        self.v.identity_fusion=Mock();self.v.acoustic_tick()
        self.assertTrue(self.request['motion_returned']);self.assertTrue(self.lane.valid())
        self.assertIs(self.v.sound_attention,self.request);self.v.identity_fusion.reset.assert_not_called()
        self.v.side_task.set_result(None);self.v.acoustic_tick()
        self.assertIsNone(self.v.sound_attention);self.v.identity_fusion.reset.assert_called_once()

    async def test_capture_failure_logs_the_specific_end_reason(self):
        self.lane.feed(10.,bytes(640),bytes(640));self.lane.feed(10.2,bytes(640),bytes(640))
        self.v.log=Mock()
        with patch('vivis.speech.side_dialogue.websocket_connection',side_effect=AssertionError('network')):
            await self.lane.run()
        name=self.v.log.write.call_args.args[0];fields=self.v.log.write.call_args.kwargs
        self.assertEqual(name,'side_dialogue_finished');self.assertEqual(fields['reason'],'capture_clock_gap')
        self.assertEqual(fields['captured_frames'],1)

    async def test_saved_late_onset_sequence_still_produces_side_audio_through_virtual_dac(self):
        audio=(np.sin(np.arange(320)*.12)*4000).astype('<i2').tobytes();heard=[]
        events=[dict(type='conversation.item.input_audio_transcription.completed',transcript='您听见我的介绍了吗？'),
            dict(type='response.created',response=dict(id='side_reply')),
            dict(type='response.audio.delta',response_id='side_reply',delta=base64.b64encode(audio).decode()),
            dict(type='response.audio_transcript.done',response_id='side_reply',transcript='听见了，请继续。'),
            dict(type='response.done',response=dict(id='side_reply',status='completed'))]
        owner=self
        class WS:
            async def send(self,raw):
                e=json.loads(raw)
                if e['type']=='session.update':self.session=e['session']
                if e['type']=='input_audio_buffer.commit':
                    if not await owner.onset():owner.v.input_generation+=1
            async def recv(self):return json.dumps(dict(type='session.updated',session=self.session))
            def __aiter__(self):return self
            async def __anext__(self):
                if not events:raise StopAsyncIteration
                return json.dumps(events.pop(0))
        @asynccontextmanager
        async def connection(*args):yield WS()
        self.lane.feed(10.,audio,bytes(len(audio)));self.lane.end()
        async def dac():
            while not task.done():
                heard.append(self.v.player.render(1920,time.monotonic()));await asyncio.sleep(.002)
        with patch('vivis.speech.side_dialogue.websocket_connection',connection):
            task=asyncio.create_task(self.lane.run());await asyncio.gather(task,dac())
        self.assertTrue(any(np.any(x) for x in heard));self.assertEqual(self.lane.transcript,'您听见我的介绍了吗？')
        self.assertEqual(self.main,(self.v.turn_person,self.v.memory_session,self.v.input_generation))
