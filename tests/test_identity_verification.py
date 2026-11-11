"""Identity changes invalidate old context, task authority and relay audio."""
import base64
import json
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.speech.cloud_managed import ConversationReset
from vivis.speech.realtime import VoiceSession


class IdentityVerificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from test_cloud_control import CloudBackendTests
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.service=CloudBackendTests().service(temporary.name,[])
        self.addCleanup(self.service.log.close)
        self.person='alice'
        self.service.identified_person=lambda:self.person
        self.service.presence.visible=Mock(return_value=True)
        self.service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
        self.service.enabled=True
        self.state=VoiceSession(self.service.player,self.service.log)
        self.state.interrupt=Mock(wraps=self.state.interrupt)

    async def test_identity_is_audited_every_three_seconds_with_faster_change_checks(self):
        with patch.object(self.service.log,'write') as log,patch('vivis.speech.cloud_managed.time.monotonic',return_value=10.) as clock:
            self.service.verify_identity(self.state)
            clock.return_value=12.99;self.service.verify_identity(self.state)
            clock.return_value=13.;self.service.verify_identity(self.state)
        checks=[c for c in log.call_args_list if c.args[0]=='identity_verified']
        self.assertEqual(len(checks),2)
        self.assertTrue(all(c.kwargs['person_id']=='alice' and c.kwargs['interval_s']==3 for c in checks))
        self.state.interrupt.assert_not_called()

    async def test_known_to_unknown_clears_context_after_three_seconds(self):
        self.service.conversation_person='alice';self.person=None
        previous=self.service.memory_session
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=10.) as clock:
            self.service.verify_identity(self.state)
            clock.return_value=12.99;self.service.verify_identity(self.state)
            clock.return_value=13.
            with self.assertRaisesRegex(ConversationReset,'identity_uncertain'):
                self.service.verify_identity(self.state)
        self.assertIsNone(self.service.conversation_person)
        self.assertNotEqual(self.service.memory_session,previous)
        self.assertTrue(self.service.turn_person.startswith('anonymous:'))

    async def test_review_preserves_conversation_without_authorizing_identity(self):
        service=self.service;service.conversation_person='alice';self.person=None
        service.reviewing_identity=lambda:dict(state='reviewing',person=dict(person_id='alice',preferred_address='甲工'))
        previous=service.memory_session
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=10.) as clock:
            service.verify_identity(self.state)
            clock.return_value=14.9;service.verify_identity(self.state)
            self.assertIsNone(service.identity_missing_since)
        self.assertEqual(service.conversation_person,'alice')
        self.assertEqual(service.memory_session,previous)
        self.state.interrupt.assert_not_called();self.assertIsNone(service.identified_person())

    async def test_uncertain_identity_finishes_queued_reply_before_context_reset(self):
        service=self.service;service.conversation_person='alice';self.person=None
        service.speech_responses.append('reply')
        pcm=(np.ones(480,dtype='<i2')*1000).tobytes()
        service.player.add('reply',pcm);service.player.finish('reply')
        self.state.responses['reply']=dict(status='completed',received_audio_s=.02,empty_dac=13.2)
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=10.) as clock:
            service.verify_identity(self.state)
            clock.return_value=13.;service.verify_identity(self.state)
            self.assertTrue(service.identity_reset_pending)
            self.assertTrue(service.can_play_response('reply'))
            self.state.interrupt.assert_not_called()
            service.player.render(960,13.1)
            # PCM has left the queue, but its scheduled DAC span is unfinished.
            service.verify_identity(self.state);self.state.interrupt.assert_not_called()
            clock.return_value=13.21
            with self.assertRaisesRegex(ConversationReset,'identity_uncertain'):service.verify_identity(self.state)
        self.assertIsNone(service.identified_person())

    async def test_different_confirmed_person_still_interrupts_an_active_reply(self):
        service=self.service;service.conversation_person='alice';self.person='bob'
        service.speech_responses.append('reply');self.state.generating='reply'
        with self.assertRaisesRegex(ConversationReset,'confirmed_person_changed'):service.verify_identity(self.state)
        self.state.interrupt.assert_called_once_with(source='confirmed_person_changed')

    async def test_lost_identity_waits_for_user_control_turn_without_restoring_authority(self):
        service=self.service;service.conversation_person='alice';self.person=None
        service.in_speech=True
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=10.) as clock:
            service.verify_identity(self.state);clock.return_value=13.;service.verify_identity(self.state)
            self.state.interrupt.assert_not_called()
            service.turn_person='alice';service.turn_item_id='stop';service.turn_ended_at=13.
            with self.assertRaisesRegex(ValueError,'speaker changed'):service.fresh_task_source()
            service.in_speech=False
            self.state.responses['control']=dict(status='completed',user_turn=dict(item_id='stop'))
            with self.assertRaisesRegex(ConversationReset,'identity_uncertain'):service.verify_identity(self.state)

    async def test_context_reset_does_not_erase_user_turn_waiting_for_a_tool_response(self):
        service=self.service;service.conversation_person='alice';self.person=None
        service.turn_item_id='stop';service.turn_ended_at=10.
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=10.) as clock:
            service.verify_identity(self.state);clock.return_value=13.;service.verify_identity(self.state)
            self.state.interrupt.assert_not_called()
            self.state.responses['control']=dict(status='completed',user_turn=dict(item_id='stop'))
            with self.assertRaisesRegex(ConversationReset,'identity_uncertain'):service.verify_identity(self.state)

    async def test_face_grace_expiry_keeps_only_the_authorized_reply_and_its_late_packets(self):
        service=self.service;service.cloud_state=self.state
        service.speech_responses.append('reply');self.state.generating='reply'
        service.presence.visible.return_value=False;service.presence.last_face=10.
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=24.99):
            self.assertTrue(service.can_speak())
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=25.):
            self.assertFalse(service.can_speak())
            # Packet arrives before context_loop notices permission expiry.
            self.assertTrue(service.can_play_response('reply'))
            self.assertFalse(service.can_play_response('new-reply'))
            service.update_speech_permission(self.state,False)
            self.state.interrupt.assert_not_called()
            service.presence.visible.return_value=True
            service.update_speech_permission(self.state,True)
            self.assertFalse(service.finish_responses)
            self.assertTrue(service.can_speak())

    async def test_departure_waits_for_playback_then_resets(self):
        service=self.service;self.person=None
        service.visit_seen=True;service.absence_since=10.;service.presence.visible.return_value=False
        service.speech_responses.append('reply');self.state.generating='reply'
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=30.):
            service.verify_identity(self.state);self.state.interrupt.assert_not_called()
            self.state.generating=None
            with self.assertRaisesRegex(ConversationReset,'visitor_departed'):service.verify_identity(self.state)

    async def test_speaker_handoff_and_explicit_interrupt_override_finish_permission(self):
        service=self.service;service.cloud_state=self.state
        service.presence.visible.return_value=False
        service.speech_responses.append('reply');self.state.generating='reply'
        self.assertTrue(service.can_play_response('reply'))
        service.pending_mode=(2,False)
        self.assertFalse(service.can_play_response('reply'))
        service.pending_mode=None
        self.state.interrupt('reply',source='valid_speech_stopped')
        self.assertFalse(service.can_play_response('reply'))

    async def test_private_relay_does_not_inherit_permission_to_finish_without_its_recipient(self):
        service=self.service;service.cloud_state=self.state;self.person=None
        service.presence.visible.return_value=False
        service.speech_responses.append('relay');self.state.generating='relay'
        service.relay_recipients['relay']='alice'
        self.assertFalse(service.can_play_response('relay'))
        service.protect_relay_audio(self.state)
        self.state.interrupt.assert_called_once_with('relay',source='relay_recipient_changed')

    async def test_stop_tool_remains_available_without_face_during_reply_completion(self):
        service=self.service;self.person=None
        service.context.update(mode='task',target=next(iter(service.program_ids)))
        service.presence.visible.return_value=False;service.response_revisions['reply']=1
        sent=[]
        async def send(value):sent.append(json.loads(value))
        with patch.object(service,'emit') as emit:
            await service.tool(SimpleNamespace(send=send),dict(name='robot_program',call_id='stop-no-face',
                response_id='reply',arguments='{"action":"stop","program":""}'))
        self.assertTrue(json.loads(sent[-1]['item']['output'])['accepted'])
        self.assertTrue(any(call.args[0].get('arguments',{}).get('action')=='stop' for call in emit.call_args_list))

    async def test_review_of_different_person_cannot_preserve_old_context(self):
        service=self.service;service.conversation_person='alice';self.person=None
        service.reviewing_identity=lambda:dict(state='reviewing',person=dict(person_id='bob'))
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=10.) as clock:
            service.verify_identity(self.state);clock.return_value=13.
            with self.assertRaisesRegex(ConversationReset,'identity_uncertain'):service.verify_identity(self.state)

    async def test_review_status_retains_name_but_personal_memory_and_relay_are_blocked(self):
        service=self.service;self.person=None;service.memory=Mock()
        service.reviewing_identity=lambda:dict(state='reviewing',person=dict(person_id='alice',display_name='测试甲',preferred_address='甲工'))
        sent=[]
        async def send(value):sent.append(json.loads(value))
        ws=SimpleNamespace(send=send);service.response_revisions['r']=1
        await service.tool(ws,dict(name='get_robot_status',call_id='review-status',response_id='r',arguments='{}'))
        status=json.loads(sent[-1]['item']['output'])
        self.assertFalse(status['identity_verified']);self.assertIsNone(status['current_person'])
        self.assertEqual(status['identity_state'],'reviewing')
        self.assertEqual(status['last_known_person']['preferred_address'],'甲工')
        for index,name in enumerate(('search_person_memory','search_robot_memory')):
            await service.tool(ws,dict(name=name,call_id='review-memory'+str(index),response_id='r',arguments='{"keywords":[]}'))
            self.assertFalse(json.loads(sent[-1]['item']['output'])['accepted'])
        service.memory.search.assert_not_called();service.memory.search_all.assert_not_called()
        service.relay_recipients['relay']='alice'
        service.player.add('relay',(np.ones(480,dtype='<i2')*1000).tobytes())
        state=VoiceSession(service.player,service.log);state.generating='relay'
        service.protect_relay_audio(state)
        self.assertEqual(service.player.count,0);self.assertTrue(service.relay_identity_hold)

    async def test_anonymous_conversation_cannot_continue_with_newly_identified_person(self):
        self.service.resume_history.append(dict(role='user',text='上一个匿名访客的内容'))
        with self.assertRaisesRegex(ConversationReset,'confirmed_person_changed'):
            self.service.verify_identity(self.state)
        self.assertEqual(self.service.conversation_person,'alice')

    async def test_reset_clears_previous_history_and_task_turn_without_restarting_hardware(self):
        service=self.service;service.conversation_person='alice';self.person='bob'
        service.turn_item_id='old-turn';service.turn_ended_at=time.monotonic()
        service.in_speech=True
        service.resume_history.append(dict(role='user',text='甲的私人聊天'))
        calls=[]
        async def connection():
            calls.append(True)
            if len(calls)==1:service.verify_identity(self.state)
            self.assertFalse(service.resume_history)
            self.assertIsNone(service.turn_item_id)
            self.assertFalse(service.in_speech)
            self.assertEqual(service.conversation_person,'bob')
        service._connection=connection
        await service.cloud(0)
        self.assertEqual(len(calls),2)
        self.state.interrupt.assert_called_once_with(source='confirmed_person_changed')

    async def test_lost_identity_cannot_authorize_a_prior_known_speakers_task(self):
        service=self.service;service.turn_person='alice';service.turn_item_id='human'
        service.turn_ended_at=time.monotonic();self.person=None
        with self.assertRaisesRegex(ValueError,'speaker changed'):
            service.fresh_task_source()

    async def test_relay_remains_protected_after_an_acknowledgement_clears_pending_task(self):
        service=self.service;player=service.player
        pcm=(np.ones(480,dtype='<i2')*1000).tobytes()
        player.add('relay',pcm)
        state=VoiceSession(player,service.log)
        state.generating='relay'
        service.relay_recipients['relay']='alice';service.relay_pending.clear()
        self.person=None
        service.protect_relay_audio(state)
        self.assertIn('relay',state.blocked)
        self.assertEqual(player.count,0)
        self.assertTrue(service.relay_identity_hold)
        state.event(dict(type='response.audio.delta',response_id='relay',delta=base64.b64encode(pcm).decode()))
        self.assertEqual(player.count,0)

    async def test_finished_relay_does_not_mute_later_conversations(self):
        service=self.service;player=service.player
        state=VoiceSession(player,service.log)
        state.responses['old']=dict(status='completed',received_audio_s=.02,empty_dac=0)
        player.consumed['old']=960
        service.relay_recipients['old']='alice';self.person='bob'
        service.protect_relay_audio(state)
        self.assertFalse(service.relay_recipients)
        self.assertFalse(service.relay_identity_hold)

    async def test_finished_tool_only_response_cannot_mute_new_visitors(self):
        service=self.service
        state=VoiceSession(service.player,service.log)
        state.responses['tool']=dict(status='completed')
        service.relay_recipients['tool']='alice';self.person='bob'
        service.protect_relay_audio(state)
        self.assertFalse(service.relay_recipients)
        self.assertFalse(service.relay_identity_hold)

    async def test_old_response_cannot_query_the_new_persons_memory(self):
        service=self.service;service.memory=Mock()
        service.reply_people['old']='alice';service.response_revisions['old']=1
        self.person='bob';sent=[]
        async def send(value):sent.append(json.loads(value))
        await service.tool(SimpleNamespace(send=send),dict(name='search_person_memory',call_id='stale-read',
            response_id='old',arguments=json.dumps(dict(keywords=[]))))
        self.assertFalse(json.loads(sent[-1]['item']['output'])['accepted'])
        service.memory.search.assert_not_called()


if __name__=='__main__':unittest.main()
