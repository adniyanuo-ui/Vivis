"""Reception identity acquisition is fast; replacement/loss has one bounded hold."""
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from vivis.interaction.identity_stability import IdentityStability


class IdentityStabilityTests(unittest.TestCase):
    def test_verified_first_identity_is_immediate_and_change_waits_eight_seconds(self):
        gate=IdentityStability(8)
        self.assertIsNone(gate.observe(None,10)['person_id'])
        self.assertEqual(gate.observe('alice',11)['person_id'],'alice')
        self.assertTrue(gate.observe('bob',12)['held'])
        self.assertEqual(gate.observe('bob',19.99)['person_id'],'alice')
        result=gate.observe('bob',20)
        self.assertEqual(result['person_id'],'bob');self.assertFalse(result['held'])

    def test_loss_does_not_extend_hold_and_old_identity_cannot_resurrect(self):
        gate=IdentityStability(8);gate.observe('alice',10)
        for t in (11,12,15,18.99):self.assertEqual(gate.observe(None,t)['person_id'],'alice')
        self.assertIsNone(gate.observe(None,19)['person_id'])
        self.assertEqual(gate.released_since,11)
        self.assertIsNone(gate.observe(None,50)['person_id'])

    def test_return_of_original_cancels_hold_and_candidates_cannot_extend_it(self):
        gate=IdentityStability(8);gate.observe('alice',10)
        gate.observe('bob',11);self.assertFalse(gate.observe('alice',18)['held'])
        gate.observe(None,19);gate.observe('bob',21);gate.observe(None,23)
        self.assertEqual(gate.observe('carol',26.99)['person_id'],'alice')
        self.assertEqual(gate.observe('carol',27)['person_id'],'carol')


class StableBusinessIdentityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from test_cloud_control import CloudBackendTests
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.s=CloudBackendTests().service(temp.name,[]);self.addCleanup(self.s.log.close)
        self.s.enabled=True;self.s.context.update(revision=1,mode='interaction',target='interaction',ready=True)
        self.s.knowledge.people={'alice':dict(display_name='甲',preferred_address='甲工',professional_summary='甲的资料'),
                                 'bob':dict(display_name='乙',preferred_address='小乙',professional_summary='乙的资料')}
        self.s.recognized_person=Mock(return_value='alice')
        self.s.presence.visible=Mock(return_value=True)

    async def test_labels_use_preferred_names_and_unknown_without_rewriting_old_snapshot(self):
        s=self.s
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=10) as clock:
            s.recognized_person.return_value=None
            self.assertEqual(s.identity_labels()['identity_label'],'other-other')
            s.recognized_person.return_value='alice'
            self.assertEqual(s.identity_labels()['identity_label'],'甲工-甲工')
            clock.return_value=11;s.recognized_person.return_value=None
            recorded=s.identity_labels()
            self.assertEqual(recorded['identity_label'],'甲工-other')
            clock.return_value=12;s.recognized_person.return_value='bob'
            self.assertEqual(s.identity_labels()['identity_label'],'甲工-小乙')
            clock.return_value=19
            self.assertEqual(s.identity_labels()['identity_label'],'小乙-小乙')
            self.assertEqual(recorded['identity_label'],'甲工-other')
            self.assertEqual(s.identity_labels(dict(person_id=None,candidate_id='bob'))['identity_label'],'other-小乙')

    async def test_missing_face_keeps_request_and_profile_then_expires_without_second_grace(self):
        from vivis.interaction.message_lane import StaleMessageTurn
        from vivis.speech.cloud_managed import ConversationReset
        from vivis.speech.realtime import VoiceSession
        s=self.s;state=VoiceSession(s.player,s.log)
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=10) as clock:
            s.verify_identity(state);s.begin_user_turn('request');request=s.message_request()
            clock.return_value=11;s.recognized_person.return_value=None;s.presence.visible.return_value=False
            request.guard()
            stable=s.stable_identity()
            context=s.knowledge.person_context(11,allow_review=True,retained_person_id='alice',stable_identity=stable)
            self.assertEqual(context['person_id'],'alice');self.assertIn('甲的资料',context['instructions'])
            clock.return_value=18.99;request.guard();self.assertEqual(s.identified_person(),'alice')
            clock.return_value=19
            with self.assertRaises(StaleMessageTurn):request.guard()
            # No extra eight seconds after the business hold already expired.
            s.turn_item_id=None
            with self.assertRaisesRegex(ConversationReset,'identity_uncertain'):s.verify_identity(state)
            self.assertIsNone(s.conversation_person)

    async def test_replacement_preserves_old_context_until_boundary_then_invalidates_old_work(self):
        from vivis.interaction.message_lane import StaleMessageTurn
        from vivis.speech.cloud_managed import ConversationReset
        from vivis.speech.realtime import VoiceSession
        s=self.s;state=VoiceSession(s.player,s.log)
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=10) as clock:
            s.verify_identity(state);s.begin_user_turn('request');request=s.message_request()
            clock.return_value=11;s.recognized_person.return_value='bob';s.verify_identity(state);request.guard()
            self.assertEqual(s.conversation_person,'alice')
            clock.return_value=18;stable=s.stable_identity()
            context=s.knowledge.person_context(18,allow_review=True,retained_person_id='alice',stable_identity=stable)
            self.assertEqual(context['person_id'],'alice');self.assertNotIn('乙的资料',context['instructions'])
            # The countdown must not rewrite cloud instructions on every tick.
            clock.return_value=18.1
            again=s.knowledge.person_context(18.1,allow_review=True,retained_person_id='alice',stable_identity=s.stable_identity())
            self.assertEqual(context['instructions'],again['instructions'])
            clock.return_value=19
            with self.assertRaises(StaleMessageTurn):request.guard()
            with self.assertRaisesRegex(ConversationReset,'confirmed_person_changed'):s.verify_identity(state)
            self.assertEqual(s.conversation_person,'bob')
