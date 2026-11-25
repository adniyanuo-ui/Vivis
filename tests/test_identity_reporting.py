"""Public claims must follow current typed evidence, not a retained address."""
import copy
import unittest

from vivis.interaction.identity_fusion import recognition_report,unavailable


PEOPLE=dict(alice=dict(display_name='何裕德',preferred_address='何工',phone='PRIVATE_CANARY'),
    bob=dict(display_name='王润章',preferred_address='小王'))


def evidence(state='confirmed',person='alice'):return dict(state=state,person_id=person,reason='matched')


class RecognitionReportingTests(unittest.TestCase):
    def report(self,face=None,voice=None,person='alice',decision=None):
        return recognition_report(person,PEOPLE,face or unavailable(),voice or unavailable(),decision)

    def test_pending_voice_does_not_turn_retained_name_into_a_match(self):
        r=self.report(voice=unavailable('current_turn_not_ready'))
        self.assertEqual(r['business_basis'],'retained_previous')
        self.assertEqual(r['business_address'],'何工')
        self.assertFalse(r['can_claim_current_voice']);self.assertFalse(r['can_claim_current_image'])
        self.assertIn('尚未返回',r['notice']);self.assertIn('沿用此前',r['notice'])

    def test_repeated_suspects_never_upgrade_or_mutate_evidence(self):
        voice=dict(evidence('suspected'),score=.6483639479);before=copy.deepcopy(voice)
        for i in range(8):
            r=self.report(voice=voice)
            self.assertFalse(r['can_claim_current_voice']);self.assertEqual(r['current_voice']['state'],'suspected')
            self.assertIn('只是疑似何工',r['notice'])
        self.assertEqual(voice,before)

    def test_confirmed_voice_can_be_acknowledged_before_business_admission(self):
        r=self.report(voice=evidence(),person=None)
        self.assertTrue(r['can_claim_current_voice']);self.assertIsNone(r['business_person_id'])
        self.assertIn('本轮声音已确认：何工',r['notice']);self.assertIn('业务身份尚未接纳',r['notice'])
        self.assertFalse(r['can_claim_current_image'])

    def test_confirmed_image_does_not_imply_confirmed_voice(self):
        r=self.report(face=evidence(),voice=evidence('suspected'))
        self.assertEqual(r['business_basis'],'current_image')
        self.assertTrue(r['can_claim_current_image']);self.assertFalse(r['can_claim_current_voice'])

    def test_conflicting_channels_keep_their_own_names(self):
        r=self.report(face=evidence(),voice=evidence(person='bob'))
        self.assertEqual(r['business_basis'],'current_image');self.assertEqual(r['current_voice']['address'],'小王')
        self.assertIn('分别确认了不同人物',r['notice'])

    def test_joint_business_confirmation_does_not_fake_a_single_channel_match(self):
        r=self.report(face=evidence('suspected'),voice=evidence('suspected'),decision='joint_confirmed')
        self.assertEqual(r['business_basis'],'joint_evidence')
        self.assertFalse(r['can_claim_current_voice']);self.assertFalse(r['can_claim_current_image'])

    def test_other_is_not_pending_and_cannot_erase_a_business_actor(self):
        r=self.report(voice=evidence('other',None))
        self.assertEqual(r['current_voice']['state'],'other');self.assertFalse(r['can_claim_current_voice'])
        self.assertEqual(r['business_person_id'],'alice');self.assertIn('未匹配到熟人',r['notice'])

    def test_unknown_profiles_cannot_gain_claims_or_leak_private_fields(self):
        r=self.report(face=evidence(person='unknown'),voice=evidence(person='unknown'),person='unknown')
        self.assertEqual(r['business_basis'],'unknown');self.assertFalse(r['can_claim_current_voice'])
        self.assertNotIn('PRIVATE_CANARY',str(self.report(voice=evidence())))


if __name__=='__main__':unittest.main()
