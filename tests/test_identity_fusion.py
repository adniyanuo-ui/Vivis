import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from vivis.interaction.identity_fusion import IdentityFusion,configuration,evidence_label,fresh,unavailable
from vivis.paths import ROOT


PEOPLE={'alice':dict(preferred_address='何工'),'bob':dict(preferred_address='小王'),'carol':dict(preferred_address='小梁')}


def policy():return configuration('configs/perception/identity_fusion.json',ROOT)


def evidence(state='confirmed',person='alice',stamp=10.,uid=None,**extra):
    return dict(state=state,person_id=person,stamp=stamp,evidence_id=uid or state+str(person)+str(stamp),**extra)


class FusionRulesTests(unittest.TestCase):
    def setUp(self):self.gate=IdentityFusion(policy())
    def observe(self,face=None,voice=None,now=10):return self.gate.observe(face,voice,now,PEOPLE)

    def test_confirmed_image_wins_every_voice_state(self):
        for voice in (None,evidence('suspected','bob'),evidence('other',None),evidence('confirmed','bob')):
            result=self.observe(evidence(),voice)
            self.assertEqual(result['person_id'],'alice')
        self.assertTrue(result['conflict'])

    def test_voice_confirmation_works_without_image_and_overrides_weak_image(self):
        for face in (None,evidence('suspected','bob'),evidence('other',None)):
            result=self.observe(face,evidence());self.assertEqual(result['person_id'],'alice')
            self.assertEqual(result['reason'],'voice_confirmed')

    def test_missing_evidence_holds_but_two_definite_other_results_clear(self):
        self.observe(evidence())
        for a,b in ((None,None),(None,evidence('other',None)),(evidence('suspected','bob'),None)):
            result=self.observe(a,b);self.assertEqual(result['person_id'],'alice');self.assertTrue(result['held'])
        result=self.observe(evidence('other',None),evidence('other',None))
        self.assertIsNone(result['person_id']);self.assertEqual(result['reason'],'both_other')

    def test_joint_weak_confirmation_requires_two_disjoint_sources_and_new_frames(self):
        image=evidence('suspected','alice');voice=evidence('suspected','alice',source_turn_ids=['a','b'])
        for _ in range(20):self.assertIsNone(self.observe(image,voice)['person_id'])
        # A fresh frame with the exact same voice is not new evidence.
        image=evidence('suspected','alice',10.2)
        self.assertIsNone(self.observe(image,voice,10.2)['person_id'])
        voice2=evidence('suspected','alice',10.2,source_turn_ids=['b','c'])
        self.assertIsNone(self.observe(image,voice2,10.2)['person_id'])
        # Missing evidence pauses, not fabricates or destroys the first pair.
        self.assertIsNone(self.observe(None,None,10.3)['person_id'])
        result=self.observe(evidence('suspected','alice',10.4),evidence('suspected','alice',10.4,source_turn_ids=['d']),10.4)
        self.assertEqual(result['person_id'],'alice');self.assertEqual(result['reason'],'joint_confirmed')

    def test_weak_history_rules_do_not_bootstrap_single_candidate(self):
        combinations=[(evidence('suspected','alice'),evidence('other',None)),
            (evidence('suspected','alice'),evidence('suspected','bob'))]
        for a,b in combinations:
            gate=IdentityFusion(policy());self.assertIsNone(gate.observe(a,b,10,PEOPLE)['person_id'])
            gate.observe(evidence(),None,10,PEOPLE)
            self.assertEqual(gate.observe(a,b,10,PEOPLE)['person_id'],'alice')
            self.assertIsNone(gate.observe(evidence('suspected','bob'),evidence('suspected','carol'),10,PEOPLE)['person_id'])

    def test_stale_future_unregistered_or_quality_rejected_is_none_not_other(self):
        for value in (evidence(stamp=0),evidence(stamp=11),evidence(person='unknown'),
                evidence(stamp=float('nan')),dict(state='confirmed',person_id='alice',stamp=10)):
            self.assertEqual(fresh(value,10,1.5,PEOPLE)['state'],'unavailable')
        self.observe(evidence());self.assertEqual(self.observe(None,None,100)['person_id'],'alice')
        self.gate.reset(101);self.assertIsNone(self.observe(None,None,101)['person_id'])

    def test_joint_evidence_expiration_skew_and_clock_reset(self):
        self.observe(evidence('suspected'),evidence('suspected'))
        self.observe(None,None,19)
        result=self.observe(evidence('suspected',stamp=19),evidence('suspected',stamp=19),19)
        self.assertIsNone(result['person_id'])
        result=self.observe(evidence('suspected',stamp=20),evidence('suspected',stamp=17),20)
        self.assertIsNone(result['person_id'])
        self.observe(evidence(stamp=20),None,20)
        self.assertIsNone(self.observe(None,None,1)['person_id'])

    def test_labels_distinguish_none_other_and_suspected(self):
        self.assertEqual(evidence_label(unavailable(),PEOPLE,'图'),'None/图')
        self.assertEqual(evidence_label(evidence('other',None),PEOPLE,'声'),'other/声')
        self.assertEqual(evidence_label(evidence('suspected'),PEOPLE,'图'),'何工/图/疑')

    def test_staging_observes_aligned_pairs_without_reassigning_business_actor(self):
        self.gate.person='bob'
        first=self.gate.stage_voice(evidence('suspected'),evidence('suspected'),10,PEOPLE)
        self.assertEqual(first['state'],'suspected');self.assertEqual(self.gate.person,'bob')
        second=self.gate.stage_voice(evidence('suspected',stamp=11),evidence('suspected',stamp=11),11,PEOPLE)
        self.assertEqual(second['state'],'confirmed');self.assertEqual(self.gate.person,'bob')
        result=self.observe(evidence('suspected',stamp=15),second,15)
        self.assertEqual(result['person_id'],'alice');self.assertEqual(result['reason'],'joint_confirmed')

    def test_bad_fusion_policy_fails_before_devices_are_opened(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'policy.json'
            for key,value in [('voice_max_age_s',float('nan')),('pair_confirmation_samples',1),('enabled','true')]:
                config=policy();config[key]=value;path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):configuration('policy.json',root)

    def test_intervening_confirmed_person_clears_old_joint_votes(self):
        self.gate.stage_voice(evidence('suspected'),evidence('suspected'),10,PEOPLE)
        self.assertEqual(len(self.gate.pairs),1)
        self.observe(None,evidence(person='bob',stamp=10.5),10.5)
        self.assertEqual(self.gate.pairs,[])


class FusedVoiceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from test_cloud_control import CloudBackendTests
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.events=[];self.s=CloudBackendTests().service(temp.name,self.events);self.addCleanup(self.s.log.close)
        self.s.knowledge.people=copy.deepcopy(PEOPLE)
        self.s.continuity=SimpleNamespace(speaker=None,visual_evidence={},sampling_epoch=0,audio_epoch=0,submit=Mock(),
            blocked_people=set(),finalized={})
        self.s.init_fused_identity();self.s.context.update(revision=1,mode='interaction',target='interaction',ready=True)
        self.s.enabled=True;self.s.can_speak=Mock(return_value=True);self.s.presence.visible=Mock(return_value=True)
        self.s.presence.allowed=Mock(return_value=True)

    def face(self,value):self.s.continuity.visual_evidence=dict(identity_evidence=value)
    def result(self,state='confirmed',person='alice',**kw):
        s=self.s
        return dict(voice_context=s.sync_voice_binding(),session=s.memory_session,item_id=s.turn_item_id,
            source_generation=s.input_generation,sample_epoch=0,audio_epoch=0,turn_id='original',
            identity_evidence=evidence(state,person),**kw)

    async def test_voice_result_waits_for_next_boundary_and_never_reassigns_original_round(self):
        s=self.s
        with patch('time.monotonic',return_value=10):
            s.begin_user_turn('first');original=s.turn_person
            s.accept_speaker_identity(self.result())
            self.assertIsNone(s.identified_person());self.assertEqual(s.turn_person,original)
            self.assertIn('何工/声',s.identity_labels()['identity_label'])
            s.input_generation+=1  # The receive loop advances before begin_user_turn.
            s.begin_user_turn('second')
            self.assertEqual(s.turn_person,'alice');self.assertEqual(s.memory_turns['first']['person'],original)
            self.assertEqual(s.identified_person(),'alice')
            self.assertIn('None/声',s.identity_labels()['identity_label'])
            # The new input is already bound to the freshly confirmed person.
            # Do not reconnect and lose its audio merely because old public
            # anonymous chat exists on this still-live connection.
            from vivis.speech.realtime import VoiceSession
            s.resume_history.append(dict(role='user',content='匿名交流'))
            s.in_speech=True;s.verify_identity(VoiceSession(s.player,s.log))
            self.assertEqual(s.conversation_person,'alice')

    async def test_current_image_is_immediate_without_old_eight_second_replacement(self):
        s=self.s
        with patch('time.monotonic',return_value=10):
            self.face(evidence());self.assertEqual(s.identified_person(),'alice')
            self.face(evidence(person='bob',uid='new_face'));self.assertEqual(s.identified_person(),'bob')
            self.face(unavailable());self.assertEqual(s.identified_person(),'bob')

    async def test_late_wrong_item_session_connection_epoch_or_generation_cannot_apply(self):
        s=self.s
        with patch('time.monotonic',return_value=10):
            s.begin_user_turn('current')
            result=self.result()
            for key,value in [('item_id','old'),('session','old'),('voice_context','old'),('audio_epoch',1),('source_generation',999)]:
                invalid=dict(result);invalid[key]=value;s.accept_speaker_identity(invalid)
                self.assertIsNone(s.voice_pending)
            s.message_connection=object();s.accept_speaker_identity(result);self.assertIsNone(s.voice_pending)
            self.assertIsNone(s.identified_person())

    async def test_new_connection_cannot_reuse_previous_weak_joint_vote(self):
        s=self.s
        with patch('time.monotonic',return_value=10):
            self.face(evidence('suspected'));s.begin_user_turn('current')
            s.accept_speaker_identity(self.result('suspected'))
            self.assertEqual(len(s.identity_fusion.pairs),1)
            s.message_connection=object();s.sync_voice_binding()
            self.assertEqual(s.identity_fusion.pairs,[])
            self.assertIsNone(s.voice_pending)

    async def test_invalidated_audio_never_applies_voice(self):
        s=self.s
        with patch('time.monotonic',return_value=10):
            s.begin_user_turn('current')
            s.accept_speaker_identity(self.result());s.continuity.audio_epoch=1;s.admit_voice_identity()
            self.assertIsNone(s.identified_person())

    async def test_visual_collection_epoch_and_conflict_do_not_revoke_valid_voice(self):
        s=self.s
        with patch('time.monotonic',return_value=10):
            s.begin_user_turn('current');result=self.result(sampling_conflict=True)
            s.continuity.sampling_epoch=9
            s.accept_speaker_identity(result)
            self.assertIsNotNone(s.voice_pending)
            s.input_generation+=1;s.begin_user_turn('next')
            self.assertEqual(s.turn_person,'alice')

    async def test_speaker_status_distinguishes_gallery_readiness_and_current_result(self):
        self.s.continuity.speaker=SimpleNamespace(c=dict(enrollment_samples=3),gallery=dict(alice=[{},{}]))
        status=self.s.speaker_status('alice')
        self.assertTrue(status['enabled']);self.assertTrue(status['library_present'])
        self.assertFalse(status['current_person_eligible']);self.assertEqual(status['current_person_active_references'],2)
        self.assertEqual(status['current_result']['state'],'unavailable')

    async def test_conflicting_voice_cannot_serve_current_actor_without_confirmed_image(self):
        from vivis.interaction.message_lane import StaleMessageTurn
        s=self.s
        with patch('time.monotonic',return_value=10):
            self.face(evidence());s.begin_user_turn('current');request=s.message_request()
            self.face(unavailable());s.accept_speaker_identity(self.result(person='bob'))
            with self.assertRaises(StaleMessageTurn):request.guard()
            self.face(evidence());request.guard()  # User's explicit image-priority policy.

    async def test_private_context_uses_fused_person_and_preserves_frozen_actor(self):
        s=self.s
        with patch('time.monotonic',return_value=10):
            s.begin_user_turn('anonymous');s.accept_speaker_identity(self.result())
            s.begin_user_turn('identified')
            context=s.knowledge.person_context(10,allow_identification=True,allow_review=True,
                retained_person_id='alice',stable_identity=s.stable_identity())
            self.assertEqual(context['person_id'],'alice');self.assertIn('local_image_voice_fusion',context['instructions'])
            self.assertEqual(s.memory_turns['anonymous']['person'].split(':')[0],'anonymous')

    async def test_weak_joint_proof_is_paired_on_arrival_not_after_long_reply(self):
        s=self.s
        with patch('time.monotonic',return_value=10) as clock:
            self.face(evidence('suspected'));s.begin_user_turn('first')
            s.accept_speaker_identity(self.result('suspected'))
            clock.return_value=12;self.face(evidence('suspected',stamp=12));s.input_generation+=1;s.begin_user_turn('second')
            self.assertIsNone(s.identified_person())
            clock.return_value=12.5;self.face(evidence('suspected',stamp=12.5))
            result=self.result('suspected');result['identity_evidence']=evidence('suspected',stamp=12.5)
            s.accept_speaker_identity(result)
            self.assertEqual(s.voice_pending['adjudicated_evidence']['state'],'confirmed')
            self.assertIsNone(s.identified_person())
            clock.return_value=16;self.face(evidence('suspected',stamp=16));s.input_generation+=1;s.begin_user_turn('third')
            self.assertEqual(s.turn_person,'alice')

    async def test_recorder_to_model_to_business_boundary_preserves_original_database_actor(self):
        import numpy as np
        from vivis.speech.round_recording import RoundRecorder
        from vivis.speech.speaker_observation import SpeakerObserver
        from test_short_speaker_sampling import pcm
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        recorder=RoundRecorder(json.loads((ROOT/'configs/interaction/continuity.json').read_text()),root)
        self.addCleanup(recorder.close)
        c=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text())
        recorder.speaker=SpeakerObserver(c,root,lambda e:None,lambda *a,**k:None,extractor=lambda x:[1.,0.])
        for i in range(3):recorder.speaker._store_reference('alice',np.ones(16000,np.float32)*.1,np.array([1.,0.]),'ref'+str(i),'v')
        s=self.s;s.continuity=recorder;s.init_fused_identity()
        s.knowledge.current_identity=Mock(return_value=dict(state='unknown',person=None))
        s.recognized_person=Mock(return_value=None)
        with patch('time.monotonic',return_value=10.9):
            s.sync_voice_binding();recorder.queue.join()
            raw=pcm(.8)
            for i in range(40):recorder.capture(10+i*.02,raw[i*640:(i+1)*640],bytes(640))
            s.input_generation+=1;s.begin_user_turn('recorded');original=s.turn_person
            s.continuity_start(dict(item_id='recorded',audio_start_ms=0),SimpleNamespace(at=lambda ms:10.))
            recorder.final('recorded','现在请听听我的声音。')
            recorder.end('recorded',10.8,True,s.continuity_evidence(original));recorder.queue.join()
            recorder.speaker.job.wait(2);recorder.speaker.poll()
            self.assertIsNotNone(s.voice_pending)
            self.assertIsNone(s.identified_person());self.assertEqual(s.turn_person,original)
            s.input_generation+=1;s.begin_user_turn('next')
            self.assertEqual(s.turn_person,'alice')
            with recorder.store.db() as db:person=db.execute("SELECT person FROM rounds WHERE item='recorded'").fetchone()[0]
            self.assertEqual(person,'other')
            self.assertEqual(len(recorder.speaker.gallery['alice']),3)


if __name__=='__main__':unittest.main()
