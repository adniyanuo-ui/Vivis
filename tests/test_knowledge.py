"""Persona injection, private-source boundaries and optional local identity."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.paths import ROOT
from vivis.interaction.knowledge import Knowledge,professional_profile
from vivis.speech.managed import voice_configuration,refresh_knowledge
from vivis.vision.familiar import FamiliarFaces,match


def config():
    c=json.loads((ROOT/'configs/perception/familiar_faces.json').read_text())
    c['backend']='sface' # Existing 128-d mock tests target the retained SFace backend.
    return c
def vector(i):
    v=np.zeros(128,dtype=np.float32);v[i]=1.;return v


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.robot=self.root/'configs/robot.json';self.robot.parent.mkdir()
        self.robot.write_text(json.dumps(dict(knowledge='knowledge.json')))
        self.c=dict(schema_version='vivis-knowledge-1.0',enabled=True,
            persona_document='docs/knowledge/characters/persona.md',persona_sections=[1,2,3,4],
            people_directory='private/people',allow_cloud_professional_context=True,
            recognition='configs/recognition.json')
        (self.root/'configs/knowledge.json').write_text(json.dumps(self.c))
        (self.root/'configs/recognition.json').write_text(json.dumps(config()))
        persona=self.root/self.c['persona_document'];persona.parent.mkdir(parents=True)
        persona.write_text('# Persona\n## 1. Identity\n青年机器人\n## 2. Likes\n篮球\n## 3. Style\n温暖\n## 4. Values\n友善\n## 5. Future\nFUTURE_UNIMPLEMENTED\n')
        self.directory=self.root/'private/people/testperson';self.directory.mkdir(parents=True)
        self.profile=dict(schema_version='vivis-person-1.0',allow_cloud_context=True,
            display_name='测试同事',preferred_address='测试工',organization='测试公司',role='工程师',
            professional_summary='喜欢讨论算法验证。',raw_resume='RAW_RESUME_CANARY',phone='13900000000',
            photo='/private/PHOTO_CANARY.jpg')
        (self.directory/'profile.json').write_text(json.dumps(self.profile))
        (self.directory/'resume.local.txt').write_text('RAW_FILE_CANARY')

    def knowledge(self):return Knowledge(self.robot,self.root)

    def test_only_selected_public_sections_and_professional_fields_reach_prompt(self):
        k=self.knowledge();text=k.base_instructions()
        for included in ('青年机器人','篮球','测试同事','工程师','不挖苦用户或同事','标准普通话'):
            self.assertIn(included,text)
        for excluded in ('FUTURE_UNIMPLEMENTED','RAW_RESUME_CANARY','RAW_FILE_CANARY','13900000000','PHOTO_CANARY','photo'):
            self.assertNotIn(excluded,text)
        self.assertIsNone(k.current_person(20))
        self.assertIn('不要沿用',k.person_instructions(20))

    def test_no_consent_no_person_context_and_sensitive_summary_rejected(self):
        self.profile['allow_cloud_context']=False
        (self.directory/'profile.json').write_text(json.dumps(self.profile))
        self.assertNotIn('测试同事',self.knowledge().base_instructions())
        for value in ('手机号13900000000','me@example.org','病史详细记录'):
            with self.assertRaises(ValueError):professional_profile(dict(self.profile,allow_cloud_context=True,professional_summary=value))

    def test_cloud_control_base_does_not_preload_unverified_person_profiles(self):
        k=self.knowledge()
        self.assertNotIn('测试同事',k.base_instructions(include_people=False))
        self.assertNotIn('喜欢讨论算法验证',k.base_instructions(include_people=False))
        self.assertIn('青年机器人',k.base_instructions(include_people=False))
        path=self.root/config()['state_file'];path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(schema_version='vivis-familiar-person-1.0',state='recognized',
            person_id='testperson',observed_monotonic=20.)))
        self.assertIn('测试同事',k.person_context(20.)['instructions'])

    def test_explicit_business_sections_without_source_notes(self):
        path=self.root/'docs/knowledge/people/business.md';path.parent.mkdir()
        path.write_text('## 1. Facts\nBUSINESS_FACT\n## 2. Program\nPROGRAM_FACT\n## 3. Notes\nNOT_FOR_RUNTIME\n')
        self.c['business_documents']=[dict(path=str(path.relative_to(self.root)),sections=[1,2])]
        (self.root/'configs/knowledge.json').write_text(json.dumps(self.c))
        text=self.knowledge().base_instructions()
        self.assertIn('BUSINESS_FACT',text);self.assertIn('PROGRAM_FACT',text);self.assertNotIn('NOT_FOR_RUNTIME',text)

    def test_state_expires_and_corrupt_or_unknown_identity_cannot_crash_chat(self):
        k=self.knowledge();path=self.root/config()['state_file'];path.parent.mkdir(parents=True)
        good=dict(schema_version='vivis-familiar-person-1.0',state='recognized',person_id='testperson',observed_monotonic=20.)
        path.write_text(json.dumps(good));self.assertEqual(k.current_person(21)['display_name'],'测试同事')
        self.assertIsNone(k.current_person(20+config()['max_result_age_s']+.1));self.assertIsNone(k.current_person(19))
        for bad in ([],None,dict(good,person_id=[]),dict(good,person_id='other'),dict(good,observed_monotonic='20'),dict(good,state='unknown')):
            path.write_text(json.dumps(bad));self.assertIsNone(k.current_person(20))
        path.write_text('{broken');self.assertIsNone(k.current_person(20))

    def test_current_identity_and_third_party_knowledge_are_separate(self):
        k=self.knowledge();path=self.root/config()['state_file'];path.parent.mkdir(parents=True)
        good=dict(schema_version='vivis-familiar-person-1.0',state='recognized',person_id='testperson',observed_monotonic=20.)
        path.write_text(json.dumps(good));snapshot=k.person_context(20.)
        self.assertTrue(snapshot['recognized']);self.assertEqual(snapshot['person_id'],'testperson')
        self.assertIn('"person_id": "testperson"',snapshot['instructions'])
        for instruction in ('云端没有收到摄像头图像','姓名、称谓与职务不能套用','用户纠正优先','不证明它就是发声者'):
            self.assertIn(instruction,snapshot['instructions'])
        path.write_text(json.dumps(dict(good,person_id='weiyufei')))
        unknown=k.person_context(20.)
        self.assertFalse(unknown['recognized']);self.assertIsNone(unknown['person_id'])
        self.assertIn('"state": "unknown", "person": null',unknown['instructions'])
        self.assertIn('unknown应坦诚说暂时没有确认身份',unknown['instructions'])

    def test_review_retains_only_a_social_name_and_never_a_verified_profile(self):
        k=self.knowledge();path=self.root/config()['state_file'];path.parent.mkdir(parents=True)
        review=dict(schema_version='vivis-familiar-person-1.0',state='reviewing',person_id='testperson',
            observed_monotonic=22.,confirmed_monotonic=20.)
        path.write_text(json.dumps(review))
        self.assertIsNone(k.current_person(22.))
        self.assertFalse(k.person_context(22.)['recognized'])
        social=k.person_context(22.,allow_identification=False,allow_review=True)
        self.assertTrue(social['reviewing']);self.assertEqual(social['person_id'],'testperson')
        self.assertIn('测试工',social['instructions'])
        for private in ('喜欢讨论算法验证','测试公司','工程师'):
            self.assertNotIn(private,social['instructions'])
        retained=k.person_context(22.,allow_review=True,retained_person_id='testperson')
        self.assertIn('"identity_retained": true',retained['instructions'])
        self.assertIn('喜欢讨论算法验证',retained['instructions'])
        wrong=k.person_context(22.,allow_review=True,retained_person_id='another-person')
        self.assertNotIn('"identity_retained": true',wrong['instructions'])
        stale=22.+config()['max_result_age_s']+.1
        self.assertEqual(k.current_identity(stale),dict(state='unknown',person=None)) # stale producer
        expired=k.person_context(stale,allow_review=True,retained_person_id='testperson')
        self.assertIsNone(expired['person_id']);self.assertNotIn('"identity_retained": true',expired['instructions'])
        for changed in (dict(review,observed_monotonic=20.+config()['review_hold_s']),dict(review,confirmed_monotonic=23.),
                        dict(review,confirmed_monotonic=None)):
            path.write_text(json.dumps(changed))
            self.assertEqual(k.current_identity(changed['observed_monotonic']),dict(state='unknown',person=None))

    def test_paths_and_symlinks_cannot_include_arbitrary_source_files(self):
        for key,value in [('persona_document','private/people/testperson/resume.local.txt'),('people_directory','docs')]:
            c=dict(self.c,**{key:value});(self.root/'configs/knowledge.json').write_text(json.dumps(c))
            with self.assertRaises(ValueError):self.knowledge()
        (self.root/'configs/knowledge.json').write_text(json.dumps(self.c))
        source=self.root/'elsewhere.json';source.write_text(json.dumps(self.profile))
        target=self.directory/'profile.json';target.unlink();target.symlink_to(source)
        with self.assertRaises(ValueError):self.knowledge()

    def test_real_voice_receives_ip_and_business_facts_without_changing_voice(self):
        speech=json.loads((ROOT/'configs/perception/speech.json').read_text())
        speech['managed_voice']=json.loads((ROOT/'configs/interaction/p1.json').read_text())['realtime_voice']
        c,_,_,session=voice_configuration(speech,ROOT/'configs/robots/xiaoshuai/robot.json')
        for text in ('前台接待兼首席情绪陪伴官','六层仿生皮肤','首钢','魏宇飞','CEO','只介绍能力类别'):
            self.assertIn(text,session['instructions'])
        self.assertEqual(session['voice'],c['voice']);self.assertEqual(session['speech_rate'],c['speech_rate'])
        self.assertEqual({t['name'] for t in session['tools']},{'robot_program','get_robot_status','search_person_memory','set_expression','set_speech_dialect','search_robot_memory','courier_pickup'})
        self.assertIsNone(Knowledge(ROOT/'configs/robots/mojia/robot.json').config)
        self.assertEqual(Knowledge(ROOT/'configs/robots/mojia/robot.json').person_instructions(),'')

    def test_text_chat_also_uses_same_knowledge(self):
        from vivis.interaction.conversation import ConversationRouter
        c=json.loads((ROOT/'configs/interaction/conversation.json').read_text())
        catalog=json.loads((ROOT/'configs/interaction/programs.json').read_text())
        router=ConversationRouter(c,catalog,dict(display_name='测试机器人'),self.knowledge())
        prompt=router.payload('你好')['messages'][0]['content']
        self.assertIn('青年机器人',prompt);self.assertIn('测试同事',prompt);self.assertNotIn('RAW_FILE_CANARY',prompt)


class FamiliarTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.c=config()
        self.f=FamiliarFaces(self.c,self.root,model=Mock(),gallery=(np.array([vector(0),vector(1)]),np.array(['alpha','beta'])))
        self.addCleanup(self.f.close)

    def infer(self,v=0,stamp=10,epoch=None):
        with patch('vivis.vision.familiar.feature',return_value=vector(v)):
            self.f.infer(None,None,stamp,self.f.epoch if epoch is None else epoch)

    def test_confidence_and_runner_up_margin_both_required(self):
        bank=np.array([vector(0),vector(1)])
        self.assertEqual(match(vector(0),bank,['alpha','beta'],.5,.08)[0],'alpha')
        self.assertIsNone(match(vector(2),bank,['alpha','beta'],.5,.08)[0])
        self.assertIsNone(match((vector(0)+vector(1))/np.sqrt(2),bank,['alpha','beta'],.5,.08)[0])
        self.assertEqual(match(vector(0),np.array([vector(0),vector(0)]),['alpha','alpha'],.5,.08)[0],'alpha')

    def test_confirmation_no_face_expiry_and_identity_switch(self):
        for _ in range(self.f.c['confirmation_samples']-1):self.infer();self.assertIsNone(self.f.result['person_id'])
        self.infer();self.assertEqual(self.f.result['person_id'],'alpha')
        self.infer(1);self.assertIsNone(self.f.result['person_id'])
        self.infer(1);self.infer(1);self.assertEqual(self.f.result['person_id'],'beta')
        row=np.array([0,0,80,80]+[0]*11,dtype=np.float32)
        stale=10.+self.f.c['max_result_age_s']+.1
        self.f.last_submit=stale
        self.assertIsNone(self.f.step((1,10,None),row,now=stale)['person_id'])
        self.assertIsNone(self.f.step((1,10,None),None,now=10)['person_id'])

    def test_late_inference_does_not_restore_old_identity_after_reset_or_close(self):
        epoch=self.f.epoch;self.f.reset()
        for _ in range(4):self.infer(epoch=epoch)
        self.assertIsNone(self.f.result['person_id'])
        self.f.close()
        for _ in range(4):self.infer()
        self.assertIsNone(self.f.result['person_id'])

    def test_low_score_reviews_same_person_until_required_recovery_samples(self):
        for _ in range(3):self.infer(stamp=10.)
        self.infer(2,10.5)
        self.assertEqual(self.f.result['state'],'reviewing')
        self.assertEqual(self.f.result['person_id'],'alpha')
        self.assertEqual(self.f.result['confirmed_monotonic'],10.)
        for i in range(self.f.c['confirmation_samples']-1):
            self.infer(0,11.+i*.25);self.assertEqual(self.f.result['state'],'reviewing')
        self.infer(0,12.)
        self.assertEqual(self.f.result['state'],'recognized')
        self.assertEqual(self.f.result['confirmed_monotonic'],12.)

    def test_review_has_a_fixed_deadline_not_refreshed_by_intermittent_matches(self):
        for _ in range(3):self.infer(stamp=10.)
        deadline=10.+self.f.c['review_hold_s']
        for stamp in np.arange(10.5,deadline,.5):
            self.infer(2 if int(stamp*2)%2 else 0,float(stamp))
            self.assertEqual(self.f.result['state'],'reviewing')
            self.assertEqual(self.f.result['confirmed_monotonic'],10.)
        self.infer(2,deadline)
        self.assertEqual(self.f.result['state'],'unknown');self.assertIsNone(self.f.result['person_id'])

    def test_multiple_faces_no_face_or_large_bbox_jump_clears_held_identity(self):
        row=np.array([0,0,80,80]+[0]*11,dtype=np.float32)
        self.f.job.offer=Mock(return_value=False)
        sample=(1,10.5,np.zeros((100,100,3),dtype=np.uint8))
        for reason in ('many','none','jump'):
            for _ in range(3):self.infer(stamp=10.)
            self.infer(2,10.5);self.f.last_bbox=row[:4].copy()
            changed=row.copy()
            if reason=='jump':changed[0]=200
            result=self.f.step(sample,None if reason=='none' else changed,now=10.5,faces=2 if reason=='many' else 1)
            self.assertEqual(result['state'],'unknown');self.assertIsNone(result['person_id'])

    def test_stale_review_invalidates_pending_old_inference(self):
        for _ in range(3):self.infer(stamp=10.)
        self.infer(2,10.5);epoch=self.f.epoch
        stale=10.5+self.f.c['max_result_age_s']+.1
        self.f.last_submit=stale
        row=np.array([0,0,80,80]+[0]*11,dtype=np.float32)
        result=self.f.step((1,stale,None),row,now=stale)
        self.assertEqual(result['state'],'unknown')
        for _ in range(3):self.infer(stamp=stale,epoch=epoch)
        self.assertIsNone(self.f.result['person_id'])

    def test_first_pending_inference_is_not_invalidated_by_empty_initial_result(self):
        self.f.job.offer=Mock(return_value=True)
        epoch=self.f.epoch
        sample=(1,10.,np.zeros((100,100,3),dtype=np.uint8))
        row=np.array([0,0,80,80]+[0]*11,dtype=np.float32)
        self.f.step(sample,row,now=10.)
        self.assertEqual(self.f.epoch,epoch)
        for _ in range(3):self.infer(stamp=10.,epoch=epoch)
        self.assertEqual(self.f.result['state'],'recognized')

    def test_first_sample_after_stale_camera_must_start_confirmation_again(self):
        for _ in range(3):self.infer(stamp=10.)
        def offer(_fn,_image,_row,stamp,epoch):
            self.infer(stamp=stamp,epoch=epoch);return True
        self.f.job.offer=offer
        stale=10.+self.f.c['max_result_age_s']+.1
        sample=(1,stale,np.zeros((100,100,3),dtype=np.uint8))
        row=np.array([0,0,80,80]+[0]*11,dtype=np.float32)
        result=self.f.step(sample,row,now=stale)
        self.assertEqual(result['state'],'unknown');self.assertEqual(self.f.hits,1)

    def test_missing_or_corrupt_gallery_degrades_optional_recognition(self):
        absent=FamiliarFaces(self.c,self.root);self.addCleanup(absent.close)
        self.assertFalse(absent.available)
        broken=FamiliarFaces(self.c,self.root,model=Mock(),gallery=(np.zeros((1,128)),np.array(['alpha'])))
        self.addCleanup(broken.close);self.assertFalse(broken.available);self.assertIsNotNone(broken.error)

    def test_state_is_private_atomic_and_old_owner_cannot_remove_new_owner(self):
        self.f.publish(self.f.unknown())
        self.assertEqual(self.f.path.stat().st_mode & 0o777,0o600)
        data=json.loads(self.f.path.read_text())
        self.assertNotIn('photo',data);self.assertNotIn('embedding',data);self.assertNotIn('display_name',data)
        data['owner']='another';self.f.path.write_text(json.dumps(data));self.f.close()
        self.assertTrue(self.f.path.exists())

    def test_slow_inference_has_no_waiting_queue(self):
        self.f.job.offer=Mock(return_value=False)
        sample=(1,10,np.zeros((100,100,3),dtype=np.uint8));row=np.array([0,0,80,80]+[0]*11,dtype=np.float32)
        self.f.step(sample,row,now=10);self.f.step(sample,row,now=10.1)
        self.assertEqual(self.f.job.offer.call_count,2);self.assertEqual(self.f.last_submit,-float('inf'))

    def test_interaction_only_shares_existing_camera_and_standby_stays_camera_off(self):
        from vivis.interaction.standby import plan
        from vivis.runtime.robot import Robot
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        awake=json.loads((ROOT/'configs/interaction/interaction.json').read_text())
        sleep=json.loads((ROOT/'configs/interaction/standby.json').read_text())
        self.assertIn('familiar_faces',plan(robot,awake,1.)[4].tracking)
        self.assertIsNone(plan(robot,sleep,1.)[4].tracking)


class ContextRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_updates_without_response_creation_and_stops_at_epoch_change(self):
        person=[None];active=[True];sent=[];log=Mock()
        knowledge=SimpleNamespace(config={},person_context=lambda:dict(instructions=person[0] or 'unknown',recognized=person[0] is not None,person_id='testperson' if person[0] else None))
        knowledge.config={'enabled':True}
        async def send(message):sent.append(json.loads(message))
        task=asyncio.create_task(refresh_knowledge(SimpleNamespace(send=send),knowledge,dict(instructions='BASE '),'unknown',lambda:active[0],log,7))
        try:
            person[0]='KNOWN';await asyncio.sleep(.56)
            person[0]=None;await asyncio.sleep(.56)
            active[0]=False;await asyncio.wait_for(task,timeout=1.)
        finally:
            task.cancel();await asyncio.gather(task,return_exceptions=True)
        self.assertEqual([m['session']['instructions'] for m in sent],['BASE KNOWN','BASE unknown'])
        self.assertTrue(all(m['type']=='session.update' and set(m['session'])=={'instructions'} for m in sent))
        self.assertEqual(log.write.call_count,2)
        self.assertEqual(log.write.call_args_list[0].kwargs['person_id'],'testperson')
        self.assertIsNone(log.write.call_args_list[1].kwargs['person_id'])
