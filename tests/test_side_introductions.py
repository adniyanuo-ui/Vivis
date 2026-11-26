"""Offline source attribution, bootstrap/hot load and actual fake-DAC dialogue."""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,Mock,patch

import numpy as np

from vivis.interaction.introductions import Introductions
from vivis.interaction.knowledge import Knowledge
from vivis.speech.side_dialogue import SideDialogue
from vivis.vision.photo_pool import PhotoPool
from vivis.vision.visitor_registration import VisitorRegistration


def vector(index):
    v=np.zeros(128,np.float32);v[index]=1;return v


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.c=dict(people_directory='private/people',sha256='model',score_threshold=.6,minimum_margin=.08,
            photo_sampling=dict(min_face_size_px=60,brightness_range=[.1,.95],min_laplacian_variance=25))
        self.owner='a'*32;self.reg=VisitorRegistration(self.root,self.c,self.owner);self.store=Introductions(self.root)
        self.image=np.random.default_rng(5).integers(40,220,(120,120,3),dtype=np.uint8)
        self.row=np.array([0,0,120,120],np.float32)
        self.reg.track(self.row,10.,1)
        self.token=self.reg.token
        self.reg.observe(self.token,self.image,self.row,vector(1),{},10.,None,np.dot)
        self.guest=self.reg.status(10.)
        self.info=dict(display_name='杨明扬',preferred_address='杨教授',background='人民大学信息学院，研究灵巧手')
        self.source=dict(turn_id='intro1',text='这是杨明扬，杨教授，人民大学信息学院，研究灵巧手。')

    def request(self,**kw):
        return self.store.request(kw.pop('information',self.info),guest=kw.pop('guest',self.guest),
            introducer=kw.pop('introducer','heyude'),source=kw.pop('source',self.source),session='visit1',now=10.,**kw)

    def test_full_bootstrap_activates_locked_bases_hotloads_and_survives_restart(self):
        out=self.request();self.assertEqual(out['status'],'pending_face');pid=out['person_id']
        self.assertFalse((self.root/'private/people'/pid/'profile.json').exists())
        self.reg.complete(10.1)
        profile=json.loads((self.root/'private/people'/pid/'profile.json').read_text())
        self.assertEqual(profile['preferred_address'],'杨教授')
        self.assertFalse(profile['introduction']['background_verified'])
        self.assertEqual(profile['introduction']['introducer'],'heyude')
        pool=PhotoPool(self.root,self.c,np.array([vector(0)]),['alpha'],['base'])
        self.addCleanup(pool.close);before=pool.anchors.copy()
        self.reg.refresh(pool)
        self.assertIn(pid,pool.pools);self.assertTrue(pool.pools[pid]['rows'][0]['locked'])
        np.testing.assert_array_equal(pool.anchors[:1],before)
        self.assertIn(pid,pool.gallery()[1]);self.assertEqual(len(pool.pools[pid]['active']),1)
        renewed=PhotoPool(self.root,self.c,np.array([vector(0)]),['alpha'],['base']);self.addCleanup(renewed.close)
        VisitorRegistration(self.root,self.c,'b'*32).refresh(renewed)
        self.assertIn(pid,renewed.gallery()[1]);self.assertEqual(len(list((self.root/'private/people'/pid/'photo_pool').glob('*.jpg'))),1)

    def test_title_alone_is_sufficient_and_not_expanded_to_invented_name(self):
        out=self.request(information=dict(preferred_address='杨教授',background=''),source=dict(turn_id='title',text='这位是杨教授'))
        self.reg.complete(10.1)
        profile=json.loads((self.root/'private/people'/out['person_id']/'profile.json').read_text())
        self.assertEqual(profile['display_name'],'杨教授')

    def test_explicit_full_name_and_title_support_short_address(self):
        out=self.request(source=dict(turn_id='fullname',text='这位是杨明扬教授'))
        self.assertTrue(out['accepted'])

    def test_no_name_or_title_requests_clarification_and_writes_nothing(self):
        out=self.request(information=dict(background='软件工程师'))
        self.assertFalse(out['accepted']);self.assertEqual(out['status'],'needs_address')
        self.assertFalse(self.store.statuses())

    def test_unconfirmed_introducer_and_invented_name_cannot_register(self):
        with self.assertRaises(ValueError):self.request(introducer=None)
        with self.assertRaises(ValueError):self.request(information=dict(display_name='王小明'))
        self.assertFalse(self.store.statuses())

    def test_retry_uses_one_id_and_does_not_rewrite_a_profile(self):
        a=self.request();b=self.request();self.assertEqual(a['person_id'],b['person_id'])
        self.reg.complete(10.1);self.reg.complete(10.2)
        self.assertEqual(len(self.store.statuses()),1)
        self.assertEqual(self.request()['status'],'ready')

    def test_changed_target_waits_instead_of_enrolling_current_stranger(self):
        out=self.request();self.reg.track(None,10.1,0);self.reg.track(self.row,10.2,1)
        self.reg.observe(self.reg.token,self.image,self.row,vector(2),{},10.2,None,np.dot)
        self.reg.complete(10.3)
        self.assertFalse((self.root/'private/people'/out['person_id']).exists())
        self.reg.complete(131.)
        self.assertEqual(self.store.statuses()[0]['status'],'failed')

    def test_same_box_new_face_revokes_old_token_without_registering_wrong_person(self):
        out=self.request()
        self.reg.observe(self.token,self.image,self.row,vector(2),{},10.2,None,np.dot)
        self.assertNotEqual(self.reg.token,self.token)
        self.reg.complete(10.3)
        self.assertFalse((self.root/'private/people'/out['person_id']).exists())

    def test_multi_face_has_no_current_guest_and_poor_images_are_not_seeds(self):
        self.reg.track(self.row,10.1,2);self.assertIsNone(self.reg.status(10.1))
        self.reg.track(self.row,10.2,1)
        self.reg.observe(self.reg.token,np.zeros_like(self.image),self.row,vector(2),{},10.2,None,np.dot)
        self.assertEqual(self.reg.status(10.2)['seed_count'],0)

    def test_camera_geometry_alone_does_not_republish_identity_state_at_sixty_hz(self):
        before=self.reg.status(10.)
        for n in range(1,9):
            self.reg.track(self.row,10.+n/60,1)
            self.assertEqual(self.reg.status(10.+n/60),before)

    def test_unknown_token_can_be_reacquired_after_glance_without_claiming_identity(self):
        self.reg.track(None,10.1,0);self.reg.track(self.row,10.2,1)
        new=self.reg.token;self.assertNotEqual(new,self.token)
        self.reg.observe(new,self.image,self.row,vector(1),{},10.2,None,np.dot)
        self.assertEqual(self.reg.token,self.token);self.assertIsNone(self.reg.status(10.2)['person_id'])

    def test_old_owner_pending_work_cannot_register_after_camera_restart(self):
        self.request();other=VisitorRegistration(self.root,self.c,'b'*32);other.complete(10.1)
        self.assertEqual(self.store.statuses()[0]['status'],'failed')

    def test_profiles_hotload_in_existing_dictionary_and_keep_sources_local(self):
        out=self.request();self.reg.complete(10.1)
        k=object.__new__(Knowledge);k.people={};prior=k.people;k.people_directory=self.root/'private/people'
        self.assertEqual(k.refresh_people(),[out['person_id']]);self.assertIs(k.people,prior)
        self.assertNotIn('introduction',k.people[out['person_id']]);self.assertEqual(k.refresh_people(),[])

    def test_background_read_does_not_resize_the_live_people_dictionary(self):
        out=self.request();self.reg.complete(10.1)
        k=object.__new__(Knowledge);k.people={};k.people_directory=self.root/'private/people'
        updates=k.new_profiles();self.assertEqual(k.people,{})
        self.assertIn(out['person_id'],updates)


class SideDialogueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from test_cloud_control import CloudBackendTests
        from vivis.speech.realtime import VoiceSession
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.v=CloudBackendTests().service(self.tmp.name,[]);self.v.health_available=True;self.v.enabled=True
        self.addCleanup(self.v.log.close);self.addCleanup(self.v.recorders.close)
        self.addCleanup(self.v.conversation_animation.close)
        self.v.shutdown.clear();self.v.context=dict(revision=8,mode='interaction',ready=True)
        self.v.cloud_state=VoiceSession(self.v.player,self.v.log)
        self.v.knowledge.root=Path(self.tmp.name);self.v.knowledge.persona='小帅20岁';self.v.knowledge.business='翌脑科技前台'
        self.request=dict(id='1'*32,person=None,session=self.v.memory_session,revision=8,
            connection=self.v.message_connection,guest_key='a'*32,guest=None)
        self.lane=SideDialogue(self.v,self.request)
        self.events=[];self.v.emit=self.events.append
        self.original_person=self.v.turn_person

    def audio(self):
        return (np.sin(np.arange(320)*.12)*4000).astype('<i2').tobytes()

    def business(self,action='create'):
        from vivis.paths import ROOT
        from vivis.interaction.memory import PersonMemory
        from vivis.interaction.robot_tasks import RobotTasks
        self.v.memory=PersonMemory(json.loads((ROOT/'configs/interaction/memory.json').read_text()),self.tmp.name)
        self.v.tasks=RobotTasks(self.v.memory,json.loads((ROOT/'configs/interaction/tasks.json').read_text()))
        self.v.knowledge.people={'heyude':dict(display_name='何工',preferred_address='何工'),
            'wangrunzhang':dict(display_name='小王',preferred_address='小王')}
        self.v.extract_message=AsyncMock(return_value=dict(action=action,recipient_id='wangrunzhang',content='下午三点开会'))
        self.lane.actor='heyude';self.request['person']='wangrunzhang'
        self.lane.transcript='给小王留言，下午三点开会。';self.lane.say=AsyncMock(return_value=True)

    async def test_side_message_uses_own_actor_shared_ledger_and_verified_receipt_only(self):
        self.business();out=await self.lane.tool('side_message',{})
        self.assertTrue(out['saved']);self.assertTrue(self.lane.terminal_tool)
        rows=self.v.tasks.list();self.assertEqual(len(rows),1);self.assertEqual(rows[0]['creator'],'heyude')
        self.assertEqual(rows[0]['recipient'],'wangrunzhang');self.assertEqual(self.request['person'],'wangrunzhang')
        self.assertEqual(self.v.turn_person,self.original_person)
        self.assertEqual(out['spoken_receipts'],['已保存，等见到小王后转告。'])

    async def test_failed_speech_does_not_fake_complete_receipt_or_undo_write(self):
        self.business();self.lane.say=AsyncMock(return_value=False)
        out=await self.lane.tool('side_message',{})
        self.assertTrue(out['saved']);self.assertEqual(out['spoken_receipts'],[])
        self.assertTrue(self.lane.terminal_tool);self.assertEqual(len(self.v.tasks.list()),1)
        self.assertTrue(any('未完整播出' in e.get('message','') for e in self.events))

    async def test_semantic_non_message_resumes_normal_answer_without_a_ledger_write(self):
        self.business(action='none');out=await self.lane.tool('side_message',{})
        self.assertFalse(self.lane.terminal_tool);self.assertEqual(out['operation_status'],'chat')
        self.assertEqual(self.v.tasks.list(),[])

    async def test_provider_pretool_success_audio_is_not_played_or_followed_by_another_claim(self):
        self.business();sent=[]
        evidence=dict(identity_evidence=dict(state='confirmed',person_id='heyude'))
        self.v.continuity=SimpleNamespace(speaker=SimpleNamespace(audio_epoch=0,
            recognize_external=AsyncMock(return_value=evidence)),external_turn=Mock())
        packets=[dict(type='conversation.item.input_audio_transcription.completed',transcript=self.lane.transcript),
            dict(type='response.created',response=dict(id='premature')),
            dict(type='response.audio.delta',response_id='premature',delta=__import__('base64').b64encode(self.audio()).decode()),
            dict(type='response.audio_transcript.done',response_id='premature',transcript='已完成。'),
            dict(type='response.function_call_arguments.done',response_id='premature',call_id='msg1',name='side_message',arguments='{}'),
            dict(type='response.done',response=dict(id='premature',status='completed'))]
        class WS:
            async def send(self,raw):
                value=json.loads(raw);sent.append(value)
                if value['type']=='session.update':self.session=value['session']
            async def recv(self):return json.dumps(dict(type='session.updated',session=self.session))
            def __aiter__(self):return self
            async def __anext__(self):
                if not packets:raise StopAsyncIteration
                return json.dumps(packets.pop(0))
        @asynccontextmanager
        async def connection(*args):yield WS()
        self.lane.feed(10.,self.audio(),bytes(640));self.lane.end()
        with patch('vivis.speech.side_dialogue.websocket_connection',connection):await self.lane.run()
        self.assertEqual(len(self.v.tasks.list()),1)
        self.assertNotIn('premature',self.v.player.started_responses)
        self.assertEqual(self.v.player.count,0)
        self.assertEqual(sum(p['type']=='response.create' for p in sent),1)
        self.assertFalse(any(e.get('text')=='已完成。' for e in self.events))

    async def test_public_question_sends_original_audio_and_answers_through_existing_dac(self):
        sent=[];audio=self.audio();packets=[
            dict(type='conversation.item.input_audio_transcription.completed',transcript='今天星期几？'),
            dict(type='response.created',response=dict(id='side_reply')),
            dict(type='response.audio.delta',response_id='side_reply',delta=__import__('base64').b64encode(audio).decode()),
            dict(type='response.audio_transcript.done',response_id='side_reply',transcript='今天星期四。'),
            dict(type='response.done',response=dict(id='side_reply',status='completed'))]
        class WS:
            async def send(self,raw):
                sent.append(json.loads(raw))
                if sent[-1]['type']=='session.update':self.session=sent[-1]['session']
            async def recv(self):return json.dumps(dict(type='session.updated',session=self.session))
            def __aiter__(self):return self
            async def __anext__(self):
                if not packets:raise StopAsyncIteration
                return json.dumps(packets.pop(0))
        @asynccontextmanager
        async def connection(*args):yield WS()
        self.lane.feed(10.,audio,bytes(len(audio)));self.lane.end()
        pcm=[]
        async def dac():
            while not task.done():
                pcm.append(self.v.player.render(1920,time.monotonic()));await asyncio.sleep(.002)
        with patch('vivis.speech.side_dialogue.websocket_connection',connection):
            task=asyncio.create_task(self.lane.run());await asyncio.gather(task,dac())
        self.assertTrue(any(np.any(x) for x in pcm));self.assertEqual(self.lane.transcript,'今天星期几？')
        self.assertTrue(any(e.get('text')=='今天星期四。' for e in self.events))
        session=sent[0]['session'];self.assertFalse(session['enable_search']);self.assertIsNone(session['turn_detection'])
        self.assertNotIn('私人',self.v.knowledge.persona);self.assertNotIn('历史聊天',session['instructions'])
        chunks=[s['audio'] for s in sent if s['type']=='input_audio_buffer.append']
        self.assertEqual(__import__('base64').b64decode(''.join(chunks)),audio)
        self.assertEqual(self.v.memory_session,self.request['session']);self.assertIsNone(self.lane.actor)

    async def test_side_business_cannot_borrow_main_person_and_public_status_is_allowed(self):
        self.request['person']='heyude'
        with self.assertRaises(ValueError):await self.lane.tool('side_message',{})
        output=await self.lane.tool('get_robot_status',{})
        self.assertIsNone(output['side_speaker']);self.assertEqual(output['main_person_retained'],'heyude')

    async def test_side_introduction_requires_own_voice_and_locked_unknown_guest(self):
        self.lane.transcript='这是杨教授'
        with self.assertRaises(ValueError):await self.lane.tool('introduce_visitor',dict(preferred_address='杨教授'))
        self.lane.actor='heyude'
        with self.assertRaises(ValueError):await self.lane.tool('introduce_visitor',dict(preferred_address='杨教授'))
        self.request['guest']=dict(token='a'*32,owner='b'*32,stamp=time.monotonic())
        out=await self.lane.tool('introduce_visitor',dict(preferred_address='杨教授'))
        self.assertTrue(out['accepted']);self.assertIn('introduced_guest',vars(self.v))
        self.assertIsNone(self.request['person']);self.assertEqual(self.v.turn_person,self.original_person)

    async def test_missing_address_can_be_completed_by_same_confirmed_introducer(self):
        self.lane.actor='heyude';self.request['guest']=dict(token='a'*32,owner='b'*32,stamp=time.monotonic())
        self.lane.transcript='这是我们的软件工程师'
        out=await self.lane.tool('introduce_visitor',dict(background='软件工程师'))
        self.assertEqual(out['status'],'needs_address')
        self.lane.transcript='这位叫杨教授'
        out=await self.lane.tool('introduce_visitor',dict(preferred_address='杨教授',background='软件工程师'))
        self.assertTrue(out['accepted']);document=Introductions(Path(self.tmp.name)).statuses()[0]
        self.assertIn('软件工程师',document['source']['text']);self.assertIsNone(self.v.introduction_draft)

    async def test_side_voice_is_external_only_never_main_fusion_or_enrollment(self):
        self.lane.feed(10.,self.audio(),bytes(640))
        result=dict(identity_evidence=dict(state='confirmed',person_id='heyude'))
        observer=SimpleNamespace(audio_epoch=4,recognize_external=AsyncMock(return_value=result))
        self.v.continuity=SimpleNamespace(speaker=observer)
        await self.lane.identify(self.audio(),bytes(640))
        self.assertEqual(self.lane.actor,'heyude')
        source=observer.recognize_external.await_args.args[2]
        self.assertTrue(source['voice_context'].startswith('side:'));self.assertEqual(self.v.turn_person,self.original_person)

    async def test_stale_connection_or_new_main_turn_revokes_old_side_actions(self):
        self.v.message_connection=object()
        with self.assertRaises(ValueError):await self.lane.tool('get_robot_status',{})
        self.assertFalse(self.lane.valid())

    async def test_new_main_input_generation_revokes_late_side_tools(self):
        self.request['generation']=self.v.input_generation
        self.v.input_generation+=1
        with self.assertRaises(ValueError):await self.lane.tool('set_expression',dict(expression='joy',source='command'))

    async def test_released_output_lease_prevents_old_side_actions(self):
        self.v.enabled=False
        with self.assertRaises(ValueError):await self.lane.tool('get_robot_status',{})

    async def test_malformed_tool_arguments_return_clarification_without_side_effects(self):
        with self.assertRaises(ValueError):await self.lane.tool('introduce_visitor',None)
        with self.assertRaises(ValueError):await self.lane.tool('side_message',[])
        self.assertFalse(Introductions(Path(self.tmp.name)).statuses())

    async def test_side_social_expression_does_not_publish_an_unbound_late_pose(self):
        with patch.object(self.v.expression,'publish',side_effect=AssertionError('unbound social pose')):
            result=await self.lane.tool('set_expression',dict(expression='joy',source='social'))
        self.assertEqual(result['status'],'keep_existing_social_animation')

    async def test_capture_gap_and_ninety_second_cap_do_not_reach_qwen(self):
        self.lane.feed(10.,self.audio(),bytes(640));self.lane.feed(10.1,self.audio(),bytes(640))
        self.assertFalse(self.lane.valid());self.assertTrue(self.lane.capture_done.is_set())
        with patch('vivis.speech.side_dialogue.websocket_connection',side_effect=AssertionError('network')):
            await self.lane.run()

    async def test_title_request_does_not_relabel_the_introducer_as_the_guest(self):
        self.lane.actor='heyude';self.request['guest']=dict(token='a'*32,owner='b'*32,stamp=time.monotonic())
        self.lane.transcript='这是杨教授，是大学老师'
        out=await self.lane.tool('introduce_visitor',dict(preferred_address='杨教授',background='大学老师'))
        self.assertNotEqual(out['person_id'],'heyude')
        document=Introductions(Path(self.tmp.name)).statuses()[0]
        self.assertEqual(document['introducer'],'heyude');self.assertEqual(document['guest']['token'],'a'*32)


class ExternalSpeakerWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_model_worker_does_not_publish_to_main_or_enroll_side_audio(self):
        from vivis.paths import ROOT
        from vivis.speech.speaker_observation import SpeakerObserver
        with tempfile.TemporaryDirectory() as folder:
            c=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text())
            observer=SpeakerObserver(c,Path(folder),Mock(),Mock(),extractor=lambda x:[1.,0.])
            self.addCleanup(observer.job.close)
            pcm=np.rint(np.sin(np.arange(48000)*.1)*3000).astype('<i2').tobytes();far=bytes(len(pcm))
            for i in range(3):
                observer.infer(pcm,far,dict(turn_id=str(i),session='v',logical_person='heyude',visual_person='heyude',enrollment_allowed=True))
            while not observer.results.empty():observer.results.get()
            before=observer.gallery_path.read_bytes();main=observer.recognition_window
            source=dict(turn_id='side_1',session='v',voice_context='side:guest:1',audio_started=10.,audio_ended=13.,audio_epoch=observer.audio_epoch)
            result=await observer.recognize_external(pcm,far,source,lambda:True)
            self.assertEqual(result['identity_evidence']['state'],'confirmed')
            self.assertEqual(result['speaker_candidate'],'heyude');self.assertTrue(observer.results.empty())
            self.assertEqual(observer.gallery_path.read_bytes(),before);self.assertIs(observer.recognition_window,main)
            self.assertIsNot(observer.external_windows['side:guest:1'],main)

    async def test_manual_mode_cannot_silently_fall_back_to_server_vad(self):
        from vivis.speech.realtime import connect_session
        for detection in ({'type':'server_vad'},{}):
            ws=SimpleNamespace(send=AsyncMock(),recv=AsyncMock(return_value=json.dumps(dict(
                type='session.updated',session=dict(voice='voice',turn_detection=detection)))))
            with self.assertRaises(RuntimeError):await connect_session(ws,dict(voice='voice',turn_detection=None),Mock())

    async def test_live_provider_omits_disabled_turn_detection_in_ack(self):
        from vivis.speech.realtime import connect_session
        ws=SimpleNamespace(send=AsyncMock(),recv=AsyncMock(return_value=json.dumps(dict(
            type='session.updated',session=dict(voice='voice')))))
        result=await connect_session(ws,dict(voice='voice',turn_detection=None),Mock())
        self.assertEqual(result['voice'],'voice')


class ResidentFaceBootstrapTests(unittest.TestCase):
    def test_registered_guest_becomes_real_two_hit_identity_in_same_face_worker(self):
        import cv2
        from test_seeta_identity import FakeSeeta,config,vector as seeta_vector
        from vivis.vision.familiar import FamiliarFaces
        from vivis.vision.identity_config import backend_config
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);c=backend_config(config());model=FakeSeeta();model.index=2
            image=np.random.default_rng(6).integers(40,220,(120,120,3),dtype=np.uint8)
            base=root/'private/people/alpha';base.mkdir(parents=True);(base/'profile.json').write_text('{}')
            cv2.imwrite(str(base/'photo.png'),image);digest=hashlib.sha256((base/'photo.png').read_bytes()).hexdigest()
            pool=PhotoPool(root,c,np.array([seeta_vector(0)]),['alpha'],[digest])
            f=FamiliarFaces(c,root,model=model,gallery=(np.array([seeta_vector(0)]),np.array(['alpha'])))
            f.photo_pool=pool;row=np.array([0,0,120,120],np.float32);now=time.monotonic()
            try:
                f.visitors.track(row,now,1);token=f.visitors.token
                f.infer(image,row,now,f.epoch,False,token)
                self.assertIsNone(f.result['person_id'])
                out=Introductions(root).request(dict(preferred_address='杨教授'),guest=f.visitors.status(now),
                    introducer='heyude',source=dict(turn_id='intro1',text='这是杨教授'),session='v',now=now)
                f.infer(image,row,time.monotonic(),f.epoch,False,token)
                self.assertEqual(Introductions(root).statuses()[0]['status'],'ready')
                f.infer(image,row,time.monotonic(),f.epoch,False,token)
                self.assertEqual(f.result['identity_evidence']['state'],'suspected')
                f.infer(image,row,time.monotonic(),f.epoch,False,token)
                self.assertEqual(f.result['identity_evidence']['state'],'confirmed')
                self.assertEqual(f.result['person_id'],out['person_id']);self.assertIs(f.model,model)
                self.assertEqual(len(pool.pools[out['person_id']]['active']),1)
            finally:f.close()


class SideJournalTests(unittest.TestCase):
    def test_side_round_has_own_audio_actor_and_never_enters_sampling(self):
        from vivis.paths import ROOT
        from vivis.speech.round_recording import RoundRecorder
        import wave
        with tempfile.TemporaryDirectory() as folder:
            c=json.loads((ROOT/'configs/interaction/continuity.json').read_text())
            recorder=RoundRecorder(c,Path(folder));recorder.offer_speaker=Mock()
            try:
                pcm=np.arange(16000,dtype='<i2').tobytes()
                recorder.external_turn('side_1','visit','heyude',pcm,'这是杨教授')
                recorder.queue.join()
                with recorder.store.db() as db:row=dict(db.execute('SELECT * FROM rounds').fetchone())
                self.assertEqual(row['person'],'heyude');self.assertEqual(row['user_text'],'这是杨教授')
                self.assertEqual(row['alignment'],'side_source_contiguous')
                with wave.open(str(recorder.store.directory/row['audio']),'rb') as audio:self.assertEqual(audio.readframes(16000),pcm)
                recorder.offer_speaker.assert_not_called()
            finally:recorder.close()
