"""Cloud-only control permission, visit greeting and per-person memory invariants."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from vivis.paths import ROOT
from vivis.interaction.cloud_control import controller_intent,SpeechPresence,PresencePublisher,ReceptionGate,tools
from vivis.interaction.memory import PersonMemory
from vivis.interaction.knowledge import professional_profile
from vivis.interaction.voice_bridge import VoiceBridge
from vivis.speech.cloud_managed import CloudManagedVoice
from vivis.speech.managed import voice_configuration


def config(path):return json.loads((ROOT/path).read_text())


class CloudPolicyTests(unittest.TestCase):
    def test_tools_are_limited_to_programme_and_readonly_context(self):
        declared=tools(['song','script'])
        self.assertEqual({t['name'] for t in declared},{'robot_program','get_robot_status','search_person_memory','set_expression','set_speech_dialect','search_robot_memory','manage_robot_task'})
        self.assertEqual(controller_intent(dict(action='wake',program='song'),{'song'},'standby'),dict(kind='wake',followup=dict(kind='select',program='song')))
        self.assertEqual(controller_intent(dict(action='stop',program=''),{'song'},'task'),dict(kind='interaction'))
        for args in (dict(action='play',program='missing'),dict(action='play',program=''),dict(action='shell',program=''),dict(action='play',program='song',position=0)):
            with self.assertRaises(ValueError):controller_intent(args,{'song'},'task')
        with self.assertRaises(ValueError):controller_intent(dict(action='play',program='song'),{'song'},'standby')

    def test_presence_age_wake_exception_and_return_to_default_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=config('configs/perception/speech_presence.json');p=SpeechPresence(c,tmp);publisher=PresencePublisher(c,tmp)
            self.assertFalse(p.allowed(10));p.wake(10);self.assertTrue(p.allowed(11))
            publisher.publish(1,12,12);self.assertTrue(p.allowed(12));self.assertEqual(p.wake_until,0)
            publisher.publish(0,13,13);self.assertTrue(p.allowed(13))
            expired=12+c['face_loss_grace_s']+.1
            publisher.publish(0,expired,expired);self.assertFalse(p.allowed(expired))
            publisher.publish(1,expired+1,expired+1);self.assertFalse(p.allowed(expired+3))
            p.wake(expired+5);p.continued_turn(expired+10)
            self.assertTrue(p.allowed(expired+39));self.assertFalse(p.allowed(expired+41))
            publisher.close();self.assertFalse(p.visible(expired+43))

    def test_greeting_once_per_visit_and_not_after_brief_detection_loss(self):
        c=config('configs/interaction/reception.json');gate=ReceptionGate(c)
        self.assertFalse(gate.observe(True,'standby',10));self.assertTrue(gate.observe(True,'standby',12))
        self.assertFalse(gate.observe(True,'interaction',20))
        gate.observe(False,'interaction',30);gate.observe(True,'interaction',31)
        self.assertFalse(gate.observe(True,'interaction',40))
        gate.observe(False,'interaction',50);gate.observe(False,'interaction',54)
        gate.observe(True,'interaction',140);self.assertTrue(gate.observe(True,'interaction',143))
        self.assertFalse(gate.observe(True,'interaction',143))

    def test_new_registered_visitor_is_not_blocked_by_another_visitors_cooldown(self):
        gate=ReceptionGate(config('configs/interaction/reception.json'))
        gate.observe(True,'standby',0,person_id='alpha')
        self.assertTrue(gate.observe(True,'standby',2,person_id='alpha'))
        gate.observe(False,'interaction',2);gate.observe(False,'interaction',6)
        gate.observe(True,'interaction',7,person_id='beta')
        self.assertTrue(gate.observe(True,'interaction',9,person_id='beta'))

    def test_live_idle_gaze_has_no_artificial_end_of_song_return(self):
        from vivis.runtime.robot import Robot
        from vivis.interaction.standby import plan,StandbyGaze
        c=config('configs/interaction/interaction.json')
        profile,ids,_,_,source,neutral=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),c,1.)
        receiver=Mock();receiver.poll.return_value=None
        gaze=StandbyGaze(source.tracking,profile,ids,receiver,source);gaze.publish=Mock()
        gaze.prepare(neutral,neutral[0],[],time.monotonic())
        gaze.set_audio_time(100000.,None)
        import numpy as np
        result=gaze.apply(neutral[0],neutral[0],1/60,time.monotonic())
        self.assertTrue(np.isfinite(result).all());self.assertFalse(gaze.returning);self.assertFalse(gaze.platform_returning)
        self.assertIsNone(gaze.publish.call_args.args[0]['remaining_seconds'])

    def test_age_is_explicit_validated_profile_fact(self):
        p=dict(schema_version='vivis-person-1.0',allow_cloud_context=True,display_name='测试工',age_years=44)
        self.assertEqual(professional_profile(p)['age_years'],44)
        for age in (True,-1,121,'44'):
            with self.assertRaises(ValueError):professional_profile(dict(p,age_years=age))

    def test_reception_standby_opens_only_vision_not_neck_gaze(self):
        from vivis.runtime.robot import Robot
        from vivis.interaction.standby import plan
        c=config('configs/interaction/standby.json');c['presence_only']=True
        _,_,_,_,source,_=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),c,1.)
        self.assertTrue(source.presence_only);self.assertFalse(source.tracking_enabled)
        self.assertFalse(source.response['enabled']);self.assertIsNotNone(source.tracking)


class PersonMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        c=config('configs/interaction/memory.json');c['working_token_budget']=100
        self.m=PersonMemory(c,self.tmp.name)

    def test_persist_reload_and_no_cross_person_recall(self):
        self.m.append('alpha','one','user','下次提醒我调试机器人，计划在周三下午。')
        self.m.append('beta','one','user','我想学习做饭。')
        reloaded=PersonMemory(self.m.c,self.tmp.name)
        self.assertIn('机器人',reloaded.context('alpha'));self.assertNotIn('做饭',reloaded.context('alpha'))
        fallback=reloaded.search('beta',['机器人'])
        self.assertEqual(fallback['match_type'],'keyword_miss_recent_history_fallback')
        self.assertTrue(all('机器人' not in r['text'] for r in fallback['matches']))
        self.assertIn('做饭',fallback['matches'][0]['text'])
        self.assertEqual(reloaded.context('anonymous:'+'a'*32),'')
        self.assertEqual(self.m.path.stat().st_mode&0o777,0o600)

    def test_compaction_threshold_and_raw_facts_stay_searchable(self):
        mid=self.m.append('alpha','one','user','计划明天上午十点测试机器头。'*8)
        self.assertEqual(self.m.pending(),'alpha');batch=self.m.batch('alpha')
        result=dict(summary='计划真机测试。',facts=[dict(key='测试时间',value='明天上午十点',source_ids=[mid])])
        self.m.compact(batch,result)
        self.assertIsNone(self.m.pending());self.assertIn('明天上午十点',self.m.context('alpha'))
        self.assertEqual(len(self.m.search('alpha',['机器头'])['matches']),1)
        self.assertEqual(self.m.search('alpha',['测试时间'])['facts'][0]['value'],'明天上午十点')
        self.assertIn('明天上午十点',PersonMemory(self.m.c,self.tmp.name).context('alpha'))

    def test_recall_previous_visit_does_not_mix_current_visit(self):
        self.m.append('alpha','previous','user','上次约好调试机器人。')
        self.m.append('alpha','current','user','这次先喝茶。')
        matches=self.m.search('alpha',[],'current')['matches']
        self.assertEqual(len(matches),1);self.assertIn('调试机器人',matches[0]['text'])

    def test_compaction_batch_counts_metadata_and_not_just_short_text(self):
        self.m.c['compression_batch_max_chars']=2000
        for _ in range(50):self.m.append('alpha','previous','user','短句记录。')
        batch=self.m.batch('alpha')
        self.assertLess(len(batch['messages']),50)
        self.assertLessEqual(len(json.dumps(batch['messages'],ensure_ascii=False)),2000)

    def test_do_not_promote_assistant_guesses_to_facts_or_mix_person_ids(self):
        first=self.m.append('alpha','one','assistant','您是CEO。')
        self.m.append('alpha','one','user','不是，我是工程师。')
        batch=self.m.batch('alpha')
        with self.assertRaises(ValueError):self.m.compact(batch,dict(summary='',facts=[dict(key='职业',value='CEO',source_ids=[first])]))
        other=self.m.append('beta','one','user','我是导游。')
        with self.assertRaises(ValueError):self.m.compact(batch,dict(summary='',facts=[dict(key='职业',value='导游',source_ids=[other])]))
        self.assertIn('工程师',self.m.context('alpha'))


class CloudBackendTests(unittest.IsolatedAsyncioTestCase):
    def service(self,tmp,events):
        speech=config('configs/perception/speech.json');speech['managed_voice']=config('configs/interaction/p1.json')['realtime_voice']
        speech['managed_voice']['lip_sync']['enabled']=False;speech['programs']=str(ROOT/'configs/interaction/programs.json')
        with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'unit-test-placeholder'}):
            with patch('vivis.interaction.memory.PersonMemory'):
                service=CloudManagedVoice(speech,ROOT/'configs/robots/xiaoshuai/robot.json',Path(tmp),events.append)
        service.memory=None
        return service

    async def test_cloud_connection_survives_output_handoff_and_no_decoder_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=self.service(tmp,[]);fake=Mock()
            service.audio=fake;service.cloud_task=Mock()
            service.request(dict(kind='voice_mode',epoch=1,enabled=True));service.release_at=0
            await service.apply_mode();fake.enable_output.assert_called_once()
            service.request(dict(kind='voice_mode',epoch=2,enabled=False));service.release_at=0
            await service.apply_mode();self.assertFalse(service.enabled)
            service.cloud_task.cancel.assert_not_called();self.assertIsNone(service.decoder)
            service.log.close()

    async def test_tool_validated_deduplicated_and_old_state_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            events=[];service=self.service(tmp,events);sent=[]
            async def send(text):sent.append(json.loads(text))
            ws=SimpleNamespace(send=send)
            service.context.update(revision=4,mode='standby',target='standby',ready=True)
            service.response_revisions['r']=4
            event=dict(call_id='c',name='robot_program',arguments=json.dumps(dict(action='wake',program='',explicit_wake=True)),response_id='r')
            await service.tool(ws,event);await service.tool(ws,event)
            self.assertEqual(len(events),1);self.assertEqual(events[0]['kind'],'cloud_action')
            self.assertEqual(len(sent),1)
            service.context['revision']=5
            await service.tool(ws,dict(event,call_id='late'))
            self.assertEqual(len(events),1);self.assertFalse(json.loads(sent[-1]['item']['output'])['accepted'])
            service.log.close()

    async def test_background_memory_compaction_is_cancelled_on_interaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=self.service(tmp,[])
            service.memory_task=asyncio.create_task(asyncio.sleep(30))
            service.request(dict(kind='voice_context',revision=1,mode='interaction',target='interaction',ready=False))
            await asyncio.gather(service.memory_task,return_exceptions=True)
            self.assertTrue(service.memory_task.cancelled());service.log.close()

    async def test_no_face_blocks_voice_unless_explicit_wake(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=self.service(tmp,[]);service.enabled=True
            service.context.update(mode='interaction',target='interaction',ready=True)
            with patch.object(service.presence,'visible',return_value=False):
                self.assertFalse(service.can_speak());service.presence.wake();self.assertTrue(service.can_speak())
                service.context['mode']='task';self.assertFalse(service.can_speak())
            service.log.close()

class ContinuousCloudAudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_local_decoder_quiet_without_face_and_same_connection_during_programme(self):
        import base64
        from test_managed_voice import FakeAudio
        from unittest.mock import AsyncMock
        import numpy as np
        events=[];messages=asyncio.Queue();sent=[]
        class WS:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            async def send(self,text):sent.append(json.loads(text))
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await messages.get())
        async def connected(ws,session,log):return session
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{'DASHSCOPE_API_KEY':'unit-test-placeholder'}), \
             patch('vivis.speech.managed.AudioIO',FakeAudio), \
             patch('vivis.speech.managed.Decoder',side_effect=AssertionError('Local ASR must not start')) as decoder, \
             patch('vivis.interaction.memory.PersonMemory'), \
             patch('vivis.speech.cloud_managed.websocket_connection',return_value=WS()), \
             patch('vivis.speech.cloud_managed.connect_session',side_effect=connected):
            speech=config('configs/perception/speech.json');speech['managed_voice']=config('configs/interaction/p1.json')['realtime_voice']
            speech['managed_voice']['lip_sync']['enabled']=False;speech['programs']=str(ROOT/'configs/interaction/programs.json')
            service=CloudManagedVoice(speech,ROOT/'configs/robots/xiaoshuai/robot.json',Path(tmp),events.append)
            service.memory=None;service.presence=SpeechPresence(config('configs/perception/speech_presence.json'),tmp)
            service.reception.c=dict(service.reception.c,enabled=False)
            task=asyncio.create_task(service.run())
            async def capture():
                while not service.shutdown.is_set():
                    stamp=time.monotonic();service.audio.last_capture=stamp
                    if service.audio.output_stream is not None:service.audio.last_output=stamp
                    service.audio.uplink.put((stamp,b'\0'*640,b'\0'*640,b'\0'*640));await asyncio.sleep(.02)
            producer=asyncio.create_task(capture())
            async def until(predicate):
                deadline=time.monotonic()+2
                while not predicate():
                    if task.done():task.result()
                    if time.monotonic()>deadline:raise AssertionError('Cloud audio test timed out')
                    await asyncio.sleep(.01)
            try:
                await until(lambda:service.ws is not None)
                service.controls.put(dict(kind='voice_context',revision=1,mode='interaction',target='interaction',ready=True))
                service.controls.put(dict(kind='voice_mode',epoch=1,enabled=True))
                await until(lambda:service.audio.output_stream is not None)
                await messages.put(dict(type='response.created',response=dict(id='muted')))
                pcm=base64.b64encode((np.ones(240,dtype='<i2')*100).tobytes()).decode()
                await messages.put(dict(type='response.audio.delta',response_id='muted',delta=pcm))
                await asyncio.sleep(.05);self.assertEqual(service.player.count,0)
                await messages.put(dict(type='response.function_call_arguments.done',response_id='muted',call_id='wake',name='robot_program',arguments=json.dumps(dict(action='wake',program='',explicit_wake=True))))
                await until(lambda:service.presence.wake_until>time.monotonic())
                await messages.put(dict(type='response.created',response=dict(id='audible')))
                await messages.put(dict(type='response.audio.delta',response_id='audible',delta=pcm))
                await until(lambda:service.player.count>0)
                service.controls.put(dict(kind='voice_context',revision=2,mode='task',target='baigufuren',ready=True))
                service.controls.put(dict(kind='voice_mode',epoch=2,enabled=False))
                await until(lambda:service.audio.output_stream is None)
                self.assertIsNotNone(service.ws);self.assertEqual(service.summary['cloud_connections'],1)
                await messages.put(dict(type='response.created',response=dict(id='stop-response')))
                await messages.put(dict(type='response.function_call_arguments.done',response_id='stop-response',call_id='stop',name='robot_program',arguments=json.dumps(dict(action='stop',program='',explicit_wake=False))))
                await until(lambda:any(e.get('arguments',{}).get('action')=='stop' for e in events))
                self.assertEqual(service.audio.opens,1);decoder.assert_not_called();self.assertEqual(service.local.qsize(),0)
                person=['alpha']
                with patch.object(service.presence,'visible',return_value=True),patch.object(service,'identified_person',side_effect=lambda:person[0]):
                    service.controls.put(dict(kind='voice_context',revision=3,mode='interaction',target='interaction',ready=True))
                    service.controls.put(dict(kind='voice_mode',epoch=3,enabled=True))
                    await until(lambda:service.conversation_person=='alpha')
                    person[0]='beta'
                    await until(lambda:service.summary['cloud_connections']==2)
                    self.assertEqual(service.audio.opens,1)

            finally:
                service.shutdown.set();await asyncio.wait_for(task,timeout=3)
                producer.cancel();await asyncio.gather(producer,return_exceptions=True)
