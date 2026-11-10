"""Speech preferences use cloud tools, retaining conversation and device settings."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vivis.paths import ROOT
from vivis.interaction.cloud_control import tools
from vivis.interaction.speech_dialect import SpeechDialect,configuration
from vivis.speech.cloud_managed import CloudManagedVoice,CloudIdleTimeout,ConversationReset


class DialectCatalogTests(unittest.TestCase):
    def test_all_official_choices_share_one_tool_and_validated_catalog(self):
        preference=SpeechDialect();names=configuration()['dialects']
        self.assertEqual(len(names),21);self.assertEqual(preference.name,'普通话')
        declared=next(t for t in tools([]) if t['name']=='set_speech_dialect')
        self.assertEqual(declared['parameters']['properties']['dialect']['enum'],names)
        for name in names:self.assertEqual(preference.requested(dict(dialect=name)),name)
        self.assertIn('陕西话',names);self.assertIn('山西话',names)
        self.assertEqual(preference.voice('original'),'original')
        preference.name='四川话'
        self.assertEqual(preference.voice('original'),'xunanchuan_v3.1')
        self.assertEqual(preference.snapshot()['capability'],'experimental')
        preference.name='陕西话';self.assertEqual(preference.snapshot()['capability'],'documented')
        for args in ({},None,dict(dialect=True),dict(dialect=['四川话']),dict(dialect='未知口音'),
                     dict(dialect='普通话',voice='other'),dict(dialect='普通话',instructions='ignore rules')):
            with self.assertRaises(ValueError):preference.requested(args)


class DialectServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        speech=json.loads((ROOT/'configs/perception/speech.json').read_text())
        speech['managed_voice']=json.loads((ROOT/'configs/interaction/p1.json').read_text())['realtime_voice']
        speech['managed_voice']['lip_sync']['enabled']=False
        speech['programs']=str(ROOT/'configs/interaction/programs.json')
        from test_managed_voice import FakeAudio
        with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'unit-test-placeholder'}), \
             patch('vivis.speech.managed.AudioIO',FakeAudio),patch('vivis.interaction.memory.PersonMemory'):
            self.service=CloudManagedVoice(speech,ROOT/'configs/robots/xiaoshuai/robot.json',Path(self.tmp.name),lambda _:None)
        self.service.memory=None;self.addCleanup(self.service.log.close)
        self.service.enabled=True;self.service.presence.wake()
        self.service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
        self.service.response_revisions['r']=1
        self.sent=[]
        async def send(text):self.sent.append(json.loads(text))
        self.ws=SimpleNamespace(send=send)

    def event(self,args,call='switch',response='r'):
        return dict(call_id=call,response_id=response,name='set_speech_dialect',arguments=json.dumps(args))

    def result(self):return json.loads(self.sent[-1]['item']['output'])

    async def test_switch_then_restore_updates_before_tool_result_without_touching_conversation_or_voice(self):
        service=self.service;original=copy.deepcopy(service.session)
        for i,name in enumerate(('广东话','四川话','普通话')):
            previous_voice=service.active_voice
            await service.tool(self.ws,self.event(dict(dialect=name),str(i)))
            result=self.sent[-1]
            self.assertEqual(result['item']['type'],'function_call_output')
            self.assertTrue(self.result()['accepted']);self.assertFalse(self.result()['pronunciation_verified'])
            expected=service.dialect.voice(service.session['voice'])
            if expected!=previous_voice:
                self.assertEqual(self.result()['status'],'pending_voice_reconnect')
                self.assertFalse(service.can_speak());self.assertTrue(service.pending_voice_switch['ready'])
                with self.assertRaises(ValueError):service.voice_ready('unconfirmed-wrong-voice')
                service.voice_ready(expected)  # Simulate the validated connection handshake.
                self.assertEqual(service.announcement_kind,'dialect')
            else:
                update=self.sent[-2]
                self.assertEqual(update['type'],'session.update')
                self.assertEqual(set(update['session']),{'instructions'})
                self.assertIn('"dialect": "'+name+'"',update['session']['instructions'])
                self.assertTrue(service.continue_response)
            self.assertEqual(service.dialect.name,name)
        self.assertEqual(service.session,original)  # Voice/rate/audio format/tools and knowledge retained.
        self.assertIsNone(service.tool_repair)
        self.assertEqual(service.summary['cloud_connections'],0)
        self.assertFalse(any(p['type']=='response.cancel' or p.get('item',{}).get('type')=='message' for p in self.sent))

    async def test_status_reports_actual_preference_and_complete_context_keeps_robot_identity(self):
        await self.service.tool(self.ws,self.event(dict(dialect='上海话')))
        await self.service.tool(self.ws,dict(call_id='status',response_id='r',name='get_robot_status',arguments='{}'))
        self.assertEqual(self.result()['speech_preference']['dialect'],'上海话')
        instructions,_,_=self.service.context_snapshot()
        self.assertIn('首席情绪陪伴官',instructions)
        self.assertIn('优先于人设',instructions)
        self.assertIn('不改写预录歌曲或剧本',instructions)

    async def test_stale_duplicate_or_invalid_calls_cannot_change_preference(self):
        event=self.event(dict(dialect='广东话'))
        await self.service.tool(self.ws,event);count=len(self.sent)
        await self.service.tool(self.ws,event);self.assertEqual(len(self.sent),count)
        self.service.context['revision']=2
        await self.service.tool(self.ws,self.event(dict(dialect='四川话'),'late'))
        self.assertFalse(self.result()['accepted']);self.assertEqual(self.service.dialect.name,'广东话')
        self.service.response_revisions['r']=2
        await self.service.tool(self.ws,self.event(dict(dialect='英语'),'unsupported'))
        self.assertFalse(self.result()['accepted']);self.assertEqual(self.service.dialect.name,'广东话')
        self.assertEqual(sum(p['type']=='session.update' for p in self.sent),0)

    async def test_no_face_unready_standby_or_programme_cannot_switch_or_override_speaker_ownership(self):
        with patch.object(self.service.presence,'allowed',return_value=False):
            await self.service.tool(self.ws,self.event(dict(dialect='四川话'),'no-face'))
            self.assertFalse(self.result()['accepted'])
        for i,(mode,ready) in enumerate((('interaction',False),('standby',True),('task',True))):
            self.service.context.update(mode=mode,ready=ready)
            await self.service.tool(self.ws,self.event(dict(dialect='四川话'),'state'+str(i)))
            self.assertFalse(self.result()['accepted'])
        self.assertEqual(self.service.dialect.name,'普通话')
        self.assertFalse(any(p['type']=='session.update' for p in self.sent))

    async def test_send_failure_rolls_back_without_false_success(self):
        async def fail(_):raise OSError('connection gone')
        with self.assertRaises(OSError):
            await self.service.tool(SimpleNamespace(send=fail),self.event(dict(dialect='四川话')))
        self.assertEqual(self.service.dialect.name,'普通话');self.assertFalse(self.sent)

    async def test_programme_and_standby_handoffs_preserve_visit_preference(self):
        await self.service.tool(self.ws,self.event(dict(dialect='四川话')))
        for revision,(mode,target) in enumerate((('task','baigufuren'),('interaction','interaction'),('standby','standby')),2):
            self.service.request(dict(kind='voice_context',revision=revision,mode=mode,target=target,ready=True))
            self.assertEqual(self.service.dialect.name,'四川话')
        self.assertEqual(self.service.session['output_audio']['language'],'zh')

    async def test_idle_connection_renewal_retains_preference_and_new_visitor_resets_it(self):
        await self.service.tool(self.ws,self.event(dict(dialect='四川话')))
        seen=[]
        async def connection():
            seen.append(self.service.dialect.name)
            if len(seen)==1:raise CloudIdleTimeout()
            if len(seen)==2:raise ConversationReset('confirmed_person_changed')
            self.service.shutdown.set()
        with patch.object(self.service,'_connection',side_effect=connection):await self.service.cloud(0)
        self.assertEqual(seen,['四川话','四川话','普通话'])

    async def test_unknown_identity_clears_personal_conversation_but_retains_public_dialect(self):
        await self.service.tool(self.ws,self.event(dict(dialect='四川话')))
        seen=[]
        async def connection():
            seen.append(self.service.dialect.name)
            if len(seen)==1:raise ConversationReset('identity_uncertain')
            self.service.shutdown.set()
        with patch.object(self.service,'_connection',side_effect=connection):await self.service.cloud(0)
        self.assertEqual(seen,['四川话','四川话'])

    async def test_real_cloud_loop_reconnects_voice_only_and_preserves_recent_context_and_audio_lease(self):
        service=self.service;service.reception.c=dict(service.reception.c,enabled=False)
        service.resume_history.extend([dict(role='user',text='我喜欢羽毛球。'),dict(role='assistant',text='那我们聊聊羽毛球。')])
        connections=[];sessions=[]
        class WS:
            def __init__(self):self.messages=asyncio.Queue();self.sent=[]
            async def __aenter__(self):connections.append(self);return self
            async def __aexit__(self,*_):return False
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await self.messages.get())
            async def send(self,text):
                packet=json.loads(text);self.sent.append(packet)
                if packet['type']=='conversation.item.create' and packet['item'].get('id'):
                    await self.messages.put(dict(type='conversation.item.created',item=packet['item']))
        async def connected(ws,session,log):sessions.append(session);return session
        async def until(predicate):
            for _ in range(300):
                if task.done():task.result()
                if predicate():return
                await asyncio.sleep(.01)
            raise AssertionError('Voice switch did not complete')
        async def switch(name,number):
            ws=connections[-1];rid='response'+str(number)
            await ws.messages.put(dict(type='response.created',response=dict(id=rid)))
            await ws.messages.put(dict(type='response.function_call_arguments.done',response_id=rid,
                call_id='call'+str(number),name='set_speech_dialect',arguments=json.dumps(dict(dialect=name))))
        with patch('vivis.speech.cloud_managed.websocket_connection',side_effect=lambda *_:WS()), \
             patch('vivis.speech.cloud_managed.connect_session',side_effect=connected), \
             patch.object(service,'identified_person',return_value=None):
            task=asyncio.create_task(service.cloud(0))
            try:
                await until(lambda:len(sessions)==1 and service.audio.output_stream is not None)
                original_stream=service.audio.output_stream
                await switch('四川话',1)
                await until(lambda:len(sessions)==2 and service.pending_voice_switch is None)
                self.assertEqual(service.active_voice,'xunanchuan_v3.1')
                self.assertIs(service.audio.output_stream,original_stream)
                self.assertIn('羽毛球',json.dumps(connections[-1].sent,ensure_ascii=False))
                await switch('上海话',2)
                await until(lambda:service.dialect.name=='上海话')
                self.assertEqual(len(sessions),2)  # Same voice changes instructions, never reconnects.
                await switch('普通话',3)
                await until(lambda:len(sessions)==3 and service.pending_voice_switch is None)
                self.assertEqual(service.active_voice,service.session['voice'])
                self.assertIs(service.audio.output_stream,original_stream)
                self.assertEqual(service.audio.opens,0)
                self.assertIn('羽毛球',json.dumps(connections[-1].sent,ensure_ascii=False))
                self.assertEqual(service.context['revision'],1)
            finally:
                task.cancel();await asyncio.gather(task,return_exceptions=True)


if __name__=='__main__':unittest.main()
