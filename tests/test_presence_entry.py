"""Entry/exit gates: no offline ASR, no device or identity-policy changes."""
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import test_cloud_control as fixtures
from vivis.interaction.cloud_control import SpeechPresence,ReceptionGate,controller_intent,normalize_program
from vivis.interaction.presence_entry import configuration,whole_greeting
from vivis.speech.realtime import VoiceSession


class EntryEvidenceTests(unittest.TestCase):
    def test_only_whole_direct_greeting_is_recovered_not_quotation_or_negation(self):
        c=configuration()
        for text in ('你好','您好呀','喂！','嗨','Hi','HELLO！','hello，你好。','你好，您好呀','Hi! Hello.','喂啊，你好呀'):
            self.assertTrue(whole_greeting(text,c),text)
        for text in ('你好，有人在吗？','你好，有人吗？','喂，在吗？','您好，你在吗？','Hello，你好，有人在吗？','嗨，有人在吗呀？'):
            self.assertTrue(whole_greeting(text,c),text)
        for text in ('他说你好','不要说你好','你好小王','“你好”是什么意思','hello王教授','好了吗','我走了',
                     '他说hello你好','不要说hello你好','hello，你好，小王','“hello，你好”','hi' * 5,
                     '他说你好有人在吗','“你好，有人在吗？”是什么意思','你好小王有人在吗','你好有人在吗小王',
                     '不要说你好有人在吗','你好不要唤醒','你好播放歌曲','有人在吗','在吗'):
            self.assertFalse(whole_greeting(text,c),text)
        self.assertFalse(whole_greeting('你好',{**c,'enabled':False}))

    def test_legacy_config_retains_pure_greetings_without_accepting_arbitrary_suffix(self):
        c=configuration();c.pop('contact_checks',None)
        self.assertTrue(whole_greeting('Hello，你好。',c))
        self.assertFalse(whole_greeting('你好，有人在吗？',c))

    def test_single_face_requires_fresh_valid_publisher_data(self):
        with tempfile.TemporaryDirectory() as folder:
            p=SpeechPresence(fixtures.config('configs/perception/speech_presence.json'),folder)
            p.path.parent.mkdir(parents=True,exist_ok=True)
            self.assertFalse(p.single_visible(10))
            for faces in (1,2,0,True,-1):
                p.path.write_text(json.dumps(dict(schema_version='vivis-face-presence-1.0',observed_monotonic=10,faces=faces)))
                self.assertEqual(p.single_visible(10),type(faces) is int and faces==1)
                self.assertFalse(p.single_visible(20))

    def test_greeting_has_no_named_no_face_wake_authority(self):
        args=dict(action='wake',wake_reason='greeting',explicit_wake=False)
        self.assertFalse(normalize_program(args,{'song'})['explicit_wake'])
        self.assertEqual(controller_intent(args,{'song'},'standby'),{'kind':'wake'})
        with self.assertRaises(ValueError):normalize_program({**args,'explicit_wake':True},{'song'})
        with self.assertRaises(ValueError):normalize_program(dict(action='wake',explicit_wake=False),{'song'})

    def test_arrival_is_half_second_and_brief_pass_resets(self):
        gate=ReceptionGate(fixtures.config('configs/interaction/reception.json'))
        self.assertEqual(gate.c['arrival_hold_s'],.5)
        self.assertFalse(gate.observe(True,'standby',10))
        self.assertFalse(gate.observe(True,'standby',10.49))
        gate.observe(False,'standby',10.49)
        self.assertFalse(gate.observe(True,'standby',11))
        self.assertTrue(gate.observe(True,'standby',11.5))
        self.assertFalse(gate.observe(True,'interaction',12))


class EntryFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.events=[];self.s=fixtures.CloudBackendTests().service(folder.name,self.events);self.addCleanup(self.s.log.close)
        self.s.context.update(mode='standby',target='standby',ready=True,revision=1)
        self.s.presence.visible=Mock(return_value=True);self.s.presence.single_visible=Mock(return_value=True);self.s.presence.greeting_visible=Mock(return_value=True)
        self.s.identified_person=Mock(return_value=None)
        self.state=VoiceSession(self.s.player,self.s.log)

    def final(self,text='你好',transcribed=True):
        s=self.s;s.input_generation+=1;s.begin_user_turn('u');s.question_started('u',self.state)
        s.turn_ended_at=time.monotonic()
        if transcribed:s.memory_turns['u']['text']=text;s.question_onset['final_text']=text
        return dict(item_id='u',transcript=text)

    async def call(self,args,call_id='call'):
        s=self.s;s.response_revisions['r']=s.context['revision'];sent=[]
        s.response_inputs['r']=s.input_generation
        if s.turn_item_id:self.state.responses['r']=dict(user_turn=dict(item_id=s.turn_item_id))
        async def send(text):sent.append(json.loads(text))
        await s.tool(SimpleNamespace(send=send),dict(call_id=call_id,response_id='r',name='robot_program',arguments=json.dumps(args)))
        return json.loads(sent[-1]['item']['output']) if sent else None

    async def test_greeting_wakes_standby_once_without_no_face_exception(self):
        event=self.final();s=self.s
        self.assertTrue(s.admit_explicit_wake(event,self.state))
        self.assertEqual(self.events[-1]['arguments'],dict(action='wake',program='',explicit_wake=False,wake_reason='greeting'))
        self.assertEqual(s.presence.wake_until,0)
        self.assertIsNone(s.identified_person())
        self.assertFalse(s.admit_explicit_wake(event,self.state))
        result=await self.call(dict(action='wake',wake_reason='greeting',explicit_wake=False))
        self.assertTrue(result['accepted'])
        self.assertEqual(len([e for e in self.events if e['kind']=='cloud_action']),1)

    async def test_tool_then_final_does_not_send_second_wake(self):
        event=self.final(transcribed=False)
        result=await self.call(dict(action='wake',wake_reason='greeting',explicit_wake=False))
        self.assertFalse(result['accepted']);self.assertEqual(result['status'],'greeting_final_text_pending')
        self.assertFalse(self.events)
        self.assertTrue(self.s.admit_explicit_wake(event,self.state))
        self.assertEqual(len([e for e in self.events if e['kind']=='cloud_action']),1)

    async def test_model_cannot_wake_for_quote_or_reuse_another_turn(self):
        for text in ('他刚才说你好','不要说你好','你好小王'):
            self.final(text)
            result=await self.call(dict(action='wake',wake_reason='greeting',explicit_wake=False),text)
            self.assertFalse(result['accepted']);self.assertEqual(result['status'],'not_a_direct_greeting')
        self.final();self.s.input_generation+=1
        result=await self.call(dict(action='wake',wake_reason='greeting',explicit_wake=False),'stale')
        self.assertFalse(result['accepted']);self.assertEqual(result['status'],'greeting_source_not_current')
        self.assertFalse(self.events)

    async def test_no_person_and_disabled_block_both_greeting_paths(self):
        for situation in ('no-face','disabled'):
            self.s.control_pending=False;self.s.explicit_wake_seen.clear();self.s.entry=configuration()
            self.s.presence.visible.return_value=situation!='no-face'
            self.s.presence.greeting_visible.return_value=situation=='disabled'
            if situation=='disabled':self.s.entry['visible_greeting_wake']=False
            self.assertFalse(self.s.admit_explicit_wake(self.final(),self.state))
            result=await self.call(dict(action='wake',wake_reason='greeting',explicit_wake=False),situation)
            self.assertFalse(result['accepted'],situation)
            self.assertEqual(self.s.presence.wake_until,0)
        self.assertFalse([e for e in self.events if e['kind']=='cloud_action'])

    async def test_face_lost_while_waiting_for_audio_permission_cancels_greeting(self):
        s=self.s;s.admit_explicit_wake(self.final(),self.state)
        s.context.update(mode='interaction',target='interaction');s.enabled=True
        async def lose_face(*args):s.presence.greeting_visible.return_value=False
        gate=SimpleNamespace(enable=AsyncMock(side_effect=lose_face));controller=SimpleNamespace(append=AsyncMock())
        self.assertFalse(await s.complete_explicit_wake(None,self.state,controller,gate,'instructions'))
        controller.append.assert_not_awaited();self.assertFalse(s.cloud_audio_enabled)
        self.assertEqual(s.presence.wake_until,0)

    async def test_greeting_waits_through_homing_without_old_welcome(self):
        s=self.s;s.admit_explicit_wake(self.final(),self.state)
        gate=SimpleNamespace(enable=AsyncMock());controller=SimpleNamespace(append=AsyncMock())
        s.request(dict(kind='voice_context',revision=2,mode='interaction',target='interaction',ready=False))
        self.assertFalse(await s.complete_explicit_wake(None,self.state,controller,gate,'instructions'))
        self.assertIsNotNone(s.explicit_wake_pending);gate.enable.assert_not_awaited()
        s.request(dict(kind='voice_context',revision=3,mode='interaction',target='interaction',ready=True))
        self.assertIsNone(s.announcement_revision)
        s.enabled=True
        self.assertTrue(await s.complete_explicit_wake(None,self.state,controller,gate,'instructions'))
        controller.append.assert_awaited_once();self.assertEqual(s.presence.wake_until,0)

    async def test_greeting_echo_or_unknown_turn_does_not_wake(self):
        event=self.final();event['item_id']='missing'
        self.assertFalse(self.s.admit_explicit_wake(event,self.state))
        self.state.generating='robot';self.state.responses['robot']=dict(transcript='你好')
        self.assertFalse(self.s.admit_explicit_wake(self.final(),self.state))
        self.s.cloud_state=self.state
        result=await self.call(dict(action='wake',wake_reason='greeting',explicit_wake=False))
        self.assertFalse(result['accepted']);self.assertEqual(result['status'],'greeting_echo_not_excluded')
        self.assertFalse(self.events)

    async def test_named_wake_still_works_without_visible_face(self):
        self.s.presence.visible.return_value=False;self.s.presence.greeting_visible.return_value=False
        self.assertTrue(self.s.admit_explicit_wake(self.final('小帅你好'),self.state))
        self.assertGreater(self.s.presence.wake_until,time.monotonic())

    async def test_stop_and_rest_do_not_require_face_or_name_and_deduplicate(self):
        self.s.presence.visible.return_value=False;self.s.presence.greeting_visible.return_value=False
        self.s.context.update(mode='task',target=next(iter(self.s.program_ids)))
        self.s.cloud_state=self.state
        for action in ('stop','standby'):
            self.final('停止。' if action=='stop' else '你休息吧。')
            result=await self.call(dict(action=action),action)
            self.assertTrue(result['accepted'])
        self.assertEqual([e['arguments']['action'] for e in self.events if e['kind']=='cloud_action'],['stop','standby'])
        self.assertIsNone(await self.call(dict(action='stop'),'stop'))
        self.assertEqual(len([e for e in self.events if e['kind']=='cloud_action']),2)

    async def test_explicit_rest_holds_current_visit_until_real_departure(self):
        s=self.s;s.context.update(mode='interaction',target='interaction')
        s.request(dict(kind='voice_context',revision=2,mode='standby',target='standby',ready=True))
        self.assertFalse(s.reception.observe(True,'standby',10))
        self.assertFalse(s.reception.observe(True,'standby',20))
        s.reception.observe(False,'standby',21);s.reception.observe(False,'standby',25)
        self.assertFalse(s.reception.observe(True,'standby',26))
        self.assertTrue(s.reception.observe(True,'standby',26.5))


if __name__=='__main__':unittest.main()
