"""Semantic-VAD false negatives recover only idle, current, read-only questions."""
import asyncio
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch

from test_cloud_control import CloudBackendTests
from vivis.speech.realtime import VoiceSession
from vivis.speech.invalid_question import complete_question,distinct_from_playback,explicit_expression_request


class InvalidQuestionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.s=CloudBackendTests().service(temp.name,[]);self.addCleanup(self.s.log.close)
        s=self.s;s.enabled=True;s.context.update(mode='interaction',target='interaction',ready=True,revision=1)
        s.presence.visible=Mock(return_value=True);s.identified_person=Mock(return_value='alice')
        s.knowledge.people={'alice':dict(display_name='甲',preferred_address='甲工')}
        self.state=VoiceSession(s.player,s.log)

    def invalid(self,item='u'):
        s=self.s;s.input_generation+=1;s.begin_user_turn(item);s.question_started(item,self.state)
        s.question_invalidated(dict(item_id=item));s.memory_turns[item]['valid']=False;s.turn_item_id=None;s.turn_ended_at=None
        return dict(type='conversation.item.input_audio_transcription.completed',item_id=item,transcript='王润章来过没？')

    async def test_logged_questions_recover_once_with_original_person_and_read_only_scope(self):
        for text in ('王润章来过没？','我让他帮我买冰淇淋的事儿，你转告了吗？','小帅，你怎么不说话了？'):
            event=self.invalid();event['transcript']=text
            self.assertTrue(self.s.recover_invalid_question(event,self.state))
            self.assertFalse(self.s.recover_invalid_question(event,self.state))
            self.assertTrue(self.s.memory_turns['u']['valid']);self.assertEqual(self.s.turn_person,'alice')
            self.assertTrue(self.s.message_request().read_only)

    async def test_backchannels_and_short_name_are_not_forced_into_responses(self):
        for text in ('嗯','对','好的','啊？','王瑞章。','收到','哈哈哈','今天阳光很好。'):
            self.assertFalse(complete_question(text),text)

    async def test_expression_admission_rejects_laughter_echo_narration_and_business(self):
        for text in ('哈哈哈哈的大笑。','小帅，你笑一个。','请你给我大笑一下吧','做一个悲伤一下'):
            self.assertTrue(explicit_expression_request(text),text)
        for text in ('哈哈哈','哈哈哈哈。','不要笑一个','他让我笑一个','小帅说笑一个','笑一个然后给小王留言','大笑','笑话真好笑'):
            self.assertFalse(explicit_expression_request(text),text)
        self.state.generating='r';self.state.responses['r']=dict(transcript='您今天想聊什么？')
        event=self.invalid();event['transcript']='哈哈哈哈的大笑。'
        self.state.generating=None
        self.assertFalse(self.s.recover_invalid_question(event,self.state))

    async def test_recovered_expression_uses_existing_tool_once_and_keeps_business_read_only(self):
        s=self.s;event=self.invalid();event['transcript']='哈哈哈哈的大笑。'
        self.assertTrue(s.recover_invalid_question(event,self.state));self.assertTrue(s.message_request().read_only)
        s.expression.publish=Mock(return_value=dict(status='pending_expression_target',intent_id='test'))
        s.response_revisions['r']=1;s.response_inputs['r']=s.input_generation
        sent=[]
        async def send(text):sent.append(json.loads(text))
        async def invoke(name,args,call):
            await s.tool(SimpleNamespace(send=send),dict(call_id=call,response_id='r',name=name,arguments=json.dumps(args)))
            return json.loads(sent[-1]['item']['output'])
        self.assertFalse((await invoke('robot_program',dict(action='play',program='baigufuren'),'program'))['accepted'])
        self.assertFalse((await invoke('courier_pickup',dict(action='register',code='9999'),'write'))['accepted'])
        self.assertTrue((await invoke('set_expression',dict(expression='smile'),'first'))['accepted'])
        self.assertFalse((await invoke('set_expression',dict(expression='smile'),'duplicate'))['accepted'])
        s.expression.publish.assert_called_once()

    async def test_expression_recovery_cannot_act_after_guard_changes(self):
        for change in ('person','connection','face','response_generation','expired','in_speech'):
            self.setUp()
            event=self.invalid();event['transcript']='哈哈哈哈的大笑。'
            self.assertTrue(self.s.recover_invalid_question(event,self.state))
            s=self.s;s.response_revisions['r']=1;s.response_inputs['r']=s.input_generation
            source=s.recovered_expression
            if change=='person':source['person']='someone-else'
            if change=='connection':source['connection']=object()
            if change=='face':s.presence.visible.return_value=False
            if change=='response_generation':s.response_inputs['r']-=1
            if change=='expired':source['ended']-=60
            if change=='in_speech':s.in_speech=True
            s.expression.publish=Mock();sent=[]
            async def send(text):sent.append(json.loads(text))
            await s.tool(SimpleNamespace(send=send),dict(call_id=change,response_id='r',name='set_expression',arguments='{"expression":"smile"}'))
            self.assertFalse(json.loads(sent[-1]['item']['output'])['accepted'],change)
            s.expression.publish.assert_not_called()
            s.presence.visible.return_value=True;s.in_speech=False

    async def test_no_recovery_during_speaker_output_or_acoustic_tail(self):
        self.state.generating='robot';event=self.invalid()
        self.state.generating=None
        self.assertFalse(self.s.recover_invalid_question(event,self.state))
        self.state.responses['robot']=dict(empty_dac=time.monotonic())
        event=self.invalid();self.assertFalse(self.s.recover_invalid_question(event,self.state))

    async def test_welcome_barge_in_final_question_cancels_only_old_speech(self):
        self.state.generating='welcome'
        self.state.responses['welcome']=dict(transcript='您好！我是小帅，有什么可以帮您的吗？')
        event=self.invalid();self.state.generating=None
        self.assertTrue(self.s.recover_invalid_question(event,self.state))
        self.assertIn('welcome',self.state.blocked)
        self.assertIn('welcome',self.s.blocked)
        self.assertTrue(self.s.message_request().read_only)
        self.assertFalse(self.s.recover_invalid_question(event,self.state))

    async def test_echo_questions_and_newer_output_cannot_trigger_barge_in(self):
        for text in ('有什么可以帮您的吗？','有什么能帮您的吗？'):
            self.state.generating='welcome'
            self.state.responses['welcome']=dict(transcript='您好！我是小帅，有什么可以帮您的吗？')
            event=self.invalid();event['transcript']=text
            self.assertFalse(self.s.recover_invalid_question(event,self.state))
        event=self.invalid();self.state.generating='new-response'
        self.assertFalse(self.s.recover_invalid_question(event,self.state))
        self.assertFalse(distinct_from_playback('王润章来过没？',[None]))

    async def test_cancel_inference_never_cancels_newer_generation(self):
        ws=SimpleNamespace(send=AsyncMock());self.s.ws=ws;self.s.cloud_state=self.state
        self.state.generating='old'
        await self.s.cancel_inference(ws,'old');ws.send.assert_awaited_once()
        self.state.generating='new'
        await self.s.cancel_inference(ws,'old');ws.send.assert_awaited_once()

    async def test_person_generation_connection_and_visibility_changes_invalidate_recovery(self):
        for change in ('person','generation','connection','face','mode','expired'):
            self.setUp()
            event=self.invalid()
            if change=='person':self.s.identified_person.return_value='bob'
            if change=='generation':self.s.input_generation+=1
            if change=='connection':self.s.message_connection=object()
            if change=='face':self.s.presence.visible.return_value=False
            if change=='mode':self.s.context.update(mode='task',target='baigufuren',revision=2)
            if change=='expired':self.s.invalid_question['ended']-=60
            self.assertFalse(self.s.recover_invalid_question(event,self.state),change)

    async def test_query_waits_for_controller_ack_and_cancels_if_new_speech_arrives(self):
        event=self.invalid();self.s.recover_invalid_question(event,self.state)
        self.s.pending_user_text=event['transcript'];request=self.s.message_request()
        gate=SimpleNamespace(append=AsyncMock(return_value='controller-test'))
        await self.s.ask_recovered_question(None,gate,self.state,request)
        gate.append.assert_awaited_once();self.assertTrue(self.s.continue_response)
        self.s.continue_response=False
        async def changed(*_):self.s.input_generation+=1
        gate.append=AsyncMock(side_effect=changed)
        await self.s.ask_recovered_question(None,gate,self.state,request)
        self.assertFalse(self.s.continue_response)

    async def test_recovery_cannot_start_program_or_register_code(self):
        event=self.invalid();self.s.recover_invalid_question(event,self.state)
        self.s.response_revisions['r']=1;sent=[]
        async def send(text):sent.append(json.loads(text))
        for i,(name,args) in enumerate((('robot_program',dict(action='play',program='baigufuren')),('courier_pickup',dict(action='register',code='9999')))):
            await self.s.tool(SimpleNamespace(send=send),dict(call_id=str(i),response_id='r',name=name,arguments=json.dumps(args)))
            self.assertFalse(json.loads(sent[-1]['item']['output'])['accepted'])

    async def test_receive_loop_recovers_logged_sequence_after_server_item_ack(self):
        await self.receive_case(False)

    async def test_receive_loop_welcome_interruption_recovers_without_replaying_welcome(self):
        await self.receive_case(True)

    async def test_receive_loop_recovers_actual_laugh_request_without_direct_motor_commands(self):
        await self.receive_case(False,expression=True)

    async def receive_case(self,playing,expression=False):
        from test_managed_voice import FakeAudio
        s=self.s;s.audio=FakeAudio();s.conversation_person='alice'
        s.reception.c=dict(s.reception.c,enabled=False);s.memory=None
        messages=asyncio.Queue();sent=[]
        class WS:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            async def send(self,text):
                data=json.loads(text);sent.append(data)
                if data.get('type')=='conversation.item.create' and data.get('item',{}).get('id','').startswith('controller_'):
                    await messages.put(dict(type='conversation.item.created',item=data['item']))
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await messages.get())
        async def connect(ws,session,log):return session
        async def until(check):
            for _ in range(150):
                if check():return
                await asyncio.sleep(.01)
            self.fail('event-loop condition did not complete')
        with patch('vivis.speech.cloud_managed.websocket_connection',return_value=WS()),patch('vivis.speech.cloud_managed.connect_session',side_effect=connect):
            task=asyncio.create_task(s._connection())
            try:
                await until(lambda:s.cloud_state is not None)
                if playing:
                    s.cloud_state.responses['welcome']=dict(transcript='您好！我是小帅，有什么可以帮您的吗？',status='completed')
                    s.player.add('welcome',b'\0'*48000);s.player.finish('welcome')
                await messages.put(dict(type='input_audio_buffer.speech_started',item_id='real-question'))
                await messages.put(dict(type='input_audio_buffer.speech_stopped',item_id='real-question',reason='turn_invalid'))
                transcript='哈哈哈哈的大笑。' if expression else '王润章来过没？'
                await messages.put(dict(type='conversation.item.input_audio_transcription.completed',item_id='real-question',transcript=transcript))
                await until(lambda:any(x.get('type')=='response.create' for x in sent))
                controller=[x for x in sent if x.get('item',{}).get('id','').startswith('controller_')]
                self.assertEqual(len(controller),1)
                self.assertIn(transcript,controller[0]['item']['content'][0]['text'])
                if expression:self.assertIn('set_expression',controller[0]['item']['content'][0]['text'])
                self.assertTrue(s.message_request().read_only)
                if playing:
                    self.assertIn('welcome',s.cloud_state.blocked)
                    self.assertNotIn('welcome',s.player.response_ids())
                await messages.put(dict(type='conversation.item.input_audio_transcription.completed',item_id='real-question',transcript=transcript))
                await asyncio.sleep(.1)
                self.assertEqual(len([x for x in sent if x.get('type')=='response.create']),1)
            finally:
                task.cancel();await asyncio.gather(task,return_exceptions=True)
