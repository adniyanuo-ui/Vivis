"""Onsite utterances: attribution, semantic admission, durable writes and sole relay."""
import asyncio
import base64
import unittest
from dataclasses import replace
from unittest.mock import Mock,AsyncMock,patch

import test_message_narrow as fixture
from vivis.interaction.message_lane import query_recipient_fields,recipient_only
from vivis.speech.message_receipts import MessageReceipts,ReceiptRejected

PEOPLE={'he':dict(display_name='何裕德',preferred_address='何工'),
        'wang':dict(display_name='王润章',preferred_address='小王')}

class QueryRolesTests(unittest.TestCase):
    def fields(self,out,text):return query_recipient_fields(out,text,PEOPLE,{'current_person_id':'he'})

    def test_incoming_and_outgoing_cannot_be_swapped(self):
        for text in ('小王有给我的留言吗？','小王王润章有有给我的留言吗？'):
            out=self.fields(dict(action='query',recipient_reference='小王',recipient_id='wang'),text)
            self.assertEqual(out['recipient_id'],'he');self.assertEqual(out['creator_id'],'wang')
            self.assertEqual(out['status'],'active')
        out=self.fields(dict(action='query',recipient_reference='我',recipient_id='he'),'我给小王的留言转告了吗？')
        self.assertEqual(out['recipient_id'],'wang');self.assertEqual(out['creator_id'],'he')

    def test_bare_inbox_excludes_closed_history(self):
        out=self.fields(dict(action='query',recipient_reference='我',recipient_id='he',status='all'),'有我的留言吗？')
        self.assertEqual(out['status'],'active')

    def test_recipient_clarification_is_not_a_new_body(self):
        for text in ('这条留言是给小王的，王润章。','给小王的。'):
            self.assertTrue(recipient_only(text,PEOPLE))
        self.assertFalse(recipient_only('问问何工，准备哪份材料。',PEOPLE))

class SemanticGateTests(unittest.TestCase):
    def setUp(self):
        self.g=MessageReceipts();self.g.bind('r',1,'u')
        self.audio=dict(type='response.audio.delta',response_id='r',delta=base64.b64encode(b'\0'*480).decode())
        self.transcript=lambda t:dict(type='response.audio_transcript.done',response_id='r',transcript=t)
        self.final=dict(type='conversation.item.input_audio_transcription.completed',item_id='u',transcript='遇到何工帮我问问准备哪份资料。')

    def test_no_keyword_request_waits_for_semantics_and_cannot_speak_false_promise(self):
        self.g.filter(self.audio);self.g.filter(self.transcript('我记下了，包在我身上！'))
        self.g.expect_intent('u');self.assertEqual(self.g.filter(self.final),[self.final])
        self.assertFalse(self.g.rejected);self.g.decide_intent('u',True)
        self.assertFalse(self.g.drain_released());self.assertEqual(self.g.filter(self.audio),[])

    def test_even_semantic_none_cannot_bypass_claim_check(self):
        self.g.expect_intent('u');self.g.filter(self.final);self.g.filter(self.audio)
        self.g.filter(self.transcript('我记下了，遇到何工告诉他。'));self.g.decide_intent('u',False)
        self.assertFalse(self.g.drain_released());self.assertTrue(self.g.rejected)

    def test_chat_releases_once_and_network_times_are_preserved(self):
        self.g.expect_intent('u');self.g.filter(dict(self.final,transcript='今天天气不错。'))
        audio=dict(self.audio,_network_received_monotonic=12.)
        self.g.filter(audio);self.g.filter(self.transcript('是啊，阳光挺好。'))
        done=dict(type='response.done',response=dict(id='r',status='completed'))
        self.g.filter(done);self.g.decide_intent('u',False)
        released=self.g.drain_released()
        self.assertEqual([e['type'] for e in released],['response.audio.delta','response.audio_transcript.done','response.done'])
        self.assertEqual(released[0]['_network_received_monotonic'],12.)
        self.assertFalse(self.g.drain_released())

    def test_decision_before_response_created_still_protects_audio(self):
        g=MessageReceipts();g.expect_intent('u');g.filter(self.final);g.decide_intent('u',True)
        g.bind('r',1,'u');self.assertEqual(g.filter(self.audio),[])

    def test_control_function_call_does_not_wait_for_semantic_extraction(self):
        self.g.expect_intent('u');self.g.filter(self.final)
        call=dict(type='response.function_call_arguments.done',response_id='r',name='robot_program',arguments='{"action":"stop"}')
        self.assertEqual(self.g.filter(call),[call])

class DurableAndDirectionTests(unittest.IsolatedAsyncioTestCase):
    setUp=fixture.FixedMessageFlowTests.setUp
    request=fixture.FixedMessageFlowTests.request
    save=fixture.FixedMessageFlowTests.save
    async def test_commit_then_exception_reconciles_before_saying_unknown(self):
        original=self.store.create
        def commit_then_fail(*a,**k):original(*a,**k);raise RuntimeError('lost return AFTER commit')
        self.store.create=commit_then_fail
        request=self.request();await self.lane.run(request)
        self.assertTrue(self.lane.result['saved']);self.assertEqual(len(self.store.list('all')),1)
        await self.lane.run(request);self.assertEqual(len(self.store.list('all')),1)

    async def test_single_character_keyword_and_sender_filter(self):
        self.answer=dict(action='create',recipient_id='bob',content='帮我买包烟。');await self.save()
        self.answer=dict(action='create',recipient_id='alice',content='问问抽什么烟。')
        await self.lane.run(self.request('bob','遇到甲工帮我问一下抽什么烟。'))
        self.answer=dict(action='query',recipient_id='alice',creator_id='bob',status='active',content_keyword='烟')
        await self.lane.run(self.request('alice','小乙给我的留言，关于烟的。'))
        self.assertIn('问问抽什么烟',self.spoken[-1]);self.assertNotIn('帮我买包烟',self.spoken[-1])
        self.assertEqual(len(self.lane.result['result']['tasks']),1)

    async def test_only_routing_words_never_create_message(self):
        self.answer=dict(action='create',recipient_id='bob',content='这条留言是给小乙的。')
        await self.lane.run(self.request(text='这条留言是给小乙的。'))
        self.assertFalse(self.store.list('all'));self.assertIn('具体内容' if '具体内容' in self.spoken[-1] else '内容',self.spoken[-1])

    async def test_anonymous_smalltalk_does_not_require_face_verification(self):
        self.answer=dict(action='none');r=replace(self.request('anonymous:'+'a'*32,'今天天气很好。'),verified_person=None);self.current=r
        await self.lane.run(r);self.assertEqual(self.lane.result['operation_status'],'chat');self.assertFalse(self.spoken)

    async def test_failed_extraction_never_says_commit_unknown(self):
        self.lane.extract=AsyncMock(side_effect=TimeoutError())
        await self.lane.run(self.request());self.assertFalse(self.store.list('all'))
        self.assertEqual(self.lane.result['operation_status'],'failed');self.assertIn('尚未执行写入',self.spoken[-1])

class FinalTurnAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_unmatched_natural_request_enters_semantic_lane(self):
        from test_cloud_control import CloudBackendTests
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            service=CloudBackendTests().service(tmp,[])
            try:
                service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
                service.enabled=True;service.presence.wake();service.tasks=Mock();service.memory=Mock()
                service.turn_item_id='u';service.pending_user_text='遇到何工，帮我问一下他抽什么烟。'
                service.final_courier_read=Mock(return_value=False);lane=Mock();service.messages=Mock(return_value=lane)
                request=object();service.message_request=Mock(return_value=request)
                service.final_message(dict(type='conversation.item.input_audio_transcription.completed',item_id='u',transcript=service.pending_user_text),Mock())
                lane.start.assert_called_once_with(request)
                self.assertEqual(service.message_receipts.intents['u'],'waiting')
            finally:service.log.close()

    async def test_unrelated_chat_does_not_create_lane_or_hold_audio_for_intent(self):
        from test_cloud_control import CloudBackendTests
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            service=CloudBackendTests().service(tmp,[])
            try:
                service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
                service.enabled=True;service.presence.wake();service.tasks=Mock();service.memory=Mock()
                service.final_courier_read=Mock(return_value=False);service.messages=Mock()
                service.message_request=Mock()
                for index,text in enumerate(('今天天气真好。','你会唱什么歌？','小帅笑一个。',
                        '我今天有点累，想聊聊天。','你是哪个公司的？','第一件事是好好休息。')):
                    with self.subTest(text=text):
                        item=str(index);service.turn_item_id=item;service.pending_user_text=text
                        event=dict(type='conversation.item.input_audio_transcription.completed',item_id=item,transcript=text)
                        self.assertFalse(service.ordinary_message_turn(event))
                        service.final_message(event,Mock())
                        self.assertNotIn(item,service.message_receipts.intents)
                        service.message_receipts.bind('r'+item,index,item)
                        service.message_receipts.filter(event)
                        audio=dict(type='response.audio.delta',response_id='r'+item,delta=base64.b64encode(b'\0'*480).decode())
                        self.assertEqual(service.message_receipts.filter(audio),[audio])
                service.messages.assert_not_called();service.message_request.assert_not_called()
            finally:service.log.close()

    async def test_synonyms_and_pending_reply_still_enter_without_literal_message_word(self):
        from test_cloud_control import CloudBackendTests
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            service=CloudBackendTests().service(tmp,[])
            try:
                service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
                service.enabled=True;service.presence.wake();service.tasks=Mock();service.memory=Mock()
                service.final_courier_read=Mock(return_value=False);lane=Mock();service.messages=Mock(return_value=lane)
                service.message_request=Mock(return_value=object())
                for index,text in enumerate(('帮我捎句话给小王。','碰到小王提醒他带资料。',
                        '麻烦问一下何工明天几点到。','给小王留个口信。','请转达何工明天早上九点到。')):
                    with self.subTest(text=text):
                        service.turn_item_id=str(index);service.pending_user_text=text
                        event=dict(type='conversation.item.input_audio_transcription.completed',item_id=str(index),transcript=text)
                        self.assertTrue(service.ordinary_message_turn(event));service.final_message(event,Mock())
                self.assertEqual(lane.start.call_count,5)
                service.message_lane=Mock();service.message_lane.accepts_followup.return_value=True
                service.turn_item_id='body';service.pending_user_text='打火机买好了。'
                event=dict(type='conversation.item.input_audio_transcription.completed',item_id='body',transcript=service.pending_user_text)
                self.assertTrue(service.ordinary_message_turn(event))
                service.final_message(event,Mock());self.assertEqual(lane.start.call_count,6)
            finally:service.log.close()

class SemanticCompletionTests(unittest.TestCase):
    def test_discarded_chat_still_delivers_transport_completion(self):
        gate=MessageReceipts();gate.expect_intent('u');gate.bind('r',1,'u')
        gate.filter(dict(type='conversation.item.input_audio_transcription.completed',item_id='u',transcript='遇到甲工帮我问问。'))
        done=dict(type='response.done',response=dict(id='r',status='completed'))
        gate.filter(done);gate.decide_intent('u',True)
        self.assertEqual(gate.drain_released(),[done])
        self.assertEqual(gate.filter(done),[done])

    def test_implicit_sender_is_not_a_missing_recipient(self):
        out=query_recipient_fields(dict(action='query',recipient_id='wang',recipient_reference='小王',creator_id='he',creator_reference='我'),
            '刚才给小王的留言保存了吗？',PEOPLE,dict(current_person_id='he'))
        self.assertEqual(out['action'],'query');self.assertEqual(out['recipient_id'],'wang')
        out=query_recipient_fields(dict(action='query',recipient_reference='小王',recipient_id='wang',creator_reference='我'),
            '刚才给小王的留言保存了吗？',PEOPLE,dict(current_person_id='he'))
        self.assertEqual(out['action'],'query')

    def test_repeat_requires_current_persons_actual_delivery_handle(self):
        hint=dict(current_person_id='he',delivery_complete=True,task_id='a'*32)
        result=query_recipient_fields(dict(action='relay',recipient_id='he',recipient_reference='何工'),
            '没听清，再说一遍。',PEOPLE,hint)
        self.assertEqual(result['task_id'],'a'*32)
        wrong=query_recipient_fields(dict(action='relay',task_id='b'*32),'没听清，再说一遍。',PEOPLE,hint)
        self.assertEqual(wrong,dict(action='clarify',missing='target'))

class ManagedMessageConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,business):
        import json,tempfile,time
        from pathlib import Path
        from test_cloud_control import CloudBackendTests,config
        from test_managed_voice import FakeAudio
        from vivis.interaction.memory import PersonMemory
        from vivis.interaction.robot_tasks import RobotTasks
        queue=asyncio.Queue();sent=[];finish=asyncio.Event();entered=asyncio.Event()
        class WS:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await queue.get())
            async def send(self,data):
                packet=json.loads(data);sent.append(packet)
                if packet['type']=='response.cancel':
                    await queue.put(dict(type='response.done',response=dict(id='r',status='cancelled')))
        async def connected(ws,session,log):return session
        async def extracted(*_):
            entered.set();await finish.wait()
            return dict(action='create',recipient_id='beta',content='准备测试资料。') if business else dict(action='none')
        with tempfile.TemporaryDirectory() as tmp,patch('vivis.speech.managed.AudioIO',FakeAudio), \
             patch('vivis.speech.cloud_managed.websocket_connection',return_value=WS()), \
             patch('vivis.speech.cloud_managed.connect_session',side_effect=connected):
            service=CloudBackendTests().service(tmp,[])
            service.memory=PersonMemory(config('configs/interaction/memory.json'),tmp)
            service.tasks=RobotTasks(service.memory,config('configs/interaction/tasks.json'))
            service.knowledge.people={'alpha':dict(display_name='甲工'),'beta':dict(display_name='小乙')}
            service.identified_person=lambda:'alpha'
            service.context_snapshot=lambda:('synthetic instructions',dict(recognized=True,person_id='alpha'),True)
            service.verify_identity=Mock();service.schedule_task_notice=Mock()
            service.extract_message=extracted;service.speak_message=AsyncMock(return_value=True)
            service.reception.c=dict(service.reception.c,enabled=False)
            service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
            service.enabled=True;service.presence.wake()
            task=asyncio.create_task(service._connection())
            async def until(predicate):
                deadline=time.monotonic()+3
                while not predicate():
                    if task.done():task.result()
                    self.assertLess(time.monotonic(),deadline)
                    await asyncio.sleep(.01)
            try:
                await until(lambda:service.cloud_state is not None)
                for event in [dict(type='input_audio_buffer.speech_started',item_id='u'),
                    dict(type='input_audio_buffer.speech_stopped',item_id='u'),
                    dict(type='conversation.item.input_audio_transcription.completed',item_id='u',transcript='遇到小乙帮我问问资料。' if business else '天气真好。'),
                    dict(type='response.created',response=dict(id='r')),
                    dict(type='response.audio.delta',response_id='r',delta=base64.b64encode(b'\0'*480).decode()),
                    dict(type='response.audio_transcript.done',response_id='r',transcript='我记下了，包在我身上！' if business else '是啊，阳光真好。'),
                    dict(type='response.done',response=dict(id='r',status='completed'))]:await queue.put(event)
                if business:
                    await asyncio.wait_for(entered.wait(),3);await asyncio.sleep(.05)
                    self.assertEqual(service.player.count,0)
                    finish.set();await until(lambda:service.message_lane is not None and service.message_lane.result is not None)
                    await until(lambda:service.cloud_state.generating is None)
                    self.assertEqual(len(service.tasks.list('all')),1);self.assertEqual(service.player.count,0)
                    service.speak_message.assert_awaited_once()
                    self.assertIn('已保存',service.speak_message.call_args.args[0])
                else:
                    await until(lambda:service.player.count>0)
                    self.assertFalse(entered.is_set());self.assertIsNone(service.message_lane)
                    self.assertNotIn('u',service.message_receipts.intents)
                    await until(lambda:service.cloud_state.generating is None)
                    self.assertFalse(service.tasks.list('all'));service.speak_message.assert_not_awaited()
            finally:
                task.cancel();await asyncio.gather(task,return_exceptions=True);service.log.close()

    async def test_business_owns_write_and_voice_without_stuck_generation(self):await self.exercise(True)
    async def test_ordinary_chat_is_released_without_database_or_duplicate_reply(self):await self.exercise(False)

class OptionalGreetingTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_welcome_ack_does_not_disconnect_voice_or_request_unacknowledged_reply(self):
        import json,tempfile
        from test_cloud_control import CloudBackendTests
        from test_managed_voice import FakeAudio
        from vivis.speech.runtime_policy import policy
        sent=[];queue=asyncio.Queue();settings=dict(policy(),controller_ack_s=.01)
        class WS:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await queue.get())
            async def send(self,data):sent.append(json.loads(data))
        async def connected(ws,session,log):return session
        with tempfile.TemporaryDirectory() as tmp,patch('vivis.speech.managed.AudioIO',FakeAudio), \
             patch('vivis.speech.cloud_managed.websocket_connection',return_value=WS()), \
             patch('vivis.speech.cloud_managed.connect_session',side_effect=connected), \
             patch('vivis.speech.cloud_managed.policy',return_value=settings):
            service=CloudBackendTests().service(tmp,[])
            service.context_snapshot=lambda:('synthetic',dict(recognized=False,person_id=None),True)
            service.verify_identity=Mock();service.schedule_task_notice=Mock()
            service.reception.c=dict(service.reception.c,enabled=False)
            service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
            service.enabled=True;service.presence.wake();service.announcement_revision=1
            task=asyncio.create_task(service._connection())
            try:
                await asyncio.sleep(.25)
                if task.done():task.result()
                self.assertFalse(task.done())
                self.assertTrue(any(e['type']=='conversation.item.create' for e in sent))
                self.assertFalse(any(e['type']=='response.create' for e in sent))
                self.assertIsNone(service.announcement_revision)
            finally:
                task.cancel();await asyncio.gather(task,return_exceptions=True);service.log.close()
