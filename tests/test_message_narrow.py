"""First-round acceptance: fixed saves/delivery, real DB, synthetic Qwen fields."""
import asyncio
from dataclasses import replace
import json
import tempfile
import unittest

from vivis.paths import ROOT
from vivis.interaction.memory import PersonMemory
from vivis.interaction.robot_tasks import RobotTasks
from vivis.interaction.message_delivery import MessageRequest
from vivis.interaction.message_lane import MessageLane


class FixedMessageFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.config=json.loads((ROOT/'configs/interaction/memory.json').read_text())
        self.policy=json.loads((ROOT/'configs/interaction/tasks.json').read_text())
        self.memory=PersonMemory(self.config,self.tmp.name);self.store=RobotTasks(self.memory,self.policy)
        self.people={'alice':dict(preferred_address='甲工'),'bob':dict(preferred_address='小乙')}
        self.sequence=0;self.current=None;self.spoken=[];self.answer=dict(action='create',recipient_id='bob',content='明天买一个USB hub。')
        async def extract(text,people,hint):return dict(self.answer)
        async def speak(text,request):self.spoken.append(text);return True
        self.lane=MessageLane(self.store,self.people,extract,speak,lambda r:r==self.current,lambda *a,**k:None)

    def request(self,actor='alice',text='请给小乙留言，明天买一个USB hub。'):
        self.sequence+=1;mid=self.memory.append(actor,'visit','user',text)
        self.current=MessageRequest(actor,'visit',str(self.sequence),mid,text,actor)
        return self.current

    async def save(self):
        r=self.request();await self.lane.run(r);self.assertTrue(self.lane.result['saved']);return r

    async def test_supplement_during_extraction_restarts_with_same_person_draft(self):
        entered=asyncio.Event();finish=asyncio.Event();hints=[]
        async def extract(text,people,hint):
            hints.append(dict(hint))
            if len(hints)==1:entered.set();await finish.wait()
            return dict(action='create',recipient_id='bob',content='今天下午买一个冰淇淋。')
        self.lane.extract=extract
        first=self.request(text='让小乙买一个冰淇淋。');self.lane.start(first)
        await entered.wait();self.lane.next_turn('alice','visit')
        next_request=self.request(text='下午，今天下午。')
        finish.set();await self.lane.task
        self.assertFalse(self.store.list('all'))
        self.lane.start(next_request);await self.lane.task
        self.assertEqual(hints[-1]['draft_source'],first.source_text)
        self.assertEqual(len(self.store.list('all')),1)
        self.assertEqual(self.store.list('all')[0]['content'],'今天下午买一个冰淇淋。')
        self.assertFalse(self.lane.has_continuation('alice','visit'))

    async def test_pending_draft_cannot_cross_person_or_session_or_edit_saved_message(self):
        r=self.request();self.lane.extracting=r
        self.lane.next_turn('alice','visit');self.assertTrue(self.lane.has_continuation('alice','visit'))
        self.lane.next_turn('bob','visit');self.assertFalse(self.lane.continuation)
        self.lane.next_turn('alice','new-session');self.assertFalse(self.lane.continuation)
        self.lane.extracting=None
        await self.lane.run(r);self.lane.next_turn('alice','visit')
        self.assertFalse(self.lane.continuation)

    async def test_recovered_query_cannot_write_but_can_read_ledger(self):
        request=replace(self.request(),read_only=True);self.current=request
        await self.lane.run(request);self.assertFalse(self.store.list('all'))
        self.answer=dict(action='query',status='all')
        await self.lane.run(request);self.assertEqual(self.lane.result['operation_status'],'read')

    async def test_courier_query_uses_owner_ledger_not_colleague_messages(self):
        from vivis.interaction.courier import CourierHandoffs
        self.lane.courier=CourierHandoffs(self.memory)
        mid=self.memory.append('alice','owner','user','退件取件码4321。')
        row=self.lane.courier.register('alice','甲工',['甲工'],'4321',[mid],source_reason='synthetic')
        await self.save();self.answer=dict(action='courier_query')
        await self.lane.run(self.request(text='我还有其他的留言吗？给快递的有没有？'))
        self.assertIn('4、3、2、1',self.spoken[-1]);self.assertNotIn('USB',self.spoken[-1])
        self.assertEqual(self.lane.courier.get(row['id'])['status'],'pending')
        self.assertEqual(len(self.store.list('all')),1)

    async def test_identity_loss_during_extraction_notifies_without_reading_private_data(self):
        from unittest.mock import AsyncMock
        self.lane.notify=AsyncMock()
        async def extract(*_):self.current=None;return dict(action='query')
        self.lane.extract=extract
        await self.lane.run(self.request(text='留言保存了吗？'))
        self.lane.notify.assert_awaited_once()
        self.assertIn('身份正在复核',self.lane.notify.call_args.args[0])
        self.assertFalse(self.spoken);self.assertFalse(self.store.list('all'))

    async def test_say_once_stores_once_and_same_request_retries_are_idempotent(self):
        request=await self.save();await self.lane.run(request)
        self.assertEqual(len(self.store.list('all')),1)
        self.assertTrue(self.lane.result['saved']);self.assertIn('小乙',self.spoken[0])

    async def test_different_new_requests_do_not_scan_or_semantically_merge_history(self):
        await self.save();await self.save()
        self.assertEqual(len(self.store.list('all')),2)
        self.answer=dict(action='retry');await self.lane.run(self.request(text='重新保存刚才那条。'))
        self.assertEqual(len(self.store.list('all')),2)

    async def test_a_speaker_change_during_extraction_cannot_commit_old_request(self):
        entered=asyncio.Event();finish=asyncio.Event()
        async def extract(*_):entered.set();await finish.wait();return dict(self.answer)
        self.lane.extract=extract;r=self.request();task=asyncio.create_task(self.lane.run(r))
        await entered.wait();self.current=replace(r,actor='bob',verified_person='bob');finish.set();await task
        self.assertFalse(self.store.list('all'));self.assertFalse(self.spoken)

    async def test_interrupted_relay_stays_pending_and_complete_delivery_is_durable(self):
        await self.save();row=self.store.list('all')[0]
        request=MessageRequest('bob','visit',None,None,None,'bob');self.current=request
        async def interrupted(*_):return False
        self.lane.speak=interrupted;await self.lane.relay(row['task_id'],request)
        self.assertEqual(self.store.get(row['task_id'])['status'],'pending')
        async def complete(*_):return True
        self.lane.speak=complete;await self.lane.relay(row['task_id'],request)
        self.assertEqual(self.store.get(row['task_id'])['status'],'announced')
        fresh=RobotTasks(PersonMemory(self.config,self.tmp.name),self.policy)
        self.assertEqual(fresh.get(row['task_id'])['status'],'announced');self.assertFalse(fresh.due('bob'))
        self.assertFalse(fresh.list('active',recipient='bob'))
        self.assertNotIn('收到了吗',self.spoken[-1])

    async def test_a_rollback_never_speaks_success_and_unknown_commit_checks_on_retry(self):
        original=self.store.get
        def unavailable(*_):raise RuntimeError('synthetic readback unavailable AFTER commit')
        self.store.get=unavailable;await self.lane.run(self.request())
        self.assertEqual(self.lane.result['operation_status'],'unconfirmed');self.assertIsNone(self.lane.result['saved'])
        self.assertNotIn('已保存',self.spoken[-1]);self.store.get=original
        self.answer=dict(action='retry');await self.lane.run(self.request(text='重试刚才那条。'))
        self.assertEqual(len(self.store.list('all')),1);self.assertTrue(self.lane.result['saved'])

    async def test_only_explicit_supplement_updates_an_exact_owned_message(self):
        await self.save();tid=self.store.list('all')[0]['task_id']
        self.answer=dict(action='supplement',content='放在前台。')
        req=self.request(text='补充刚才那条，放在前台。');await self.lane.run(req);await self.lane.run(req)
        self.assertEqual(len(self.store.list('all')),1)
        self.assertEqual(self.store.get(tid)['content'].count('放在前台'),1)

    async def test_queries_read_saved_status_without_another_write(self):
        await self.save();self.answer=dict(action='query',status='active')
        await self.lane.run(self.request(text='刚才的留言保存了吗？'))
        self.assertIn('已保存，等待转告',self.spoken[-1]);self.assertEqual(len(self.store.list('all')),1)

    async def test_specific_query_filters_before_pagination_and_keeps_database_unchanged(self):
        await self.save()
        self.answer=dict(action='create',recipient_id='bob',content='今天下午买一个冰淇淋。')
        await self.lane.run(self.request(text='让小乙今天下午买一个冰淇淋。'))
        row=self.store.list('all')[-1];self.store.mark_announced(row['task_id'],'bob','complete-audio')
        self.store.c['list_limit']=1
        self.answer=dict(action='query',status='all',recipient_id='bob',content_keyword='冰淇淋')
        await self.lane.run(self.request(text='我让小乙买冰淇淋的事，你转告了吗？'))
        self.assertIn('查询到1条',self.spoken[-1]);self.assertIn('已转达',self.spoken[-1])
        self.assertIn('冰淇淋',self.spoken[-1]);self.assertNotIn('USB',self.spoken[-1])
        self.assertEqual(sum(self.store.counts().values()),2)

    async def test_query_keyword_cannot_be_invented_or_disclose_other_people(self):
        await self.save()
        self.answer=dict(action='query',content_keyword='不存在的东西')
        await self.lane.run(self.request(text='给小乙的留言保存了吗？'))
        self.assertNotIn('USB',self.spoken[-1])
        # The database predicate never relaxes the existing actor permission.
        self.assertFalse(self.store.page('all',visible_to='other',content_keyword='USB')['tasks'])

    async def test_delivered_message_is_not_pending_after_restart_and_can_be_queried(self):
        await self.save();tid=self.store.list('all')[0]['task_id'];self.store.mark_announced(tid,'bob','previous-complete-audio')
        self.lane.offered.clear();self.lane.last.clear()
        self.answer=dict(action='query',status='all');await self.lane.run(self.request('bob','我的留言有哪些？'))
        self.assertIn('已转达',self.spoken[-1]);self.assertNotIn('等待确认',self.spoken[-1])
        self.assertEqual(self.store.get(tid)['status'],'announced')
        self.assertFalse(self.store.due('bob','2099-01-01T00:00:00+00:00'))

    async def test_recipient_change_at_delivery_commit_keeps_pending(self):
        await self.save();tid=self.store.list('all')[0]['task_id']
        request=MessageRequest('bob','visit',None,None,None,'bob');self.current=request
        original=self.store.mark_announced
        def changed(*args,**kwargs):
            self.current=None
            return original(*args,**kwargs)
        self.store.mark_announced=changed
        await self.lane.relay(tid,request)
        self.assertEqual(self.store.get(tid)['status'],'pending')

    async def test_explicit_repeat_reads_delivered_message_without_reopening(self):
        await self.save();tid=self.store.list('all')[0]['task_id']
        request=MessageRequest('bob','visit',None,None,None,'bob');self.current=request
        await self.lane.relay(tid,request)
        self.answer=dict(action='relay',task_id=tid)
        await self.lane.run(self.request('bob','没听清，再说一遍。'))
        self.assertIn('USB hub',self.spoken[-1]);self.assertFalse(self.store.due('bob'))

    async def test_same_request_supplement_cannot_mutate_another_persons_message(self):
        await self.save();tid=self.store.list('all')[0]['task_id']
        self.answer=dict(action='supplement',content='未经授权的补充。',task_id=tid)
        await self.lane.run(self.request('bob','补充这条。'))
        self.assertNotIn('未经授权',self.store.get(tid)['content'])
        self.assertFalse(self.lane.result['saved'])

    async def test_voice_decoder_failure_after_commit_cannot_report_unsaved_or_duplicate(self):
        async def broken(*_):raise ValueError('synthetic malformed TTS WAV')
        self.lane.speak=broken
        request=self.request();await self.lane.run(request)
        self.assertTrue(self.lane.result['saved']);self.assertEqual(len(self.store.list('all')),1)
        await self.lane.run(request)
        self.assertTrue(self.lane.result['saved']);self.assertEqual(len(self.store.list('all')),1)
        self.assertEqual(self.store.list('all')[0]['status'],'pending')

    async def test_query_voice_failure_preserves_read_result_without_retrying_as_database_failure(self):
        await self.save();calls=[]
        async def broken(*_):calls.append(True);raise ValueError('bad TTS WAV')
        self.lane.speak=broken;self.answer=dict(action='query',status='active')
        await self.lane.run(self.request(text='保存了吗？'))
        self.assertEqual(self.lane.result['operation_status'],'read')
        self.assertEqual(self.lane.result['speech_status'],'failed')
        self.assertEqual(len(calls),1);self.assertEqual(len(self.store.list('all')),1)

    async def test_sender_readback_misclassified_as_relay_keeps_requested_recipient(self):
        await self.save();self.lane.last.clear()
        self.answer=dict(action='relay',recipient_id='bob')
        await self.lane.run(self.request(text='给小乙的留言保存了吗？完整说一下。'))
        self.assertEqual(self.lane.result['operation_status'],'read')
        self.assertIn('明天买一个USB hub',self.spoken[-1])
        self.assertEqual(self.store.list('all')[0]['status'],'pending')

    async def test_unknown_query_recipient_asks_instead_of_reporting_empty_or_unsaved(self):
        await self.save();self.answer=dict(action='query',recipient_id='xiaoer')
        await self.lane.run(self.request(text='给小儿的留言保存了吗？'))
        self.assertEqual(self.lane.result['operation_status'],'needs_clarification')
        self.assertIsNone(self.lane.result['saved'])
        self.assertIn('给谁',self.spoken[-1]);self.assertNotIn('尚未保存',self.spoken[-1])
        self.assertEqual(len(self.store.list('all')),1)

    def test_query_target_cannot_be_guessed_as_current_person(self):
        from vivis.interaction.message_lane import query_recipient_fields
        people={'he':dict(display_name='何裕德',preferred_address='何工'),
                'wang':dict(display_name='王润章',preferred_address='小王')}
        wrong=dict(action='query',recipient_id='he',recipient_reference='我')
        self.assertEqual(query_recipient_fields(wrong,'给小儿的留言保存了吗？',people,{'current_person_id':'he'}),dict(action='clarify',missing='recipient'))
        right=dict(action='query',recipient_id='wang',recipient_reference='小王')
        self.assertEqual(query_recipient_fields(right,'给小王的留言保存了吗？',people,{'current_person_id':'he'}),right)
        own=dict(action='relay',recipient_reference='我')
        self.assertEqual(query_recipient_fields(own,'把给我的留言说一下',people,{'current_person_id':'he'})['recipient_id'],'he')

    def test_pronoun_query_uses_nearest_unique_user_topic_only(self):
        from vivis.interaction.message_lane import query_recipient_fields
        people={'he':dict(display_name='何裕德',preferred_address='何工'),'wang':dict(display_name='王润章',preferred_address='小王')}
        query=dict(action='query',recipient_reference='他',recipient_id='wang',content_keyword='冰淇淋')
        source='我让你转告的，让他买冰淇淋的事儿，你转告了吗？'
        hint=dict(current_person_id='he',recent_user_turns=['小王来过没？','王润章来过没？'])
        self.assertEqual(query_recipient_fields(query,source,people,hint),query)
        for turns in ([],['小王和何工来过没？'],['小王来过没？','何工来过没？']):
            self.assertEqual(query_recipient_fields(query,source,people,dict(hint,recent_user_turns=turns)),dict(action='clarify',missing='recipient'))
        self.assertEqual(query_recipient_fields(dict(query,action='relay'),source,people,hint),dict(action='clarify',missing='recipient'))

    def test_visit_question_gate_does_not_capture_ordinary_statement(self):
        from vivis.speech.message_receipts import MESSAGE_REQUEST
        for text in ('小王来过没？','王润章来过没有','小王来了没？'):
            self.assertTrue(MESSAGE_REQUEST.search(text))
        for text in ('我来过北京。','我没有见过这么大的雨。'):
            self.assertFalse(MESSAGE_REQUEST.search(text))

    async def test_visit_query_uses_delivery_event_not_assistant_claim(self):
        await self.save();tid=self.store.list('all')[0]['task_id']
        self.memory.append('alice','visit','assistant','小乙还没来过呢。')
        self.store.mark_announced(tid,'bob','finished-audio')
        self.answer=dict(action='visit_query',recipient_id='bob')
        await self.lane.run(self.request(text='小乙来过没？'))
        self.assertIn('向小乙完整转达过留言',self.spoken[-1])
        self.assertNotIn('还没来过',self.spoken[-1])
        self.assertEqual(self.store.get(tid)['status'],'announced')

    async def test_no_visit_evidence_means_unknown_not_absence(self):
        self.memory.append('bob','old','assistant','欢迎小乙，您又来了。')
        self.answer=dict(action='visit_query',recipient_id='bob')
        await self.lane.run(self.request(text='小乙来过没？'))
        self.assertIn('不能据此说没有来过',self.spoken[-1])
        self.assertIsNone(self.lane.result['result']['last_interaction'])
        self.memory.append('bob','actual','user','你好。')
        await self.lane.run(self.request(text='小乙来过没？'))
        self.assertIn('有与小乙的交流记录',self.spoken[-1])
