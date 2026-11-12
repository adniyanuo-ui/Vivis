"""Actual 18:36/18:40 self-query and sender follow-up regressions."""
import unittest
from dataclasses import replace
import test_message_narrow as fixture
from vivis.interaction.message_lane import query_recipient_fields

PEOPLE={'alice':dict(display_name='测试甲',preferred_address='甲工'),
        'bob':dict(display_name='测试乙',preferred_address='小乙')}

class ScopeTests(unittest.TestCase):
    def fields(self,out,source):return query_recipient_fields(out,source,PEOPLE,dict(current_person_id='bob'))

    def test_implicit_self_is_not_missing_recipient(self):
        for text in ('还有别的留言吗？','还有其他留言吗','有留言吗？'):
            result=self.fields(dict(action='query',recipient_reference='我',status='active'),text)
            self.assertEqual(result,dict(action='query',recipient_id='bob',status='active'))

    def test_related_means_visibility_union_not_sender_and_recipient_intersection(self):
        out=dict(action='query',recipient_reference='我',creator_reference='我',recipient_id='bob',status='active')
        self.assertEqual(self.fields(out,'还有别的跟我有关的事吗？'),dict(action='query',status='active'))
        result=self.fields(out,'还有给我的留言吗？')
        self.assertEqual(result['recipient_id'],'bob');self.assertNotIn('creator_id',result)

    def test_sender_question_is_not_a_write_recipient_clarification(self):
        for text in ('哦，这是谁给我的留言？','谁留的？'):
            result=self.fields(dict(action='clarify',missing='recipient'),text)
            self.assertEqual(result,dict(action='query',recipient_id='bob',status='active',query_detail='sender'))

    def test_explicit_other_target_is_not_replaced_by_self(self):
        result=self.fields(dict(action='query',recipient_reference='甲工',recipient_id='alice'),'还有给甲工的留言吗？')
        self.assertEqual(result['recipient_id'],'alice')
        result=self.fields(dict(action='query',recipient_reference='小儿',recipient_id='alice'),'给小儿的留言呢？')
        self.assertEqual(result['action'],'clarify')

class DeliveryQueryTests(unittest.IsolatedAsyncioTestCase):
    setUp=fixture.FixedMessageFlowTests.setUp
    request=fixture.FixedMessageFlowTests.request
    save=fixture.FixedMessageFlowTests.save

    async def prepare(self):
        await self.save();row=self.store.list('all')[0]
        self.answer=dict(action='query',recipient_id='bob',status='active')
        return row['task_id']

    async def test_complete_inbox_read_counts_as_delivery_and_sender_followup_uses_that_record(self):
        tid=await self.prepare()
        await self.lane.run(self.request('bob','还有给我的留言吗？'))
        self.assertIn('甲工留言：',self.spoken[-1]);self.assertEqual(self.store.get(tid)['status'],'announced')
        self.assertFalse(self.store.due('bob'))
        await self.lane.run(self.request('bob','还有别的留言吗？'))
        self.assertIn('目前没有待转达',self.spoken[-1])
        self.answer=dict(action='query',recipient_id='bob',status='all',query_detail='sender')
        await self.lane.run(self.request('bob','这是谁给我的留言？'))
        self.assertIn('甲工给您的',self.spoken[-1]);self.assertNotIn('USB',self.spoken[-1])
        self.assertNotIn('给谁',self.spoken[-1])

    async def test_interrupted_inbox_read_remains_pending(self):
        tid=await self.prepare()
        async def interrupted(*_):return False
        self.lane.speak=interrupted
        await self.lane.run(self.request('bob','还有我的留言吗？'))
        self.assertEqual(self.store.get(tid)['status'],'pending');self.assertFalse(self.lane.readback)

    async def test_sender_question_alone_does_not_mark_delivery(self):
        tid=await self.prepare();self.answer.update(query_detail='sender')
        await self.lane.run(self.request('bob','谁给我留言？'))
        self.assertEqual(self.store.get(tid)['status'],'pending')

    async def test_sender_read_and_recovered_readonly_never_mark_recipient_delivery(self):
        tid=await self.prepare();await self.lane.run(self.request('alice','给小乙的留言保存了吗？'))
        self.assertEqual(self.store.get(tid)['status'],'pending')
        request=replace(self.request('bob','还有给我的留言吗？'),read_only=True);self.current=request
        await self.lane.run(request);self.assertEqual(self.store.get(tid)['status'],'pending')

    async def test_new_content_during_read_cannot_be_marked_delivered(self):
        tid=await self.prepare()
        async def supplement_during_read(*_):
            self.store.supplement(tid,'alice','visit','addition',None,'放前台')
            return True
        self.lane.speak=supplement_during_read
        await self.lane.run(self.request('bob','还有给我的留言吗？'))
        self.assertEqual(self.store.get(tid)['status'],'pending')
        self.assertIn('放前台',self.store.get(tid)['content'])

    async def test_identity_change_after_audio_prevents_delivery_commit(self):
        tid=await self.prepare()
        async def changed(*_):self.current=None;return True
        self.lane.speak=changed
        await self.lane.run(self.request('bob','还有给我的留言吗？'))
        self.assertEqual(self.store.get(tid)['status'],'pending')
