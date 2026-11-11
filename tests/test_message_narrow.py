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

    async def test_interrupted_relay_never_marks_announced_and_complete_then_ack_is_durable(self):
        await self.save();row=self.store.list('all')[0]
        request=MessageRequest('bob','visit',None,None,None,'bob');self.current=request
        async def interrupted(*_):return False
        self.lane.speak=interrupted;await self.lane.relay(row['task_id'],request)
        self.assertEqual(self.store.get(row['task_id'])['status'],'pending')
        async def complete(*_):return True
        self.lane.speak=complete;await self.lane.relay(row['task_id'],request)
        self.assertEqual(self.store.get(row['task_id'])['status'],'announced')
        self.answer=dict(action='acknowledge');await self.lane.run(self.request('bob','收到了。'))
        fresh=RobotTasks(PersonMemory(self.config,self.tmp.name),self.policy)
        self.assertEqual(fresh.get(row['task_id'])['status'],'completed');self.assertFalse(fresh.due('bob'))

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

    async def test_ack_after_restart_can_bind_only_a_single_announced_message(self):
        await self.save();tid=self.store.list('all')[0]['task_id'];self.store.mark_announced(tid,'bob','previous-complete-audio')
        self.lane.offered.clear();self.lane.last.clear()
        self.answer=dict(action='acknowledge');await self.lane.run(self.request('bob','收到了。'))
        self.assertEqual(self.store.get(tid)['status'],'completed')

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
