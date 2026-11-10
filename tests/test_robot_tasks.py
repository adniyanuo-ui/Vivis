"""Durable work memory, attributed global recall and evidence-based relay state."""
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vivis.paths import ROOT
from vivis.interaction.memory import PersonMemory
from vivis.interaction.robot_tasks import RobotTasks,relay_matches


def read(name):return json.loads((ROOT/name).read_text())


class RobotTaskStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.memory=PersonMemory(read('configs/interaction/memory.json'),self.tmp.name)
        self.tasks=RobotTasks(self.memory,read('configs/interaction/tasks.json'))
        self.people={'alice':{},'bob':{}}

    def create(self,**extra):
        args=dict(action='create',kind='message',title='测试转告',content='请准备两份资料。',recipient_id='bob',**extra)
        return self.tasks.create(args,'alice','session','utterance',None,self.people)

    def test_task_and_plan_survive_new_store_without_relying_on_chat_summary(self):
        task=self.create(next_action='明天见到小王后转告')
        fresh=RobotTasks(PersonMemory(self.memory.c,self.tmp.name),self.tasks.c)
        self.assertEqual(fresh.get(task['task_id'])['next_action'],'明天见到小王后转告')
        self.assertEqual(fresh.list()[0]['status'],'pending')
        self.assertEqual(len(fresh.due('bob')),1);self.assertFalse(fresh.due('alice'))
        self.assertEqual(self.memory.path.stat().st_mode&0o777,0o600)

    def test_next_day_task_survives_a_fresh_python_process(self):
        task=self.create(not_before='2026-11-11T09:00:00+08:00')
        code='''import json,sys
from vivis.interaction.memory import PersonMemory
from vivis.interaction.robot_tasks import RobotTasks
m=PersonMemory(json.loads(sys.argv[1]),sys.argv[2]);t=RobotTasks(m,json.loads(sys.argv[3]))
print(json.dumps(t.due('bob','2026-11-11T09:00:01+08:00')))
'''
        result=subprocess.run([str(ROOT/'.venv/bin/python'),'-c',code,json.dumps(self.memory.c),self.tmp.name,json.dumps(self.tasks.c)],
            capture_output=True,text=True,check=True,env={**os.environ,'PYTHONPATH':str(ROOT/'src')})
        self.assertEqual(json.loads(result.stdout)[0]['task_id'],task['task_id'])

    def test_duplicate_current_turn_does_not_create_two_tasks_and_source_can_bind_later(self):
        task=self.create();again=self.create()
        self.assertEqual(task['task_id'],again['task_id']);self.assertEqual(len(self.tasks.list()),1)
        mid=self.memory.append('alice','session','user','请转告小王准备资料。')
        self.tasks.bind_source('session','utterance',mid)
        self.assertEqual(self.tasks.get(task['task_id'])['source_message_id'],mid)

    def test_due_dates_have_timezone_and_respect_next_day_gate(self):
        task=self.create(not_before='2026-11-11T08:00:00+08:00')
        self.assertFalse(self.tasks.due('bob','2026-11-11T07:59:59+08:00'))
        self.assertEqual(self.tasks.due('bob','2026-11-11T08:00:01+08:00')[0]['task_id'],task['task_id'])
        for extra in (dict(not_before='明早'),dict(not_before='2026-11-11T08:00:00'),dict(recipient_id='missing'),dict(recipient_id=[])):
            args=dict(action='create',kind='message',title='测试',content='具体内容。',recipient_id='bob');args.update(extra)
            with self.assertRaises(ValueError):self.tasks.create(args,'alice','s','u',None,self.people)

    def test_audio_playback_is_not_recipient_confirmation_or_physical_completion(self):
        task=self.create();tid=task['task_id']
        with self.assertRaises(ValueError):self.tasks.prepare_relay(tid,'alice')
        self.tasks.prepare_relay(tid,'bob');self.tasks.mark_announced(tid,'bob','response')
        self.assertEqual(self.tasks.get(tid)['status'],'announced')
        with self.assertRaises(ValueError):self.tasks.update(dict(action='update',task_id=tid,status='completed'),'alice','u')
        self.tasks.update(dict(action='update',task_id=tid,status='completed'),'bob','ack-user-turn')
        self.assertEqual(len(self.tasks.list('completed')),1);self.assertFalse(self.tasks.due('bob'))
        with self.assertRaises(ValueError):self.tasks.update(dict(action='update',task_id=tid,status='pending'),'bob','u')

    def test_cancelled_and_future_tasks_do_not_relay_and_last_attempt_throttles_restart(self):
        task=self.create();tid=task['task_id']
        self.tasks.prepare_relay(tid,'bob')
        self.assertFalse(RobotTasks(PersonMemory(self.memory.c,self.tmp.name),self.tasks.c).due('bob'))
        self.tasks.update(dict(action='update',task_id=tid,status='cancelled'),'alice','cancel')
        with self.assertRaises(ValueError):self.tasks.prepare_relay(tid,'bob')

    def test_global_history_is_attributed_paginated_and_includes_anonymous_without_relabelling(self):
        self.memory.append('bob','old','user','资料是蓝色的。')
        for i in range(12):self.memory.append('alice','s','user','全局检索'+str(i))
        self.memory.append('anonymous:'+'a'*32,'visit','user','全局检索匿名记录')
        first=self.memory.search_all(['全局检索'])
        self.assertTrue(first['has_more']);self.assertFalse(first['matches'][0]['speaker_confirmed'])
        second=self.memory.search_all(['全局检索'],before_id=first['next_before_id'])
        self.assertFalse({r['id'] for r in first['matches']}&{r['id'] for r in second['matches']})
        self.assertTrue(any(r['person']=='alice' for r in first['matches']))
        self.assertEqual(self.memory.search_all(['蓝色'])['matches'][0]['person'],'bob')
        self.assertNotIn('蓝色',self.memory.context('alice'))  # Existing per-person injection still attributed.

    def test_global_search_escapes_wildcards_and_relay_requires_complete_content(self):
        self.memory.append('alice','s','user','进度50%_完成')
        self.memory.append('bob','s','user','进度5000完成')
        self.assertEqual(len(self.memory.search_all(['50%_'])['matches']),1)
        self.assertTrue(relay_matches('请准备两份资料。','何工说：请准备两份资料！收到吗？'))
        self.assertFalse(relay_matches('请准备两份资料。','请准备资料。'))

    def test_task_pages_and_counts_do_not_hide_older_or_newer_work(self):
        self.tasks.c=dict(self.tasks.c,list_limit=2)
        for i in range(5):
            self.tasks.create(dict(action='create',kind='todo',title='计划'+str(i),content='任务'+str(i)),'alice','s','u'+str(i),None,self.people)
        first=self.tasks.page();second=self.tasks.page(after_task_id=first['next_after_task_id'])
        third=self.tasks.page(after_task_id=second['next_after_task_id'])
        self.assertEqual(sum(len(page['tasks']) for page in (first,second,third)),5)
        self.assertEqual(first['counts']['pending'],5);self.assertFalse(third['has_more'])


class RobotTaskRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from test_cloud_control import CloudBackendTests
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.events=[];self.service=CloudBackendTests().service(self.tmp.name,self.events);self.addCleanup(self.service.log.close)
        self.service.memory=PersonMemory(read('configs/interaction/memory.json'),self.tmp.name)
        self.service.tasks=RobotTasks(self.service.memory,read('configs/interaction/tasks.json'))
        self.service.knowledge.people={'alice':dict(display_name='测试甲',preferred_address='甲工'),'bob':dict(display_name='测试乙',preferred_address='小王')}
        self.service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
        self.service.response_revisions['r']=1;self.service.enabled=True;self.service.presence.wake()
        self.service.turn_person='alice';self.service.turn_item_id='human-turn';self.service.turn_ended_at=time.monotonic()
        self.service.identified_person=lambda:'alice'
        self.sent=[]
        async def send(s):self.sent.append(json.loads(s))
        self.ws=SimpleNamespace(send=send)

    async def tool(self,args,call='task'):
        await self.service.tool(self.ws,dict(call_id=call,name='manage_robot_task',response_id='r',arguments=json.dumps(args)))
        return json.loads(self.sent[-1]['item']['output'])

    async def create(self):
        result=await self.tool(dict(action='create',kind='message',title='准备资料',content='请准备两份资料。',recipient_id='bob'))
        self.assertTrue(result['accepted']);return result['result']

    async def test_create_requires_current_user_not_historical_context_and_update_has_actor(self):
        task=await self.create();self.assertEqual(task['creator'],'alice')
        self.service.turn_item_id=None
        bad=await self.tool(dict(action='update',task_id=task['task_id'],status='cancelled'),'historical')
        self.assertFalse(bad['accepted']);self.assertEqual(self.service.tasks.get(task['task_id'])['status'],'pending')

    async def test_speaker_change_cannot_reuse_another_persons_current_turn(self):
        self.service.identified_person=lambda:'bob'
        result=await self.tool(dict(action='create',kind='message',title='测试',content='请准备资料。',recipient_id='bob'))
        self.assertFalse(result['accepted']);self.assertFalse(self.service.tasks.list())

    async def test_recipient_lookup_after_rejected_guess_can_continue_and_create_once(self):
        bad=await self.tool(dict(action='create',kind='message',title='测试',content='请准备资料。',recipient_id='guessed'))
        self.assertFalse(bad['accepted']);self.assertIsNotNone(self.service.tool_repair)
        await self.service.tool(self.ws,dict(call_id='status',name='get_robot_status',response_id='r',arguments='{}'))
        self.assertIsNone(self.service.tool_repair);self.assertTrue(self.service.continue_response)
        result=await self.tool(dict(action='create',kind='message',title='测试',content='请准备资料。',recipient_id='bob'),'corrected')
        self.assertTrue(result['accepted']);self.assertEqual(len(self.service.tasks.list()),1)

    async def test_wrong_person_cannot_prepare_and_unplayed_text_does_not_mark_announced(self):
        task=await self.create()
        wrong=await self.tool(dict(action='prepare_relay',task_id=task['task_id']),'wrong')
        self.assertFalse(wrong['accepted'])
        self.service.identified_person=lambda:'bob';self.service.tool_repair=None
        result=await self.tool(dict(action='prepare_relay',task_id=task['task_id']),'right')
        self.assertTrue(result['accepted'])
        relay=self.service.relay_pending[task['task_id']];relay['response_id']='spoken'
        state=SimpleNamespace(blocked=set(),responses={'spoken':dict(status='completed',transcript=task['content'],received_audio_s=1,empty_dac=0)})
        self.service.check_task_delivery(state)
        self.assertEqual(self.service.tasks.get(task['task_id'])['status'],'pending')
        self.service.player.consumed['spoken']=48000
        self.service.check_task_delivery(state)
        self.assertEqual(self.service.tasks.get(task['task_id'])['status'],'announced')

    async def test_incomplete_speech_cancellation_or_identity_change_cannot_claim_delivery(self):
        task=await self.create();self.service.identified_person=lambda:'bob'
        await self.tool(dict(action='prepare_relay',task_id=task['task_id']),'prepare')
        relay=self.service.relay_pending[task['task_id']];relay['response_id']='spoken'
        self.service.player.consumed['spoken']=48000
        state=SimpleNamespace(blocked=set(),responses={'spoken':dict(status='completed',transcript='请准备资料',received_audio_s=1,empty_dac=0)})
        self.service.check_task_delivery(state)
        self.assertEqual(self.service.tasks.get(task['task_id'])['status'],'pending')
        self.service.tool_repair=None
        await self.tool(dict(action='prepare_relay',task_id=task['task_id']),'prepare2')
        self.service.identified_person=lambda:'alice'
        self.service.check_task_delivery(state)
        self.assertEqual(self.service.tasks.get(task['task_id'])['status'],'pending')

    async def test_due_inbox_triggers_one_notice_per_visit_and_global_search_is_readonly(self):
        task=await self.create();service=self.service;service.identified_person=lambda:'bob'
        service.continue_response=False;service.announcement_revision=None
        service.schedule_task_notice(SimpleNamespace(generating=None))
        self.assertEqual(service.announcement_kind,'tasks');self.assertIn(task['task_id'],service.task_notice_attempts)
        self.assertNotIn(task['task_id'],service.task_visit_announced)
        service.announcement_revision=None;service.schedule_task_notice(SimpleNamespace(generating=None))
        self.assertIsNone(service.announcement_revision)
        service.memory.append('alice','previous','user','请给小王准备资料。')
        await service.tool(self.ws,dict(call_id='search',response_id='r',name='search_robot_memory',arguments=json.dumps(dict(keywords=['小王']))))
        result=json.loads(self.sent[-1]['item']['output'])
        self.assertTrue(result['accepted']);self.assertEqual(result['result']['matches'][0]['person'],'alice')

    async def test_only_known_response_create_race_is_coalesced_and_pending_request_blocks_new_notice(self):
        service=self.service;service.last_response_request_at=time.monotonic()
        event=dict(type='error',error=dict(code='invalid_value',param='response.create',message='Cannot create response while another response is in progress.'))
        self.assertTrue(service.response_busy(event,SimpleNamespace(generating='existing')))
        self.assertFalse(service.response_request_pending)
        wrong=dict(type='error',error=dict(code='invalid_value',param='response.create',message='conversation has no messages'))
        self.assertFalse(service.response_busy(wrong,SimpleNamespace(generating=None)))
        task=await self.create();service.identified_person=lambda:'bob'
        service.response_request_pending=True;service.continue_response=False;service.announcement_revision=None
        service.schedule_task_notice(SimpleNamespace(generating=None))
        self.assertIsNone(service.announcement_revision)


if __name__=='__main__':unittest.main()
