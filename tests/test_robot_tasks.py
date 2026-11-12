"""Durable work memory, attributed global recall and evidence-based relay state."""
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import asyncio
import base64
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from vivis.paths import ROOT
from vivis.interaction.memory import PersonMemory
from vivis.interaction.robot_tasks import RobotTasks,relay_matches,MessageMeaningUncertain


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

    def test_same_request_retry_does_not_reopen_acknowledged_message(self):
        task=self.create();self.tasks.acknowledge(task['task_id'],'bob','received')
        again=self.create()
        self.assertEqual(again['task_id'],task['task_id']);self.assertEqual(again['effect'],'already_acknowledged')
        self.assertEqual(len(self.tasks.list('all')),1);self.assertFalse(self.tasks.due('bob'))

    def test_new_request_does_not_scan_history_even_when_content_is_similar(self):
        task=self.create()
        other=self.tasks.create(dict(action='create',kind='message',title='新请求',content='请准备两份资料。',recipient_id='bob'),
            'alice','new-session','new-turn',None,self.people)
        self.assertNotEqual(task['task_id'],other['task_id']);self.assertEqual(len(self.tasks.list('all')),2)

    def test_explicit_supplement_is_idempotent_and_cannot_update_another_owner(self):
        task=self.create()
        changed=self.tasks.supplement(task['task_id'],'alice','s2','u2',None,'请带到前台。')
        again=self.tasks.supplement(task['task_id'],'alice','s2','u2',None,'请带到前台。')
        self.assertEqual(changed['content'],again['content']);self.assertEqual(changed['content'].count('前台'),1)
        with self.assertRaises(ValueError):self.tasks.supplement(task['task_id'],'bob','s','u3',None,'不属于我。')

    def test_authority_change_before_commit_rolls_back_the_entire_request(self):
        calls=[]
        def guard():
            calls.append(1)
            if len(calls)==2:raise ValueError('new speaker')
        with self.assertRaises(ValueError):
            self.tasks.create(dict(action='create',kind='message',title='测试',content='准备资料。',recipient_id='bob'),
                'alice','s','u',None,self.people,guard=guard)
        self.assertFalse(self.tasks.list('all'))
        with self.memory.db() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM message_writes').fetchone()[0],0)

    def test_acknowledgement_is_persistent_delivery_only_after_restart(self):
        task=self.create();result=self.tasks.acknowledge(task['task_id'],'bob','收到')
        self.assertEqual(result['delivery_state'],'acknowledged');self.assertFalse(result['physical_execution_tracked'])
        fresh=RobotTasks(PersonMemory(self.memory.c,self.tmp.name),self.tasks.c)
        self.assertEqual(fresh.get(task['task_id'])['status'],'completed');self.assertFalse(fresh.due('bob'))

    def test_known_ambiguous_source_cannot_be_silently_replaced_or_saved_without_clarification(self):
        args=dict(action='create',kind='message',title='采购',content='买一个U盘。',recipient_id='bob')
        with self.assertRaises(MessageMeaningUncertain):
            self.tasks.create(args,'alice','s','u',None,self.people,source_text='让小王买一个uber。')
        self.assertFalse(self.tasks.list('all'))
        result=self.tasks.create(args,'alice','s','u2',None,self.people,source_text='我说的不是Uber，是U盘。',clarified=True)
        self.assertEqual(result['content'],'买一个U盘。')

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
        self.assertFalse(self.tasks.due('bob'));self.assertFalse(self.tasks.list('active',recipient='bob'))
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

    def test_verified_message_visibility_keeps_requester_and_recipient_scopes(self):
        task=self.create()
        hidden=self.tasks.create(dict(action='create',kind='message',title='其他人的留言',content='OTHER_MESSAGE_CANARY',recipient_id='dave'),
            'carol','s','other',None,{'carol':{},'dave':{}})
        plan=self.tasks.create(dict(action='create',kind='todo',title='公共工作计划',content='整理资料'),
            'carol','s','plan',None,{})
        for person in ('alice','bob'):
            page=self.tasks.page(visible_to=person)
            self.assertEqual({t['task_id'] for t in page['tasks']},{task['task_id'],plan['task_id']})
            self.assertEqual(page['counts']['pending'],2)
            self.assertNotIn(hidden['content'],json.dumps(page))


class RobotTaskRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from test_cloud_control import CloudBackendTests
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.events=[]
        self.service=CloudBackendTests().service(self.tmp.name,self.events);self.addCleanup(self.service.log.close)
        s=self.service;s.memory=PersonMemory(read('configs/interaction/memory.json'),self.tmp.name)
        s.tasks=RobotTasks(s.memory,read('configs/interaction/tasks.json'));s.tasks.c=dict(s.tasks.c,auto_announce=False)
        s.knowledge.people={'alice':dict(display_name='测试甲',preferred_address='甲工'),'bob':dict(display_name='测试乙',preferred_address='小乙')}
        s.context.update(revision=1,mode='interaction',target='interaction',ready=True)
        s.enabled=True;s.presence.wake();self.person='alice';s.identified_person=lambda:self.person
        self.answers=dict(action='create',recipient_id='bob',content='请准备两份资料。');self.spoken=[]
        async def extract(*_):return dict(self.answers)
        async def speak(text,request):self.spoken.append(text);return True
        s.extract_message=extract;s.speak_message=speak
        self.sent=[]
        async def send(text):self.sent.append(json.loads(text))
        self.ws=SimpleNamespace(send=send);s.cloud_state=SimpleNamespace(interrupt=Mock(),generating=None)

    def final(self,text='给小乙留言，请准备两份资料。'):
        s=self.service;s.input_generation+=1;s.begin_user_turn('turn'+str(s.input_generation));s.turn_ended_at=time.monotonic()
        event=dict(type='conversation.item.input_audio_transcription.completed',item_id=s.turn_item_id,transcript=text)
        s.remember(event);return event

    async def finish(self):
        await asyncio.wait_for(self.service.messages().task,2.)

    async def test_final_text_is_the_only_start_and_duplicate_final_does_not_save_twice(self):
        s=self.service;s.begin_user_turn('early')
        s.final_message(dict(type='conversation.item.input_audio_transcription.delta',item_id='early',text='给小乙'),s.cloud_state)
        self.assertIsNone(s.message_lane)
        event=self.final();s.final_message(event,s.cloud_state);await self.finish()
        s.final_message(event,s.cloud_state)
        self.assertEqual(len(s.tasks.list('all')),1);self.assertEqual(len(self.spoken),1)
        self.assertEqual(self.spoken[0],'已保存，等见到小乙后转告。')
        self.assertFalse(hasattr(s,'source_waiters'))

    async def test_an_early_chat_tool_can_never_write_or_queue_a_message(self):
        s=self.service;s.response_revisions['r']=1
        await s.tool(self.ws,dict(name='manage_robot_task',call_id='early',response_id='r',arguments=json.dumps(dict(action='create',kind='message',title='测试',content='准备资料。',recipient_id='bob'))))
        self.assertFalse(json.loads(self.sent[-1]['item']['output'])['accepted']);self.assertFalse(s.tasks.list('all'))

    async def test_changed_speaker_invalidates_extraction_before_write(self):
        s=self.service;started=asyncio.Event();end=asyncio.Event()
        async def extract(*_):started.set();await end.wait();return dict(self.answers)
        s.extract_message=extract;s.final_message(self.final(),s.cloud_state)
        await started.wait();self.person='bob';end.set();await self.finish()
        self.assertFalse(s.tasks.list('all'));self.assertFalse(self.spoken)

    async def test_readback_failure_does_not_speak_success_or_claim_rollback(self):
        s=self.service
        with patch.object(s.tasks,'get',side_effect=RuntimeError('readback failed after commit')):
            s.final_message(self.final(),s.cloud_state);await self.finish()
        self.assertEqual(len(s.tasks.list('all')),1)
        self.assertIn('尚未确认',self.spoken[-1]);self.assertNotIn('已保存',self.spoken[-1])
        self.assertIsNone(s.messages().result['saved'])

    async def test_automatic_relay_marks_announced_only_after_complete_playback(self):
        s=self.service;s.final_message(self.final(),s.cloud_state);await self.finish()
        tid=s.tasks.list('all')[0]['task_id'];self.person='bob';s.tasks.c=dict(s.tasks.c,auto_announce=True)
        s.continue_response=False;s.announcement_revision=None
        started=asyncio.Event();end=asyncio.Event()
        async def speak(*_):started.set();await end.wait();return True
        s.messages().speak=speak;s.schedule_task_notice(SimpleNamespace(generating=None))
        await started.wait();self.assertEqual(s.tasks.get(tid)['status'],'pending')
        s.schedule_task_notice(SimpleNamespace(generating=None))  # No duplicate notice while busy.
        end.set();await self.finish()
        self.assertEqual(s.tasks.get(tid)['status'],'announced');self.assertIn(tid,s.task_visit_announced)

    async def test_unknown_self_report_cannot_query_private_tasks_or_global_history(self):
        s=self.service;s.final_message(self.final(),s.cloud_state);await self.finish()
        self.person=None;s.response_revisions['r']=1
        await s.tool(self.ws,dict(name='get_robot_status',call_id='status',response_id='r',arguments='{}'))
        output=json.loads(self.sent[-1]['item']['output'])
        self.assertFalse(output['identity_verified']);self.assertFalse(output['robot_tasks']['active'])
        await s.tool(self.ws,dict(name='search_robot_memory',call_id='private',response_id='r',arguments='{"keywords":[]}'))
        self.assertFalse(json.loads(self.sent[-1]['item']['output'])['accepted'])

    async def test_current_database_status_says_saved_pending_not_waiting_for_verification(self):
        s=self.service;s.final_message(self.final(),s.cloud_state);await self.finish();s.response_revisions['r']=1
        await s.tool(self.ws,dict(name='get_robot_status',call_id='status',response_id='r',arguments='{}'))
        output=json.loads(self.sent[-1]['item']['output']);row=output['robot_tasks']['active'][0]
        self.assertTrue(row['saved']);self.assertEqual(row['delivery_label'],'已保存，等待转告')
        self.assertEqual(output['message_operation']['operation_status'],'saved')

    async def test_model_receive_loop_waits_for_final_text_and_uses_only_trusted_result_speech(self):
        from test_managed_voice import FakeAudio
        s=self.service;s.audio=FakeAudio();s.verify_identity=Mock();s.reception.c=dict(s.reception.c,enabled=False)
        s.announcement_revision=None;messages=asyncio.Queue()
        class WS:
            async def __aenter__(inner):return inner
            async def __aexit__(inner,*_):pass
            async def send(inner,text):self.sent.append(json.loads(text))
            def __aiter__(inner):return inner
            async def __anext__(inner):return json.dumps(await messages.get())
        async def connect(_ws,session,_log):return session
        with patch('vivis.speech.cloud_managed.websocket_connection',return_value=WS()),patch('vivis.speech.cloud_managed.connect_session',side_effect=connect):
            task=asyncio.create_task(s._connection())
            async def until(predicate):
                end=time.monotonic()+2
                while not predicate():
                    if task.done():task.result()
                    if time.monotonic()>end:raise AssertionError('fixed-flow receive timeout')
                    await asyncio.sleep(.01)
            try:
                await until(lambda:s.ws is not None)
                await messages.put(dict(type='input_audio_buffer.speech_started',item_id='final-only'))
                await messages.put(dict(type='input_audio_buffer.speech_stopped',item_id='final-only'))
                await messages.put(dict(type='response.created',response=dict(id='chat-claim')))
                await messages.put(dict(type='response.audio.delta',response_id='chat-claim',delta=base64.b64encode(b'\1\0'*480).decode()))
                await asyncio.sleep(.03);self.assertFalse(s.tasks.list('all'));self.assertEqual(s.player.count,0)
                await messages.put(dict(type='conversation.item.input_audio_transcription.completed',item_id='final-only',transcript='给小乙留言，请准备两份资料。'))
                await until(lambda:bool(self.spoken))
                self.assertEqual(len(s.tasks.list('all')),1);self.assertEqual(s.player.count,0)
                self.assertEqual(self.spoken[0],'已保存，等见到小乙后转告。')
            finally:task.cancel();await asyncio.gather(task,return_exceptions=True)

    async def test_user_speaking_response_race_is_recoverable_without_replaying(self):
        s=self.service;s.last_response_request_at=time.monotonic();s.response_request_pending=True
        event=dict(type='error',error=dict(code='invalid_value',param='response.create',message='Cannot create response while user is speaking.'))
        self.assertTrue(s.response_busy(event,SimpleNamespace(generating=None)));self.assertTrue(s.response_waiting_user_end)
        self.assertFalse(s.continue_response);self.assertFalse(s.response_request_pending)


if __name__=='__main__':unittest.main()
