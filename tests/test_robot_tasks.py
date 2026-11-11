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
from vivis.interaction.robot_tasks import RobotTasks,relay_matches,ReconciliationRequired,MessageMeaningUncertain


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

    def test_exact_restatement_survives_sessions_and_does_not_reopen_acknowledged_message(self):
        task=self.create()
        self.tasks.acknowledge(task['task_id'],'bob','received')
        again=self.tasks.create(dict(action='create',kind='message',title='换个标题',content='请准备两份资料！',recipient_id='bob'),
            'alice','another-session','another-turn',None,self.people)
        self.assertEqual(again['task_id'],task['task_id']);self.assertEqual(again['effect'],'already_acknowledged')
        self.assertEqual(len(self.tasks.list('all')),1);self.assertFalse(self.tasks.due('bob'))

    def test_semantic_restatement_requires_review_then_merges_complete_content(self):
        task=self.create()
        args=dict(action='create',kind='message',title='补充',content='请准备两份资料，并带到会议室。',recipient_id='bob')
        with self.assertRaises(ReconciliationRequired) as caught:
            self.tasks.create(args,'alice','s2','u2',None,self.people)
        self.assertEqual(caught.exception.candidates[0]['task_id'],task['task_id'])
        result=self.tasks.create(dict(args,merge_task_id=task['task_id'],merge_reason='同一委托的地点补充'),
            'alice','s2','u2',None,self.people)
        self.assertEqual(result['task_id'],task['task_id']);self.assertEqual(result['content'],args['content'])
        self.assertEqual(len(self.tasks.list('all')),1)

    def test_distinct_messages_need_explicit_semantic_review_and_cannot_merge_across_recipients(self):
        task=self.create();args=dict(action='create',kind='message',title='另件',content='下午请带钥匙。',recipient_id='bob')
        other=self.tasks.create(dict(args,distinct_from=[task['task_id']],merge_reason='带钥匙与准备资料是不同委托'),
            'alice','s2','u2',None,self.people)
        self.assertNotEqual(task['task_id'],other['task_id'])
        with self.assertRaisesRegex(ValueError,'same recipient'):
            self.tasks.create(dict(args,recipient_id='alice',merge_task_id=task['task_id'],merge_reason='错误对象'),
                'alice','s3','u3',None,self.people)

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
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.events=[];self.service=CloudBackendTests().service(self.tmp.name,self.events);self.addCleanup(self.service.log.close)
        self.service.memory=PersonMemory(read('configs/interaction/memory.json'),self.tmp.name)
        self.service.tasks=RobotTasks(self.service.memory,read('configs/interaction/tasks.json'))
        self.service.knowledge.people={'alice':dict(display_name='测试甲',preferred_address='甲工'),'bob':dict(display_name='测试乙',preferred_address='小王')}
        self.service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
        self.service.response_revisions['r']=1;self.service.enabled=True;self.service.presence.wake()
        self.service.turn_person='alice';self.service.turn_item_id='human-turn';self.service.turn_ended_at=time.monotonic()
        self.service.pending_user_text='请给小王留言，准备两份资料。'
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

    async def test_fresh_receipt_ack_can_close_message_while_old_relay_guard_remains(self):
        task=await self.create();service=self.service
        service.identified_person=lambda:'bob';service.turn_person='bob'
        service.turn_item_id='recipient-ack';service.turn_ended_at=time.monotonic()
        service.relay_pending[task['task_id']]=dict(recipient='bob',content=task['content'],revision=1,response_id=None)
        result=await self.tool(dict(action='acknowledge',task_id=task['task_id']),'ack')
        self.assertTrue(result['accepted']);self.assertTrue(result['durable'])
        self.assertIn('收到',result['spoken_receipt']);self.assertNotIn('办完',result['spoken_receipt'])
        self.assertEqual(service.tasks.get(task['task_id'])['status'],'completed')
        self.assertFalse(service.tasks.due('bob'));self.assertFalse(service.relay_pending)

    async def test_late_relay_request_from_prior_user_turn_cannot_reinstall_guard(self):
        task=await self.create();service=self.service;service.identified_person=lambda:'bob'
        service.response_inputs['r']=0;service.input_generation=1
        result=await self.tool(dict(action='prepare_relay',task_id=task['task_id']),'late-relay')
        self.assertFalse(result['accepted']);self.assertEqual(result['status'],'stale_user_turn')
        self.assertFalse(service.relay_pending);self.assertIsNone(service.tasks.get(task['task_id'])['last_attempt'])

    async def test_failed_commit_rolls_back_and_returns_only_failure_receipt(self):
        task=await self.create();service=self.service;service.identified_person=lambda:'bob';service.turn_person='bob'
        with patch.object(service.tasks,'event',side_effect=sqlite3.OperationalError('simulated commit failure')):
            result=await self.tool(dict(action='acknowledge',task_id=task['task_id']),'failed-write')
        self.assertFalse(result['accepted']);self.assertIn('操作失败',result['spoken_receipt'])
        self.assertEqual(service.tasks.get(task['task_id'])['status'],'pending')

    async def test_physical_work_tracking_is_not_exposed(self):
        result=await self.tool(dict(action='create',kind='todo',title='采购',content='买资料。'),'no-todo')
        self.assertFalse(result['accepted']);self.assertFalse(self.service.tasks.list('all'))

    async def test_ambiguous_object_is_forced_to_clarify_even_if_model_attempts_create(self):
        service=self.service;service.pending_user_text='让小王帮我买一个uber。'
        result=await self.tool(dict(action='create',kind='message',title='采购',content='买一个Uber。',recipient_id='bob'),'ambiguous')
        self.assertFalse(result['accepted']);self.assertEqual(result['status'],'meaning_clarification_required')
        self.assertIn('还没有保存',result['spoken_receipt']);self.assertFalse(service.tasks.list('all'))

    async def test_mutation_waits_for_final_source_transcript_instead_of_guessing(self):
        service=self.service;service.begin_user_turn('human-turn');service.turn_ended_at=time.monotonic()
        args=dict(action='create',kind='message',title='资料',content='准备两份资料。',recipient_id='bob')
        event=dict(call_id='before-asr',name='manage_robot_task',response_id='r',arguments=json.dumps(args))
        await service.tool(self.ws,event)
        await service.tool(self.ws,event)  # Duplicate network event is not a second job.
        self.assertFalse(self.sent);self.assertEqual(len(service.source_waiters),1)
        self.assertTrue(service.message_source_waiting);self.assertFalse(service.tasks.list('all'))
        service.remember(dict(type='conversation.item.input_audio_transcription.completed',item_id='human-turn',transcript='给小王留一句，准备两份资料。'))
        await service.resume_source_tools(self.ws)
        self.assertEqual(self.sent[-1]['item']['call_id'],'before-asr')
        result=json.loads(self.sent[-1]['item']['output'])
        self.assertTrue(result['accepted']);self.assertEqual(len(service.tasks.list('all')),1)
        self.assertEqual(service.tasks.list('all')[0]['source_message_id'],service.turn_message_id)
        await service.resume_source_tools(self.ws);self.assertEqual(len(self.sent),1)
        self.assertFalse(service.message_source_waiting)

    async def test_deferred_source_cannot_write_after_new_turn_or_mode_change(self):
        for change in ('turn','mode','person'):
            s=self.service;s.pending_user_text=None;s.turn_item_id='original-'+change
            args=dict(action='create',kind='message',title='资料',content='准备资料。',recipient_id='bob')
            await s.tool(self.ws,dict(call_id='pending-'+change,name='manage_robot_task',response_id='r',arguments=json.dumps(args)))
            if change=='turn':s.input_generation+=1
            elif change=='mode':s.context['revision']+=1
            else:s.turn_person='bob'
            s.pending_user_text='新话轮不能授权旧请求。'
            await s.resume_source_tools(self.ws)
            self.assertEqual(json.loads(self.sent[-1]['item']['output'])['status'],'stale_user_turn')
            self.assertFalse(s.tasks.list('all'))
            s.context['revision']=1;s.response_revisions['r']=1;s.turn_person='alice'

    async def test_source_timeout_reports_not_saved_without_requesting_full_repetition(self):
        s=self.service;s.pending_user_text=None
        args=dict(action='create',kind='message',title='资料',content='准备资料。',recipient_id='bob')
        await s.tool(self.ws,dict(call_id='pending',name='manage_robot_task',response_id='r',arguments=json.dumps(args)))
        s.source_waiters['pending']['deadline']=0
        await s.resume_source_tools(self.ws)
        result=json.loads(self.sent[-1]['item']['output'])
        self.assertEqual(result['status'],'source_transcript_timeout');self.assertIn('尚未保存',result['spoken_receipt'])
        self.assertFalse(s.tasks.list('all'));self.assertFalse(s.source_waiters)

    async def test_semantic_review_does_not_turn_internal_check_into_a_question(self):
        s=self.service
        first=await self.create()
        result=await self.tool(dict(action='create',kind='message',title='采购',content='买一个USB扩展坞。',recipient_id='bob'),'different')
        self.assertEqual(result['status'],'semantic_reconciliation_required')
        self.assertNotIn('spoken_receipt',result);self.assertEqual(s.message_receipts.pending['kind'],'needs_tool')
        self.assertEqual(s.message_reconciliation['request']['content'],'买一个USB扩展坞。')
        result=await self.tool(dict(action='create',kind='message',title='采购',content='买一个USB扩展坞。',recipient_id='bob',
            distinct_from=[first['task_id']],merge_reason='资料准备和购买扩展坞是不同事项。'),'reviewed')
        self.assertTrue(result['accepted']);self.assertEqual(len(s.tasks.list('all')),2)
        self.assertFalse(s.message_reconciliation)

    async def test_unknown_identity_can_be_asked_to_clarify_without_reading_or_writing_tasks(self):
        self.service.identified_person=lambda:None
        result=await self.tool(dict(action='clarify',clarify_topic='meaning'),'clarify-unknown')
        self.assertTrue(result['accepted']);self.assertFalse(result['durable'])
        self.assertIn('没有听清',result['spoken_receipt']);self.assertIn('还没有保存',result['spoken_receipt'])
        self.assertFalse(self.service.tasks.list('all'))

    async def test_actual_receive_loop_holds_false_success_and_plays_verified_database_receipt(self):
        from test_managed_voice import FakeAudio
        from vivis.speech.message_receipts import RECEIPTS
        service=self.service;service.audio=FakeAudio();service.conversation_person='alice'
        service.presence.visible=Mock(return_value=True);service.announcement_revision=None
        service.reception.c=dict(service.reception.c,enabled=False)
        messages=asyncio.Queue()
        class WS:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            async def send(self,_):pass
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await messages.get())
        async def connect(_ws,session,_log):return session
        with patch('vivis.speech.cloud_managed.websocket_connection',return_value=WS()), \
             patch('vivis.speech.cloud_managed.connect_session',side_effect=connect):
            task=asyncio.create_task(service._connection())
            async def until(predicate):
                deadline=time.monotonic()+2
                while not predicate():
                    if task.done():task.result()
                    if time.monotonic()>deadline:raise AssertionError('Receipt integration timed out')
                    await asyncio.sleep(.01)
            try:
                await until(lambda:service.ws is not None)
                service.last_response_request_at=time.monotonic()
                await messages.put(dict(type='error',error=dict(type='invalid_request_error',code='invalid_value',
                    param='response.create',message='Cannot create response while user is speaking.')))
                await until(lambda:service.response_waiting_user_end)
                self.assertFalse(task.done());self.assertFalse(service.shutdown.is_set())
                await messages.put(dict(type='input_audio_buffer.speech_started',item_id='fresh-message'))
                await messages.put(dict(type='input_audio_buffer.speech_stopped',item_id='fresh-message'))
                await until(lambda:not service.response_waiting_user_end)
                # Reproduce the real failure: model promises without invoking
                # ANY tool, and the final ASR text arrives after its reply.
                await messages.put(dict(type='response.created',response=dict(id='skipped-tool')))
                pcm=base64.b64encode(b'\1\0'*480).decode()
                await messages.put(dict(type='response.audio.delta',response_id='skipped-tool',delta=pcm))
                await messages.put(dict(type='response.audio_transcript.done',response_id='skipped-tool',transcript='我记下了，会告诉小王的。'))
                await messages.put(dict(type='conversation.item.input_audio_transcription.completed',item_id='fresh-message',transcript='请给小王留言，让他准备两份资料。'))
                await until(lambda:'skipped-tool' in service.blocked)
                self.assertEqual(service.player.count,0);self.assertFalse(service.tasks.list('all'))
                await messages.put(dict(type='response.done',response=dict(id='skipped-tool',status='completed')))
                await messages.put(dict(type='input_audio_buffer.speech_started',item_id='late-source'))
                await messages.put(dict(type='input_audio_buffer.speech_stopped',item_id='late-source'))
                await messages.put(dict(type='response.created',response=dict(id='create')))
                args=dict(action='create',kind='message',title='测试留言',content='请准备两份资料。',recipient_id='bob')
                await messages.put(dict(type='response.function_call_arguments.done',response_id='create',name='manage_robot_task',call_id='write',arguments=json.dumps(args)))
                await until(lambda:'write' in service.source_waiters)
                self.assertFalse(service.tasks.list('all'))
                await messages.put(dict(type='conversation.item.input_audio_transcription.completed',item_id='late-source',transcript='给小王留言，请准备两份资料。'))
                await until(lambda:len(service.tasks.list('all'))==1)
                self.assertFalse(service.source_waiters)
                await messages.put(dict(type='response.done',response=dict(id='create',status='completed')))
                await messages.put(dict(type='response.created',response=dict(id='false-receipt')))
                await messages.put(dict(type='response.audio.delta',response_id='false-receipt',delta=pcm))
                await asyncio.sleep(.05);self.assertEqual(service.player.count,0)
                await messages.put(dict(type='response.audio_transcript.done',response_id='false-receipt',transcript='好的，事情我已经办完了。'))
                await until(lambda:'false-receipt' in service.blocked)
                self.assertEqual(service.player.count,0)
                await messages.put(dict(type='response.done',response=dict(id='false-receipt',status='completed')))
                await messages.put(dict(type='response.created',response=dict(id='verified-receipt')))
                await messages.put(dict(type='response.audio.delta',response_id='verified-receipt',delta=pcm))
                await asyncio.sleep(.05);self.assertEqual(service.player.count,0)
                await messages.put(dict(type='response.audio_transcript.done',response_id='verified-receipt',transcript=RECEIPTS['created']))
                await until(lambda:service.player.count>0)
                self.assertEqual(service.tasks.list('all')[0]['status'],'pending')
                self.assertFalse(any(e.get('role')=='assistant' and '办完' in e.get('text','') for e in self.events))
            finally:
                task.cancel();await asyncio.gather(task,return_exceptions=True)

    async def test_recipient_lookup_after_rejected_guess_can_continue_and_create_once(self):
        bad=await self.tool(dict(action='create',kind='message',title='测试',content='请准备资料。',recipient_id='guessed'))
        self.assertFalse(bad['accepted']);self.assertIn('没有保存',bad['spoken_receipt'])
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
        state=SimpleNamespace(interrupt=Mock(),blocked=set(),responses={'spoken':dict(status='completed',transcript='请准备资料',received_audio_s=1,empty_dac=0)})
        self.service.check_task_delivery(state)
        self.assertEqual(self.service.tasks.get(task['task_id'])['status'],'pending')
        self.service.tool_repair=None
        await self.tool(dict(action='prepare_relay',task_id=task['task_id']),'prepare2')
        self.service.identified_person=lambda:'alice'
        self.service.check_task_delivery(state)
        self.assertEqual(self.service.tasks.get(task['task_id'])['status'],'pending')
        state.interrupt.assert_called_once_with(None,source='relay_recipient_changed')

    async def test_due_inbox_triggers_one_notice_per_visit_and_global_search_is_readonly(self):
        task=await self.create();service=self.service;service.identified_person=lambda:'bob'
        service.message_receipts.pending=None  # The creation receipt is already spoken in this scenario.
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

    async def test_user_speaking_response_race_is_deferred_without_replaying_or_faulting(self):
        service=self.service;service.last_response_request_at=time.monotonic();service.response_request_pending=True
        service.continue_response=True
        event=dict(type='error',error=dict(code='invalid_value',param='response.create',message='Cannot create response while user is speaking.'))
        self.assertTrue(service.response_busy(event,SimpleNamespace(generating=None)))
        self.assertTrue(service.response_waiting_user_end)
        self.assertFalse(service.response_request_pending);self.assertFalse(service.continue_response)
        self.assertTrue(service.enabled)
        service.last_response_request_at=time.monotonic()-6
        self.assertFalse(service.response_busy(event,SimpleNamespace(generating=None)))

    async def test_self_report_cannot_expose_someone_elses_tasks_or_global_history(self):
        await self.create()
        service=self.service
        service.turn_person='anonymous:'+'a'*32
        service.pending_user_text='我是小王'
        service.identified_person=lambda:None
        await service.tool(self.ws,dict(call_id='status',name='get_robot_status',response_id='r',arguments='{}'))
        status=json.loads(self.sent[-1]['item']['output'])
        self.assertFalse(status['identity_verified'])
        self.assertIsNone(status['current_person'])
        self.assertEqual(status['known_people'],[])
        self.assertEqual(status['robot_tasks']['active'],[])
        self.assertEqual(status['robot_tasks']['due_task_inbox'],[])
        self.assertNotIn('准备资料',json.dumps(status,ensure_ascii=False))
        for index,args in enumerate((dict(action='list'),dict(action='list',recipient_id='bob'))):
            result=await self.tool(args,'unverified-list'+str(index))
            self.assertFalse(result['accepted'])
            self.assertEqual(result['status'],'identity_confirmation_required')
            self.assertIsNone(service.tool_repair)
            self.assertTrue(service.continue_response)
        await service.tool(self.ws,dict(call_id='global',name='search_robot_memory',response_id='r',arguments=json.dumps(dict(keywords=[]))))
        self.assertFalse(json.loads(self.sent[-1]['item']['output'])['accepted'])

    async def test_status_uses_the_verified_person_id_instead_of_a_second_identity_read(self):
        self.service.knowledge.current_person=lambda:dict(person_id='bob',display_name='测试乙')
        await self.service.tool(self.ws,dict(call_id='status',name='get_robot_status',response_id='r',arguments='{}'))
        status=json.loads(self.sent[-1]['item']['output'])
        self.assertTrue(status['identity_verified'])
        self.assertEqual(status['current_person']['person_id'],'alice')
        self.assertEqual(status['current_person']['display_name'],'测试甲')


if __name__=='__main__':unittest.main()
