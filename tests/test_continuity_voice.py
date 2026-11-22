"""Manual standby archiving, source guard and nonfatal optional-worker errors."""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,Mock,patch

import test_cloud_control
from test_continuity import config
from vivis.speech.round_recording import RoundRecorder


class ArchiveVoiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.events=[]
        self.voice=test_cloud_control.CloudBackendTests().service(self.tmp.name,self.events);self.addCleanup(self.voice.log.close)
        self.voice.continuity=RoundRecorder(config(),Path(self.tmp.name));self.addCleanup(self.voice.continuity.close)
        self.store=self.voice.continuity.store
        self.voice.context.update(mode='standby',ready=True)

    def round(self):
        rid,_=self.store.begin('visit','item','heyude');self.store.final(rid,'我喜欢篮球。');self.store.close(rid,'complete');return rid

    def state(self):
        with self.store.db() as d:return dict(d.execute('SELECT * FROM jobs').fetchone())

    async def test_success_only_after_committed_batch_and_recent_not_deleted(self):
        self.round();jid=self.store.request_archive('heyude')
        with patch('vivis.speech.memory_compaction.extract_facts',new=AsyncMock(return_value=dict(facts=[]))):
            await self.voice.archive_run(jid)
        self.assertEqual(self.state()['state'],'complete')
        with self.store.db() as d:self.assertEqual(d.execute('SELECT archived FROM rounds').fetchone()[0],1)

    async def test_cloud_failure_keeps_sources_and_records_specific_reason(self):
        self.round();jid=self.store.request_archive('heyude')
        with patch('vivis.speech.memory_compaction.extract_facts',new=AsyncMock(side_effect=TimeoutError('network timeout'))):
            await self.voice.archive_run(jid)
        self.assertEqual(self.state()['state'],'failed');self.assertIn('TimeoutError',self.state()['error'])
        with self.store.db() as d:self.assertEqual(d.execute('SELECT archived FROM rounds').fetchone()[0],0)

    async def test_new_interaction_prevents_background_commit(self):
        self.round();jid=self.store.request_archive('heyude')
        async def extract(*args):
            self.voice.context['mode']='interaction';return dict(facts=[])
        with patch('vivis.speech.memory_compaction.extract_facts',new=extract):await self.voice.archive_run(jid)
        self.assertEqual(self.state()['state'],'paused')
        with self.store.db() as d:self.assertEqual(d.execute('SELECT archived FROM rounds').fetchone()[0],0)

    async def test_optional_journal_error_does_not_crash_voice(self):
        with patch.object(self.voice,'_continuity_tick',new=AsyncMock(side_effect=OSError('disk full'))):
            await self.voice.continuity_tick(None);await self.voice.continuity_tick(None)
        self.assertEqual(len([e for e in self.events if e.get('kind')=='voice_warning']),1)

    async def test_admin_request_requires_final_text_and_same_actor(self):
        v=self.voice;v.context.update(mode='interaction',ready=True);v.turn_item_id='item';v.turn_ended_at=time.monotonic()
        v.turn_person='heyude';v.identified_person=lambda:'heyude';v.in_speech=False
        v.pending_user_text='请整理档案';v.response_inputs['reply']=v.input_generation
        result=await v.organize_archives({},dict(response_id='reply'))
        self.assertEqual(result['status'],'queued');self.assertTrue(self.events[-1]['enter_standby'])
        v.pending_user_text=None
        with self.assertRaises(ValueError):await v.organize_archives({},dict(response_id='reply'))

    async def test_non_admin_request_denied(self):
        v=self.voice;v.context.update(mode='interaction',ready=True);v.turn_item_id='item';v.turn_ended_at=time.monotonic()
        v.turn_person='wangrunzhang';v.identified_person=lambda:'wangrunzhang';v.pending_user_text='请整理档案'
        v.response_inputs['reply']=v.input_generation
        with self.assertRaises(PermissionError):await v.organize_archives({},dict(response_id='reply'))

    async def test_new_turn_during_database_queue_cannot_force_old_standby(self):
        v=self.voice;v.context.update(mode='interaction',ready=True);v.turn_item_id='item';v.turn_ended_at=time.monotonic()
        v.turn_person='heyude';v.identified_person=lambda:'heyude';v.pending_user_text='请整理档案'
        v.response_inputs['reply']=v.input_generation
        entered=threading.Event();release=threading.Event();original=self.store.request_archive
        def delayed(person):
            entered.set();release.wait(timeout=2);return original(person)
        with patch.object(self.store,'request_archive',side_effect=delayed):
            task=asyncio.create_task(v.organize_archives({},dict(response_id='reply')))
            for _ in range(100):
                if entered.is_set():break
                await asyncio.sleep(.01)
            self.assertTrue(entered.is_set());v.input_generation+=1;release.set();result=await task
        self.assertEqual(result['status'],'queued_without_state_change')
        self.assertFalse(any(e.get('kind')=='archive_notice' for e in self.events))

    async def test_completion_notice_not_marked_spoken_until_playback_success(self):
        self.round();jid=self.store.request_archive('heyude')
        with patch('vivis.speech.memory_compaction.extract_facts',new=AsyncMock(return_value=dict(facts=[]))):await self.voice.archive_run(jid)
        self.voice.identified_person=lambda:None;self.voice.in_speech=False
        state=SimpleNamespace(responses={},player=self.voice.player)
        await self.voice.continuity_tick(state)
        notice=next(e for e in self.events if e.get('kind')=='archive_notice')
        self.assertEqual(notice['text'],'档案更新完毕。');self.assertEqual(self.state()['notice'],0)
        self.voice.farewell=dict(id='notice',archive_job=jid)
        self.voice.finish_farewell(self.voice.farewell,'interrupted')
        self.assertEqual(self.state()['notice'],0)


if __name__=='__main__':unittest.main()
