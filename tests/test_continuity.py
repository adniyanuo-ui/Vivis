"""Private temporary stores, PCM taps and immutable source attribution; no devices."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import wave

import numpy as np

from vivis.paths import ROOT
from vivis.interaction.continuity import RoundStore,configuration
from vivis.speech.round_recording import RoundRecorder,SentTimeline
from vivis.speech.realtime_audio import VoiceBuffer


def config():return json.loads((ROOT/'configs/interaction/continuity.json').read_text())


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.c=config();self.c.update(familiar_rounds=2,other_rounds=3)
        self.s=RoundStore(self.c,self.root)

    def make(self,item,person='heyude',text='我喜欢篮球。',session='visit'):
        rid,notice=self.s.begin(session,item,person);self.assertIsNone(notice)
        self.s.final(rid,text);self.s.close(rid,'complete');return rid

    def archive(self):
        jid=self.s.request_archive('heyude')
        while batch:=self.s.next_batch(jid):self.s.commit_batch(jid,batch,dict(facts=[]))
        return jid

    def test_full_unarchived_preserves_old_and_pauses_new(self):
        self.make('a');self.make('b');rid,notice=self.s.begin('visit','c','heyude')
        self.assertIsNone(rid);self.assertIn('不保存',notice)
        with self.s.db() as d:self.assertEqual(d.execute('SELECT count(*) FROM rounds').fetchone()[0],2)

    def test_archiving_does_not_delete_recent_then_pipeline_evicts_only_oldest(self):
        first=self.make('a');self.make('b');self.archive()
        with self.s.db() as d:self.assertEqual(d.execute('SELECT count(*) FROM rounds').fetchone()[0],2)
        self.make('c')
        with self.s.db() as d:self.assertIsNone(d.execute('SELECT id FROM rounds WHERE id=?',(first,)).fetchone())

    def test_unknown_quota_shared_but_visits_separate_no_person_facts(self):
        for i in range(4):self.make(str(i),'anonymous:'+str(i)*32,session='visit'+str(i))
        with self.s.db() as d:
            self.assertEqual(d.execute('SELECT count(*) FROM rounds WHERE person="other"').fetchone()[0],3)
            self.assertEqual(d.execute('SELECT count(DISTINCT session) FROM rounds').fetchone()[0],3)
        self.assertEqual(self.s.context('other'),'')

    def test_same_request_begin_idempotent(self):
        rid=self.make('a');again,_=self.s.begin('visit','a','heyude');self.assertEqual(rid,again)

    def test_current_open_oldest_cannot_be_rotated(self):
        self.s.begin('visit','a','other');self.make('b','other');self.make('c','other')
        self.assertIsNone(self.s.begin('visit','d','other')[0])

    def test_unknown_identity_cannot_read_known_context(self):
        self.make('a');self.assertNotIn('篮球',self.s.context(None))

    def test_administrator_only_and_job_retry_idempotent(self):
        with self.assertRaises(PermissionError):self.s.request_archive('wangrunzhang')
        jid=self.s.request_archive('heyude');self.assertEqual(jid,self.s.request_archive('heyude'))

    def test_sources_and_corrections_survive_round_rotation(self):
        rid=self.make('a',text='我喜欢篮球。');jid=self.s.request_archive('heyude');batch=self.s.next_batch(jid)
        fact=dict(key='爱好',value='篮球',source_id=rid,quote='我喜欢篮球。')
        self.s.commit_batch(jid,batch,dict(facts=[fact]));self.s.next_batch(jid)
        rid2=self.make('b',text='我现在更喜欢羽毛球。');jid=self.s.request_archive('heyude');batch=self.s.next_batch(jid)
        self.s.commit_batch(jid,batch,dict(facts=[dict(key='爱好',value='羽毛球',source_id=rid2,quote='我现在更喜欢羽毛球。')]))
        self.make('c')
        text=self.s.context('heyude');self.assertIn('羽毛球',text)
        with self.s.db() as d:
            rows=d.execute('SELECT active,quote,supersedes FROM facts ORDER BY id').fetchall()
            self.assertEqual([r['active'] for r in rows],[0,1]);self.assertIn('篮球',rows[0]['quote']);self.assertIsNotNone(rows[1]['supersedes'])

    def test_fabricated_fact_or_other_person_source_rejected_atomically(self):
        rid=self.make('a');other=self.make('b','wangrunzhang')
        jid=self.s.request_archive('heyude');batch=self.s.next_batch(jid)
        for source,value in [(rid,'足球'),(other,'篮球')]:
            with self.assertRaises(ValueError):self.s.commit_batch(jid,batch,dict(facts=[dict(key='爱好',value=value,source_id=source,quote='我喜欢篮球。')]))
        with self.s.db() as d:self.assertEqual(d.execute('SELECT count(*) FROM facts').fetchone()[0],0)

    def test_commit_cannot_be_replayed(self):
        self.make('a');jid=self.s.request_archive('heyude');batch=self.s.next_batch(jid)
        self.s.commit_batch(jid,batch,dict(facts=[]))
        with self.assertRaises(ValueError):self.s.commit_batch(jid,batch,dict(facts=[]))

    def test_recover_open_round_and_job_after_restart(self):
        rid,_=self.s.begin('visit','a','heyude');self.s.final(rid,'明天见。')
        jid=self.s.request_archive('heyude')
        with self.s.db() as d:d.execute("UPDATE jobs SET state='running' WHERE id=?",(jid,))
        second=RoundStore(self.c,self.root);second.recover()
        with second.db() as d:
            self.assertEqual(d.execute('SELECT outcome FROM rounds').fetchone()[0],'process_interrupted')
            self.assertEqual(d.execute('SELECT state FROM jobs').fetchone()[0],'queued')

    def test_legacy_database_untouched(self):
        legacy=self.root/'private/memory/conversations.sqlite3';legacy.parent.mkdir();legacy.write_bytes(b'legacy-business-evidence')
        self.make('a');self.archive();self.assertEqual(legacy.read_bytes(),b'legacy-business-evidence')

    def test_frozen_legacy_user_sources_archived_readonly_not_new_copies_or_assistant_guesses(self):
        from vivis.interaction.memory import PersonMemory
        legacy=PersonMemory(json.loads((ROOT/'configs/interaction/memory.json').read_text()),self.root)
        mid=legacy.append('heyude','old','user','我喜欢篮球。')
        legacy.append('heyude','old','assistant','您可能是一位教授。')
        c=dict(self.c,directory='private/parallel');store=RoundStore(c,self.root,legacy.path)
        legacy.append('heyude','new','user','新文字兼容副本，不属于旧基线。')
        before=legacy.path.read_bytes();jid=store.request_archive('heyude');batch=store.next_batch(jid)
        self.assertEqual(len(batch['messages']),1);self.assertEqual(batch['messages'][0]['id'],'legacy:'+str(mid))
        store.commit_batch(jid,batch,dict(facts=[dict(key='爱好',value='篮球',quote='我喜欢篮球。',source_id='legacy:'+str(mid))]))
        self.assertIsNone(store.next_batch(jid));self.assertEqual(legacy.path.read_bytes(),before)

    def test_interruption_fade_stays_in_one_output_recording(self):
        recorder=RoundRecorder(self.c,self.root);self.addCleanup(recorder.close)
        recorder.on_played('reply',np.ones(960,np.float32)*.1,10.,0)
        recorder.on_reply('reply','v',None,'一段回答',960,'interrupted')
        recorder.on_played('reply',np.linspace(.1,0,480,dtype=np.float32),10.02,960)
        recorder.on_reply('reply','v',None,'一段回答',1440,'interrupted')
        recorder.finish_output('reply')
        with recorder.store.db() as d:row=d.execute('SELECT * FROM replies').fetchone()
        self.assertEqual(row['recorded_samples'],1440);self.assertEqual(row['audio_status'],'recorded')
        with wave.open(str(recorder.store.directory/row['audio'])) as w:self.assertEqual(w.getnframes(),1440)

    def test_missing_dac_observer_chunk_marked_gapped(self):
        recorder=RoundRecorder(self.c,self.root);self.addCleanup(recorder.close)
        recorder.on_played('reply',np.ones(960,np.float32)*.1,10.,960)
        recorder.on_reply('reply','v',None,'回答',1920,'complete');recorder.finish_output('reply')
        with recorder.store.db() as d:row=d.execute('SELECT * FROM replies').fetchone()
        self.assertEqual(row['audio_status'],'gapped');self.assertEqual(row['recorded_samples'],960)

    def test_recovered_invalid_turn_never_silently_claims_full_alignment(self):
        recorder=RoundRecorder(self.c,self.root);self.addCleanup(recorder.close)
        recorder.on_capture(1.,bytes(640),bytes(640));recorder.on_start('a','v','heyude',1.,{})
        recorder.on_end('a',1.02,False);recorder.on_final('a','小帅，您好。',accepted=True)
        with recorder.store.db() as d:row=d.execute('SELECT * FROM rounds').fetchone()
        self.assertEqual(row['alignment'],'recovered_invalid_turn_partial');self.assertEqual(row['user_text'],'小帅，您好。')

    def test_path_and_quota_configuration_rejected(self):
        bad=dict(self.c,directory='../public')
        with self.assertRaises(ValueError):RoundStore(bad,self.root)
        bad=dict(self.c,familiar_rounds=0)
        (self.root/'c.json').write_text(json.dumps(bad))
        with self.assertRaises(ValueError):configuration('c.json',self.root)

    def test_reply_bound_to_original_round_and_partial_not_complete(self):
        rid=self.make('a');self.s.reply('r','visit',rid,'未完整播出的长回答',samples=960,outcome='interrupted')
        with self.s.db() as d:
            row=d.execute('SELECT * FROM replies').fetchone();self.assertEqual(row['round_id'],rid);self.assertEqual(row['outcome'],'interrupted')

    def test_server_offsets_map_to_adc_not_arrival_time(self):
        line=SentTimeline();line.sent(100.,bytes(640));line.sent(100.02,bytes(640))
        self.assertAlmostEqual(line.at(30),100.03);self.assertIsNone(line.at(90));self.assertIsNone(line.at(None))

    def test_dac_observer_records_post_gain_and_failure_cannot_break_playback(self):
        p=VoiceBuffer(gain=.5);seen=[];p.audio_observer=lambda *a:seen.append(a)
        p.add('r',np.full(240,16000,dtype='<i2').tobytes());out=p.render(480,1.)
        np.testing.assert_allclose(seen[0][1],out);self.assertEqual(seen[0][2:],(1.,0))
        p.audio_observer=lambda *a:(_ for _ in ()).throw(OSError())
        p.add('r',bytes(480));p.render(480,2.)

    def test_user_clip_trims_late_endpoint_and_same_round_keeps_final(self):
        recorder=RoundRecorder(self.c,self.root);self.addCleanup(recorder.close)
        for i in range(50):recorder.on_capture(10+i*.02,bytes(640),bytes(640))
        recorder.on_start('a','v','heyude',10.2,dict(person_id='heyude',confirmed=True))
        recorder.on_final('a','我喜欢篮球。');recorder.on_end('a',10.6,True,dict(person_id='heyude',confirmed=True))
        with recorder.store.db() as d:row=d.execute('SELECT * FROM rounds').fetchone()
        with wave.open(str(recorder.store.directory/row['audio'])) as w:self.assertEqual(w.getnframes(),6400)
        self.assertEqual(row['alignment'],'aligned');self.assertEqual(row['user_text'],'我喜欢篮球。')

    def test_invalid_audio_candidate_not_counted_as_round(self):
        recorder=RoundRecorder(self.c,self.root);self.addCleanup(recorder.close)
        recorder.on_capture(1.,bytes(640),bytes(640));recorder.on_start('a','v','heyude',1.,{})
        recorder.on_end('a',1.02,False)
        with recorder.store.db() as d:self.assertEqual(d.execute('SELECT count(*) FROM rounds').fetchone()[0],0)

    def test_second_recorder_cannot_recover_or_modify_live_owner(self):
        recorder=RoundRecorder(self.c,self.root);self.addCleanup(recorder.close)
        rid,_=recorder.store.begin('v','ongoing','heyude')
        with self.assertRaises(RuntimeError):RoundRecorder(self.c,self.root)
        with recorder.store.db() as d:self.assertIsNone(d.execute('SELECT closed FROM rounds WHERE id=?',(rid,)).fetchone()[0])

    def test_proactive_welcome_not_attached_to_previous_user_turn(self):
        recorder=RoundRecorder(self.c,self.root);self.addCleanup(recorder.close)
        recorder.on_start('old','previous_visit','heyude',None,{})
        player=VoiceBuffer()
        state=SimpleNamespace(player=player,blocked=set(),responses=dict(welcome=dict(
            user_turn=dict(item_id='old'),archive_no_user=True,archive_session='new_visit',
            status='completed',received_audio_s=0,transcript='您好，请问怎么称呼您？')))
        recorder.poll(state,'new_visit');recorder.queue.join()
        with recorder.store.db() as d:
            row=d.execute('SELECT round_id,session FROM replies WHERE id="welcome"').fetchone()
        self.assertIsNone(row['round_id']);self.assertEqual(row['session'],'new_visit')

    def test_recording_overflow_only_marks_affected_clip_not_future_conversation(self):
        recorder=RoundRecorder(self.c,self.root);self.addCleanup(recorder.close)
        for i in range(30):recorder.on_capture(1+i*.02,bytes(640),bytes(640))
        recorder.on_start('first','v','heyude',1.,{});recorder.drop_generation+=1
        recorder.on_end('first',1.1,True);recorder.on_final('first','第一句。')
        recorder.on_start('second','v','heyude',1.2,{});recorder.on_end('second',1.4,True)
        with recorder.store.db() as d:rows={r['item']:r['alignment'] for r in d.execute('SELECT item,alignment FROM rounds')}
        self.assertEqual(rows['first'],'recording_queue_gap');self.assertEqual(rows['second'],'aligned')

    def test_long_session_does_not_requeue_completed_old_responses(self):
        recorder=RoundRecorder(self.c,self.root);self.addCleanup(recorder.close)
        recorder.submit=Mock(return_value=True);player=VoiceBuffer()
        state=SimpleNamespace(player=player,blocked=set(),responses={
            str(i):dict(status='completed',received_audio_s=0,transcript='回答') for i in range(1000)})
        recorder.poll(state,'visit');self.assertEqual(recorder.submit.call_count,256)
        recorder.last_poll=0;recorder.submit.reset_mock();recorder.poll(state,'visit')
        recorder.submit.assert_not_called()

    def test_unengaged_standby_speech_not_kept_as_visitor_round(self):
        recorder=RoundRecorder(self.c,self.root);self.addCleanup(recorder.close)
        recorder.on_start('background','v',None,None,{})
        recorder.on_final('background','旁边的人在聊天。',accepted=False)
        with recorder.store.db() as d:self.assertEqual(d.execute('SELECT count(*) FROM rounds').fetchone()[0],0)

    def test_speaker_reference_waits_for_accepted_final_transcript(self):
        recorder=RoundRecorder(self.c,self.root);self.addCleanup(recorder.close);recorder.speaker=Mock()
        for i in range(20):recorder.on_capture(1+i*.02,bytes(640),bytes(640))
        evidence=dict(person_id='heyude',confirmed=True)
        recorder.on_start('a','v','heyude',1.,evidence);recorder.on_end('a',1.2,True,evidence)
        recorder.speaker.offer.assert_not_called()
        recorder.on_final('a','这句已经正式接受。',accepted=True);recorder.speaker.offer.assert_called_once()


if __name__=='__main__':unittest.main()
