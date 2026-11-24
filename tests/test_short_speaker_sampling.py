import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.speech.speaker_observation import SpeakerObserver
from vivis.speech.speaker_sampling import ShortSpeechCollector,visual_ranges

ROOT=Path(__file__).resolve().parents[1]


def pcm(seconds):
    return np.rint(np.sin(np.arange(round(seconds*16000))*.1)*3000).astype('<i2').tobytes()


def source(turn,seconds=.4,person='heyude',epoch=0,**extra):
    return dict(turn_id=turn,session='run:visit_'+turn,item_id='item_'+turn,logical_person='other',
        visual_person=None,enrollment_allowed=False,_collection_managed=True,
        _sampling_ranges=[dict(start_s=0.,end_s=seconds,person_id=person,epoch=epoch)],**extra)


class ShortCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.c=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text())
        self.events=[];self.log=[]
        self.s=SpeakerObserver(self.c,Path(self.tmp.name),self.events.append,
            lambda *a,**kw:self.log.append((a,kw)),extractor=lambda x:[1.,0.])
        self.addCleanup(self.s.job.close)

    def feed(self,turn,seconds=.4,**kw):
        x=pcm(seconds);self.s.infer(x,bytes(len(x)),source(turn,seconds,**kw))

    def test_four_short_turns_form_one_reference_without_changing_their_identity(self):
        for i in range(3):self.feed(str(i))
        self.assertEqual(self.s.gallery,{})
        self.feed('3');rows=self.s.gallery['heyude']
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['source_turn_ids'],['0','1','2','3'])
        self.assertEqual(len(rows[0]['source_fragments']),4)
        self.s.poll()
        observations=[kw for kind,kw in self.log if kind==('speaker_observation',)]
        self.assertTrue(all(r['logical_person']=='other' and r['speaker_candidate'] is None for r in observations))
        self.assertTrue(any('聚合参考已入库' in e['message'] for e in self.events))
        self.assertTrue(any('尚未入库' in e['message'] for e in self.events))

    def test_used_turns_not_reenrolled_after_restart(self):
        for i in range(4):self.feed(str(i))
        second=SpeakerObserver(self.c,Path(self.tmp.name),lambda e:None,lambda *a,**kw:None,extractor=lambda x:[1.,0.])
        self.addCleanup(second.job.close)
        for i in range(4):
            x=pcm(.4);second.infer(x,bytes(len(x)),source(str(i)))
        self.assertEqual(len(second.gallery['heyude']),1);self.assertIsNone(second.collector.batch)

    def test_none_pauses_collection_and_context_or_person_change_discards_it(self):
        self.feed('a');first=self.s.collector.batch
        x=pcm(.4);s=source('none');s['_sampling_ranges']=[]
        self.s.infer(x,bytes(len(x)),s);self.assertIs(self.s.collector.batch,first)
        self.s.set_collection_epoch(1);self.feed('b',epoch=1)
        self.assertEqual(self.s.collector.batch['source_turns'],['b'])
        self.feed('c',person='wangrunzhang',epoch=1)
        self.assertEqual(self.s.collector.batch['source_turns'],['c'])
        s=source('new_run',epoch=1,person='wangrunzhang');s['session']='another_run:v'
        self.s.infer(x,bytes(len(x)),s)
        self.assertEqual(self.s.collector.batch['source_turns'],['new_run'])

    def test_playback_is_excluded_but_clean_portion_still_usable(self):
        x=pcm(2);far=pcm(1)+bytes(32000)
        self.s.infer(x,far,source('double',2))
        b=self.s.collector.batch;self.assertIsNotNone(b)
        self.assertGreater(b['effective_s'],.6);self.assertLess(b['effective_s'],1.)
        self.assertGreaterEqual(b['fragments'][0]['start_sample'],17600)
        self.assertEqual(self.s.gallery,{})

    def test_confirmed_voice_conflict_and_visual_conflict_clear_batch(self):
        self.feed('a');x=pcm(.4)
        self.s.collector.accept(x,bytes(len(x)),source('b'),dict(speaker_candidate='wangrunzhang'),0,32,[])
        self.assertIsNone(self.s.collector.batch)
        self.feed('c');self.feed('d',_sampling_conflict=True)
        self.assertIsNone(self.s.collector.batch);self.assertEqual(self.s.gallery,{})

    def test_inconsistent_composite_waits_for_more_speech_then_discards_at_limit(self):
        with patch('vivis.speech.speaker_observation.consistency_windows',return_value=(None,{})):
            for i in range(4):self.feed(str(i))
            self.assertIsNotNone(self.s.collector.batch)
            self.assertGreater(self.s.collector.batch['target'],1.6)
            for i in range(4,8):self.feed(str(i))
        self.assertEqual(self.s.gallery,{})
        self.assertIsNone(self.s.collector.batch)

    def test_disk_failure_never_reports_success_and_leaves_no_reference(self):
        with patch.object(self.s,'save_gallery',side_effect=OSError('disk full')):
            for i in range(4):self.feed(str(i))
        self.assertEqual(self.s.gallery,{});self.assertFalse(list(self.s.directory.glob('*/*.wav')))
        self.s.poll();self.assertFalse(any('已入库' in e['message'] for e in self.events))

    def test_manual_edit_during_batch_save_is_not_overwritten(self):
        from vivis.speech.speaker_gallery import read_gallery,edit_reference
        self.s._store_reference('heyude',np.ones(32000,dtype=np.float32)*.1,np.array([1.,0.]),'old','old')
        original=self.s.save_gallery
        def concurrent(people):
            data=read_gallery(self.s.gallery_path,self.c['sha256'])
            edit_reference(self.s.gallery_path,self.c['sha256'],'heyude','old','deactivate',data['revision'])
            original(people)
        for i in range(3):self.feed(str(i))
        with patch.object(self.s,'save_gallery',side_effect=concurrent):self.feed('last')
        data=read_gallery(self.s.gallery_path,self.c['sha256'])
        self.assertEqual(len(data['people']['heyude']),1)
        self.assertFalse(data['people']['heyude'][0]['active'])
        self.assertEqual(len(list(self.s.directory.glob('*/*.wav'))),1)

    def test_pool_limit_counts_inactive_and_deletion_reopens_only_one_slot(self):
        for i in range(32):
            self.s._store_reference('heyude',np.ones(32000,dtype=np.float32)*.1,np.array([1.,0.]),'old'+str(i),'old')
        self.s.gallery['heyude'][0]['active']=False;self.s.save_gallery(self.s.gallery)
        self.feed('full',2);self.assertEqual(len(self.s.gallery['heyude']),32)
        (self.s.directory/'heyude/old0.wav').unlink()
        self.feed('new',2);self.feed('full_again',2)
        self.assertEqual(len(self.s.gallery['heyude']),32)
        self.assertEqual(self.s.gallery['heyude'][-1]['source_turn_ids'],['new'])

    def test_epoch_change_during_inference_cannot_commit_old_batch(self):
        for i in range(3):self.feed(str(i))
        def delayed(x):self.s.set_collection_epoch(1);return np.array([1.,0.])
        self.s.feature=delayed;self.feed('last')
        self.assertEqual(self.s.gallery,{});self.assertIsNone(self.s.collector.batch)

    def test_busy_observer_skips_optional_work_without_queueing(self):
        entered=threading.Event();release=threading.Event()
        def feature(x):entered.set();release.wait(2);return [1.,0.]
        self.s.extractor=feature;x=pcm(2)
        clip=dict(near=[x],far=[bytes(len(x))],rid='turn',session='v',item='item',person='heyude',
            evidence=dict(person_id='heyude',confirmed=True),alignment='aligned',
            sampling_ranges=source('turn',2)['_sampling_ranges'])
        try:
            self.assertTrue(self.s.offer(clip));self.assertTrue(entered.wait(1))
            self.assertFalse(self.s.offer(dict(clip,rid='skipped')))
        finally:release.set();self.s.job.wait(2)
        self.assertEqual(len(self.s.gallery['heyude']),1)
        self.assertEqual(self.s.gallery['heyude'][0]['source_turn_ids'],['turn'])


class SamplingBoundsTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text())['short_collection']
        self.collector=ShortSpeechCollector(self.config)

    def test_adjacent_frames_merge_without_extending_through_missing_face(self):
        points=[dict(stamp=1+i*.2,person_id='heyude',confirmed=True,epoch=0) for i in range(3)]
        spans,conflict=visual_ranges(points,1.,1.8,.5)
        self.assertFalse(conflict);self.assertEqual(len(spans),1)
        self.assertAlmostEqual(spans[0]['end_s'],.8)
        points.insert(2,dict(stamp=1.3,person_id=None,confirmed=False,epoch=0))
        spans,_=visual_ranges(points,1.,1.8,.5)
        self.assertEqual(len(spans),2);self.assertAlmostEqual(spans[0]['end_s'],.3)
        points.append(dict(stamp=1.7,person_id='wangrunzhang',confirmed=True,epoch=1))
        self.assertEqual(visual_ranges(points,1.,1.8,.5),([],True))

    def test_missing_reference_never_treated_as_silence(self):
        x=pcm(1);ref=bytes(16000);spans=source('a',1)['_sampling_ranges']
        fragments,error=self.collector.fragments(x,ref,spans)
        self.assertEqual(error,'reference_alignment_unavailable');self.assertEqual(fragments,[])
        fragments,_=self.collector.fragments(x,ref,spans,True)
        self.assertTrue(fragments);self.assertGreaterEqual(fragments[0]['start_sample'],8000)
        fragments,_=self.collector.fragments(x,b'',spans,True)
        self.assertEqual(fragments,[])

    def test_expiration_memory_bound_and_retry_idempotence(self):
        x=pcm(.4);s=source('a')
        self.collector.accept(x,bytes(len(x)),s,{},0,32,[],now=1)
        self.assertEqual(self.collector.accept(x,bytes(len(x)),s,{},0,32,[],now=2)[1]['status'],'already_used')
        self.assertFalse(self.collector.expire(30,0));self.assertTrue(self.collector.expire(32,0))
        self.collector.remember([str(i) for i in range(1100)]);self.assertEqual(len(self.collector.used),1024)
        source_ids=['a',*[str(i) for i in range(1100)]]
        self.assertEqual(self.collector.accept(x,bytes(len(x)),s,{},0,32,source_ids,now=32)[1]['status'],'already_used')

    def test_bad_policy_rejected(self):
        for key,value in [('minimum_fragment_s',0),('maximum_window_s',1000),('maximum_speech_s',float('nan'))]:
            c=copy.deepcopy(self.config);c[key]=value
            with self.assertRaises(ValueError):ShortSpeechCollector(c)


class RecorderWitnessTests(unittest.TestCase):
    def test_slow_visual_file_reader_has_no_waiting_queue_or_shutdown_dependency(self):
        from vivis.speech.round_recording import RoundRecorder
        with tempfile.TemporaryDirectory() as folder:
            c=json.loads((ROOT/'configs/interaction/continuity.json').read_text())
            recorder=RoundRecorder(c,Path(folder));entered=threading.Event();release=threading.Event()
            def slow(now):
                entered.set();release.wait(2)
                return dict(stamp=now,confirmed=False,person_id=None)
            try:
                self.assertTrue(recorder.offer_sampling_identity(slow,1))
                self.assertTrue(entered.wait(1));self.assertFalse(recorder.offer_sampling_identity(slow,2))
                recorder.close();self.assertTrue(recorder.sampling_job.closed.is_set())
            finally:release.set();recorder.sampling_job.wait(2);recorder.close()

    def test_raw_sampling_evidence_does_not_use_held_or_stale_identity(self):
        from vivis.speech.continuity_voice import ContinuityVoice
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);v=ContinuityVoice()
            v.knowledge=SimpleNamespace(root=root,recognition=dict(state_file='face.json',max_result_age_s=1),people={'heyude':{}})
            v.presence=SimpleNamespace(path=root/'presence.json',c=dict(max_age_s=1),person_count=lambda now:1)
            (root/'presence.json').write_text(json.dumps(dict(observed_monotonic=10,faces=1)))
            (root/'face.json').write_text(json.dumps(dict(observed_monotonic=10,person_id='heyude',state='recognized',owner='vision')))
            self.assertTrue(v.collection_visual_evidence(10.2)['confirmed'])
            self.assertFalse(v.collection_visual_evidence(12)['confirmed'])
            (root/'face.json').write_text(json.dumps(dict(observed_monotonic=10,person_id='heyude',state='reviewing',owner='vision')))
            self.assertFalse(v.collection_visual_evidence(10.2)['confirmed'])
            (root/'presence.json').write_text(json.dumps(dict(observed_monotonic=10,faces=2)))
            self.assertTrue(v.collection_visual_evidence(10.2)['blocked'])

    def test_interior_visual_evidence_labels_samples_without_reassigning_original_turn(self):
        from vivis.speech.round_recording import RoundRecorder
        with tempfile.TemporaryDirectory() as folder:
            c=json.loads((ROOT/'configs/interaction/continuity.json').read_text())
            recorder=RoundRecorder(c,Path(folder))
            try:
                recorder.speaker=SimpleNamespace(c={},offer=Mock(),set_collection_epoch=Mock(),job=Mock())
                for i in range(3):recorder.on_sampling_identity(dict(stamp=1+i*.2,person_id='heyude',confirmed=True,blocked=False,owner='vision'))
                clip=dict(start=1.,near=[pcm(.6)],far=[bytes(19200)],person='other',evidence={},sampling_epoch=0)
                recorder.offer_speaker(clip)
                offered=recorder.speaker.offer.call_args.args[0]
                self.assertEqual(offered['person'],'other');self.assertTrue(offered['sampling_ranges'])
                self.assertEqual(clip['evidence'],{})
                recorder.on_sampling_identity(dict(stamp=1.5,person_id='wangrunzhang',confirmed=True,blocked=False,owner='vision'))
                recorder.offer_speaker(clip)
                self.assertTrue(recorder.speaker.offer.call_args.args[0]['sampling_conflict'])
            finally:recorder.close()


if __name__=='__main__':unittest.main()
