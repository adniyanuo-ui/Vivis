"""Explicit local selection; disposable journals and mocked embeddings only."""
import json
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np

from vivis.paths import ROOT
from vivis.interaction.continuity import RoundStore
from vivis.interaction.review_data import ReviewData
from vivis.speech.speaker_gallery import read_gallery,edit_reference
from vivis.speech.speaker_import import import_rounds,import_separately
from vivis.speech.speaker_observation import SpeakerObserver


class SpeakerImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        for name in ('configs/interaction/continuity.json','configs/perception/familiar_faces.json','configs/perception/speaker_observation.json'):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
        for person in ['heyude','wangrunzhang']:
            folder=self.root/'private/people'/person;folder.mkdir(parents=True)
            (folder/'profile.json').write_text(json.dumps(dict(display_name=person)))
        self.c=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text())
        self.store=RoundStore(json.loads((ROOT/'configs/interaction/continuity.json').read_text()),self.root)
        self.gallery=self.root/self.c['directory']/'gallery.json'
        self.counter=0

    def make_round(self,seconds=2,session='visit',person='heyude',alignment='aligned'):
        self.counter+=1
        rid,_=self.store.begin(session,'source'+str(self.counter),person)
        if rid is None:raise AssertionError('round not created')
        path=self.store.directory/'audio'/rid/'user.wav';path.parent.mkdir(parents=True,exist_ok=True)
        pcm=np.rint(np.sin(np.arange(round(seconds*16000))*(.1+self.counter*.0001))*3000).astype('<i2').tobytes()
        with wave.open(str(path),'wb') as w:w.setparams((1,2,16000,0,'NONE','PCM'));w.writeframes(pcm)
        self.store.input_audio(rid,str(path.relative_to(self.store.directory)),alignment)
        self.store.final(rid,'测试原话');self.store.close(rid,'complete')
        return rid,path

    def run_import(self,ids,person='heyude',revision=0,**kw):
        return import_rounds(self.root,person,ids,revision,extractor=lambda x:[1.,0.],**kw)

    def test_import_preserves_original_and_stays_inactive_until_manual_activation(self):
        rid,path=self.make_round();before=path.read_bytes();database=self.store.path.read_bytes()
        result=self.run_import([rid],confirmed=True)
        row=read_gallery(self.gallery,self.c['sha256'])['people']['heyude'][0]
        self.assertFalse(row['active']);self.assertTrue(row['administrator_verified']);self.assertEqual(row['source_turn_ids'],[rid])
        self.assertEqual(before,path.read_bytes());self.assertEqual(database,self.store.path.read_bytes())
        edit_reference(self.gallery,self.c['sha256'],'heyude',result['sample_id'],'activate',1)
        self.assertTrue(read_gallery(self.gallery,self.c['sha256'])['people']['heyude'][0]['locked'])

    def test_retry_with_old_revision_returns_existing_reference_without_duplicate(self):
        rid,_=self.make_round();first=self.run_import([rid],confirmed=True)
        second=self.run_import([rid],confirmed=True)
        self.assertTrue(second['already_saved']);self.assertEqual(first['sample_id'],second['sample_id'])
        self.assertEqual(len(read_gallery(self.gallery,self.c['sha256'])['people']['heyude']),1)

    def test_missing_admin_attestation_rejected_and_prepare_does_not_claim_it(self):
        rid,_=self.make_round()
        with self.assertRaises(ValueError):self.run_import([rid])
        self.run_import([rid],prepare=True)
        row=read_gallery(self.gallery,self.c['sha256'])['people']['heyude'][0]
        self.assertFalse(row['active']);self.assertFalse(row['administrator_verified'])
        edit_reference(self.gallery,self.c['sha256'],'heyude',row['turn_id'],'activate',1)
        row=read_gallery(self.gallery,self.c['sha256'])['people']['heyude'][0]
        self.assertTrue(row['administrator_verified']);self.assertIsNone(row['candidate_reason'])
        self.assertEqual(row['origin'],'manual_review')

    def test_three_short_turns_form_one_reference_with_all_sources(self):
        ids=[self.make_round(.5,session='same')[0] for i in range(3)]
        self.run_import(ids,confirmed=True)
        rows=read_gallery(self.gallery,self.c['sha256'])['people']['heyude']
        self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]['source_turn_ids']),set(ids))
        with self.assertRaisesRegex(ValueError,'重复|交叉'):self.run_import(ids[:1],revision=1,confirmed=True)

    def test_unaligned_and_external_symlink_audio_rejected(self):
        rid,path=self.make_round(alignment='recording_queue_gap')
        with self.assertRaisesRegex(ValueError,'完整'):self.run_import([rid],confirmed=True)
        self.store.input_audio(rid,str(path.relative_to(self.store.directory)),'aligned')
        outside=self.root/'external.wav';path.replace(outside);path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'越界'):self.run_import([rid],confirmed=True)

    def test_strong_other_person_match_cannot_be_imported_under_wrong_face(self):
        observer=SpeakerObserver(self.c,self.root,lambda e:None,lambda *a,**k:None,extractor=lambda x:[1.,0.])
        self.addCleanup(observer.job.close)
        for i in range(3):observer._store_reference('heyude',np.ones(24000,np.float32)*.1,np.array([1.,0.]),'ref'+str(i),'v')
        rid,_=self.make_round(person='wangrunzhang')
        with self.assertRaisesRegex(ValueError,'另一人'):self.run_import([rid],person='wangrunzhang',revision=3,confirmed=True)
        self.assertNotIn('wangrunzhang',read_gallery(self.gallery,self.c['sha256'])['people'])

    def test_snapshot_shows_thirty_two_rounds_and_marks_used_sources(self):
        for i in range(35):
            rid,_=self.store.begin('v','item'+str(i),'heyude');self.store.final(rid,'原话');self.store.close(rid,'complete')
        state,_=ReviewData(self.root).snapshot();p=next(p for p in state['people'] if p['id']=='heyude')
        self.assertEqual(len(p['rounds']),32)
        rid,_=self.make_round(session='new')
        self.run_import([rid],confirmed=True)
        state,_=ReviewData(self.root).snapshot();p=next(p for p in state['people'] if p['id']=='heyude')
        self.assertTrue(next(r for r in p['rounds'] if r['id']==rid)['used_for_voice'])

    def test_two_selected_long_recordings_can_make_two_candidates_without_merging(self):
        first,_=self.make_round(6.5,session='first_visit');second,_=self.make_round(6.5,session='second_visit')
        with self.assertRaisesRegex(ValueError,'同一次来访'):self.run_import([first,second],confirmed=True)
        result=import_separately(self.root,'heyude',[first,second],0,confirmed=True,extractor=lambda x:[1.,0.])
        self.assertEqual(result['saved_count'],2);self.assertEqual(result['failed_count'],0)
        rows=read_gallery(self.gallery,self.c['sha256'])['people']['heyude']
        self.assertEqual(len(rows),2);self.assertTrue(all(not r['active'] for r in rows))
        retry=import_separately(self.root,'heyude',[first,second],2,confirmed=True,extractor=lambda x:[1.,0.])
        self.assertEqual(retry['existing_count'],2);self.assertEqual(retry['saved_count'],0)

    def test_one_short_failure_does_not_discard_another_valid_selected_recording(self):
        first,_=self.make_round(.5);second,_=self.make_round(2)
        result=import_separately(self.root,'heyude',[first,second],0,confirmed=True,extractor=lambda x:[1.,0.])
        self.assertTrue(result['partial']);self.assertEqual(result['saved_count'],1);self.assertEqual(result['failed_count'],1)
        self.assertFalse(result['items'][0]['ok']);self.assertTrue(result['items'][1]['ok'])
        self.assertEqual(len(read_gallery(self.gallery,self.c['sha256'])['people']['heyude']),1)

    def test_near_threshold_rejection_shows_distinguishable_numbers_without_relaxing_gate(self):
        rid,_=self.make_round(6)
        similarity=.5995825529;vectors=iter([[1.,0.],[similarity,np.sqrt(1-similarity**2)]])
        with self.assertRaisesRegex(ValueError,'0.5996低于0.6000'):
            import_rounds(self.root,'heyude',[rid],0,confirmed=True,extractor=lambda x:next(vectors))
        self.assertFalse(self.gallery.exists())


if __name__=='__main__':unittest.main()
