import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vivis.paths import ROOT
from vivis.speech.speaker_observation import SpeakerObserver
from vivis.speech.speaker_window import ShortRecognitionWindow
from test_short_speaker_sampling import pcm


class ShortRecognitionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.c=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text())
        self.s=SpeakerObserver(self.c,Path(temp.name),lambda e:None,lambda *a,**kw:None,extractor=lambda x:[1.,0.])
        self.addCleanup(self.s.job.close)
        for i in range(3):self.s._store_reference('alice',np.ones(16000,np.float32)*.1,np.array([1.,0.]),'ref'+str(i),'v')

    def feed(self,i,seconds=.4,start=None,**extra):
        start=10+i if start is None else start;x=pcm(seconds)
        source=dict(turn_id='turn'+str(i),item_id='item'+str(i),session='visit',logical_person='anonymous',visual_person=None,
            enrollment_allowed=False,_collection_managed=True,_sampling_ranges=[],audio_started=start,audio_ended=start+seconds,
            sample_epoch=0,voice_context='context',source_generation=i)
        source.update(extra);self.s.infer(x,bytes(len(x)),source)
        results=[]
        while not self.s.results.empty():results.append(self.s.results.get())
        return next(r for r in results if not r.get('collection_event'))

    def test_short_sentences_match_as_one_disjoint_window_without_enrollment(self):
        for i in range(3):self.assertEqual(self.feed(i)['identity_evidence']['state'],'unavailable')
        result=self.feed(3)
        self.assertEqual(result['reason'],'short_window');self.assertEqual(result['speaker_candidate'],'alice')
        self.assertEqual(result['identity_evidence']['source_turn_ids'],['turn0','turn1','turn2','turn3'])
        self.assertIsNone(self.s.recognition_window.batch);self.assertEqual(len(self.s.gallery['alice']),3)
        self.assertEqual(result['identity_evidence']['stamp'],13.4)
        self.assertEqual(self.feed(4)['identity_evidence']['state'],'unavailable')

    def test_current_sentence_still_matches_immediately(self):
        result=self.feed(0,seconds=.8)
        self.assertEqual(result['reason'],'accepted');self.assertEqual(result['speaker_candidate'],'alice')
        self.assertNotIn('window_match',result);self.assertIsNone(self.s.recognition_window.batch)

    def test_session_epoch_overlap_long_gap_and_source_retry_discard_window(self):
        for changes in (dict(session='another'),dict(sample_epoch=1),dict(voice_context='new'),dict(audio_started=9,audio_ended=9.4)):
            self.s.recognition_window=ShortRecognitionWindow(self.c['short_recognition'])
            self.feed(0);self.feed(1,**changes)
            batch=self.s.recognition_window.batch
            self.assertTrue(batch is None or batch['turns']==['turn1'])
        self.s.recognition_window=ShortRecognitionWindow(self.c['short_recognition'])
        self.feed(0);self.feed(1,start=20)
        self.assertEqual(self.s.recognition_window.batch['turns'],['turn1'])
        self.feed(1,start=20);self.assertEqual(self.s.recognition_window.batch['turns'],['turn1'])

    def test_mixed_voice_window_cannot_publish_matching_whole_feature(self):
        for i in range(3):self.feed(i)
        with patch.object(self.s,'feature',side_effect=[np.array([1.,0.]),np.array([0.,1.])]):result=self.feed(3)
        self.assertIsNone(result['speaker_candidate']);self.assertEqual(result['identity_evidence']['state'],'unavailable')
        self.assertEqual(result['window_match']['consistency'],0.)

    def test_insufficient_window_collects_more_before_retry_without_lowering_gates(self):
        for i in range(3):self.feed(i)
        with patch('vivis.speech.speaker_observation.consistency_windows',return_value=(None,{})):first=self.feed(3)
        self.assertIsNotNone(self.s.recognition_window.batch)
        self.assertEqual(self.feed(4)['identity_evidence']['state'],'unavailable')
        result=self.feed(5)
        self.assertEqual(result['identity_evidence']['state'],'confirmed')
        self.assertEqual(len(result['identity_evidence']['source_turn_ids']),6)
        self.assertEqual(len(first['window_match']['source_turn_ids']),4)
        self.assertIsNone(self.s.recognition_window.batch)

    def test_last_short_speaker_cannot_inherit_earlier_dominant_speaker(self):
        for i in range(3):self.feed(i)
        a=np.array([1.,0.]);b=np.array([0.,1.])
        with patch.object(self.s,'feature',side_effect=[a,a,a,b]):result=self.feed(3)
        self.assertIsNone(result['speaker_candidate'])
        self.assertEqual(result['identity_evidence']['state'],'unavailable')
        self.assertEqual(result['window_match']['reason'],'latest_fragment_not_supported')

    def test_manual_reference_deactivation_is_loaded_before_composite_match(self):
        from vivis.speech.speaker_gallery import edit_reference,read_gallery
        for i in range(3):self.feed(i)
        data=read_gallery(self.s.gallery_path,self.c['sha256'])
        edit_reference(self.s.gallery_path,self.c['sha256'],'alice','ref0','deactivate',data['revision'])
        result=self.feed(3)
        self.assertEqual(result['identity_evidence']['state'],'unavailable');self.assertIsNone(result['speaker_candidate'])

    def test_suspect_threshold_does_not_replace_confirmed_threshold_or_margin(self):
        self.s.extractor=lambda x:[.5,.8660254]
        result=self.feed(0,.8);self.assertEqual(result['identity_evidence']['state'],'suspected')
        self.assertIsNone(result['speaker_candidate']);self.assertAlmostEqual(result['score'],.5,places=5)
        self.s.extractor=lambda x:[.3,.9539392]
        self.assertEqual(self.feed(1,.8)['identity_evidence']['state'],'other')

    def test_no_gallery_is_none_and_robot_playback_cannot_fill_window(self):
        window=self.s.recognition_window;x=pcm(.4)
        source=dict(turn_id='robot',session='v',audio_started=10,audio_ended=10.4,sample_epoch=0,voice_context='v')
        self.assertIsNone(window.add(x,x,source,dict(reason='robot_playback'),0));self.assertIsNone(window.batch)
        self.s.gallery={};self.s.refresh_gallery=lambda:None
        self.assertEqual(self.feed(0,.8)['identity_evidence']['state'],'unavailable')

    def test_recognition_conflict_gate_does_not_depend_on_reference_collection(self):
        self.s.collector=None
        self.feed(0,sampling_conflict=True)
        self.assertIsNone(self.s.recognition_window.batch)

    def test_full_reference_pool_keeps_matching_without_new_samples(self):
        for i in range(3,32):self.s._store_reference('alice',np.ones(16000,np.float32)*.1,np.array([1.,0.]),'ref'+str(i),'v')
        result=self.feed(0,.8)
        self.assertEqual(result['identity_evidence']['state'],'confirmed')
        self.assertEqual(len(self.s.gallery['alice']),32)
        self.assertEqual(len(list((self.s.directory/'alice').glob('*.wav'))),32)

    def test_invalid_recognition_policy_cannot_start(self):
        for key,value in [('minimum_fragment_s',.1),('maximum_window_s',100),('maximum_speech_s',float('nan'))]:
            c=copy.deepcopy(self.c['short_recognition']);c[key]=value
            with self.assertRaises(ValueError):ShortRecognitionWindow(c)

    def test_disabled_window_preserves_single_sentence_behavior(self):
        self.s.recognition_window=None
        for i in range(6):self.assertEqual(self.feed(i)['identity_evidence']['state'],'unavailable')


if __name__=='__main__':unittest.main()
