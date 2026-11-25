"""Speaker provenance, playback exclusion and independent mixed-speech checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vivis.paths import ROOT
from vivis.speech.speaker_observation import SpeakerObserver,consistency_windows
from vivis.speech.speaker_sampling import ShortSpeechCollector
from vivis.speech.speaker_segments import clean_fragments,recognition_audio


def pcm(seconds,value=3000):return np.full(round(seconds*16000),value,dtype='<i2').tobytes()


class SegmentTests(unittest.TestCase):
    def test_recognition_and_collection_exclude_identical_playback_and_keep_original_gain(self):
        near=pcm(4);far=bytes(32000)+pcm(1)+bytes(64000)
        original=near;derived,reference,detail,error=recognition_audio(near,far)
        self.assertIsNone(error);self.assertEqual(near,original)
        spans=[dict(start_s=0,end_s=4)]
        c=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text())['short_collection']
        fragments,_=ShortSpeechCollector(c).fragments(near,far,spans)
        self.assertEqual(detail['clean_fragments'],[{k:f[k] for k in ('start_sample','end_sample','effective_s')} for f in fragments])
        self.assertEqual(derived,b''.join(near[2*f['start_sample']:2*f['end_sample']] for f in fragments))
        self.assertTrue(np.all(np.frombuffer(derived,dtype='<i2')==3000))
        self.assertEqual(reference,bytes(len(derived)))
        self.assertEqual(len(fragments),2)
        self.assertLessEqual(fragments[0]['end_sample'],14400)
        self.assertGreaterEqual(fragments[1]['start_sample'],33600)

    def test_brief_playback_is_removed_even_when_whole_reference_rms_is_low(self):
        far=bytes(64000)+pcm(.02,500)+bytes(round(7.98*32000))
        derived,_,detail,error=recognition_audio(pcm(10),far)
        self.assertIsNone(error);self.assertLess(len(derived),320000)
        self.assertAlmostEqual(detail['excluded_reference_s'],.22)

    def test_unknown_prefix_and_robot_tail_never_become_silent_reference(self):
        near=pcm(3)
        self.assertEqual(recognition_audio(near,bytes(32000))[3],'reference_alignment_unavailable')
        derived,_,detail,error=recognition_audio(near,bytes(32000),reference_end_aligned=True)
        self.assertIsNone(error);self.assertEqual(len(derived),28800)
        self.assertEqual(detail['clean_fragments'][0]['start_sample'],33600)
        self.assertEqual(recognition_audio(near,b'',reference_end_aligned=True)[3],'robot_playback')

    def test_output_guard_crosses_visual_range_boundary_and_missing_reference_is_not_accepted(self):
        near=pcm(2);far=pcm(1)+bytes(32000)
        parts,error=clean_fragments(near,far,[dict(start_s=1,end_s=2)])
        self.assertIsNone(error);self.assertEqual(parts[0]['start_sample'],17600)
        self.assertEqual(clean_fragments(near,b'',[dict(start_s=0,end_s=2)])[1],'reference_alignment_unavailable')

    def test_coverage_avoids_sparse_tail_without_overlapping_or_copying_a_short_word(self):
        x=np.zeros(192000,np.float32)
        for i in [0,2,4,6,8]:x[int(i*16000):int((i+.8)*16000)]=.1
        x[-3200:]=.1
        self.assertIsNone(consistency_windows(x,.5)[0])
        windows,detail=consistency_windows(x,.5,'speech_coverage')
        self.assertEqual(len(windows),2)
        for a,b in zip(detail['probes'],detail['probes'][1:]):self.assertLessEqual(a['end_sample'],b['start_sample'])
        self.assertTrue(all(p['effective_s']>=.5 for p in detail['probes']))
        single=np.r_[np.zeros(48000),np.ones(12000)*.1,np.zeros(48000)].astype(np.float32)
        self.assertIsNone(consistency_windows(single,.5,'speech_coverage')[0])


class RecognitionSegmentsTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.c=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text())
        self.s=SpeakerObserver(self.c,Path(temp.name),lambda e:None,lambda *a,**k:None,extractor=lambda x:[1.,0.])
        self.addCleanup(self.s.job.close)
        for i in range(3):self.s._store_reference('alice',np.ones(16000,np.float32)*.1,np.array([1.,0.]),'ref'+str(i),'old')

    def infer(self,near,far=None,i=0,start=None,**extra):
        start=10+i if start is None else start
        source=dict(turn_id='turn'+str(i),session='visit',item_id='item'+str(i),logical_person='other',visual_person=None,
            enrollment_allowed=False,_collection_managed=True,_sampling_ranges=[],audio_started=start,audio_ended=start+len(near)/32000,
            audio_epoch=0,sample_epoch=0,voice_context='context',source_generation=i)
        source.update(extra);self.s.infer(near,bytes(len(near)) if far is None else far,source)
        results=[]
        while not self.s.results.empty():results.append(self.s.results.get())
        return next(r for r in results if not r.get('collection_event'))

    def test_remaining_clean_speech_can_identify_but_robot_only_and_clipped_still_cannot(self):
        r=self.infer(pcm(4),pcm(1)+bytes(96000))
        self.assertEqual(r['speaker_candidate'],'alice');self.assertIn('recognition_segments',r)
        self.assertEqual(self.infer(pcm(2),pcm(2),i=1)['identity_evidence']['state'],'unavailable')
        self.assertEqual(self.infer(pcm(1,32767)+pcm(2),pcm(1)+bytes(64000),i=2)['reason'],'clipping')
        self.assertEqual(len(self.s.gallery['alice']),3)

    def test_clean_fragments_from_same_sentence_merge_once_and_mixed_speakers_are_rejected(self):
        far=bytes(25600)+pcm(.4)+bytes(25600)
        r=self.infer(pcm(2),far)
        self.assertEqual(r['speaker_candidate'],'alice')
        self.assertEqual(len(r['recognition_segments']['clean_fragments']),2)
        self.s.extractor=lambda x:[1.,0.] if float(np.mean(x))>0 else [0.,1.]
        r=self.infer(pcm(.8)+pcm(.4)+pcm(.8,-3000),far,i=1,start=20)
        self.assertEqual(r['recognition_reason'],'consistency_rejected');self.assertIsNone(r['speaker_candidate'])

    def test_clean_short_remainders_accumulate_with_source_binding_then_clear(self):
        # Each original is1s, but only0.4s remains after playback+100ms guard.
        for i in range(4):r=self.infer(pcm(1),pcm(.5)+bytes(16000),i=i,start=10+i*1.2)
        self.assertEqual(r['speaker_candidate'],'alice')
        self.assertEqual(r['identity_evidence']['source_turn_ids'],['turn0','turn1','turn2','turn3'])
        self.assertIsNone(self.s.recognition_window.batch)
        self.assertEqual(len(self.s.gallery['alice']),3)
        self.assertEqual(self.infer(pcm(.4),i=4,start=16)['identity_evidence']['state'],'unavailable')

    def test_half_second_is_admitted_but_point_four_eight_still_waits(self):
        self.assertEqual(self.infer(pcm(.5))['speaker_candidate'],'alice')
        self.assertEqual(self.infer(bytes(16000)+pcm(.48),i=1)['reason'],'insufficient_speech')

    def test_intervening_other_speaker_is_still_checked(self):
        x=pcm(1.2)+pcm(2,-3000)+pcm(2.8)
        self.s.extractor=lambda x:[1.,0.] if np.mean(x)>0 else [0.,1.]
        r=self.infer(x)
        self.assertEqual(len(r['consistency']['probes']),2)
        self.assertEqual(r['recognition_reason'],'consistency_rejected');self.assertIsNone(r['speaker_candidate'])
        self.assertEqual(len(self.s.gallery['alice']),3)

    def test_reference_alignment_error_does_not_feed_short_window(self):
        self.infer(pcm(.4))
        r=self.infer(pcm(.6),bytes(6400),i=1)
        self.assertEqual(r['reason'],'reference_alignment_unavailable')
        self.assertIsNone(self.s.recognition_window.batch)

    def test_legacy_modes_remain_available_and_short_checks_stay_disjoint(self):
        for mode in ('fixed','speech_aligned','speech_coverage'):
            self.s.c=dict(self.c,consistency_window_mode=mode)
            similarity,detail=self.s.check_consistency(np.ones(24000,np.float32)*.1,independent=True)
            self.assertAlmostEqual(similarity,1.)
            self.assertNotEqual(detail['mode'],'fixed')


if __name__=='__main__':unittest.main()
