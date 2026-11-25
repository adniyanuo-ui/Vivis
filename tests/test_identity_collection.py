"""Automatic photo selection and diagnostic-only speaker evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import wave

import numpy as np

from vivis.paths import ROOT
from vivis.vision.photo_pool import PhotoPool,choose
from vivis.speech.speaker_observation import SpeakerObserver,audio_quality,consistency_windows


def photo(i,locked=False):
    return dict(hash=str(i),path=(str(i)+'.jpg' if locked else 'photo_pool/'+str(i)+'.jpg'),locked=locked,origin='manual' if locked else 'automatic',quality=.9,
        coverage=[i*.1,0,.5],vector=np.ones(128).tolist(),dhash=None)


class PhotoTests(unittest.TestCase):
    def test_locked_bases_stay_and_diverse_sixteen_selected(self):
        rows=[photo(i,i<4) for i in range(32)];active=choose(rows)
        self.assertEqual(len(active),16);self.assertTrue({'0','1','2','3'}<=set(active));self.assertIn('31',active)

    def test_invalid_active_and_unlocked_base_rejected(self):
        base=dict(person_id='alpha',rows=[photo(0,True),photo(1)],active=['0','1'])
        PhotoPool.validate(base,'alpha')
        for changes in (dict(active=['1']),dict(active=['missing']),dict(person_id='beta')):
            with self.assertRaises(ValueError):PhotoPool.validate(dict(base,**changes),'alpha')

    def test_duplicate_and_escaped_path_rejected(self):
        rows=[photo(0,True),photo(0)]
        with self.assertRaises(ValueError):PhotoPool.validate(dict(person_id='alpha',rows=rows,active=['0']),'alpha')
        rows=[photo(0,True)];rows[0]['path']='../other.jpg'
        with self.assertRaises(ValueError):PhotoPool.validate(dict(person_id='alpha',rows=rows,active=['0']),'alpha')

    def pool(self):
        import cv2
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        folder=root/'private/people/alpha';folder.mkdir(parents=True);(folder/'profile.json').write_text('{}')
        image=np.random.default_rng(1).integers(25,230,(120,120,3),dtype=np.uint8)
        cv2.imwrite(str(folder/'photo.png'),image);digest=hashlib.sha256((folder/'photo.png').read_bytes()).hexdigest()
        vectors=np.zeros((1,128),np.float32);vectors[0,0]=1
        pool=PhotoPool(root,dict(people_directory='private/people',score_threshold=.6,minimum_margin=.1),vectors,['alpha'],[digest])
        self.addCleanup(pool.close)
        for stamp in (8,9,10):pool.observe_identity('alpha',stamp)
        return pool,image,vectors[0]

    def test_manual_original_locked_without_rewriting_gallery(self):
        pool,_,_=self.pool();data=pool.pools['alpha'];self.assertEqual(len(data['rows']),1)
        self.assertTrue(data['rows'][0]['locked']);self.assertEqual(pool.gallery()[0].shape,(1,128))

    def test_rebuilt_manual_gallery_new_photo_not_hidden_by_old_pool(self):
        import cv2
        pool,image,vector=self.pool();folder=pool.directory/'alpha'
        path=folder/'reference.png';cv2.imwrite(str(path),255-image)
        digest=hashlib.sha256(path.read_bytes()).hexdigest();old=pool.pools['alpha']['rows'][0]['hash']
        renewed=PhotoPool(pool.root,pool.c,np.array([vector,vector]),['alpha','alpha'],[old,digest])
        self.assertEqual(len(renewed.pools['alpha']['rows']),2)
        self.assertIn(digest,renewed.pools['alpha']['active']);self.assertEqual(sum(r['locked'] for r in renewed.pools['alpha']['rows']),2)

    def test_rejected_quality_and_other_anchor_cannot_enroll(self):
        pool,image,vector=self.pool();row=np.array([0,0,120,120,35,40,85,40,60,65,40,90,80,90],np.float32)
        self.assertFalse(pool.consider('alpha',image,row,vector,dict(quality_low=True),np.dot,10))
        self.assertFalse(pool.consider('alpha',image,row,-vector,{},np.dot,10))

    def test_administrator_unlock_is_not_undone_on_restart(self):
        pool,image,vector=self.pool();folder=pool.directory/'alpha';(folder/'photo_pool').mkdir()
        (folder/'photo_pool/new.jpg').write_bytes(b'photo')
        data=copy.deepcopy(pool.pools['alpha']);digest=data['rows'][0]['hash']
        row=photo(1);row.update(hash='new',path='photo_pool/new.jpg',locked=False,excluded=False,lock_reason='administrator_unlocked')
        data['rows'].append(row);data['active'].append('new');data['revision']+=1
        (folder/'photo_pool.json').write_text(json.dumps(data))
        renewed=PhotoPool(pool.root,pool.c,np.array([vector]),['alpha'],[digest]);self.addCleanup(renewed.close)
        self.assertFalse(renewed.pools['alpha']['rows'][1]['locked']);self.assertIn('new',renewed.pools['alpha']['active'])

    def test_eligible_frame_added_and_near_duplicate_skipped(self):
        pool,image,vector=self.pool();row=np.array([0,0,120,120,35,40,85,40,60,65,40,90,80,90],np.float32)
        self.assertTrue(pool.consider('alpha',image,row,vector,{},np.dot,10))
        pool.observe_identity('alpha',11)
        self.assertFalse(pool.consider('alpha',image,row,vector,{},np.dot,11))
        self.assertEqual(len(pool.pools['alpha']['rows']),1)  # Pending samples do not change recognition.
        self.assertTrue(pool.finish_batch('alpha'))
        data=pool.pools['alpha'];self.assertEqual(len(data['rows']),2)
        self.assertTrue(all((pool.directory/'alpha'/r['path']).is_file() for r in data['rows']))


class SpeakerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.c=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text())
        self.events=[];self.log=[];self.s=SpeakerObserver(self.c,Path(self.tmp.name),self.events.append,lambda *a,**kw:self.log.append((a,kw)),extractor=lambda x:[1.,0.])
        self.pcm=np.rint(np.sin(np.arange(48000)*.1)*3000).astype('<i2').tobytes();self.far=bytes(len(self.pcm))

    def source(self,i,**kw):return dict(turn_id=str(i),session='v',item_id='item'+str(i),logical_person='heyude',visual_person='heyude',enrollment_allowed=True,**kw)

    def test_quality_rejects_short_quiet_clipped_and_robot_playback(self):
        for pcm,far in [(bytes(100),bytes(100)),(bytes(96000),bytes(96000)),(b'\xff\x7f'*48000,bytes(96000)),(self.pcm,self.pcm)]:
            self.assertIsNone(audio_quality(pcm,far)[0])

    def test_effective_speech_threshold_is_point_five_seconds(self):
        # Total WAV duration includes silence; admission counts energized20ms frames.
        frame=np.rint(np.sin(np.arange(320)*.1)*3000).astype('<i2').tobytes()
        self.assertEqual(self.c['minimum_seconds'],.5)
        for frames,expected in ((24,'insufficient_speech'),(25,'accepted')):
            pcm=bytes(16000)+frame*frames+bytes(16000)
            statistics={};_,reason=audio_quality(pcm,bytes(len(pcm)),self.c['minimum_seconds'],statistics)
            self.assertEqual(reason,expected)
            self.assertAlmostEqual(statistics['effective_speech_s'],frames*.02)
        pcm=frame*50
        self.assertEqual(audio_quality(pcm,bytes(len(pcm)),1.5)[1],'too_short')

    def test_sub_point_five_configuration_is_rejected(self):
        for value in (.49,0):
            with self.assertRaises(ValueError):
                SpeakerObserver(dict(self.c,minimum_seconds=value),Path(self.tmp.name),self.events.append,lambda *a,**kw:None)

    def test_enrollment_needs_three_independent_turns_and_same_retry_not_added(self):
        self.s.infer(self.pcm,self.far,self.source(1));self.s.infer(self.pcm,self.far,self.source(1))
        self.assertEqual(len(self.s.gallery['heyude']),1)
        self.s.infer(self.pcm,self.far,self.source(2));self.s.infer(self.pcm,self.far,self.source(3))
        self.s.infer(self.pcm,self.far,dict(self.source(4),enrollment_allowed=False,logical_person='other',visual_person=None))
        result=None
        while not self.s.results.empty():result=self.s.results.get()
        self.assertEqual(result['speaker_candidate'],'heyude');self.assertTrue(result['observation_only'])
        self.assertEqual(len(self.s.gallery['heyude']),3)

    def test_held_or_unknown_face_not_enrolled(self):
        self.s.infer(self.pcm,self.far,dict(self.source(1),enrollment_allowed=False))
        self.assertEqual(self.s.gallery,{})
        self.s.poll();result=self.log[-1][1]
        self.assertTrue(result['audio_quality_accepted']);self.assertFalse(result['reference_sample_saved'])
        self.assertEqual(result['enrollment_skip_reason'],'visual_evidence_unconfirmed')
        self.assertIn('未入库',self.events[-1]['message'])

    def test_late_results_keep_original_source_and_have_no_authority(self):
        self.s.infer(self.pcm,self.far,self.source('original'));self.s.poll()
        result=self.log[-1][1];self.assertEqual(result['turn_id'],'original');self.assertEqual(result['logical_person'],'heyude')
        self.assertNotIn('authorized',result);self.assertTrue(result['observation_only'])

    def test_reference_recordings_and_vectors_bounded_after_restart(self):
        for i in range(40):self.s.infer(self.pcm,self.far,self.source(i))
        self.assertEqual(len(self.s.gallery['heyude']),32)
        self.assertEqual([r['turn_id'] for r in self.s.gallery['heyude']],[str(i) for i in range(32)])
        self.assertEqual(len(list((self.s.directory/'heyude').glob('*.wav'))),32)
        second=SpeakerObserver(self.c,Path(self.tmp.name),lambda e:None,lambda *a,**kw:None,extractor=lambda x:[1.,0.])
        self.assertEqual(len(second.gallery['heyude']),32)
        before=second.gallery_path.read_bytes();second.infer(self.pcm,self.far,self.source(50))
        self.assertEqual(second.gallery_path.read_bytes(),before)

    def test_manual_audio_deletion_reopens_only_that_slot_while_running_and_after_restart(self):
        for i in range(32):self.s.infer(self.pcm,self.far,self.source(i))
        folder=self.s.directory/'heyude';(folder/'5.wav').unlink()
        self.s.infer(self.pcm,self.far,self.source(40));self.s.infer(self.pcm,self.far,self.source(41))
        ids=[r['turn_id'] for r in self.s.gallery['heyude']]
        self.assertEqual(len(ids),32);self.assertIn('40',ids);self.assertNotIn('5',ids);self.assertNotIn('41',ids)
        (folder/'6.wav').unlink()
        second=SpeakerObserver(self.c,Path(self.tmp.name),lambda e:None,lambda *a,**kw:None,extractor=lambda x:[1.,0.])
        self.assertEqual(len(second.gallery['heyude']),31)
        second.infer(self.pcm,self.far,self.source(42));self.assertEqual(len(second.gallery['heyude']),32)

    def test_reference_save_failure_does_not_leave_a_phantom_in_memory_or_audio(self):
        from unittest.mock import patch
        with patch.object(self.s,'save_gallery',side_effect=OSError('disk full')):
            self.s.infer(self.pcm,self.far,self.source(1))
        self.assertEqual(self.s.gallery,{})
        self.assertFalse(list((self.s.directory/'heyude').glob('*.wav')))
        self.s.infer(self.pcm,self.far,self.source(1));self.assertEqual(len(self.s.gallery['heyude']),1)

    def test_disabled_reference_not_used_for_matching_and_active_changes_reload(self):
        from vivis.speech.speaker_gallery import edit_reference,read_gallery
        for i in range(3):self.s.infer(self.pcm,self.far,self.source(i))
        source=dict(self.source('query'),enrollment_allowed=False)
        data=read_gallery(self.s.gallery_path,self.c['sha256'])
        edit_reference(self.s.gallery_path,self.c['sha256'],'heyude','0','deactivate',data['revision'])
        self.s.infer(self.pcm,self.far,source)
        while not self.s.results.empty():result=self.s.results.get()
        self.assertIsNone(result['speaker_candidate']);self.assertEqual(len(self.s.gallery['heyude']),3)
        data=read_gallery(self.s.gallery_path,self.c['sha256'])
        edit_reference(self.s.gallery_path,self.c['sha256'],'heyude','0','activate',data['revision'])
        self.s.infer(self.pcm,self.far,source);self.assertEqual(self.s.results.get()['speaker_candidate'],'heyude')

    def test_concurrent_manual_edit_cannot_be_overwritten_by_enrollment(self):
        from unittest.mock import patch
        from vivis.speech.speaker_gallery import edit_reference,read_gallery
        self.s.infer(self.pcm,self.far,self.source(0));save=self.s.save_gallery
        def concurrent(people):
            data=read_gallery(self.s.gallery_path,self.c['sha256'])
            edit_reference(self.s.gallery_path,self.c['sha256'],'heyude','0','deactivate',data['revision'])
            save(people)
        with patch.object(self.s,'save_gallery',side_effect=concurrent):self.s.infer(self.pcm,self.far,self.source(1))
        data=read_gallery(self.s.gallery_path,self.c['sha256'])
        self.assertEqual(len(data['people']['heyude']),1);self.assertFalse(data['people']['heyude'][0]['active'])
        self.assertFalse((self.s.directory/'heyude/1.wav').exists())

    def test_inconsistent_multi_speaker_windows_not_enrolled(self):
        def feature(x):return [1,0] if np.mean(x)>0 else [0,1]
        self.s.extractor=feature
        pcm=np.r_[np.full(48000,3000),np.full(48000,-3000)].astype('<i2').tobytes()
        self.s.infer(pcm,bytes(len(pcm)),self.source(1));self.assertEqual(self.s.gallery,{})

    def test_consistency_probes_ignore_padding_without_changing_main_query(self):
        # Three seconds of leading silence previously became the first probe.
        self.s.extractor=lambda x:[1.,0.] if np.mean(x*x)>.001 else [0.,1.]
        pcm=np.r_[np.zeros(48000),np.full(48000,3000)].astype('<i2').tobytes()
        original=pcm;self.s.infer(pcm,bytes(len(pcm)),self.source('padded'))
        self.assertEqual(pcm,original);result=self.s.results.get()
        self.assertTrue(result['reference_sample_saved']);self.assertAlmostEqual(result['consistency']['similarity'],1.)
        with wave.open(str(self.s.directory/'heyude/padded.wav')) as w:self.assertEqual(w.getnframes(),96000)

    def test_consistency_does_not_reuse_one_short_word_twice(self):
        x=np.r_[np.zeros(48000),np.full(12000,.1),np.zeros(48000)].astype(np.float32)
        windows,evidence=consistency_windows(x,.6)
        self.assertIsNone(windows);self.assertEqual(evidence['reason'],'insufficient_independent_speech')

    def test_failed_consistency_never_publishes_an_otherwise_matching_candidate(self):
        for i in range(3):self.s.infer(self.pcm,self.far,self.source(i))
        while not self.s.results.empty():self.s.results.get()
        self.s.feature=Mock(side_effect=[np.array([1.,0.]),np.array([1.,0.]),np.array([0.,1.])])
        pcm=np.full(96000,3000,dtype='<i2').tobytes()
        self.s.infer(pcm,bytes(len(pcm)),self.source('mixed'))
        result=self.s.results.get();self.assertEqual(result['best_candidate'],'heyude')
        self.assertIsNone(result['speaker_candidate']);self.assertEqual(result['recognition_reason'],'consistency_rejected')
        self.assertFalse(result['reference_sample_saved']);self.assertEqual(len(self.s.gallery['heyude']),3)

    def test_low_score_is_explained_without_lowering_threshold_or_enrolling(self):
        for i in range(3):self.s.infer(self.pcm,self.far,self.source(i))
        while not self.s.results.empty():self.s.results.get()
        self.s.feature=lambda x:np.array([.59,np.sqrt(1-.59**2)])
        self.s.infer(self.pcm,self.far,dict(self.source('low'),enrollment_allowed=False))
        self.s.poll();result=self.events[-1]['speaker_observation']
        self.assertIsNone(result['speaker_candidate']);self.assertEqual(result['recognition_reason'],'low_similarity')
        self.assertIn('0.590低于0.650',self.events[-1]['message'])


if __name__=='__main__':unittest.main()
