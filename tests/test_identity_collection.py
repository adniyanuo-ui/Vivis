"""Automatic photo selection and diagnostic-only speaker evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vivis.paths import ROOT
from vivis.vision.photo_pool import PhotoPool,choose
from vivis.speech.speaker_observation import SpeakerObserver,audio_quality


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

    def test_eligible_frame_added_and_near_duplicate_skipped(self):
        pool,image,vector=self.pool();row=np.array([0,0,120,120,35,40,85,40,60,65,40,90,80,90],np.float32)
        self.assertTrue(pool.consider('alpha',image,row,vector,{},np.dot,10))
        self.assertFalse(pool.consider('alpha',image,row,vector,{},np.dot,20))
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

    def test_late_results_keep_original_source_and_have_no_authority(self):
        self.s.infer(self.pcm,self.far,self.source('original'));self.s.poll()
        result=self.log[-1][1];self.assertEqual(result['turn_id'],'original');self.assertEqual(result['logical_person'],'heyude')
        self.assertNotIn('authorized',result);self.assertTrue(result['observation_only'])

    def test_reference_recordings_and_vectors_bounded_after_restart(self):
        for i in range(12):self.s.infer(self.pcm,self.far,self.source(i))
        self.assertEqual(len(self.s.gallery['heyude']),8)
        self.assertEqual(len(list((self.s.directory/'heyude').glob('*.wav'))),8)
        second=SpeakerObserver(self.c,Path(self.tmp.name),lambda e:None,lambda *a,**kw:None,extractor=lambda x:[1.,0.])
        self.assertEqual(len(second.gallery['heyude']),8)

    def test_inconsistent_multi_speaker_windows_not_enrolled(self):
        def feature(x):return [1,0] if np.mean(x)>0 else [0,1]
        self.s.extractor=feature
        pcm=np.r_[np.full(48000,3000),np.full(48000,-3000)].astype('<i2').tobytes()
        self.s.infer(pcm,bytes(len(pcm)),self.source(1));self.assertEqual(self.s.gallery,{})


if __name__=='__main__':unittest.main()
