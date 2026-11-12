"""Backend separation, quality review and asynchronous native-engine lifetime."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

import numpy as np
from vivis.paths import ROOT
from vivis.vision.familiar import FamiliarFaces
from vivis.vision.identity_config import backend_config
from vivis.vision.enrollment import save_gallery,SEETA_CONTRACT


def config():return json.loads((ROOT/'configs/perception/familiar_faces.json').read_text())
def vector(index):
    v=np.zeros(1024,np.float32);v[index]=1.;return v


class FakeSeeta:
    def __init__(self):
        self.quality=True;self.index=0;self.closed=threading.Event()
        self.compare=Mock(side_effect=lambda a,b:float(a@b))
        self.entered=None;self.release=None
    def evaluate_selected(self,image,row,gate):
        if self.entered:self.entered.set();self.release.wait(2)
        return dict(quality_rejected=not self.quality,face_source='yunet_selected'),vector(self.index) if self.quality else None
    def close(self):self.closed.set()


class SeetaIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.c=config();self.c['backend']='seeta6';self.model=FakeSeeta()
        self.f=FamiliarFaces(self.c,self.root,model=self.model,gallery=(np.array([vector(0),vector(1)]),np.array(['alpha','beta'])))
        self.addCleanup(self.f.close)

    def infer(self,stamp):self.f.infer(None,np.array([1,2,100,100]),stamp,self.f.epoch)

    def test_seeta_1024_api_scores_confirm_without_loading_sface(self):
        self.assertTrue(self.f.available)
        for t in (10.,10.5,11.):self.infer(t)
        self.assertEqual(self.f.result['person_id'],'alpha')
        self.assertEqual(self.f.result['backend'],'seeta6')
        self.assertEqual(self.model.compare.call_count,6)
        self.assertEqual(self.f.result['observation']['confirmation_hits'],3)
        self.assertEqual(self.f.result['observation']['decision_reason'],'confirmed')

    def test_rejected_candidate_margin_is_reported_without_changing_threshold(self):
        self.model.compare.side_effect=[.72,.69]
        self.infer(10.)
        observation=self.f.result['observation']
        self.assertEqual(observation['decision_reason'],'ambiguous_candidates')
        self.assertAlmostEqual(observation['score_margin'],.03)
        self.assertEqual(observation['confirmation_hits'],0)
        self.assertIsNone(self.f.result['person_id'])

    def test_rejected_score_reports_why_confirmations_cannot_accumulate(self):
        self.model.compare.side_effect=[.55,.1]
        self.infer(10.)
        self.assertEqual(self.f.result['observation']['decision_reason'],'below_threshold')
        self.assertEqual(self.f.result['observation']['score_threshold'],.6)

    def test_low_quality_holds_name_but_skips_comparison_and_requires_qualified_recovery(self):
        for t in (10.,10.5,11.):self.infer(t)
        self.model.compare.reset_mock();self.model.quality=False;self.infer(11.5)
        self.assertEqual(self.f.result['state'],'reviewing')
        self.assertEqual(self.f.result['person_id'],'alpha')
        self.assertTrue(self.f.result['observation']['quality_rejected'])
        self.assertIsNone(self.f.result['similarity']);self.model.compare.assert_not_called()
        self.model.quality=True
        for i in range(self.c['confirmation_samples']-1):
            self.infer(12.+i*.25);self.assertEqual(self.f.result['state'],'reviewing')
        self.infer(13.);self.assertEqual(self.f.result['state'],'recognized')

    def test_new_credible_person_revokes_old_name_before_confirmation(self):
        for t in (10.,10.5,11.):self.infer(t)
        self.model.index=1;self.infer(11.5)
        self.assertIsNone(self.f.result['person_id'])
        self.infer(12.);self.infer(12.5);self.assertEqual(self.f.result['person_id'],'beta')

    def test_slow_identity_sampling_keeps_results_valid_between_samples(self):
        self.f.c.update(hz=.5,max_result_age_s=3.,review_hold_s=8.)
        stamps=[]
        def offer(fn,*args):
            stamps.append(args[2]);fn(*args);return True
        self.f.job.offer=offer
        image=np.zeros((120,120,3),np.uint8);row=np.array([0,0,100,100])
        for stamp in (10.,10.5,11.,11.5,12.,12.5,13.,13.5,14.,14.5,15.):
            result=self.f.step((1,stamp,image),row,now=stamp)
            if stamp>=14:self.assertEqual(result['person_id'],'alpha')
        self.assertEqual(stamps,[10.,12.,14.])
        self.model.quality=False
        for stamp in (16.,18.,20.):
            result=self.f.step((1,stamp,image),row,now=stamp)
            self.assertEqual(result['state'],'reviewing')
        self.assertEqual(self.f.step((1,22.,image),row,now=22.)['state'],'unknown')

    def test_current_sampling_confirms_two_good_samples_at_four_hz(self):
        stamps=[]
        def offer(fn,*args):stamps.append(args[2]);fn(*args);return True
        self.f.job.offer=offer
        image=np.zeros((120,120,3),np.uint8);row=np.array([0,0,100,100])
        self.assertEqual(self.c['hz'],4.);self.assertEqual(self.c['confirmation_samples'],2)
        for stamp in (10.,10.125,10.25):
            result=self.f.step((1,stamp,image),row,now=stamp)
            if stamp<10.25:self.assertNotEqual(result['state'],'recognized')
        self.assertEqual(stamps,[10.,10.25])
        self.assertEqual(result['state'],'recognized')

    def test_native_engine_is_not_destroyed_until_inflight_inference_finishes(self):
        self.model.entered=threading.Event();self.model.release=threading.Event()
        self.assertTrue(self.f.job.offer(self.f.infer,None,np.array([0,0,100,100]),10.,self.f.epoch))
        self.assertTrue(self.model.entered.wait(1))
        self.f.close();self.assertFalse(self.model.closed.is_set())
        self.model.release.set();self.assertTrue(self.model.closed.wait(1))
        self.assertIsNone(self.f.result['person_id'])

    def test_busy_optional_job_has_no_waiting_queue_or_main_loop_inference(self):
        self.model.entered=threading.Event();self.model.release=threading.Event()
        sample=(1,time.monotonic(),np.zeros((120,120,3),np.uint8));row=np.array([0,0,100,100])
        try:
            self.f.step(sample,row);self.assertTrue(self.model.entered.wait(1))
            first=self.f.job.thread
            for _ in range(3):self.f.step(sample,row)
            self.assertIs(self.f.job.thread,first)
        finally:self.model.release.set();self.f.job.wait(1)

    def test_128_dimension_gallery_is_not_accepted_by_seeta(self):
        wrong=FamiliarFaces(self.c,self.root,model=self.model,gallery=(np.ones((1,128),np.float32),np.array(['alpha'])))
        self.addCleanup(wrong.close);self.assertFalse(wrong.available)

    def test_gallery_metadata_and_model_contract_reject_mixed_embeddings(self):
        c=backend_config(self.c);path=self.root/c['gallery'];path.parent.mkdir(parents=True)
        for backend,sha,contract in (('sface',c['sha256'],SEETA_CONTRACT),('seeta6','wrong',SEETA_CONTRACT),('seeta6',c['sha256'],'other-alignment')):
            np.savez(path,vectors=np.array([vector(0)]),ids=np.array(['alpha']),backend=np.asarray(backend),model_sha256=np.asarray(sha),embedding_contract=np.asarray(contract))
            f=FamiliarFaces(self.c,self.root,model=FakeSeeta());self.assertFalse(f.available);f.close()

    def test_gallery_rebuild_preserves_other_backend_and_backs_up_current(self):
        c=backend_config(self.c);path=self.root/c['gallery'];other=self.root/backend_config(self.c,'sface')['gallery']
        other.parent.mkdir(parents=True,exist_ok=True);other.write_bytes(b'original-sface')
        save_gallery(path,self.c,[vector(0)],['alpha'],['photo-a'])
        save_gallery(path,self.c,[vector(1)],['beta'],['photo-b'])
        self.assertEqual(other.read_bytes(),b'original-sface')
        self.assertEqual(path.stat().st_mode&0o777,0o600)
        with np.load(path,allow_pickle=False) as d:
            self.assertEqual(str(d['embedding_contract']),SEETA_CONTRACT);self.assertEqual(d['ids'].tolist(),['beta'])
        self.assertEqual(len(list((path.parent/'.backups').glob('*.npz'))),1)

    def test_backend_selection_retains_independent_thresholds_and_paths(self):
        a=backend_config(self.c,'sface');b=backend_config(self.c,'seeta6')
        self.assertEqual(a['score_threshold'],.5);self.assertEqual(b['score_threshold'],.6)
        self.assertNotEqual(a['model'],b['model']);self.assertNotEqual(a['gallery'],b['gallery'])
        with self.assertRaises(ValueError):backend_config(self.c,'unknown')


if __name__=='__main__':unittest.main()
