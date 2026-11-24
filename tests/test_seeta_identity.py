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
from vivis.vision.enrollment import save_gallery,registration_photos,SEETA_CONTRACT


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

    def test_typed_evidence_uses_raw_frame_not_review_hold(self):
        self.infer(10.)
        self.assertEqual(self.f.result['identity_evidence']['state'],'suspected')
        self.infer(10.25)
        self.assertEqual(self.f.result['identity_evidence']['state'],'confirmed')
        self.model.quality=False;self.infer(10.5)
        self.assertEqual(self.f.result['person_id'],'alpha')
        self.assertEqual(self.f.result['identity_evidence']['state'],'unavailable')

    def test_weak_identity_does_not_require_margin_or_weaken_confirmed_gate(self):
        self.model.compare.side_effect=[.55,.54]
        self.infer(10.)
        e=self.f.result['identity_evidence'];self.assertEqual(e['state'],'suspected');self.assertEqual(e['person_id'],'alpha')
        self.assertIsNone(self.f.result['person_id'])
        self.model.compare.side_effect=[.39,.3];self.infer(10.25)
        self.assertEqual(self.f.result['identity_evidence']['state'],'other')

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

    def test_brightness_reject_is_recorded_without_comparison_or_enrollment(self):
        self.model.evaluate_selected=Mock(return_value=(dict(quality_rejected=True,
            brightness=dict(level=0,score=.01),quality_rejection_reasons=['brightness']),None))
        self.f.infer(np.zeros((120,120,3),np.uint8),np.array([0,0,100,100]),10.,self.f.epoch)
        self.f.brightness_samples.job.wait(2)
        self.assertEqual(self.f.result['state'],'unknown');self.model.compare.assert_not_called()
        self.assertEqual(len(list((self.root/'private/diagnostics/face_brightness').glob('*.jpg'))),1)
        self.assertEqual(self.f.ids.tolist(),['alpha','beta'])

    def test_diagnostic_failure_does_not_revoke_a_good_identity(self):
        self.f.brightness_samples.offer=Mock(side_effect=OSError('diagnostic unavailable'))
        for t in (10.,10.5,11.):self.infer(t)
        self.assertEqual(self.f.result['person_id'],'alpha')
        self.assertEqual(self.f.brightness_error,'OSError');self.assertIsNone(self.f.error)

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


class MultiPhotoGalleryTests(unittest.TestCase):
    def test_same_person_templates_do_not_compete_in_margin(self):
        from vivis.vision.familiar import match_scores
        details={}
        pid,score=match_scores(np.array([.82,.81,.35]),np.array(['alpha','alpha','beta']),.6,.08,details)
        self.assertEqual(pid,'alpha');self.assertAlmostEqual(details['score_margin'],.47)

    def test_multiple_templates_persist_and_runtime_loads_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=config();path=root/backend_config(c)['gallery']
            save_gallery(path,c,[vector(0),vector(1),vector(2)],['alpha','alpha','beta'],['a','b','c'])
            model=FakeSeeta();f=FamiliarFaces(c,root,model=model)
            try:
                self.assertTrue(f.available);self.assertEqual(f.ids.tolist(),['alpha','alpha','beta'])
                model.index=1
                for i in range(c['confirmation_samples']):f.infer(None,np.array([0,0,100,100]),10.+i*.5,f.epoch)
                self.assertEqual(f.result['person_id'],'alpha')
            finally:f.close()
            with self.assertRaises(ValueError):
                save_gallery(path,c,[vector(0),vector(1)],['alpha','alpha'],['a','a'])
            with np.load(path,allow_pickle=False) as data:self.assertEqual(len(data['ids']),3)

    def test_all_primary_reference_and_additional_sources_deduplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'photos').mkdir()
            for name,data in [('photo.jpg',b'first'),('reference.png',b'second'),('photos/third.JPG',b'third'),('photos/copy.png',b'first'),('resume.md',b'ignored')]:
                (folder/name).write_bytes(data)
            self.assertEqual([p.name for p in registration_photos(folder)],['photo.jpg','reference.png','third.JPG'])


class BrightnessDiagnosticTests(unittest.TestCase):
    def test_dark_and_overexposed_faces_are_distinguished_without_changing_pixels(self):
        from vivis.vision.seeta import Seeta
        for level,hint in ((20,'too_dark'),(120,'not_at_native_low_boundary'),(255,'too_bright')):
            with self.subTest(level=level):
                image=np.full((30,30,3),level,np.uint8);before=image.copy()
                result=Seeta.brightness_diagnostics(image,[0,0,30,30])
                self.assertTrue(result['valid']);self.assertTrue(result['diagnostic_only'])
                self.assertEqual(result['hint'],hint)
                self.assertAlmostEqual(result['mean_gray'],level)
                np.testing.assert_array_equal(image,before)

    def test_only_selected_face_is_measured_and_outside_rectangles_are_invalid(self):
        from vivis.vision.seeta import Seeta
        image=np.full((90,90,3),255,np.uint8);image[30:60,30:60]=20
        result=Seeta.brightness_diagnostics(image,[30,30,30,30])
        self.assertEqual(result['hint'],'too_dark');self.assertAlmostEqual(result['dark_pixel_fraction'],1)
        for box in ((-40,0,10,30),(100,0,20,30),(0,0,0,30),(0,0,2,2)):
            with self.subTest(box=box):self.assertFalse(Seeta.brightness_diagnostics(image,box)['valid'])

    def test_rejection_reasons_explain_native_gate_but_do_not_override_it(self):
        from vivis.vision.seeta import Seeta
        meta=np.zeros(16,np.float64);meta[0]=1;meta[9]=2
        self.assertEqual(Seeta.observation(meta,True)['quality_rejection_reasons'],['brightness'])
        self.assertTrue(Seeta.observation(meta,True)['quality_rejected'])
        self.assertFalse(Seeta.observation(meta,False)['quality_rejected'])
        meta[7]=2;meta[9]=0
        self.assertEqual(Seeta.observation(meta,True)['quality_rejection_reasons'],['clarity'])
        meta[9]=2
        self.assertFalse(Seeta.observation(meta,True)['quality_rejected'])
