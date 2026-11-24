"""Vacant-slot batches, completion after restart and recognition isolation."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.vision.photo_pool import PhotoPool
from vivis.vision.photo_sampling import plan,apply_batch,administrator_change


def photo(i,locked=False):
    return dict(hash=str(i),path=('base.jpg' if locked else 'photo_pool/'+str(i)+'.jpg'),locked=locked,
        origin='manual' if locked else 'automatic',quality=.8,coverage=[i*.01,0,.5],vector=[1.]*128,dhash=None)


def full_pool():
    return dict(person_id='alpha',rows=[photo(i,i==0) for i in range(32)],active=[str(i) for i in range(16)],revision=1)


class ScheduleTests(unittest.TestCase):
    def test_full_pool_never_collects_or_replaces_even_without_completion_metadata(self):
        data=full_pool();before=json.dumps(data)
        for day in ('2026-11-23','2026-12-01','2027-01-01'):self.assertIsNone(plan(data,day))
        self.assertEqual(json.dumps(data),before)

    def test_initial_fill_uses_active_slots_before_reserve_without_replacement(self):
        data=full_pool();data['rows']=data['rows'][:3];data['active']=data['active'][:3]
        work=plan(data,'2026-11-01');self.assertEqual((work['active'],work['reserve']),(13,16))
        originals=json.loads(json.dumps(data['rows']))
        data=apply_batch(data,[photo(100+i) for i in range(29)],work)
        self.assertEqual(data['rows'][:3],originals);self.assertTrue({'0','1','2'}<=set(data['active']))
        self.assertEqual(len(data['rows']),32);self.assertEqual(len(data['active']),16)
        self.assertTrue(data['collection']['completed']);self.assertIsNone(plan(data,'2026-11-02'))

    def test_short_visit_keeps_remaining_vacancies_across_restart_without_daily_limit(self):
        data=full_pool();data['rows']=data['rows'][:30];work=plan(data,'2026-11-01')
        data=apply_batch(data,[photo(100)],work)
        self.assertFalse(data['collection']['completed'])
        data=json.loads(json.dumps(data));work=plan(data,'2026-11-01')
        self.assertEqual(work['active']+work['reserve'],1)
        data=apply_batch(data,[photo(200)],work)
        self.assertTrue(data['collection']['completed']);self.assertIsNone(plan(data,'2026-11-03'))

    def test_more_than_four_manual_locks_are_retained_and_cannot_be_overwritten(self):
        data=full_pool()
        for row in data['rows'][:8]:row.update(locked=True,lock_reason='administrator')
        data['rows'].pop();work=plan(data,'2026-11-01')
        data=apply_batch(data,[photo(100)],work)
        PhotoPool.validate(data,'alpha');self.assertTrue({str(i) for i in range(8)}<=set(data['active']))

    def test_manual_deletion_reopens_only_vacancies_even_on_the_same_day(self):
        data=full_pool();data['collection']=dict(policy='fill-once-1',completed=True,last_batch_day='2026-11-01')
        administrator_change(data);self.assertTrue(data['collection']['completed'])
        data['rows'].pop();administrator_change(data)
        self.assertFalse(data['collection']['completed']);self.assertEqual(plan(data,'2026-11-01')['reserve'],1)


class FakeTimer:
    def __init__(self,seconds,callback,args):self.callback=callback;self.args=args;self.cancelled=False
    def start(self):pass
    def cancel(self):self.cancelled=True


class BatchTests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.folder=self.root/'private/people/alpha';self.folder.mkdir(parents=True)
        (self.folder/'profile.json').write_text('{}');cv2.imwrite(str(self.folder/'photo.png'),self.image(1))
        self.digest=hashlib.sha256((self.folder/'photo.png').read_bytes()).hexdigest()
        self.c=dict(people_directory='private/people',score_threshold=.6,minimum_margin=.1)
        self.vector=np.eye(1,128,dtype=np.float32)[0];self.day=['2026-11-23']
        self.row=np.array([0,0,120,120,35,40,85,40,60,65,40,90,80,90],np.float32)
        self.pool=self.reopen();self.addCleanup(lambda:self.pool.close())

    def image(self,seed):return np.random.default_rng(seed).integers(25,230,(120,120,3),dtype=np.uint8)
    def reopen(self):
        return PhotoPool(self.root,self.c,np.array([self.vector]),['alpha'],[self.digest],day=lambda:self.day[0],timer_factory=FakeTimer)
    def stable(self,start=10):
        for stamp in (start-2,start-1,start):self.pool.observe_identity('alpha',stamp)
    def consider(self,seed,stamp=10,observation=None,compare=np.dot):
        return self.pool.consider('alpha',self.image(seed),self.row,self.vector,observation or {},compare,stamp)

    def test_no_stable_identity_or_quality_or_anchor_means_no_capture(self):
        self.assertFalse(self.consider(2));self.assertEqual(self.pool.last_attempt['reason'],'identity_not_stable');self.stable()
        self.assertFalse(self.consider(2,observation=dict(quality_low=True)))
        self.assertFalse(self.consider(2,compare=lambda a,b:0.))
        self.pool.observe_identity(None,10.5);self.assertFalse(self.consider(2,10.5))
        self.assertFalse(self.pool.pending)

    def test_sampling_diagnostics_explain_size_and_clarity_without_changing_admission(self):
        self.stable();self.row[2]=80
        self.assertFalse(self.consider(2));self.assertEqual(self.pool.last_attempt['reason'],'face_too_small')
        self.assertEqual(self.pool.last_attempt['crop_size_px'],[120,80])
        self.row[2]=120
        self.assertFalse(self.pool.consider('alpha',np.full((120,120,3),127,np.uint8),self.row,self.vector,{},np.dot,10))
        self.assertEqual(self.pool.last_attempt['reason'],'image_quality_low')
        self.assertEqual(self.pool.last_attempt['rejected_by'],['clarity'])
        self.assertEqual(len(self.pool.pools['alpha']['rows']),1)

    def test_normal_chat_sampling_accepts_smaller_softer_face_after_one_second(self):
        import cv2
        from vivis.paths import ROOT
        sampling=json.loads((ROOT/'configs/perception/familiar_faces.json').read_text())['photo_sampling']
        gray=np.random.default_rng(7).integers(95,100,(120,120),dtype=np.uint8)
        image=np.repeat(gray[:,:,None],3,axis=2);self.row[2]=80
        variance=cv2.Laplacian(gray[:,:80],cv2.CV_64F).var()
        self.assertGreaterEqual(variance,25);self.assertLess(variance,45)
        self.stable();self.assertFalse(self.pool.consider('alpha',image,self.row,self.vector,{},np.dot,10))
        self.pool.close()
        self.pool=PhotoPool(self.root,dict(self.c,photo_sampling=sampling),np.array([self.vector]),['alpha'],[self.digest],timer_factory=FakeTimer)
        self.pool.observe_identity('alpha',10);self.pool.observe_identity('alpha',11)
        self.assertTrue(self.pool.consider('alpha',image,self.row,self.vector,{},np.dot,11))
        self.assertEqual(len(self.pool.pools['alpha']['rows']),1)
        self.assertTrue(self.pool.finish_batch('alpha'));self.assertEqual(len(self.pool.pools['alpha']['rows']),2)
        self.assertTrue(self.pool.pools['alpha']['rows'][0]['locked'])

    def test_relaxed_sampling_still_rejects_unusable_images_and_invalid_configuration(self):
        from vivis.paths import ROOT
        sampling=json.loads((ROOT/'configs/perception/familiar_faces.json').read_text())['photo_sampling']
        self.pool.close();self.pool=PhotoPool(self.root,dict(self.c,photo_sampling=sampling),np.array([self.vector]),['alpha'],[self.digest],timer_factory=FakeTimer)
        self.stable()
        for level in (0,127,255):
            self.assertFalse(self.pool.consider('alpha',np.full((120,120,3),level,np.uint8),self.row,self.vector,{},np.dot,10))
        for changes in (dict(min_face_size_px=0),dict(brightness_range=[.9,.1]),dict(min_laplacian_variance=float('nan'))):
            with self.assertRaises(ValueError):
                PhotoPool(self.root,dict(self.c,photo_sampling=dict(sampling,**changes)),np.array([self.vector]),['alpha'],[self.digest],timer_factory=FakeTimer)

    def test_batch_is_atomic_and_can_continue_after_restart_on_same_day(self):
        self.stable();self.assertTrue(self.consider(2))
        self.pool.observe_identity('alpha',11);self.assertTrue(self.consider(3,11))
        self.assertEqual(len(self.pool.gallery()[0]),1)
        self.assertFalse(list((self.folder/'photo_pool').glob('*.jpg')))
        self.assertTrue(self.pool.finish_batch('alpha'));data=json.loads((self.folder/'photo_pool.json').read_text())
        self.assertEqual(len(data['active']),3);self.assertEqual(data['collection']['last_batch']['total'],2)
        self.pool.close();self.pool=self.reopen();self.stable(20)
        self.assertTrue(self.consider(4,20))

    def test_administrator_revision_wins_over_pending_batch(self):
        self.stable();self.assertTrue(self.consider(2));path=self.folder/'photo_pool.json'
        data=json.loads(path.read_text());data['revision']+=1;path.write_text(json.dumps(data))
        self.assertFalse(self.pool.finish_batch('alpha'));self.assertEqual(len(json.loads(path.read_text())['rows']),1)
        self.assertFalse(list((self.folder/'photo_pool').glob('*.jpg')))

    def test_failed_manifest_write_rolls_back_only_new_files_and_does_not_consume_day(self):
        self.stable();self.assertTrue(self.consider(2))
        with patch.object(self.pool,'save',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):self.pool.finish_batch('alpha')
        self.assertEqual(len(self.pool.pools['alpha']['rows']),1)
        self.assertFalse(list((self.folder/'photo_pool').glob('*.jpg')))
        self.assertTrue(self.consider(3))

    def test_close_flushes_one_pending_batch_and_cancels_timer(self):
        self.stable();self.assertTrue(self.consider(2));timer=self.pool.pending['alpha']['timer']
        self.pool.close();self.assertTrue(timer.cancelled);self.assertFalse(self.pool.pending)
        self.assertEqual(len(self.pool.pools['alpha']['rows']),2);self.assertFalse(self.consider(3))

    def test_full_pool_stays_unchanged_after_restart_and_only_deleted_slot_is_filled(self):
        self.stable()
        for i in range(31):
            stamp=10+i;self.pool.observe_identity('alpha',stamp)
            self.assertTrue(self.consider(100+i,stamp))
        self.assertTrue(self.pool.finish_batch('alpha'));self.pool.close()
        self.pool=self.reopen();before=(self.folder/'photo_pool.json').read_bytes();self.stable(50)
        compare=Mock(side_effect=AssertionError('full pool must not compare enrollment anchors'))
        self.assertFalse(self.consider(500,50,compare=compare));compare.assert_not_called()
        self.assertEqual((self.folder/'photo_pool.json').read_bytes(),before)
        from tools.manage_photo_pool import edit_active
        old=list(self.pool.pools['alpha']['rows']);victim=old[-1]
        edit_active(self.folder/'photo_pool.json','alpha',delete=victim['hash']);self.pool.refresh();self.stable(60)
        self.assertTrue(self.consider(500,60));self.assertTrue(self.pool.finish_batch('alpha'))
        data=self.pool.pools['alpha'];self.assertEqual(len(data['rows']),32)
        self.assertEqual(len(data['active']),16)
        self.assertTrue({r['hash'] for r in old[:-1]}<={r['hash'] for r in data['rows']})

    def test_recognition_can_read_gallery_without_waiting_for_photo_writer(self):
        entered=threading.Event();release=threading.Event()
        def writing():
            with self.pool.lock:entered.set();release.wait(2)
        thread=threading.Thread(target=writing);thread.start();self.assertTrue(entered.wait(1))
        try:
            self.assertFalse(self.pool.refresh());self.assertEqual(len(self.pool.gallery()[0]),1)
            self.assertFalse(self.pool.observe_identity('alpha',10));self.assertFalse(self.consider(2))
        finally:release.set();thread.join(1)

    def test_legacy_manual_activation_is_locked_and_filesystem_deletion_reconciles_on_restart(self):
        self.stable();self.assertTrue(self.consider(2));self.pool.finish_batch('alpha');self.pool.close()
        path=self.folder/'photo_pool.json';data=json.loads(path.read_text());row=data['rows'][-1]
        row['excluded']=False;path.write_text(json.dumps(data))
        self.pool=self.reopen();self.assertTrue(self.pool.pools['alpha']['rows'][-1]['locked'])
        (self.folder/row['path']).unlink();self.pool.close();self.pool=self.reopen()
        self.assertEqual(len(self.pool.pools['alpha']['rows']),1);self.stable(20);self.assertTrue(self.consider(3,20))


class RecognitionIsolationTests(unittest.TestCase):
    def test_photo_failure_does_not_change_native_calls_confirmation_or_identity(self):
        from test_seeta_identity import FakeSeeta,config,vector
        from vivis.vision.familiar import FamiliarFaces
        with tempfile.TemporaryDirectory() as folder:
            model=FakeSeeta();model.evaluate_selected=Mock(wraps=model.evaluate_selected)
            f=FamiliarFaces(config(),Path(folder),model=model,gallery=(np.array([vector(0),vector(1)]),np.array(['alpha','beta'])))
            self.addCleanup(f.close)
            pool=Mock(error=None);pool.gallery.return_value=(f.vectors,f.ids);pool.consider.side_effect=OSError('disk full')
            f.photo_pool=pool
            for stamp in (10.,10.5):f.infer(None,np.array([0,0,120,120]),stamp,f.epoch,collect=True)
            self.assertEqual(f.result['state'],'recognized');self.assertEqual(f.result['person_id'],'alpha')
            self.assertEqual(f.photo_error,'OSError');self.assertEqual(model.evaluate_selected.call_count,2)
            self.assertTrue(all(c.kwargs['gate'] for c in model.evaluate_selected.call_args_list))
            self.assertEqual(model.compare.call_count,4)
            model.quality=False;f.infer(None,np.array([0,0,120,120]),11.,f.epoch,collect=True)
            self.assertEqual(f.result['state'],'reviewing');self.assertEqual(pool.observe_identity.call_args.args,(None,11.))


if __name__=='__main__':unittest.main()
