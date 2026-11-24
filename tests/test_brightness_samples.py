"""Rejected-frame diagnostics cannot change identity or grow without bound."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vivis.vision.brightness_samples import BrightnessSamples


class BrightnessSampleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.c=dict(directory='private/diagnostics/face_brightness',maximum_images=2,sample_interval_s=2.)
        self.s=self.sampler();self.row=np.array([10,10,50,50])

    def sampler(self):
        s=BrightnessSamples(self.root,self.c,[.1,.95]);self.addCleanup(s.close);return s

    def offer(self,level,stamp=10.,observation=None,sampler=None):
        s=sampler or self.s;image=np.full((80,100,3),level,np.uint8)
        before=image.copy();accepted=s.offer(image,self.row,observation or {},stamp);s.job.wait(2)
        np.testing.assert_array_equal(image,before);return accepted

    def metadata(self):return [json.loads(p.read_text()) for p in self.s.directory.glob('*.json')]

    def test_good_brightness_does_not_write_or_create_a_folder(self):
        self.assertFalse(self.offer(128));self.assertFalse(self.s.directory.exists())

    def test_range_rejection_uses_face_brightness_not_background(self):
        image=np.full((80,100,3),200,np.uint8);image[10:60,10:60]=10
        self.assertTrue(self.s.offer(image,self.row,{},10));self.s.job.wait(2)
        row=self.metadata()[0]
        self.assertAlmostEqual(row['brightness_mean_0_1'],10/255)
        self.assertEqual(row['rejection_reasons'],['sampling_brightness'])
        self.assertTrue(row['image'].startswith('0.0392_range-low_'))
        self.assertFalse(row['identity_authority']);self.assertTrue(row['diagnostic_only'])
        for p in self.s.directory.glob('*.*'):
            if p.suffix in ('.jpg','.json'):self.assertEqual(p.stat().st_mode&0o777,0o600)

    def test_native_brightness_rejection_is_saved_even_inside_sampling_range(self):
        self.assertTrue(self.offer(80,observation=dict(brightness=dict(level=0,score=.01))))
        row=self.metadata()[0];self.assertGreater(row['brightness_mean_0_1'],.1)
        self.assertEqual(row['rejection_reasons'],['native_brightness'])
        self.assertIn('_native-low_',row['image'])

    def test_high_brightness_and_interval_limit(self):
        self.assertTrue(self.offer(250));self.assertFalse(self.offer(0,11.99))
        self.assertTrue(self.offer(0,12));self.assertEqual(len(self.metadata()),2)
        self.assertIn('_range-high_',self.metadata()[0]['image']+self.metadata()[1]['image'])

    def test_full_pool_survives_restart_and_only_manual_deletion_reopens_space(self):
        self.offer(0,10);self.offer(0,12);original={p.name:p.read_bytes() for p in self.s.directory.glob('*.jpg')}
        again=self.sampler();self.offer(0,20,sampler=again)
        self.assertEqual({p.name:p.read_bytes() for p in self.s.directory.glob('*.jpg')},original)
        self.assertEqual(again.status['retained_images'],2)
        path=next(self.s.directory.glob('*.jpg'));path.with_suffix('.json').unlink();path.unlink()
        self.offer(0,22,sampler=again);self.assertEqual(len(list(self.s.directory.glob('*.jpg'))),2)

    def test_encoding_failure_is_local_and_rolls_back_partial_files(self):
        with patch('cv2.imencode',side_effect=OSError('simulated disk/codec fault')):
            self.assertTrue(self.offer(0))
        self.assertEqual(self.s.status['error'],'OSError')
        self.assertEqual(list(self.s.directory.glob('*.jpg')),[]);self.assertEqual(self.metadata(),[])
        self.assertTrue(self.offer(0,12));self.assertIsNone(self.s.status['error'])

    def test_configuration_cannot_escape_private_or_exceed_hundred_images(self):
        for changes in (dict(directory='outputs/unsafe'),dict(directory='private'),dict(maximum_images=101),dict(maximum_images=0)):
            with self.assertRaises(ValueError):BrightnessSamples(self.root,dict(self.c,**changes),[.1,.95])

    def test_busy_diagnostic_writer_never_queues_more_frames(self):
        self.s.job.lock.acquire()
        try:self.assertFalse(self.offer(0))
        finally:self.s.job.lock.release()
        self.assertFalse(self.s.directory.exists())


if __name__=='__main__':unittest.main()
