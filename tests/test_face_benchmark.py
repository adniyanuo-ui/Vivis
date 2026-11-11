"""Offline benchmark accounting and lossless file replay; no real devices."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from vivis.vision.face_benchmark import SFace,Seeta,capture_clip,frames,rank,summarize


class FaceBenchmarkTests(unittest.TestCase):
    def test_review_excludes_invalid_framing_and_retains_valid_time_interval(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);meta=dict(schema_version='vivis-face-ab-capture-1.0',completed=True,path='raw.avi',timestamps='times.jsonl',person_id='alice',expected_faces=1,scene='normal')
            (folder/'capture.json').write_text(json.dumps(meta))
            (folder/'review.json').write_text(json.dumps(dict(usable=False)))
            with self.assertRaisesRegex(ValueError,'excluded'):capture_clip(folder)
            (folder/'review.json').write_text(json.dumps(dict(usable=True,start_s=10.,end_s=19.9)))
            result=capture_clip(folder);self.assertEqual(result['start_s'],10.);self.assertEqual(result['end_s'],19.9)
    def test_framing_page_is_live_and_guide_never_modifies_raw_pixels(self):
        import time,urllib.request
        from types import SimpleNamespace
        from vivis.vision.recording_preview import FramingPreview
        raw=np.full((120,160,3),100,np.uint8);original=raw.copy()
        preview=FramingPreview(SimpleNamespace(get=lambda:(1,time.monotonic(),raw)),port=0)
        try:
            self.assertTrue(preview.ready.wait(2));self.assertIsNone(preview.error)
            base='http://127.0.0.1:'+str(preview.preview.server.server_address[1])
            with urllib.request.urlopen(base,timeout=2) as response:page=response.read().decode()
            self.assertIn('只取景，不控制机器头',page)
            with urllib.request.urlopen(base+'/frame.jpg',timeout=2) as response:self.assertGreater(len(response.read()),100)
            np.testing.assert_array_equal(raw,original)
        finally:preview.close()

    def test_sface_baseline_preserves_production_minimum_face_size(self):
        from unittest.mock import Mock
        engine=SFace.__new__(SFace);engine.minimum_face_size=60;engine.recognizer=Mock()
        engine.vision=Mock();engine.vision.selected_face_row=np.array([0,0,59,80]+[0]*11)
        engine.vision.detect.return_value=(None,None,dict(faces=1))
        with patch('vivis.vision.familiar.feature') as extract:
            result,vector=engine.evaluate(np.zeros((100,100,3),np.uint8))
        self.assertTrue(result['size_rejected']);self.assertIsNone(vector);extract.assert_not_called()

    def test_quality_low_is_not_called_rejected_unless_gate_was_enabled(self):
        import ctypes
        from types import SimpleNamespace
        def evaluate(_handle,_image,_w,_h,gate,_features,_capacity,metadata):
            meta=np.ctypeslib.as_array(ctypes.cast(metadata,ctypes.POINTER(ctypes.c_double)),shape=(16,))
            meta[0]=1;meta[7]=0;meta[9]=2;meta[15]=int(not gate)
            return 0
        engine=Seeta.__new__(Seeta);engine.handle=1;engine.dimension=4
        engine.lib=SimpleNamespace(sf6_eval=evaluate)
        for gate in (False,True):
            result,vector=engine.evaluate(np.zeros((10,10,3),np.uint8),gate)
            self.assertTrue(result['quality_low']);self.assertEqual(result['quality_rejected'],gate)
            self.assertEqual(vector is None,gate)

    def test_score_and_runner_up_margin_are_both_required(self):
        self.assertEqual(rank({'a':.8,'b':.3},.6,.08)['prediction'],'a')
        self.assertIsNone(rank({'a':.59,'b':.3},.6,.08)['prediction'])
        self.assertIsNone(rank({'a':.8,'b':.79},.6,.08)['prediction'])
        self.assertIsNone(rank({},.6,.08)['best_score'])

    def row(self,**extra):
        return dict(clip=0,prediction='alice',truth='alice',expected_faces=1,faces=1,
                    scores={'alice':.8,'bob':.1},wall_ms=20,quality_rejected=False,**extra)

    def test_quality_rejection_and_detection_failure_stay_in_known_denominator(self):
        rows=[self.row()]
        for changes in (dict(prediction=None,quality_rejected=True),dict(prediction=None,faces=0)):
            row=self.row();row.update(changes);rows.append(row)
        result=summarize(rows,True)
        self.assertEqual(result['known_face_frames'],3)
        self.assertEqual(result['known_rejected_frames'],2)
        self.assertAlmostEqual(result['known_rejection_rate'],2/3)
        self.assertAlmostEqual(result['quality_rejection_rate'],1/3)
        self.assertEqual(result['raw_decision_transitions'],1)

    def test_wrong_known_identity_and_unknown_false_accept_have_separate_denominators(self):
        wrong=self.row();wrong['prediction']='bob'
        stranger=self.row();stranger['truth']=None
        result=summarize([wrong,stranger],False)
        self.assertEqual(result['known_wrong_id_frames'],1)
        self.assertEqual(result['unknown_false_accept_frames'],1)
        self.assertEqual(result['unknown_false_accept_rate'],1.)
        self.assertIsNone(result['raw_decision_transitions'])
        self.assertIsNone(summarize([wrong],False)['unknown_false_accept_rate'])

    def test_clip_boundaries_are_not_identity_changes(self):
        other=self.row();other.update(clip=1,prediction='bob',truth='bob')
        result=summarize([self.row(),other],True)
        self.assertEqual(result['transition_opportunities'],0)
        self.assertEqual(result['raw_decision_transitions'],0)

    def test_device_path_or_absent_file_never_opens_capture(self):
        with patch('cv2.VideoCapture') as capture:
            with self.assertRaises(ValueError):list(frames(dict(path='/dev/video0'),Path('/'),2))
            capture.assert_not_called()

    def test_lossless_video_replay_preserves_pixels_and_samples_real_timestamps(self):
        import cv2,json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'raw.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),30,(64,48))
            self.assertTrue(writer.isOpened())
            rng=np.random.default_rng(4);images=[rng.integers(0,256,(48,64,3),dtype=np.uint8) for _ in range(4)]
            for image in images:writer.write(image)
            writer.release()
            (root/'times.jsonl').write_text(''.join(json.dumps(dict(elapsed_s=t))+'\n' for t in (0.,.2,.6,1.1)))
            output=list(frames(dict(path='raw.avi',timestamps='times.jsonl'),root,2))
            self.assertEqual([r[0] for r in output],[0,2,3])
            for index,_,image in output:np.testing.assert_array_equal(image,images[index])


if __name__=='__main__':unittest.main()
