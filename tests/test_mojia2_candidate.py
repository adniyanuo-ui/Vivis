"""Offline comparison keeps timing/media and cannot pass the live safety gate."""
import copy
import importlib.util
import json
import unittest

import numpy as np

from vivis.motion.mapping import inputs
from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.trajectory import load_targets

spec = importlib.util.spec_from_file_location('prepare_mojia2_candidate', ROOT/'tools/prepare_mojia2_candidate.py')
candidate_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate_tool)


class Mojia2CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = Robot()
        cls.work = cls.robot.work('0001')
        cls.accepted = json.loads(cls.work['targets'].read_text())
        cls.calibration = ROOT/'configs/robots/mojia2/calibration.json'
        cls.directory = ROOT/'assets/works/mojia2/0001'
        cls.target_path = cls.directory/'targets2.json'
        cls.target = json.loads(cls.target_path.read_text())
        cls.audit = json.loads((cls.directory/'calibration_review.json').read_text())

    def test_metadata_correction_never_changes_position_endpoints(self):
        source = json.loads((self.calibration.parent/self.audit['source_calibration_name']).read_text())
        expected = copy.deepcopy(source)
        changed = []
        for shape in expected['blendShapes']:
            if shape['posSize'] != len(shape['positions']):
                changed.append(shape['shape'])
                shape['posSize'] = len(shape['positions'])
        self.assertEqual(changed, ['JawForward', 'JawLeft', 'JawRight'])
        self.assertEqual(json.loads(self.calibration.read_text()), expected)
        self.assertEqual(candidate_tool.digest(self.calibration), self.target['calibration_sha256'])
        self.assertEqual(candidate_tool.digest(self.calibration.parent/self.audit['source_calibration_name']),
                         self.audit['source_calibration_sha256'])

    def test_accepted_blink_recipe_reproduces_every_original_position(self):
        values, fps, order, shapes = inputs(self.work['source_clip'],self.robot.calibration,self.robot.resolve('channels'))
        ids, positions, _ = candidate_tool.replay_blinks(values,order,shapes,self.robot.motion_profile(),
                                                       self.accepted['blink_adjustment'])
        self.assertEqual(ids,self.accepted['motor_ids'])
        np.testing.assert_array_equal(positions,np.asarray(self.accepted['positions']))
        self.assertEqual(fps,60)

    def test_new_endpoints_reproduce_candidate_without_old_device_clipping(self):
        values, fps, order, shapes = inputs(self.work['source_clip'],self.calibration,self.robot.resolve('channels'))
        profile = self.robot.motion_profile()
        profile['motors'] = [dict(id=sid,neutral=axis['neutral'],safe_min=0,safe_max=4095)
                            for sid,axis in self.robot.axes.items()]
        ids, positions, report = candidate_tool.replay_blinks(values,order,shapes,profile,
                                                            self.accepted['blink_adjustment'])
        self.assertEqual(ids,self.target['motor_ids'])
        self.assertEqual(set(ids)-set(self.accepted['motor_ids']),{9,24})
        self.assertTrue(set(ids).isdisjoint({32,33,34}))
        np.testing.assert_array_equal(positions,np.asarray(self.target['positions']))
        self.assertFalse(any(report['clipped_samples_by_id'].values()))
        for sid in (9,24):
            self.assertGreater(np.ptp(positions[:,ids.index(sid)]),0)
        for side in ('Left','Right'):
            for event in self.accepted['blink_adjustment']['events']:
                if event['side'] == side:
                    for sid,endpoint in shapes['EyeBlink'+side].items():
                        self.assertEqual(positions[event['closed_start_frame'],ids.index(sid)],endpoint)

    def test_same_audio_source_fps_and_gain(self):
        for key in ('audio','source_clip'):
            self.assertEqual(ROOT/self.target[key],self.work[key])
            self.assertEqual(candidate_tool.digest(self.work[key]),self.target[key+'_sha256'])
        self.assertEqual(self.target['fps'],self.accepted['fps'])
        self.assertEqual(self.target['frames'],6774)
        self.assertEqual(self.target['motion_gain'],1.)
        self.assertEqual(candidate_tool.digest(self.target_path),self.audit['targets_sha256'])

    def test_live_loader_refuses_candidate_before_a_real_hardware_scan(self):
        profile = self.robot.profile('0001')
        profile.update(targets_sha256=candidate_tool.digest(self.target_path),
                       calibration_sha256=candidate_tool.digest(self.calibration))
        self.assertFalse(self.target['device_limits_applied'])
        self.assertFalse(self.target['playback_ready'])
        self.assertTrue(self.target['trajectory_report']['draft_only'])
        with self.assertRaisesRegex(ValueError,'complete Mojia hardware scan'):
            load_targets(self.target_path,profile,self.work['audio'],self.calibration)


if __name__ == '__main__':
    unittest.main()
