"""Named lip gains preserve media, timing, unrelated motion and hardware."""
import hashlib,io,json,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np

from vivis.motion.channel_gain import round_lip_gain_targets
from vivis.runtime.robot import Robot
from vivis.runtime.trajectory import load_targets


class RoundLipGainTests(unittest.TestCase):
    def fixture(self):
        robot=Robot();work=robot.work('wz2')
        ids,positions,fps=load_targets(work['targets'],robot.profile('wz2'),work['audio'],robot.calibration)
        return robot,work,ids,positions,fps

    def test_default_is_exact_noop_and_needs_no_source(self):
        robot,work,ids,a,fps=self.fixture();work.pop('source_clip')
        with patch('vivis.motion.channel_gain.inputs',side_effect=AssertionError('source read')):
            b,report=round_lip_gain_targets(robot,work,ids,a,fps)
        np.testing.assert_array_equal(a,b);self.assertEqual(report['gain'],1.)

    def test_two_channels_only_and_preserves_all_unrelated_columns_and_files(self):
        robot,work,ids,a,fps=self.fixture();old=a.copy();hardware=robot.motion_profile()
        paths=[work['source_clip'],work['targets'],work['audio'],robot.calibration,robot.resolve('hardware')]
        digests={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        b,report=round_lip_gain_targets(robot,work,ids,a,fps,1.2)
        affected=[10,11,12,13,15,25,26,27,28,30]
        self.assertEqual(report['changed_ids'],affected);self.assertEqual(report['baseline_reproduction_max_error_counts'],0.)
        for name,data in report['coefficients'].items():
            self.assertIn(name,('MouthFunnel','MouthPucker'))
            self.assertAlmostEqual(data['peak_after'],1.2*data['peak_before']);self.assertEqual(data['saturated_frames'],0)
        self.assertFalse(any(report['position_clipped_frames_by_id'].values()))
        np.testing.assert_array_equal(a,old)
        for j,sid in enumerate(ids):
            if sid not in affected:np.testing.assert_array_equal(a[:,j],b[:,j])
        self.assertEqual(robot.motion_profile(),hardware)
        self.assertEqual({p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},digests)

    def test_bad_gain_missing_or_corrupt_source_and_mismatched_baseline_fail(self):
        robot,work,ids,a,fps=self.fixture()
        for gain in (0,-1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):round_lip_gain_targets(robot,work,ids,a,fps,gain)
        with self.assertRaisesRegex(ValueError,'verified source'):
            round_lip_gain_targets(robot,{},ids,a,fps,1.2)
        with tempfile.TemporaryDirectory() as folder:
            wrong=Path(folder)/'source.json';wrong.write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                round_lip_gain_targets(robot,{**work,'source_clip':wrong},ids,a,fps,1.2)
        a[0,ids.index(12)]+=1
        with self.assertRaisesRegex(ValueError,'reproduce accepted'):
            round_lip_gain_targets(robot,work,ids,a,fps,1.2)

    def test_all_examples_propagate_gain_without_opening_devices(self):
        from vivis.runtime.developer import run_example
        with patch('vivis.hardware.feetech.sdk.OfficialSDKBus',side_effect=AssertionError('serial opened')), \
             patch('vivis.runtime.developer.subprocess.Popen',side_effect=AssertionError('camera opened')):
            for mode in ('face','eyes','eyes-neck'):
                with redirect_stdout(io.StringIO()) as output:
                    run_example(mode,['--check','--work','wz2','--round-lip-gain','1.2',
                                      '--mouth-speed','1500','--mouth-command-mode','lips-direct'])
                report=json.loads(output.getvalue())
                self.assertEqual(report['round_lip_gain_report']['gain'],1.2)
                self.assertEqual(report['mouth_speed_override']['speed_counts_s'],1500)
                self.assertEqual(report['mouth_gain_report']['gain'],1.)
