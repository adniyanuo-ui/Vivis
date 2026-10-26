"""Standalone distribution checks: actual bundled resources, no camera/serial I/O."""
from pathlib import Path
import time
import unittest

import numpy as np

from vivis.paths import ROOT
from vivis.runtime.assets import verify_assets, verify_model
from vivis.runtime.robot import Robot


class DistributionTests(unittest.TestCase):
    def test_bundle_has_real_media_and_local_sdk_without_legacy_directories(self):
        from vivis.hardware.feetech.sdk import PortHandler
        import inspect
        for name in ('xiaobai', 'mojia', 'exmple', 'docs/history', 'tools/xiaobai_runtime'):
            self.assertFalse((ROOT / name).exists(), name)
        sdk_path = Path(inspect.getfile(PortHandler)).resolve()
        self.assertTrue(sdk_path.is_relative_to(ROOT / 'src/vivis/hardware/feetech/_vendor'))
        rows = verify_assets(Robot())
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(Path(row['path']).stat().st_size > 1000 for row in rows))
        self.assertEqual(verify_model().stat().st_size, 232589)

    def test_bundled_model_runs_on_a_synthetic_image_without_a_camera(self):
        from vivis.vision.detection import FaceVision
        from vivis.vision.policy import VERSION
        detector = FaceVision(Robot().tracking(), ROOT)
        packet, jpeg, metrics = detector.detect((1, time.monotonic(), np.zeros((480, 640, 3), dtype=np.uint8)))
        self.assertEqual(packet['schema_version'], VERSION)
        self.assertIsNone(packet['target'])
        self.assertGreater(len(jpeg), 100)


if __name__ == '__main__':
    unittest.main()
