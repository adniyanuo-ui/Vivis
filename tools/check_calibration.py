#!/usr/bin/env python3
"""Audit the final calibration and render its axis coverage without hardware I/O."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vivis.runtime.robot import Robot
from vivis.motion.expression import ExpressionMapper

if __name__ == '__main__':
    robot = Robot()
    mapper = ExpressionMapper(robot)
    mapped = set(mapper.frame({}))
    print(json.dumps(dict(calibration_sha256=robot.calibration_sha256,
        shape_count=len(mapper.names), calibrated_axes=sorted(mapped),
        independent_neck_axes=[32,33,34], unmapped_axes=sorted(set(robot.axes)-mapped-{32,33,34})),
        ensure_ascii=False, indent=2))
