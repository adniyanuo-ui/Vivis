"""Named Blendshape input and explicit, opt-in expression composition.

This module produces nominal positions only. It never sends serial commands;
the runtime applies ownership, speed/acceleration limits and feedback checks.
"""
import csv
import math
import numpy as np

from vivis.motion.mapping import compile_trajectory, read_json


def adapt_channels(values, source_to_canonical, canonical_names):
    """Rename explicitly. Missing target channels are zero; collisions fail."""
    result = {}
    for source, value in values.items():
        if source not in source_to_canonical:
            raise ValueError('No semantic mapping for source channel: ' + source)
        name = source_to_canonical[source]
        if name not in canonical_names or name in result:
            raise ValueError('Unknown or duplicate destination channel: ' + name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Blendshape coefficients must be finite and within0..1')
        result[name] = float(value)
    return result


def fuse_layers(base, overlays, canonical_names):
    """Explicit sequential overrides: (named_weights, strength) per layer.

    Only named channels in an overlay are affected. Later layers take priority
    by interpolation. Not applied automatically to either included work.
    """
    identity = {name: name for name in canonical_names}
    result = adapt_channels(base, identity, canonical_names)
    for weights, strength in overlays:
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Layer strength must be within0..1')
        layer = adapt_channels(weights, identity, canonical_names)
        for name, value in layer.items():
            result[name] = result.get(name, 0.) * (1-strength) + value * strength
    return result


class ExpressionMapper:
    def __init__(self, robot):
        self.robot = robot
        with robot.resolve('channels').open(encoding='utf-8-sig', newline='') as handle:
            self.channels = next(csv.reader(handle))[2:]
        self.shapes = {s['shape']: {p['id']: p['position'] for p in s['positions']}
                       for s in read_json(robot.calibration)['blendShapes']}
        if len(self.channels) != 61 or len(set(self.channels)) != 61 or set(self.channels[:52]) != set(self.shapes):
            raise ValueError('Calibration and channel semantics differ')
        self.names = tuple(self.channels[:52])

    def frame(self, weights):
        named = adapt_channels(weights, {n: n for n in self.names}, self.names)
        values = np.zeros((1, 61))
        for name, value in named.items():
            values[0, self.channels.index(name)] = value
        ids, positions, report = compile_trajectory(values, self.channels, self.shapes,
                                                    self.robot.motion_profile(), scope='face_eyes')
        return dict(zip(ids, positions[0].tolist()))
