"""Mojia-only bounded horizontal pursuit, with eye-mounted or head-fixed vision.

No serial I/O. Neck motion is closed through the camera image, never subtracted
from eye commands as if commanded head rotation were measured rotation.
"""
import hashlib
import math
import numpy as np

from .person_gaze import GazePolicy, axis_normalized, axis_position, finite, validate_config
from .mapping import limit_step


def validate_mojia(config, profile, calibration, neck=False, trial=False):
    validate_config(config)
    if config.get('schema_version') != 'mojia-person-gaze-1.0' or profile.get('schema_version') != 'mojia-playback-0.1':
        raise ValueError('Need Mojia configuration and profile')
    digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
    if config.get('calibration_sha256') != digest or profile.get('calibration_sha256') != digest:
        raise ValueError('Mojia calibration changed')
    c = config['control']
    if not c.get('mount_confirmed') or c['camera_mount'] not in ('fixed', 'eye'):
        raise ValueError('Confirm camera mounting before motion')
    if c['camera_mount'] == 'eye' and c.get('camera_eye') != 'right':
        raise ValueError('This commissioning configuration uses the right eye camera')
    if not trial and not c.get('calibration_confirmed'):
        raise ValueError('Confirm eye direction before normal tracking')
    if neck and not trial and not config['neck']['direction_confirmed']:
        raise ValueError('Confirm ID32 direction before normal neck following')
    axes = c['axes']
    if [(a['name'], a['id'], a['coordinate']) for a in axes] != [('left_yaw', 8, 'x'), ('right_yaw', 23, 'x')]:
        raise ValueError('Mojia horizontal eyes must be IDs8/23')
    motors = {m['id']: m for m in profile['motors']}
    for a in axes:
        values = [a[k] for k in ('negative', 'neutral', 'positive')]
        m = motors[a['id']]
        if (any(not finite(v) for v in values) or values[1] != 2047
                or not (values[0] < values[1] < values[2] or values[2] < values[1] < values[0])
                or min(values) < m['safe_min'] or max(values) > m['safe_max']
                or m['model_raw_expected'] != 521):
            raise ValueError('Eye endpoints exceed reviewed Mojia limits')
        if trial and max(abs(v-2047) for v in values) > 40:
            raise ValueError('First eye trial is limited to +/-40 counts')
    n = config['neck']
    if (n['id'] != 32 or n['sign'] not in (-1, 1) or type(n['sign']) is not int
            or not finite(n['offset_counts']) or not 1 <= n['offset_counts'] <= (20 if trial else 160)
            or not finite(n['speed_counts_s']) or not 0 < n['speed_counts_s'] <= 25
            or not finite(n['acceleration_counts_s2']) or not 0 < n['acceleration_counts_s2'] <= 35
            or motors[32]['model_raw_expected'] != 10760
            or not motors[32]['device_min'] <= 2047-n['offset_counts'] < 2047+n['offset_counts'] <= motors[32]['device_max']):
        raise ValueError('Invalid bounded Mojia neck setup')


class MojiaPursuit:
    def __init__(self, config, ids, speeds, neck=False):
        self.config, self.ids, self.speeds, self.neck = config, ids, np.asarray(speeds), neck
        self.policy = GazePolicy(config)
        self.velocity = 0.
        self.direction = 0
        self.pending = None

    def step(self, packet, previous, dt, now, returning=False):
        c, n = self.config['control'], self.config['neck']
        anchor = next(a for a in c['axes'] if a['name'] == 'right_yaw')
        applied = [axis_normalized(anchor, previous[self.ids.index(23)]), 0.]
        xy, weight, state = self.policy.step(packet, now, dt, applied)
        result = np.full(len(previous), 2047.)
        for a in c['axes']:
            result[self.ids.index(a['id'])] = 2047. if returning else 2047*(1-weight)+weight*axis_position(a, xy[0])
        if self.neck:
            col = self.ids.index(32);position = previous[col]
            eye = applied[0]
            desired = 0.
            if returning:
                delta = 2047-position
                desired = math.copysign(min(n['speed_counts_s'], abs(delta)*2,
                                           math.sqrt(2*n['acceleration_counts_s2']*abs(delta))), delta)
            elif state == 'tracking' and weight >= .95:
                direction = int(np.sign(eye)) if abs(eye) >= .18 else self.direction if abs(eye) > .08 else 0
                if direction != self.direction:
                    self.direction, self.pending = direction, now if direction else None
                if direction and self.pending is not None and now-self.pending >= .15:
                    desired = n['sign']*n['speed_counts_s']*max(-1., min(1., eye))
            else:
                self.direction, self.pending = 0, None
            low, high = 2047-n['offset_counts'], 2047+n['offset_counts']
            distance = high-position if desired >= 0 else position-low
            cap = min(n['speed_counts_s'], math.sqrt(2*n['acceleration_counts_s2']*max(0., distance)))
            desired = float(np.clip(desired, -cap, cap))
            old = self.velocity
            self.velocity += float(np.clip(desired-old, -n['acceleration_counts_s2']*dt, n['acceleration_counts_s2']*dt))
            goal = float(np.clip(position+(old+self.velocity)*dt/2, low, high))
            if goal in (low, high):
                self.velocity = 0.
            if returning and abs(position-2047) < .25 and abs(self.velocity) <= n['acceleration_counts_s2']*dt:
                goal, self.velocity = 2047., 0.
            result[col] = goal
        return limit_step(previous, result, self.speeds, dt), dict(state=state, weight=weight, eye_error_x=float(xy[0]), neck_velocity=self.velocity)
