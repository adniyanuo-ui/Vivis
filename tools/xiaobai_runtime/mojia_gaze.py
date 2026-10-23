"""Mojia-only bounded eye pursuit, with eye-mounted or head-fixed vision.

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
    eye_speed=c.get('speed_counts_s',60)
    if not finite(eye_speed) or not 0 < eye_speed <= 300:
        raise ValueError('Invalid eye speed for commissioning stage')
    if not c.get('mount_confirmed') or c['camera_mount'] not in ('fixed', 'eye'):
        raise ValueError('Confirm camera mounting before motion')
    if c['camera_mount'] == 'eye' and c.get('camera_eye') != 'right':
        raise ValueError('This commissioning configuration uses the right eye camera')
    if not trial and not c.get('calibration_confirmed'):
        raise ValueError('Confirm eye direction before normal tracking')
    if neck and not trial and not config['neck']['direction_confirmed']:
        raise ValueError('Confirm ID32 direction before normal neck following')
    axes = c['axes']
    expected=[('left_yaw',8,'x'),('right_yaw',23,'x')]
    if c.get('vertical_enabled') is True:
        expected += [('left_pitch',7,'y'),('right_pitch',22,'y')]
        if not trial and not c.get('vertical_direction_confirmed'):
            raise ValueError('Confirm vertical eye directions before normal tracking')
    if [(a['name'], a['id'], a['coordinate']) for a in axes] != expected:
        raise ValueError('Mojia eye axes must match IDs8/23 and optional7/22')
    motors = {m['id']: m for m in profile['motors']}
    for a in axes:
        values = [a[k] for k in ('negative', 'neutral', 'positive')]
        m = motors[a['id']]
        if (any(not finite(v) for v in values) or values[1] != 2047
                or not (values[0] < values[1] < values[2] or values[2] < values[1] < values[0])
                or min(values) < m['safe_min'] or max(values) > m['safe_max']
                or m['model_raw_expected'] != 521):
            raise ValueError('Eye endpoints exceed reviewed Mojia limits')
        cap=30 if a['coordinate']=='y' else 60
        if trial and max(abs(v-2047) for v in values) > cap:
            raise ValueError('Eye trial exceeds coordinate-specific limits')
    n = config['neck']
    if (n['id'] != 32 or n['sign'] not in (-1, 1) or type(n['sign']) is not int
            or not finite(n['offset_counts']) or not 1 <= n['offset_counts'] <= (80 if trial else 160)
            or not finite(n['speed_counts_s']) or not 0 < n['speed_counts_s'] <= 900
            or not finite(n['acceleration_counts_s2']) or not 0 < n['acceleration_counts_s2'] <= 200
            or motors[32]['model_raw_expected'] != 10760
            or not motors[32]['device_min'] <= 2047-n['offset_counts'] < 2047+n['offset_counts'] <= motors[32]['device_max']):
        raise ValueError('Invalid bounded Mojia neck setup')
    for key,default,low,high in [('gain_counts_s',n['speed_counts_s'],1,180),
                                ('eye_lead_s',.15,.05,.5),('deadband',.18,.05,.5),
                                ('stop_deadband',.08,0,.3),('filter_tau_s',.12,.01,1)]:
        value=n.get(key,default)
        if not finite(value) or not low <= value <= high:
            raise ValueError('Invalid neck coordination parameter: '+key)
    if n.get('stop_deadband',.08)>=n.get('deadband',.18):
        raise ValueError('Neck hysteresis thresholds must be ordered')


class MojiaPursuit:
    def __init__(self, config, ids, speeds, neck=False):
        self.config, self.ids, self.speeds, self.neck = config, ids, np.asarray(speeds), neck
        self.policy = GazePolicy(config)
        self.velocity = 0.
        self.direction = 0
        self.pending = None
        self.filtered_eye = 0.

    def step(self, packet, previous, dt, now, returning=False):
        c, n = self.config['control'], self.config['neck']
        anchor = next(a for a in c['axes'] if a['name'] == 'right_yaw')
        vertical=next((a for a in c['axes'] if a['name']=='right_pitch'),None)
        applied = [axis_normalized(anchor, previous[self.ids.index(23)]),
                   axis_normalized(vertical,previous[self.ids.index(22)]) if vertical else 0.]
        xy, weight, state = self.policy.step(packet, now, dt, applied)
        result = np.full(len(previous), 2047.)
        for a in c['axes']:
            coordinate=0 if a['coordinate']=='x' else 1
            result[self.ids.index(a['id'])] = 2047. if returning else 2047*(1-weight)+weight*axis_position(a, xy[coordinate])
        if self.neck:
            col = self.ids.index(32);position = previous[col]
            eye = applied[0]
            desired = 0.
            if returning:
                delta = 2047-position
                desired = math.copysign(min(n['speed_counts_s'], abs(delta)*2,
                                           math.sqrt(2*n['acceleration_counts_s2']*abs(delta))), delta)
            elif state == 'tracking' and weight >= .95:
                self.filtered_eye += -math.expm1(-dt/n.get('filter_tau_s',.12))*(eye-self.filtered_eye)
                stop_band=n.get('stop_deadband',.08)
                direction = int(np.sign(eye)) if abs(eye) >= n.get('deadband',.18) else self.direction if abs(eye) > stop_band else 0
                if self.direction and eye*self.direction<0:
                    direction=0
                if direction != self.direction:
                    self.direction, self.pending = direction, now if direction else None
                if direction and self.pending is not None and now-self.pending >= n.get('eye_lead_s',.15):
                    error=direction*max(0.,direction*self.filtered_eye-stop_band)
                    desired = n['sign']*n.get('gain_counts_s',n['speed_counts_s'])*error
            else:
                self.direction, self.pending = 0, None
                self.filtered_eye=0.
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
