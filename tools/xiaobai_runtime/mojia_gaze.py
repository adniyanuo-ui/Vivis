"""Mojia-only bounded eye pursuit, with eye-mounted or head-fixed vision.

No serial I/O. Neck motion is closed through the camera image, never subtracted
from eye commands as if commanded head rotation were measured rotation.
"""
import hashlib
import math
import numpy as np

from .person_gaze import GazePolicy, axis_normalized, axis_position, finite, validate_config
from .mapping import limit_step


def camera_axes(config):
    """Select the moving camera's encoder reference from its physical mount."""
    c=config['control'];side=c.get('camera_eye') or 'right'
    if side not in ('left','right'):raise ValueError('Unknown camera eye')
    axes={a['name']:a for a in c['axes']}
    if side+'_yaw' not in axes:raise ValueError('Missing camera eye yaw axis')
    return axes[side+'_yaw'],axes.get(side+'_pitch')


def neck_command_limits(config,profile=None):
    n=config['neck'];mode=n.get('range_mode','offset')
    if mode=='device':
        if profile is None:raise ValueError('Device neck range requires the hardware profile')
        motor=next(m for m in profile['motors'] if m['id']==n['id'])
        return motor['device_min'],motor['device_max']
    if mode!='offset':raise ValueError('Unknown neck range mode')
    return 2047-n['offset_counts'],2047+n['offset_counts']


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
    if c['camera_mount'] == 'eye' and c.get('camera_eye') not in ('left','right'):
        raise ValueError('Confirm which eye carries the camera')
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
        axis_speed=a.get('speed_counts_s',eye_speed)
        if not finite(axis_speed) or not 0 < axis_speed <= 300:
            raise ValueError('Invalid per-axis eye speed')
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
    load_limit=n.get('max_load_raw',motors[32]['max_load_raw'])
    if type(load_limit) is not int or not 1<=load_limit<=1023:
        raise ValueError('Invalid neck software load threshold')
    if n.get('coordination_mode') != 'visual_error_assist' or 'eye_lead_s' in n:
        raise ValueError('Mojia neck requires visual_error_assist without a fixed eye delay')
    if type(config['camera']['width']) is not int or config['camera']['width'] <= 0:
        raise ValueError('Need positive camera width for image growth threshold')
    if type(n.get('adaptive_speed',True)) is not bool:
        raise ValueError('Invalid neck adaptive speed flag')
    low,high=neck_command_limits(config,profile)
    if n.get('range_mode','offset')=='offset' and (
            not finite(n['offset_counts']) or not 1<=n['offset_counts']<=(80 if trial else 160)):
        raise ValueError('Invalid bounded Mojia neck setup')
    if (n['id'] != 32 or n['sign'] not in (-1, 1) or type(n['sign']) is not int
            or not finite(n['speed_counts_s']) or not 0 < n['speed_counts_s'] <= 900
            or not finite(n['acceleration_counts_s2']) or not 0 < n['acceleration_counts_s2'] <= 200
            or motors[32]['model_raw_expected'] != 10760
            or not finite(low) or not finite(high)
            or not motors[32]['device_min']<=low<2047<high<=motors[32]['device_max']):
        raise ValueError('Invalid bounded Mojia neck setup')
    for key,default,low,high in [('gain_counts_s',n['speed_counts_s'],1,180),
                                ('deadband',.18,.05,.5),
                                ('stop_deadband',.08,0,.3),('filter_tau_s',.12,.01,1)]:
        value=n.get(key,default)
        if not finite(value) or not low <= value <= high:
            raise ValueError('Invalid neck coordination parameter: '+key)
    if n.get('stop_deadband',.08)>=n.get('deadband',.18):
        raise ValueError('Neck hysteresis thresholds must be ordered')


class VisualNeckAssist:
    """Recruit the neck from fresh image-error growth, never an eye-position timer.

    The existing neck hysteresis now applies to normalized image error. Two
    image pixels of accumulated growth reject subpixel trend noise. Actual eye
    feedback supplies movement evidence in production; command evidence is an
    explicit offline simulation fallback. Saturated eyes can request help even
    when they can no longer move. No command is treated as measured head motion.
    """
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.stamp = None
        self.raw = self.filtered = self.growth = 0.
        self.direction = 0
        self.reason = 'eyes_only'

    def step(self, packet, state, eye, eye_motion, eye_span):
        n = self.config['neck']
        if state != 'tracking':
            self.reset()
            self.reason = 'no_fresh_face'
            return 0.
        stamp = packet['timestamp_monotonic']
        if self.stamp is None or stamp > self.stamp:
            raw = float(packet['target']['x'])
            increasing = self.stamp is not None and raw*self.raw > 0 and abs(raw) > abs(self.raw)
            if self.stamp is None:
                self.filtered, self.growth = raw, 0.
            else:
                frame_dt = stamp-self.stamp
                same_side = raw*self.raw > 0
                self.growth = (max(0., self.growth+abs(raw)-abs(self.raw))
                               if same_side else 0.)
                self.filtered += -math.expm1(-frame_dt/n['filter_tau_s'])*(raw-self.filtered)
            self.raw, self.stamp = raw, stamp
            sign = int(np.sign(raw))
            # A reversal must first brake the old request. It does not wait for
            # the eye to cross its mechanical centre, even at a neck endpoint.
            if self.direction and (sign != self.direction or abs(raw) <= n['stop_deadband']):
                self.direction = 0
                self.growth = 0.
                self.reason = 'braking'
            saturated = eye*sign >= 1.-1./eye_span
            growing = increasing and self.growth >= 4./self.config['camera']['width']
            if not self.direction:
                if abs(raw) >= n['deadband'] and growing and eye_motion*sign > 0:
                    self.direction, self.reason = sign, 'growing_error'
                elif abs(raw) > n['stop_deadband'] and saturated:
                    self.direction, self.reason = sign, 'eye_limit'
            if self.direction:
                self.reason = 'assisting' if self.reason == 'eyes_only' else self.reason
        if not self.direction:
            return 0.
        # Filtering cannot request motion in the old direction after a reversal.
        error = self.direction*max(0., self.direction*self.filtered-n['stop_deadband'])
        return n['sign']*n['gain_counts_s']*error


class MojiaPursuit:
    def __init__(self, config, ids, speeds, neck=False,neck_bounds=None):
        self.config, self.ids, self.speeds, self.neck = config, ids, np.asarray(speeds), neck
        self.camera_yaw,self.camera_pitch=camera_axes(config)
        self.camera_eye_id=self.camera_yaw['id']
        if neck:
            if neck_bounds is None:neck_bounds=neck_command_limits(config)
            self.neck_bounds=tuple(neck_bounds)
        self.policy = GazePolicy(config)
        self.velocity = 0.
        self.assist = VisualNeckAssist(config)
        self.eye_observation = None
        self.eye_motion = 0.
        self.eye_motion_at = -math.inf
        self.measured_eye = False

    def return_window(self,position):
        if not self.neck:return 5.
        a=self.config['neck']['acceleration_counts_s2'];speed=abs(self.velocity)
        distance=abs(position-2047)+speed*speed/(2*a)
        return max(5.,speed/a+2*math.sqrt(distance/a)+1.)

    def observe_eye(self, position, now, measured=True):
        """Record the camera eye's feedback without adding serial reads."""
        if not math.isfinite(position) or not math.isfinite(now):
            raise ValueError('Invalid eye observation')
        if measured and not self.measured_eye:
            self.eye_observation = None
            self.eye_motion = 0.
        self.measured_eye = self.measured_eye or measured
        if self.eye_observation is None:
            self.eye_observation = float(position)
        delta = position-self.eye_observation
        if abs(delta) >= 1.:
            self.eye_motion, self.eye_motion_at = float(np.sign(delta)), now
            self.eye_observation = float(position)

    def step(self, packet, previous, dt, now, returning=False):
        c, n = self.config['control'], self.config['neck']
        anchor,vertical=self.camera_yaw,self.camera_pitch
        applied = [axis_normalized(anchor, previous[self.ids.index(anchor['id'])]),
                   axis_normalized(vertical,previous[self.ids.index(vertical['id'])]) if vertical else 0.]
        xy, weight, state = self.policy.step(packet, now, dt, applied)
        if not self.measured_eye:
            self.observe_eye(previous[self.ids.index(self.camera_eye_id)], now, measured=False)
        eye_motion = self.eye_motion if now-self.eye_motion_at <= self.config['transport']['max_age_s'] else 0.
        if anchor['positive'] < anchor['neutral']:
            eye_motion = -eye_motion
        result = np.full(len(previous), 2047.)
        for a in c['axes']:
            coordinate=0 if a['coordinate']=='x' else 1
            result[self.ids.index(a['id'])] = 2047. if returning else 2047*(1-weight)+weight*axis_position(a, xy[coordinate])
        if self.neck:
            col = self.ids.index(32);position = previous[col]
            eye = applied[0]
            desired = 0.
            if returning:
                self.assist.reset()
                delta = 2047-position
                desired = math.copysign(min(n['speed_counts_s'], abs(delta)*2,
                                           math.sqrt(2*n['acceleration_counts_s2']*abs(delta))), delta)
            else:
                span = abs(anchor['positive' if eye >= 0 else 'negative']-anchor['neutral'])
                desired = self.assist.step(packet,state,eye,eye_motion,span)
            low, high = self.neck_bounds
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
        return limit_step(previous, result, self.speeds, dt), dict(
            state=state, weight=weight, eye_error_x=float(xy[0]), neck_velocity=self.velocity,
            image_error_x=self.assist.raw, filtered_image_error_x=self.assist.filtered,
            image_error_growth=self.assist.growth, camera_timestamp=self.assist.stamp,
            eye_motion_direction=eye_motion, eye_motion_source='encoder' if self.measured_eye else 'command_simulation',
            camera_eye=c.get('camera_eye'),camera_eye_id=self.camera_eye_id,
            neck_assist_reason='returning' if returning else self.assist.reason,
            neck_requested_velocity=desired if self.neck else 0.,
            neck_at_limit=bool(self.neck and (result[self.ids.index(32)]<=low or result[self.ids.index(32)]>=high)))


class NeckCommandScheduler:
    """Coalesce tiny neck targets and match servo speed to average trajectory rate.

    This only shapes ID32 transport; the continuous eye/head planner and eye
    limits are unchanged. Speed remains bounded by the configured ceiling.
    """
    def __init__(self, position, now, maximum_speed, adaptive_speed=True,endpoints=()):
        self.goal=int(round(position))
        self.sent_at=now
        self.maximum_speed=int(maximum_speed)
        self.adaptive_speed=adaptive_speed
        self.endpoints=set(endpoints)
        self.speed=min(50,self.maximum_speed) if adaptive_speed else self.maximum_speed

    def step(self, planned, now, returning=False):
        goal=int(round(planned))
        dt=now-self.sent_at
        if dt<=0 or not math.isfinite(planned) or not math.isfinite(now):
            raise ValueError('Invalid neck scheduler sample')
        delta=abs(goal-self.goal)
        final_home=returning and goal==2047
        if delta and dt>=.04 and (delta>=2 or final_home or goal in self.endpoints):
            # SMS fine speed units retain approximately 50-count/s precision.
            speed=math.ceil((delta/dt)/50)*50
            self.speed=min(self.maximum_speed,max(1,speed)) if self.adaptive_speed else self.maximum_speed
            self.goal,self.sent_at=goal,now
        return self.goal,self.speed
