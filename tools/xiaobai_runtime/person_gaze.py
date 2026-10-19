"""Nonblocking vision transport and bounded eye/head arbitration. No serial I/O."""
import json
import copy
import math
import socket
import time

import numpy as np

VERSION = 'xiaobai-person-target-1.0'
AXES = {'left_yaw': 'x', 'right_yaw': 'x'}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_config(config, live=False, profile=None, ids=None, direction_trial=False):
    if config.get('schema_version') != 'xiaobai-person-gaze-1.0':
        raise ValueError('Unsupported person gaze configuration')
    transport = config['transport']
    if (transport['host'] != '127.0.0.1' or type(transport['port']) is not int
            or not 1024 <= transport['port'] <= 65535):
        raise ValueError('Vision transport must use a nonprivileged loopback UDP port')
    if not finite(transport['max_age_s']) or not 0.02 <= transport['max_age_s'] <= 1:
        raise ValueError('Invalid vision freshness limit')
    control = config['control']
    for key, low, high in [('deadband', 0, .5), ('filter_tau_s', .001, 2),
                           ('fixed_gain', .01, 2), ('eye_velocity_gain', .01, 5),
                           ('max_axis_speed', .01, 5), ('acquire_s', .05, 5),
                           ('lost_hold_s', 0, 2), ('release_s', .05, 5)]:
        if not finite(control[key]) or not low <= control[key] <= high:
            raise ValueError('Invalid gaze parameter: ' + key)
    if control['camera_mount'] not in ('unconfirmed', 'fixed', 'eye'):
        raise ValueError('Unknown camera mount')
    if not live:
        return
    if (control.get('mount_confirmed') is not True
            or control['camera_mount'] == 'unconfirmed'
            or (not direction_trial and control.get('calibration_confirmed') is not True)):
        raise ValueError('Confirm camera mount and eye calibration before live tracking')
    if control['camera_mount'] == 'eye' and control['camera_eye'] not in ('left', 'right'):
        raise ValueError('Confirm which eye carries the camera')
    axes = control['axes']
    if len(axes) != 2 or {a['name'] for a in axes} != set(AXES):
        raise ValueError('Need exactly two horizontal eye axes')
    motors = {m['id']: m for m in profile['motors']}
    trial_offset = profile.get('gaze_direction_trial_offset_counts', 40)
    if direction_trial and (type(trial_offset) is not int or trial_offset not in (40, 80)):
        raise ValueError('Direction trial offset must be 40 or 80 counts')
    trial_speed = trial_offset * 1.5
    seen = set()
    for axis in axes:
        sid = axis['id']
        if (type(sid) is not int or sid in seen or sid not in ids or not 2 <= sid <= 30
                or axis['coordinate'] != AXES[axis['name']]):
            raise ValueError('Eye axes must have unique face IDs and matching coordinates')
        seen.add(sid)
        if sid != {'left_yaw': 8, 'right_yaw': 23}[axis['name']]:
            raise ValueError('Xiaobai horizontal tracking owns only confirmed IDs 8 and 23')
        values = [axis[k] for k in ('negative', 'neutral', 'positive')]
        if (any(not finite(v) for v in values)
                or not (values[0] < values[1] < values[2] or values[2] < values[1] < values[0])):
            raise ValueError('Eye endpoints must straddle neutral with a consistent direction')
        motor = motors[sid]
        if min(values) < motor['safe_min'] or max(values) > motor['safe_max']:
            raise ValueError(f'Eye axis {sid} exceeds the playback command envelope')
        if direction_trial and (max(abs(v - axis['neutral']) for v in values) > trial_offset
                                or motor['safe_min'] < axis['neutral'] - trial_offset
                                or motor['safe_max'] > axis['neutral'] + trial_offset
                                or not 1 <= motor['max_speed_counts_s'] <= trial_speed):
            raise ValueError(f'Direction trial requires +/-{trial_offset}-count bounds and <={trial_speed:g} counts/s')


def prepare_direction_trial(config, profile, offset_counts=40):
    """Copies only: permit a narrow first test without claiming verified direction."""
    if type(offset_counts) is not int or offset_counts not in (40, 80):
        raise ValueError('Direction trial offset must be 40 or 80 counts')
    config, profile = copy.deepcopy(config), copy.deepcopy(profile)
    motors = {m['id']: m for m in profile['motors']}
    for axis in config['control']['axes']:
        motor = motors[axis['id']]
        neutral = axis['neutral']
        low, high = max(motor['safe_min'], neutral - offset_counts), min(motor['safe_max'], neutral + offset_counts)
        for key in ('negative', 'positive'):
            axis[key] = float(np.clip(axis[key], low, high))
        motor.update(safe_min=low, safe_max=high,
                     max_speed_counts_s=min(motor['max_speed_counts_s'], offset_counts * 1.5))
    config['control']['fixed_gain'] = 1.
    profile['gaze_direction_trial'] = True
    profile['gaze_direction_trial_offset_counts'] = offset_counts
    return config, profile


def validate_packet(packet, now, max_age):
    if packet.get('schema_version') != VERSION:
        raise ValueError('Unknown target packet version')
    stamp = packet['timestamp_monotonic']
    if not finite(stamp) or not 0 <= now - stamp <= max_age:
        raise ValueError('Stale or future camera frame')
    if type(packet['sequence']) is not int or packet['sequence'] < 0:
        raise ValueError('Invalid camera sequence')
    target = packet['target']
    if target is not None:
        if not isinstance(target, dict) or any(
                not finite(target[k]) or not -1 <= target[k] <= 1 for k in ('x', 'y')):
            raise ValueError('Invalid gaze target coordinates')
    return packet


def prepare_neck_follow_trial(profile, sid, sign=1, offset_counts=20, speed_multiplier=1):
    """One user-selected neck axis, within existing limits; never enable torque."""
    if type(sid) is not int or sid != 32 or type(sign) is not int or sign not in (-1, 1):
        raise ValueError('Horizontal neck trial supports only ID32 and sign -1/+1; ID34 is confirmed lateral tilt')
    if type(offset_counts) is not int or offset_counts not in (20, 80, 160):
        raise ValueError('Neck offset must be 20, 80 or 160 counts')
    if not finite(speed_multiplier) or speed_multiplier not in (1, 2, 3):
        raise ValueError('Neck speed multiplier must be 1, 2 or 3')
    result = copy.deepcopy(profile)
    motor = next(m for m in result['motors'] if m['id'] == sid)
    neutral = motor['neutral']
    if neutral != 2047:
        raise ValueError('Neck trial requires the reviewed 2047 home')
    # The initial +/-20 envelope is a software commissioning cap, not a measured
    # mechanical stroke. Explicit 80/160-count modes use retained device
    # limits; the live startup must verify that these targets still fit hardware.
    low_limit, high_limit = motor['safe_min'], motor['safe_max']
    if offset_counts > 20:
        low_limit, high_limit = motor['device_min'], motor['device_max']
        if not all(finite(v) and 0 <= v <= 4095 for v in (low_limit, high_limit)):
            raise ValueError('Expanded neck trial requires reviewed device limits')
    low, high = max(low_limit, neutral - offset_counts), min(high_limit, neutral + offset_counts)
    if not low < neutral < high or not finite(motor['max_speed_counts_s']) or motor['max_speed_counts_s'] <= 0:
        raise ValueError('Neck trial requires a bounded range on both sides of home')
    motor.update(safe_min=low, safe_max=high,
                 max_speed_counts_s=min(25*speed_multiplier, motor['max_speed_counts_s']))
    commands = result.get('diagnostic_command_ids', [m['id'] for m in result['motors'] if m['id'] < 32])
    if set(commands) & {32, 33, 34}:
        raise ValueError('Start neck trial from a face-only command profile')
    result['diagnostic_command_ids'] = list(commands) + [sid]
    # The 3x observation keeps the acceleration of the accepted 2x setting.
    acceleration = 35.*min(speed_multiplier, 2)
    return_seconds = max(3., max(neutral-low, high-neutral)/motor['max_speed_counts_s']
                         + motor['max_speed_counts_s']/acceleration + 2.)
    settings = dict(id=sid, sign=sign, offset_counts=offset_counts, speed_multiplier=speed_multiplier,
                    neutral=neutral, safe_min=low, safe_max=high,
                    gain_counts_s=60.*speed_multiplier, eye_lead_s=.08, deadband=.14, stop_deadband=.07,
                    filter_tau_s=.12, max_acceleration_counts_s2=acceleration,
                    return_home_s=return_seconds)
    result['neck_follow_trial'] = settings
    return result, settings


class TargetReceiver:
    def __init__(self, config):
        validate_config(config)
        self.max_age = config['transport']['max_age_s']
        self.expected_transform = {k: config['camera'][k] for k in ('rotation_deg', 'mirror_x')}
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind((config['transport']['host'], config['transport']['port']))
            self.socket.setblocking(False)
        except Exception:
            self.socket.close()
            raise
        self.latest = None
        self.rejected = 0

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        for _ in range(32):  # Bounded work: vision cannot monopolize the audio loop.
            try:
                data, source = self.socket.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                if source[0] != '127.0.0.1':
                    raise ValueError('Nonlocal target sender')
                packet = validate_packet(json.loads(data), now, self.max_age)
                if packet.get('camera_transform') != self.expected_transform:
                    raise ValueError('Camera orientation differs from the playback configuration')
                if self.latest and packet['timestamp_monotonic'] <= self.latest['timestamp_monotonic']:
                    raise ValueError('Out-of-order camera frame')
                self.latest = packet
            except (ValueError, KeyError, TypeError, AttributeError, UnicodeError):
                self.rejected += 1
        return self.latest

    def wait_ready(self, seconds=2):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.poll() is not None:
                return
            time.sleep(.01)
        raise RuntimeError('No fresh camera packets; start track_person_xiaobai.sh first')

    def close(self):
        self.socket.close()


class GazePolicy:
    """Normalized pursuit; x=image right, y=image down, independent of encoder sign."""
    def __init__(self, config):
        validate_config(config)
        self.control = config['control']
        self.max_age = config['transport']['max_age_s']
        self.xy = np.zeros(2)
        self.error = np.zeros(2)
        self.weight = 0.
        self.last_seen = None

    def step(self, packet, now, dt, applied_xy):
        if not finite(dt) or not 0 < dt <= .1 or not np.isfinite(applied_xy).all():
            raise ValueError('Invalid gaze clock or applied position')
        c = self.control
        visible = False
        if packet is not None:
            try:
                validate_packet(packet, now, self.max_age)
                visible = packet['target'] is not None
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
        if visible:
            target = np.array([packet['target']['x'], 0.])  # Horizontal tracking only.
            error = np.sign(target) * np.maximum(0, np.abs(target) - c['deadband'])
            alpha = -math.expm1(-dt / c['filter_tau_s'])
            self.error += alpha * (error - self.error)
            if self.weight == 0:
                self.xy = np.clip(applied_xy, -1, 1).copy()
            if c['camera_mount'] == 'eye':
                # Integrate from the rate-limited command state, avoiding wind-up when
                # the common count/s limiter cannot reach a requested position.
                desired = np.asarray(applied_xy) + c['eye_velocity_gain'] * self.error * dt
            else:
                desired = c['fixed_gain'] * self.error
            step = c['max_axis_speed'] * dt
            self.xy = np.clip(self.xy + np.clip(desired - self.xy, -step, step), -1, 1)
            self.last_seen = packet['timestamp_monotonic']
        holding = (self.last_seen is not None
                   and now - self.last_seen <= self.max_age + c['lost_hold_s'])
        engaged = visible or holding
        self.weight = float(np.clip(self.weight + dt / c['acquire_s'] if engaged
                                    else self.weight - dt / c['release_s'], 0, 1))
        state = 'tracking' if visible else 'holding' if holding else 'releasing' if self.weight else 'idle'
        return self.xy.copy(), self.weight, state


def axis_position(axis, normalized):
    endpoint = axis['positive'] if normalized >= 0 else axis['negative']
    return axis['neutral'] + abs(normalized) * (endpoint - axis['neutral'])


def axis_normalized(axis, position):
    delta = position - axis['neutral']
    positive = axis['positive'] - axis['neutral']
    denominator = positive if delta * positive >= 0 else axis['negative'] - axis['neutral']
    value = abs(delta / denominator)
    return float(np.clip(value if delta * positive >= 0 else -value, -1, 1))


class GazeOverride:
    def __init__(self, config, ids, profile, receiver=None, direction_trial=False):
        validate_config(config, live=True, profile=profile, ids=ids, direction_trial=direction_trial)
        self.config = config
        self.ids = ids
        self.axes = config['control']['axes']
        self.policy = GazePolicy(config)
        self.receiver = receiver if receiver is not None else TargetReceiver(config)
        self.last_diagnostic = None

    def apply(self, target, previous, dt, now):
        c = self.config['control']
        anchor = c['camera_eye'] if c['camera_mount'] == 'eye' else 'left'
        camera_axis = next(a for a in self.axes if a['name'] == anchor + '_yaw')
        applied = [axis_normalized(camera_axis, previous[self.ids.index(camera_axis['id'])]), 0.]
        packet = self.receiver.poll(now)
        xy, weight, state = self.policy.step(packet, now, dt, applied)
        result = target.copy()
        for axis in self.axes:
            column = self.ids.index(axis['id'])
            if c['camera_mount'] == 'eye':
                # Acquiring an already centered face must retain the current
                # eye position, rather than blend it back toward neutral.
                fallback = previous[column] if state in ('tracking', 'holding') else axis['neutral']
            else:
                fallback = target[column]
            result[column] = (1 - weight) * fallback + weight * axis_position(
                axis, xy[0 if axis['coordinate'] == 'x' else 1])
        self.last_diagnostic = {'state': state, 'weight': weight, 'normalized_xy': xy.tolist(),
                                'packet_age_s': None if packet is None else now - packet['timestamp_monotonic'],
                                'rejected_packets': self.receiver.rejected}
        return result

    def close(self):
        self.receiver.close()


class NeckFollowTrial:
    """Transfer sustained eye deflection to a slower, bounded neck motion.

    The camera is fixed to the head: real head rotation reduces image error and
    hence eye deflection. Never subtract commanded neck position from gaze;
    that double-counts camera motion and invents compensation for a stuck head.
    """
    def __init__(self, gaze, profile, ids, settings, duration, full_song=False):
        if gaze.config['control']['camera_mount'] != 'fixed':
            raise ValueError('Neck following requires a camera fixed to the head')
        if not finite(duration) or not 5 <= duration <= (7200 if full_song else 60):
            raise ValueError('Invalid neck following duration; direction trials are limited to 60 seconds')
        observation = gaze.config.get('neck_observations', {})
        if full_song and (observation.get('horizontal_id') != 32
                          or observation.get('horizontal_motion_confirmed') is not True):
            raise ValueError('Full-song neck following requires onsite confirmation of ID32 horizontal motion')
        _, expected = prepare_neck_follow_trial(
            {**profile, 'diagnostic_command_ids': [i for i in profile['diagnostic_command_ids']
                                                 if i not in (32, 33, 34)]}, settings['id'], settings['sign'],
            settings['offset_counts'], settings['speed_multiplier'])
        if settings != expected or settings['id'] not in ids:
            raise ValueError('Invalid neck trial configuration')
        if duration <= settings['return_home_s'] + 1:
            raise ValueError('Neck trial is too short for a bounded return home')
        motor = next(m for m in profile['motors'] if m['id'] == settings['id'])
        neck_commands = set(profile['diagnostic_command_ids']) & {32, 33, 34}
        if not 0 < motor['max_speed_counts_s'] <= 25*settings['speed_multiplier'] or neck_commands != {settings['id']}:
            raise ValueError('Neck following must command only the selected neck axis within the chosen speed cap')
        self.gaze, self.settings, self.duration = gaze, settings, duration
        self.receiver = gaze.receiver
        self.axes = gaze.axes + [{'id': settings['id']}]
        self.column = ids.index(settings['id'])
        self.max_speed = motor['max_speed_counts_s']
        self.start = self.pending_since = None
        self.direction = 0
        self.velocity = self.error = 0.
        self.last_diagnostic = None

    def apply(self, target, previous, dt, now):
        result = self.gaze.apply(target, previous, dt, now)
        if self.start is None:
            self.start = now - dt
        s = self.settings
        position = previous[self.column]
        # Previous is the common rate-limited command, not measured eye position.
        eye_offset = float(np.mean([axis_normalized(axis, previous[self.gaze.ids.index(axis['id'])])
                                    for axis in self.gaze.axes]))
        requested_offset = float(self.gaze.policy.xy[0])
        # Do not turn the neck for authored eye motion during vision acquisition.
        comfort_error = (math.copysign(min(abs(eye_offset), abs(requested_offset)), eye_offset)
                         if eye_offset*requested_offset > 0 else 0.)
        desired_velocity = 0.
        state = 'holding'
        if now - self.start >= self.duration - s['return_home_s']:
            delta = s['neutral'] - position
            desired_velocity = math.copysign(min(self.max_speed, abs(delta)*2.,
                                                math.sqrt(2*s['max_acceleration_counts_s2']*abs(delta))), delta)
            state = 'returning_home'
        elif self.gaze.last_diagnostic['state'] == 'tracking':
            alpha = -math.expm1(-dt/s['filter_tau_s'])
            self.error += alpha*(comfort_error-self.error)
            # Hysteresis: tiny detector movements must not start/stop the neck.
            direction = self.direction
            if (abs(comfort_error) <= s['stop_deadband']
                    or direction and np.sign(comfort_error) != direction):
                direction = 0
            elif abs(comfort_error) >= s['deadband']:
                direction = int(np.sign(comfort_error))
            if direction != self.direction:
                self.direction, self.pending_since = direction, now if direction else None
            if direction and self.pending_since is not None:
                state = 'eye_lead'
                if now - self.pending_since >= s['eye_lead_s']:
                    # Filter memory may still have the old sign after a turn.
                    # It must never request motion against the newly selected direction.
                    error = direction*max(0., direction*self.error-s['stop_deadband'])
                    desired_velocity = s['sign']*s['gain_counts_s']*error
                    state = 'following'
        else:
            self.direction, self.pending_since = 0, None
            self.error = 0.
            if abs(self.velocity) > .01:
                state = 'braking_on_loss'
        acceleration = s['max_acceleration_counts_s2']
        # Brake before a range endpoint rather than abruptly saturating a target.
        distance = s['safe_max']-position if desired_velocity >= 0 else position-s['safe_min']
        speed_cap = min(self.max_speed, math.sqrt(2*acceleration*max(0., distance)))
        desired_velocity = float(np.clip(desired_velocity, -speed_cap, speed_cap))
        new_velocity = self.velocity + float(np.clip(desired_velocity-self.velocity, -acceleration*dt, acceleration*dt))
        goal = position + .5*(self.velocity+new_velocity)*dt
        if state == 'returning_home' and abs(position-s['neutral']) < .25 and abs(new_velocity) <= acceleration*dt:
            goal, new_velocity = s['neutral'], 0.
        bounded_goal = float(np.clip(goal, s['safe_min'], s['safe_max']))
        self.velocity = 0. if bounded_goal != goal else new_velocity
        result[self.column] = bounded_goal
        self.last_diagnostic = {**self.gaze.last_diagnostic,
                                'neck': dict(id=s['id'], state=state,
                                             goal=bounded_goal, direction_sign=s['sign'],
                                             eye_offset_x=eye_offset, control_offset_x=comfort_error,
                                             filtered_error_x=self.error, requested_velocity_counts_s=desired_velocity,
                                             velocity_counts_s=self.velocity)}
        return result

    def close(self):
        self.gaze.close()
