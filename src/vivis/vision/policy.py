"""Nonblocking local person targets and bounded eye policy; no serial I/O."""
import json
import math
import socket
import time
import numpy as np

VERSION = 'vivis-person-target-1.0'


def validate_config(config):
    if config.get('schema_version') != 'mojia-person-gaze-1.0':
        raise ValueError('Unsupported person gaze configuration')
    transport = config['transport']
    if (transport['host'] != '127.0.0.1' or type(transport['port']) is not int
            or not 1024 <= transport['port'] <= 65535):
        raise ValueError('Vision transport must use a nonprivileged loopback UDP port')
    control = config['control']
    if not finite(transport['max_age_s']) or not .02 <= transport['max_age_s'] <= 1:
        raise ValueError('Invalid vision freshness limit')
    for key, low, high in [('deadband',0,.5),('filter_tau_s',.001,2),('fixed_gain',.01,2),
                           ('eye_velocity_gain',.01,5),('max_axis_speed',.01,5),
                           ('acquire_s',.05,5),('lost_hold_s',0,2),('release_s',.05,5)]:
        if not finite(control[key]) or not low <= control[key] <= high:
            raise ValueError('Invalid gaze parameter: ' + key)
    if control['camera_mount'] not in ('unconfirmed','fixed','eye'):
        raise ValueError('Unknown camera mount')

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


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
        raise RuntimeError('No fresh camera packets; start example02 or03 with the matching camera configuration')

    def close(self):
        self.socket.close()


class GazePolicy:
    """Normalized pursuit; x=image right, y=image down, independent of encoder sign."""
    def __init__(self, config):
        validate_config(config)
        self.control = config['control']
        self.vertical_enabled=(config.get('schema_version')=='mojia-person-gaze-1.0' and self.control.get('vertical_enabled') is True)
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
            vertical = self.vertical_enabled and self.control.get('camera_eye') in ('left','right')
            target = np.array([packet['target']['x'], packet['target']['y'] if vertical else 0.])
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
