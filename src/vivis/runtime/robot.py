"""Resolve one robot definition into the existing runtime contracts.

Hardware values come from hardware.json. Work manifests contain only asset
provenance; tracking never supplies a second motor speed/PID/acceleration value.
No serial, audio or camera device is opened here.
"""
import copy
import hashlib
from pathlib import Path

from vivis.paths import ROOT
from vivis.motion.mapping import read_json
from vivis.runtime.telemetry import apply_policy


class Robot:
    def __init__(self, path=None):
        self.path = Path(path or ROOT / 'configs/robots/mojia/robot.json').resolve()
        self.definition = read_json(self.path)
        if self.definition.get('schema_version') != 'vivis-robot-1.0':
            raise ValueError('Unsupported robot definition')
        self.directory = self.path.parent
        self.calibration = self.resolve('calibration')
        self.calibration_sha256 = hashlib.sha256(self.calibration.read_bytes()).hexdigest()
        if self.definition['calibration_sha256'] != self.calibration_sha256:
            raise ValueError('Robot calibration changed; review derived assets before playback')
        self.hardware = read_json(self.resolve('hardware'))
        axes = self.hardware['axes']
        if [a['id'] for a in axes] != list(range(2, 37)):
            raise ValueError('Mojia requires the complete ordered 35-axis inventory')
        if self.hardware.get('schema_version') != 'vivis-mojia-hardware-1.0':
            raise ValueError('Unsupported hardware definition')
        self.axes = {a['id']: a for a in axes}
        for a in axes:
            if (a['neutral'] != 2047 or a.get('neutral_confirmed') is not True
                    or a['model_raw_expected'] != (10760 if 31 <= a['id'] <= 35 else 521)):
                raise ValueError('Mojia neutral/model inventory mismatch')
        self.works = read_json(self.resolve('works'))
        if self.works.get('schema_version') != 'vivis-works-1.0':
            raise ValueError('Unsupported work manifest')

    def resolve(self, key):
        return (self.directory / self.definition[key]).resolve()

    def work(self, name):
        if name not in self.works['works']:
            raise ValueError('Unknown work: ' + name)
        w = copy.deepcopy(self.works['works'][name])
        if w['calibration_sha256'] != self.calibration_sha256:
            raise ValueError('Work was generated with another calibration')
        for key in ('audio', 'targets'):
            w[key] = ROOT / w[key]
        if 'source_clip' in w:w['source_clip']=ROOT/w['source_clip']
        return w

    def motion_profile(self):
        p = copy.deepcopy(self.hardware['runtime'])
        p['motors'] = [{k: copy.deepcopy(v) for k, v in a.items()
                        if k not in ('role', 'playback_enabled', 'acceleration_raw', 'pid_raw', 'deadband_raw')}
                       for a in self.axes.values() if a['playback_enabled']]
        p.update(calibration_sha256=self.calibration_sha256, calibration_path=str(self.calibration))
        shapes=read_json(self.calibration)['blendShapes']
        p['mouth_motion_ids']=sorted({point['id'] for shape in shapes
            if shape['shape'].startswith(('Mouth','Jaw','Tongue')) for point in shape['positions']})
        p['round_lip_motion_ids']=sorted({point['id'] for shape in shapes
            if shape['shape'] in ('MouthFunnel','MouthPucker') for point in shape['positions']})
        p['jaw_open_motion_ids']=sorted({point['id'] for shape in shapes
            if shape['shape']=='JawOpen' for point in shape['positions']})
        if hasattr(self,'mouth_speed_override'):
            p['mouth_speed_override']=copy.deepcopy(self.mouth_speed_override)
        apply_policy(p, self.telemetry())
        return p

    def set_mouth_speed(self,speed):
        """One-run RAM speed preset; leave files, limits, acceleration and PID intact."""
        if type(speed) is not int or not 1<=speed<=32767:
            raise ValueError('Mouth speed must be integer1..32767')
        ids=self.motion_profile()['mouth_motion_ids']
        if not ids or any(sid not in self.axes or not self.axes[sid]['playback_enabled'] for sid in ids):
            raise ValueError('Mouth speed needs a calibrated active mouth group')
        before={sid:self.axes[sid]['max_speed_counts_s'] for sid in ids}
        for sid in ids:self.axes[sid]['max_speed_counts_s']=speed
        self.mouth_speed_override=dict(speed_counts_s=speed,axes=ids,previous_speed_counts_s=before,
                                       persistence='task RAM preset; canonical hardware file unchanged')
        return copy.deepcopy(self.mouth_speed_override)

    def profile(self, name):
        w = self.work(name)
        p = self.motion_profile()
        p.update(
                 targets_sha256=w['targets_sha256'], audio_sha256=w['audio_sha256'])
        return p

    def tracking(self, camera='right', device=None):
        c = read_json(self.resolve('tracking'))
        if 'parallel_neck' in self.definition:c['parallel_neck']=read_json(self.resolve('parallel_neck'))
        if camera not in self.definition['cameras']:
            raise ValueError('Unknown camera side')
        mount = self.definition['cameras'][camera]
        selected = device or mount['device']
        if not selected:
            raise ValueError('Camera device is not configured; supply --camera-device')
        c['camera']['device'] = selected
        c['control']['camera_eye'] = camera
        c['control']['speed_counts_s'] = self.axes[8]['max_speed_counts_s']
        for a in c['control']['axes']:
            # Explicit per-axis overrides are sourced from the same hardware table.
            a['speed_counts_s'] = self.axes[a['id']]['max_speed_counts_s']
        c['calibration_sha256'] = self.calibration_sha256
        a = self.axes[32]
        c['neck'].update(speed_counts_s=a['max_speed_counts_s'],
                         observed_acceleration_raw=a['acceleration_raw'],
                         observed_pid_raw=a['pid_raw'], observed_deadband_counts=a['deadband_raw'])
        return c

    def home(self):
        data = read_json(self.resolve('homing'))
        return dict(schema_version='actuator-neutral-inventory-1.0',
                    axes=[{k: a[k] for k in ('id', 'neutral', 'neutral_confirmed', 'model_raw_expected')}
                          for a in self.axes.values()],
                    recovery_clearance=data.get('recovery_clearance'),
                    disabled_position_recovery=data.get('disabled_position_recovery'))

    def startup(self):
        axes = []
        for a in self.axes.values():
            if not a['playback_enabled']:
                continue
            row = dict(id=a['id'], model_raw=a['model_raw_expected'],
                       device_limits=[a['device_min'], a['device_max']],
                       speed_raw=a['max_speed_counts_s'], acceleration_raw=a['acceleration_raw'])
            for key in ('pid_raw', 'deadband_raw'):
                if key in a:
                    row[key] = copy.deepcopy(a[key])
            axes.append(row)
        return dict(schema_version='mojia-startup-hardware-1.0', axes=axes,
                    excluded_ids=[a['id'] for a in self.axes.values() if not a['playback_enabled']])

    def telemetry(self):
        return read_json(self.resolve('telemetry'))
