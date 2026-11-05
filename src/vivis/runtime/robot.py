"""Resolve one robot definition into the existing runtime contracts.

Hardware values come from hardware.json. Work manifests contain only asset
provenance; tracking never supplies a second motor speed/PID/acceleration value.
No serial, audio or camera device is opened here.
"""
import copy
import hashlib
from pathlib import Path

from vivis.paths import ROOT
from vivis.motion.mapping import read_json,calibrated_eye_endpoints
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
        legacy = self.hardware.get('schema_version') == 'vivis-mojia-hardware-1.0'
        if legacy and [a['id'] for a in axes] != list(range(2, 37)):
            raise ValueError('Mojia requires the complete ordered 35-axis inventory')
        if self.hardware.get('schema_version') not in ('vivis-mojia-hardware-1.0','vivis-feetech-hardware-1.0'):
            raise ValueError('Unsupported hardware definition')
        ids=[a['id'] for a in axes]
        if (not ids or any(type(sid) is not int or not 0<=sid<=253 for sid in ids)
                or ids!=sorted(set(ids))):
            raise ValueError('Robot requires an ordered unique actuator inventory')
        self.axes = {a['id']: a for a in axes}
        for a in axes:
            if legacy and (a['neutral'] != 2047 or a.get('neutral_confirmed') is not True
                    or a['model_raw_expected'] != (10760 if 31 <= a['id'] <= 35 else 521)):
                raise ValueError('Mojia neutral/model inventory mismatch')
            if not legacy:
                neutral=a['neutral'];home=a.get('home_position',neutral)
                if (type(neutral) is not int or not 0<=neutral<=4095 or a.get('neutral_confirmed') is not True
                        or a['model_raw_expected'] not in (521,10760,11272)
                        or not 0<=a['device_min']<=a['safe_min']<=a['safe_max']<=a['device_max']<=4095
                        or type(home) is not int or home!=max(a['safe_min'],min(a['safe_max'],neutral))):
                    raise ValueError('Robot neutral/model/device envelope mismatch')
        modes=self.definition.get('supported_modes',['face','eyes','eyes-neck'])
        if not modes or len(set(modes))!=len(modes) or not set(modes)<={'face','eyes','eyes-neck'}:
            raise ValueError('Invalid supported robot modes')
        self.works = read_json(self.resolve('works'))
        if self.works.get('schema_version') != 'vivis-works-1.0':
            raise ValueError('Unsupported work manifest')

    def resolve(self, key):
        return (self.directory / self.definition[key]).resolve()

    def require_mode(self,mode):
        if mode not in self.definition.get('supported_modes',['face','eyes','eyes-neck']):
            raise ValueError('Robot '+self.definition['name']+' currently supports only: '+
                             ', '.join(self.definition['supported_modes']))

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
        for motor in p['motors']:
            if motor.get('home_position',motor['neutral'])!=motor['neutral']:
                motor['mapping_neutral']=motor['neutral']
                motor['neutral']=motor['home_position']
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
        c['control']['camera_eye'] = None if mount.get('mount')=='fixed' else camera
        c['control']['speed_counts_s'] = self.axes[8]['max_speed_counts_s']
        endpoint_source=c['control'].get('endpoint_source','configured')
        if endpoint_source not in ('configured','calibration_device_clipped'):raise ValueError('Unknown eye endpoint source')
        if endpoint_source=='calibration_device_clipped':
            shapes={s['shape']:{p['id']:p['position'] for p in s['positions']} for s in read_json(self.calibration)['blendShapes']}
            for axis in c['control']['axes']:axis.update(calibrated_eye_endpoints(axis,shapes,self.axes[axis['id']]))
        for a in c['control']['axes']:
            # Explicit per-axis overrides are sourced from the same hardware table.
            a['speed_counts_s'] = self.axes[a['id']]['max_speed_counts_s']
        c['calibration_sha256'] = self.calibration_sha256
        a = self.axes[32]
        c['neck'].update(speed_counts_s=a['max_speed_counts_s'],
                         observed_acceleration_raw=a['acceleration_raw'],
                         observed_pid_raw=a.get('pid_raw'), observed_deadband_counts=a.get('deadband_raw'))
        return c

    def home(self,mode=None):
        data = read_json(self.resolve('homing'))
        if type(data.get('sensor_fault_confirmation',False)) is not bool:
            raise ValueError('Invalid homing sensor confirmation switch')
        ids=data.get('mode_axis_ids',{}).get(mode,data.get('axis_ids',list(self.axes)))
        if not ids or len(set(ids))!=len(ids) or not set(ids)<=set(self.axes):
            raise ValueError('Invalid configured home inventory')
        result=dict(schema_version='actuator-neutral-inventory-1.0',
                    axes=[dict(id=sid,neutral=self.axes[sid].get('home_position',self.axes[sid]['neutral']),
                               neutral_confirmed=self.axes[sid]['neutral_confirmed'],
                               model_raw_expected=self.axes[sid]['model_raw_expected']) for sid in ids],
                    recovery_clearance=data.get('recovery_clearance'),
                    disabled_position_recovery=data.get('disabled_position_recovery'))
        if 'sensor_fault_confirmation' in data:result['sensor_fault_confirmation']=data['sensor_fault_confirmation']
        return result

    def startup(self,active_ids=None,inventory_ids=None):
        inventory=set(self.axes) if inventory_ids is None else set(inventory_ids)
        selected={a['id'] for a in self.axes.values() if a['playback_enabled']} if active_ids is None else set(active_ids)
        if not selected<=inventory<=set(self.axes) or any(not self.axes[sid]['playback_enabled'] for sid in selected):
            raise ValueError('Invalid startup actuator selection')
        axes = []
        for a in self.axes.values():
            if a['id'] not in selected:
                continue
            row = dict(id=a['id'], model_raw=a['model_raw_expected'],
                       device_limits=[a['device_min'], a['device_max']],
                       speed_raw=a['max_speed_counts_s'], acceleration_raw=a['acceleration_raw'])
            for key in ('pid_raw', 'deadband_raw'):
                if key in a:
                    row[key] = copy.deepcopy(a[key])
            axes.append(row)
        schema='mojia-startup-hardware-1.0' if self.hardware['schema_version']=='vivis-mojia-hardware-1.0' else 'vivis-feetech-startup-hardware-1.0'
        return dict(schema_version=schema,axes=axes,excluded_ids=sorted(inventory-selected))

    def telemetry(self):
        return read_json(self.resolve('telemetry'))
