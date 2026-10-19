"""Prepare a bounded commissioning profile from final calibration and read-only scan.
No motor commands are emitted. Power/audio and model-specific write compatibility
remain explicit prerequisites, not inferred from a plausible readback alone.
"""
import hashlib,json,math,shutil
from pathlib import Path
from xiaobai_runtime.mapping import ROOT,audit,draft_profile

cal=ROOT/'xiaobai/xiaobai标定文件.json'
clip=ROOT/'outputs/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json'
header=ROOT/'xiaobai/blendshape_channels_61.csv'
scan=json.loads((ROOT/'outputs/xiaobai/scan.json').read_text())
states={s['id']:s for s in scan['servos']}
if set(states)!=set(range(2,35)) or any('error' in s for s in states.values()):raise ValueError('Need complete read-only scan')
report=audit(clip,cal,header);profile=draft_profile(report)
profile.update(calibration_path=str(cal),calibration_sha256=hashlib.sha256(cal.read_bytes()).hexdigest(),
 mixing='bounded_weighted_deltas',mixing_confirmed=True,
 mixing_confirmation_source='user requested assistant-designed rule with subsequent hardware feedback',
 jaw_owner_confirmed=True,motion_gain=.25,power_checked=False,
 hardware_stop_description='User is on site and can cut servo power via power strip',
 zero_weight_position=2047,zero_weight_position_confirmed=True,
 baseline_confirmation='User explicitly confirmed: 按公式执行，2047 是零权重基准',
 baseline_semantics='legacy motor field neutral is the formula w=0 endpoint; not the half-weight pose requested as midpoint',
 neutral_pose_policy='do not synthesize a global midpoint: each expression has its own midpoint; execute provided affine formula',
 eye_angle_endpoints_rad={n:math.radians(30) for n in report['inventory'][5]['calibration_endpoints'] if n.startswith('EyeLook')},
 eye_angle_mapping_status='proposed trial: 30 degrees preview gaze = weight 1; not measured robot eye angle',
 commissioning_scope='face_eyes',neck_angle_position_maps=None,
 limit_policy='intersection of supplied endpoint envelope and existing device limits; combined-pose mechanics still require observation')
# All eight direction channels use the same provisional artistic scale, not encoder angle conversion.
profile['eye_angle_endpoints_rad']={s['shape']:math.radians(30) for s in json.loads(cal.read_text())['blendShapes'] if s['shape'].startswith('EyeLook')}
for m in profile['motors']:
 s=states[m['id']];low,high=m['endpoint_envelope_not_safe_limits']
 m.update(safe_min=max(low,s['configured_min']),safe_max=min(high,s['configured_max']),
   device_min=s['configured_min'],device_max=s['configured_max'],
   zero_weight_position=2047,neutral_confirmed=True,
   neutral_confirmation_source='user formula fixes weight zero at 2047; physical half-pose is expression dependent',
   max_speed_counts_s=300,max_temperature_c=60,max_load_raw=350,
   model_raw_expected=s['model_raw'],register_layout_confirmed=m['id']<=30,
   register_readback_verified=True,
   register_write_verification='official SMS/STS map' if m['id']<=30 else 'pending SM85BLHV documentation or scoped write/readback verification',
   initial_position=s['position_raw'],initial_temperature_c=s['temperature_c'],
   initial_voltage_raw=s['voltage_raw'],min_voltage_raw=40 if m['id']<=30 else 90,
   max_voltage_raw=74 if m['id']<=30 else 254)
path=ROOT/'xiaobai/playback_profile.json'
backup=ROOT/'xiaobai/playback_profile_initial_draft.json'
if path.exists() and not backup.exists():shutil.copyfile(path,backup)
path.write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
report.update(calibration_path=str(cal),calibration_sha256=profile['calibration_sha256'],
 note='Final user-selected calibration replaces old calibration audit; no repeated ID31 baseline placeholders in this revision')
(ROOT/'outputs/xiaobai/calibration_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Final calibration profile prepared; no motion sent')
