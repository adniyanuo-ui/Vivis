#!/usr/bin/env python3
"""Export proposed Mojia motor targets and a calibration comparison, without serial IO."""
import argparse
import hashlib
import json
import math
import wave
from pathlib import Path

from vivis.motion.mapping import ROOT, read_json, inputs, compile_trajectory

MOJIA_IDS = set(range(2, 32)) | {35, 36}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_calibrations(old, new):
    old_ids = {sid for positions in old.values() for sid in positions}
    new_ids = {sid for positions in new.values() for sid in positions}
    return dict(
        newly_mapped_ids=sorted(new_ids-old_ids),
        no_longer_mapped_ids=sorted(old_ids-new_ids),
        empty_shapes=[name for name, positions in new.items() if not positions],
        jaw31_shapes={name:positions[31] for name, positions in new.items() if 31 in positions},
        changes=[dict(shape=name, xiaobai=old[name], mojia=new[name])
                 for name in new if new[name] != old[name]],
    )


def proposal(shapes, baseline, jaw_policy):
    """Endpoint envelopes are proposals, never measured mechanical safety limits."""
    if type(baseline) is not int or not 0 <= baseline <= 4095:
        raise ValueError('Baseline must be an integer in 0..4095')
    if jaw_policy not in ('JawOpen', 'all_calibrated'):
        raise ValueError('Unknown jaw policy')
    motors = []
    for sid in sorted({sid for positions in shapes.values() for sid in positions}):
        points = [baseline] + [positions[sid] for positions in shapes.values() if sid in positions]
        motors.append(dict(id=sid, reported_model='STS3032' if sid<=30 or sid==36 else 'SM45BL', neutral=baseline,
                           endpoint_envelope_not_safe_limits=[min(points), max(points)],
                           safe_min=None, safe_max=None))
    return dict(mixing='bounded_weighted_deltas', jaw_owner=jaw_policy,
                paired_eye_look_is_already_in_lids=True,
                eye_angle_endpoints_rad={name:math.radians(30) for name in shapes if name.startswith('EyeLook')},
                neck_angle_position_maps=None, motion_gain=1., motors=motors)


def hardware_profile(profile, scan):
    """Intersect calibration proposals with device limits; never inherit Xiaobai limits."""
    rows=scan['servos']
    if len(rows)!=35 or {row['id'] for row in rows}!=set(range(2,37)):
        raise ValueError('Need a complete Mojia scan of IDs 2..36')
    states={row['id']:row for row in rows}
    for sid,state in states.items():
        expected=521 if sid<=30 or sid==36 else 10760
        if state.get('model_raw')!=expected or state.get('mode')!=0 or state.get('status_register')!=0:
            raise ValueError(f'ID{sid}: incomplete scan, model/mode mismatch or fault')
    command_ids=[motor['id'] for motor in profile['motors']]
    for sid in (32,33,34):
        profile['motors'].append(dict(id=sid,reported_model='SM45BL',neutral=2047,
            endpoint_envelope_not_safe_limits=[2047,2047],safe_min=None,safe_max=None))
    for motor in profile['motors']:
        sid=motor['id'];state=states[sid]
        low,high=motor['endpoint_envelope_not_safe_limits']
        if sid in (32,33,34):low,high=2027,2067
        low=max(low,state['configured_min']);high=min(high,state['configured_max'])
        if not low<=motor['neutral']<=high:
            raise ValueError(f'ID{sid}: zero-weight position outside device/calibration intersection')
        motor.update(safe_min=low,safe_max=high,neutral_confirmed=True,
            device_min=state['configured_min'],device_max=state['configured_max'],
            max_speed_counts_s=50 if sid in (32,33,34) else 300,
            max_temperature_c=min(60,state['configured_max_temperature']),
            device_max_temperature_c=state['configured_max_temperature'],max_load_raw=350,
            min_voltage_raw=state['configured_min_voltage_raw'],
            max_voltage_raw=state['configured_max_voltage_raw'],
            model_raw_expected=state['model_raw'],register_layout_confirmed=True,
            register_write_verification='Pending bounded RAM write/readback during explicit arm',
            position_feedback_tolerance_counts=5)
        if sid not in (32,33,34):
            motor.update(feedback_min=state['configured_min'],feedback_max=state['configured_max'])
        if sid==31:
            motor.update(initial_settle_tolerance_counts=32,
                         initial_settle_basis='Mojia startup observation: approximately 26 counts position error at a bounded goal; target/device/load guards remain active')
    profile.update(schema_version='mojia-playback-0.1',port=scan['port'],baud=1000000,
        mixing_confirmed=True,jaw_owner_confirmed=True,
        confirmation_source='User confirmed w=0 is 2047 and requested all Mojia calibration mappings',
        physical_audio_output_confirmed=True,audio_device='USB Audio Device',audio_latency_s=.2,
        actuator_latency_ms=0,power_checked=True,serial_backend='feetech_python',
        sdk_transaction_timeout_ms=20,feedback_mode='short_registers',write_mode='position_speed',
        feedback_policy='stop_on_fault',feedback_max_age_s=1.,feedback_polls_per_frame=1,
        changed_goals_only=True,capture_serial_trace=True,confirm_sensor_transients=True,
        diagnostic_command_ids=command_ids,
        stop_method='Stop sending; preserve last bounded goal and neck holding torque',
        hardware_stop_description='Onsite operator can cut servo power',
        held_neck_ids=[32,33,34])
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', type=Path, default=ROOT/'exmple/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json')
    parser.add_argument('--audio', type=Path, default=ROOT/'exmple/0001.wav')
    parser.add_argument('--calibration', type=Path, default=ROOT/'mojia/墨甲9.23.json')
    parser.add_argument('--output', type=Path, default=ROOT/'mojia/0001_MOJIA_MOTOR_TARGETS_60fps.json')
    parser.add_argument('--baseline', type=int, required=True, help='Proposed w=0 position; must be reviewed before playback')
    parser.add_argument('--jaw-policy', choices=['JawOpen','all_calibrated'], required=True)
    parser.add_argument('--scan',type=Path,help='Complete read-only Mojia scan; apply actual device limits')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output exists; choose a new path to preserve evidence')
    header = ROOT/'xiaobai/blendshape_channels_61.csv'
    values, fps, order, shapes = inputs(args.clip, args.calibration, header, allowed_motor_ids=MOJIA_IDS)
    _, _, _, old = inputs(args.clip, ROOT/'xiaobai/xiaobai标定文件.json', header)
    profile = proposal(shapes, args.baseline, args.jaw_policy)
    if args.scan is not None:
        profile=hardware_profile(profile,read_json(args.scan))
    ids, positions, report = compile_trajectory(values, order, shapes, profile, 'face_eyes', draft=args.scan is None)
    with wave.open(str(args.audio), 'rb') as audio:
        duration = audio.getnframes()/audio.getframerate()
        if audio.getsampwidth() != 2 or abs(len(values)/fps-duration) > 1/fps:
            raise ValueError('Need PCM16 WAV with matching duration')
    diff = compare_calibrations(old, shapes)
    payload = dict(
        schema_version='mojia-motor-targets-0.1', playback_ready=False,
        device_limits_applied=args.scan is not None,
        pending_review=(['Live startup positions, torque and hardware feedback'] if args.scan else
                       ['zero-weight baseline', 'jaw31 composition', 'eye angular scale',
                        'per-motor hardware model, limits, speed and feedback',
                        'bounded initial approach and runtime rate limiting']),
        source_clip=args.clip.name, source_clip_sha256=sha256(args.clip),
        audio=args.audio.name, audio_sha256=sha256(args.audio),
        calibration=args.calibration.name, calibration_sha256=sha256(args.calibration),
        baseline=args.baseline, jaw_policy=args.jaw_policy,
        mixing=profile['mixing'], eye_mapping='30 degrees at weight 1; artistic assumption, not measured eye angle',
        paired_eye_look_is_already_in_lids=True,
        fps=read_json(args.clip)['fps'], frames=len(values), duration_seconds=duration,
        motor_ids=ids, positions=positions.tolist(),
        position_semantics='Unrounded nominal targets; time for row i is i/fps seconds. Not serial commands. No neck animation.',
        trajectory_report=report, calibration_diff=diff,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
    if args.scan is not None:
        profile.update(calibration_sha256=sha256(args.calibration),targets_sha256=sha256(args.output),
                       audio_sha256=sha256(args.audio),scan_sha256=sha256(args.scan))
        args.output.with_name(args.output.stem+'_PROFILE.json').write_text(
            json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
    report_path = args.output.with_name(args.output.stem+'_REVIEW.json')
    report_path.write_text(json.dumps(dict(proposal=profile, calibration_diff=diff, trajectory_report=report), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), review=str(report_path), playback_ready=False,
                          frames=len(values), motor_ids=ids,
                          changed_shapes=len(diff['changes']),
                          newly_mapped_ids=diff['newly_mapped_ids'],
                          no_longer_mapped_ids=diff['no_longer_mapped_ids']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
