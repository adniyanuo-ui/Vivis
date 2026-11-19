#!/usr/bin/env python3
"""Offline calibration comparison; never register a work or open hardware.

Retain the accepted work's authored blink recipe. Encoder-domain checks are
numeric checks only: an EEPROM scan is required before this candidate can play.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vivis.motion.mapping import compile_trajectory, inputs, read_json
from vivis.paths import ROOT
from vivis.runtime.robot import Robot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def replay_blinks(values, order, shapes, profile, recipe):
    """Replay accepted pulse removal and event envelopes, without new timing."""
    ids, original, report = compile_trajectory(values, order, shapes, profile, 'face_eyes')
    if recipe is None:
        return ids, original, report
    if (recipe['timing_mode'] != 'natural_asymmetric_reduced'
            or recipe['phase_profiles'] != {'closing': 'front-loaded sine',
                                            'opening': 'cosine with time exponent 1.4'}):
        raise ValueError('Unsupported accepted blink recipe')
    edited = values.copy()
    changed = np.zeros(len(values), dtype=bool)
    for candidate in recipe['frequency']['candidates']:
        if candidate['protected']:
            continue
        for side, (lo, hi) in candidate['windows'].items():
            if not 0 <= lo <= hi < len(values):
                raise ValueError('Blink pulse window exceeds clip')
            channel = order.index('EyeBlink' + side)
            edited[lo:hi+1, channel] = np.linspace(values[lo, channel], values[hi, channel], hi-lo+1)
            changed[lo:hi+1] = True
    _, edited_positions, _ = compile_trajectory(edited, order, shapes, profile, 'face_eyes')
    result = original.copy()
    for sid in (5, 6, 20, 21):
        column = ids.index(sid)
        result[changed, column] = edited_positions[changed, column]
    baseline = result.copy()
    motors = {m['id']: m for m in profile['motors']}
    for event in recipe['events']:
        start, closed, end, finish = (event[key] for key in
            ('start_frame', 'closed_start_frame', 'closed_end_frame', 'finish_frame'))
        if not 0 <= start <= closed <= end < finish-1 < len(values):
            raise ValueError('Blink event exceeds clip')
        frames = np.arange(start, finish)
        closure = np.ones(len(frames))
        before = frames < closed
        if closed > start:
            closure[before] = np.sin(np.pi/2 * (frames[before]-start)/(closed-start))
        after = frames >= end
        phase = (frames[after]-end)/(finish-end-1)
        closure[after] = .5 + .5*np.cos(np.pi*phase**1.4)
        for sid, endpoint in shapes['EyeBlink'+event['side']].items():
            column = ids.index(sid)
            base = baseline[frames, column]
            proposed = base + closure*(endpoint-base)
            result[frames, column] = (np.maximum(result[frames, column], proposed)
                if endpoint > motors[sid]['neutral'] else np.minimum(result[frames, column], proposed))
    return ids, result, report


def prepare(calibration, calibration_dir, output_dir):
    robot = Robot()
    work = robot.work('0001')
    for key in ('targets', 'audio', 'source_clip'):
        if digest(work[key]) != work[key+'_sha256']:
            raise ValueError('Accepted work hash mismatch: '+key)
    accepted = read_json(work['targets'])
    values, fps, order, old_shapes = inputs(work['source_clip'], robot.calibration, robot.resolve('channels'))
    if fps != 60 or accepted['frames'] != len(values):
        raise ValueError('Expected the accepted 60FPS 0001 source')
    recipe = accepted.get('blink_adjustment')
    old_ids, reproduced, _ = replay_blinks(values, order, old_shapes, robot.motion_profile(), recipe)
    error = float(np.max(np.abs(reproduced-np.asarray(accepted['positions']))))
    if old_ids != accepted['motor_ids'] or error > 1e-6:
        raise ValueError('Cannot reproduce accepted target timing/mapping: '+str(error))

    source = read_json(calibration)
    normalized = copy.deepcopy(source)
    corrections = []
    for shape in normalized['blendShapes']:
        count = len(shape['positions'])
        if count != shape['posSize']:
            corrections.append(dict(shape=shape['shape'], field='posSize', before=shape['posSize'], after=count))
            shape['posSize'] = count
    normalized_bytes = json_bytes(normalized)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'calibration.json'
        path.write_bytes(normalized_bytes)
        _, _, _, shapes = inputs(work['source_clip'], path, robot.resolve('channels'), set(robot.axes))

    # No old device limits are inherited. These bounds are encoder-domain only.
    profile = robot.motion_profile()
    profile['motors'] = [dict(id=sid, neutral=axis['neutral'], safe_min=0, safe_max=4095)
                        for sid, axis in robot.axes.items()]
    ids, positions, report = replay_blinks(values, order, shapes, profile, recipe)
    if not np.isfinite(positions).all() or np.any(positions < 0) or np.any(positions > 4095):
        raise ValueError('Candidate exceeds numeric encoder range')
    if any(report['clipped_samples_by_id'].values()):
        raise ValueError('Numeric-domain clipping would change the new calibration')
    ranges = {str(sid): [float(positions[:,j].min()), float(positions[:,j].max())] for j,sid in enumerate(ids)}
    report.update(draft_only=True, device_limits_verified=False, target_ranges=ranges,
                  encoder_domain_only=[0,4095], postprocessing='Replay accepted blink timing with new endpoints')
    source_hash = digest(calibration)
    normalized_hash = hashlib.sha256(normalized_bytes).hexdigest()
    with wave.open(str(work['audio'])) as wav:
        duration = wav.getnframes()/wav.getframerate()
    if abs(len(values)/fps-duration) > 1/fps:
        raise ValueError('Audio and animation duration mismatch')
    target = dict(schema_version='mojia-motor-targets-0.1', name='00012', robot='mojia2',
        playback_ready=False, device_limits_applied=False,
        pending_review=['mojia2 EEPROM limits and neutral/home verification',
                        'ID9/24 newly mapped cheek commissioning', 'Physical expression and clearance acceptance'],
        calibration=str(calibration_dir.relative_to(ROOT)/'calibration.json'),
        calibration_sha256=normalized_hash, calibration_source_sha256=source_hash,
        source_clip=str(work['source_clip'].relative_to(ROOT)), source_clip_sha256=work['source_clip_sha256'],
        audio=str(work['audio'].relative_to(ROOT)), audio_sha256=work['audio_sha256'],
        baseline=2047, baseline_basis='Existing Mojia mapping reference; mojia2 neutral remains to be verified',
        fps={'num':60,'den':1}, frames=len(values), duration_seconds=duration,
        motor_ids=ids, positions=positions.tolist(), position_semantics=accepted['position_semantics'],
        mixing=profile['mixing'], jaw_policy=profile['jaw_owner'], motion_gain=1.0,
        paired_eye_look_is_already_in_lids=profile['paired_eye_look_is_already_in_lids'],
        trajectory_report=report,
        blink_timing_replay=dict(source_targets_sha256=work['targets_sha256'],
            event_count=len(recipe['events']) if recipe else 0,
            timing_unchanged=True, endpoints_from='mojia2 calibration',
            baseline_reproduction_max_error_counts=error))
    audit = dict(robot='mojia2', work='00012', source_work='0001',
        status='offline candidate; NOT commissioned', source_calibration_name=Path(calibration).name,
        source_calibration_sha256=source_hash, calibration_sha256=normalized_hash,
        metadata_corrections=corrections, position_endpoints_unchanged=True,
        audio_unchanged=True, source_clip_unchanged=True, gain=1.0, neck_animation=False,
        accepted_targets_sha256=work['targets_sha256'], baseline_reproduction_max_error_counts=error,
        motor_ids=ids, newly_mapped_ids=sorted(set(ids)-set(old_ids)),
        pending_eeprom_ids=list(robot.axes), target_ranges=ranges,
        changes_vs_old_targets={str(sid):dict(
            changed_frames=int(np.count_nonzero(np.abs(positions[:,ids.index(sid)]-reproduced[:,j])>1e-6)),
            max_abs_delta_counts=float(np.max(np.abs(positions[:,ids.index(sid)]-reproduced[:,j]))))
            for j,sid in enumerate(old_ids)},
        limitation='No old hardware limits were applied; no device parameter or motion validation is implied.')
    # Refuse overwrites. Compute and verify the whole candidate before saving.
    destinations = [calibration_dir/Path(calibration).name, calibration_dir/'calibration.json',
                    output_dir/'targets2.json', output_dir/'calibration_review.json']
    if any(p.exists() for p in destinations):
        raise FileExistsError('Candidate already exists; choose fresh output paths')
    calibration_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(calibration, destinations[0])
    destinations[1].write_bytes(normalized_bytes)
    destinations[2].write_bytes((json.dumps(target,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())
    audit['targets_sha256'] = digest(destinations[2])
    destinations[3].write_bytes(json_bytes(audit))
    return dict(targets=str(destinations[2]), audio=str(work['audio']), frames=len(values),
                motor_ids=ids, metadata_corrections=corrections, baseline_error_counts=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, default=ROOT/'configs/robots/mojia2')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'assets/works/mojia2/0001')
    args = parser.parse_args()
    print(json.dumps(prepare(args.calibration.resolve(),args.calibration_dir.resolve(),args.output_dir.resolve()),
                     ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
