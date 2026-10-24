#!/usr/bin/env python3
"""Prepare another Mojia work using accepted limits and a 60 Hz held source."""
import argparse
import copy
import json
import math
import wave
from pathlib import Path

import numpy as np

from prepare_mojia import MOJIA_IDS, sha256
from xiaobai_runtime.mapping import ROOT, read_json, inputs, compile_trajectory


def hold_at_60(values, source_fps):
    if source_fps not in (30, 60):
        raise ValueError('Expected a 30 or 60 FPS source')
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 61 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid 61-channel source')
    # Retain the native sample times: 30 FPS samples are held for two ticks.
    return np.repeat(values, 60 // int(source_fps), axis=0)


def align_audio(decoded, output, frames):
    with wave.open(str(decoded)) as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        samples = wav.getnframes(); pcm = wav.readframes(samples)
    if width != 2:
        raise ValueError('Expected PCM16 decoded audio')
    target_samples = round(frames / 60 * rate)
    pad = target_samples - samples
    if pad < 0 or pad / rate > .1:
        raise ValueError('Audio needs trimming or more than100ms padding; inspect timestamps first')
    with wave.open(str(output), 'wb') as wav:
        wav.setnchannels(channels); wav.setsampwidth(width); wav.setframerate(rate)
        wav.writeframes(pcm + bytes(pad * channels * width))
    return dict(sample_rate=rate, channels=channels, decoded_samples=samples,
                padded_samples=pad, padded_seconds=pad/rate,
                duration_seconds=target_samples/rate, decoded_pcm_preserved=True)


def scale_expression_targets(ids, positions, profile, shapes, gain=1.):
    """Amplify composed brow/nose displacement without scaling lips or blinks."""
    if type(gain) not in (int, float) or not math.isfinite(gain) or gain <= 0:
        raise ValueError('Expression gain must be finite and positive')
    result = np.array(positions, dtype=float, copy=True)
    if result.ndim != 2 or result.shape[1] != len(ids) or not len(result) or not np.isfinite(result).all():
        raise ValueError('Invalid expression target matrix')
    names = [name for name in shapes if name.startswith(('Brow', 'NoseSneer'))]
    selected = sorted({sid for name in names for sid in shapes[name]})
    other_roles = {sid for name, values in shapes.items()
                   if name.startswith(('Eye', 'Mouth', 'Jaw', 'Tongue')) for sid in values}
    if set(selected) & other_roles:
        raise ValueError('Expression axes shared with eyes/mouth require separate review')
    motors = {m['id']:m for m in profile['motors']}
    report = dict(gain=float(gain), source_shapes=names, axes=[],
                  unmapped_cheek_shapes=[name for name in shapes if name.startswith('Cheek') and not shapes[name]])
    for sid in selected:
        if sid not in ids:
            raise ValueError('Missing calibrated expression axis')
        j = ids.index(sid); m = motors[sid]; before = result[:, j].copy()
        low, high = max(m['safe_min'], m['device_min']), min(m['safe_max'], m['device_max'])
        if not low <= m['neutral'] <= high:
            raise ValueError('Expression neutral outside limits')
        with np.errstate(over='ignore', invalid='ignore'):
            requested = m['neutral'] + gain * (before - m['neutral'])
        if not np.isfinite(requested).all():
            raise ValueError('Expression gain overflow')
        result[:, j] = np.clip(requested, low, high)
        report['axes'].append(dict(id=sid, limits=[low, high],
            original_range=[float(before.min()), float(before.max())],
            output_range=[float(result[:, j].min()), float(result[:, j].max())],
            clipped_frames=int(np.count_nonzero(requested != result[:, j]))))
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--decoded-audio', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--reference-profile', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--expression-gain', type=float, default=1.)
    args = parser.parse_args()
    calibration = ROOT/'mojia/墨甲9.23.json'
    source = read_json(args.clip); profile = read_json(args.reference_profile)
    if profile.get('schema_version') != 'mojia-playback-0.1' or profile['calibration_sha256'] != sha256(calibration):
        raise ValueError('Need a calibration-matched accepted Mojia profile')
    if any(m['neutral'] != 2047 or not m['neutral_confirmed'] for m in profile['motors']):
        raise ValueError('Expected the accepted Mojia neutral calibration')
    values, source_fps, order, shapes = inputs(args.clip, calibration,
        ROOT/'xiaobai/blendshape_channels_61.csv', allowed_motor_ids=MOJIA_IDS)
    values = hold_at_60(values, source_fps)
    paths = {name:args.prefix.with_name(args.prefix.name+suffix) for name,suffix in (
        ('audio', '.wav'), ('clip', '_BLENDSHAPES_61_60fps_HOLD.json'),
        ('targets', '_MOJIA_MOTOR_TARGETS_60fps.json'),
        ('profile', '_MOJIA_MOTOR_TARGETS_60fps_PROFILE.json'),
        ('review', '_MOJIA_PREPARATION.json'))}
    if any(p.exists() for p in paths.values()):
        raise ValueError('Output exists; preserve previous versions')
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    audio = align_audio(args.decoded_audio, paths['audio'], len(values))
    held = dict(blendShapes=values.tolist(), fps=dict(num=60, den=1), name=source['name'])
    paths['clip'].write_text(json.dumps(held, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
    profile = copy.deepcopy(profile)
    profile.pop('blink_trial', None)
    ids, positions, report = compile_trajectory(values, order, shapes, profile, 'face_eyes')
    if ids != profile['diagnostic_command_ids']:
        raise ValueError('Reference profile/calibration command IDs differ')
    positions, expression_report = scale_expression_targets(ids, positions, profile, shapes, args.expression_gain)
    for j, sid in enumerate(ids):
        report['target_ranges'][str(sid)] = [float(positions[:, j].min()), float(positions[:, j].max())]
    provenance = dict(work_id=args.prefix.name, original_clip=str(args.clip),
        original_clip_sha256=sha256(args.clip), original_fps=source_fps,
        original_frames=len(source['blendShapes']), video=str(args.video), video_sha256=sha256(args.video),
        reference_profile=str(args.reference_profile), reference_profile_sha256=sha256(args.reference_profile),
        conversion='zero-order hold at60Hz; each30FPS sample retained for two ticks', audio=audio,
        expression_gain=args.expression_gain, expression_axes=[a['id'] for a in expression_report['axes']])
    payload = dict(schema_version='mojia-motor-targets-0.1', playback_ready=False,
        device_limits_applied=True, pending_review=['Live startup and complete work acceptance'],
        source_clip=paths['clip'].name, source_clip_sha256=sha256(paths['clip']),
        audio=paths['audio'].name, audio_sha256=sha256(paths['audio']),
        calibration=calibration.name, calibration_sha256=sha256(calibration), baseline=2047,
        jaw_policy=profile['jaw_owner'], mixing=profile['mixing'],
        paired_eye_look_is_already_in_lids=profile['paired_eye_look_is_already_in_lids'],
        fps=dict(num=60, den=1), frames=len(values), duration_seconds=audio['duration_seconds'],
        motor_ids=ids, positions=positions.tolist(), trajectory_report=report,
        position_semantics='Held source samples at60Hz; original work timing; no neck animation',
        work_provenance=provenance, expression_gain_report=expression_report)
    paths['targets'].write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
    profile.update(targets_sha256=sha256(paths['targets']), audio_sha256=sha256(paths['audio']),
                   work_provenance=provenance, expression_gain_report=expression_report)
    paths['profile'].write_text(json.dumps(profile, ensure_ascii=False, indent=2)+'\n')
    paths['review'].write_text(json.dumps(dict(paths={k:str(v) for k,v in paths.items()},
        provenance=provenance, trajectory_report=report,
        expression_gain_report=expression_report), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(paths={k:str(v) for k,v in paths.items()}, frames=len(values),
                          source_fps=source_fps, audio=audio), ensure_ascii=False))


if __name__ == '__main__':
    main()
