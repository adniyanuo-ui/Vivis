#!/usr/bin/env python3
"""Create a separate Mojia blink-duration variant; never open the serial port."""
import argparse
import copy
import json
import math
import wave
from pathlib import Path

import numpy as np

from mojia_player import load_targets
from prepare_mojia import MOJIA_IDS, sha256
from xiaobai_runtime.mapping import ROOT, inputs, read_json, compile_trajectory


def audio_energy_from_wav(path, frames, fps):
    """Normalized mixed-audio envelope; this does not classify lyrics or emotion."""
    with wave.open(str(path)) as wav:
        if wav.getsampwidth() != 2:
            raise ValueError('Expected PCM16 audio')
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
        samples = samples.reshape(-1, wav.getnchannels()).astype(float) / 32768
    # Average channel power so opposite-phase stereo cannot cancel the envelope.
    power = np.mean(samples * samples, axis=1)
    cumulative = np.r_[0., np.cumsum(power)]
    centres = np.rint(np.arange(frames) * rate / fps).astype(int)
    lo = np.clip(centres - round(.04 * rate), 0, len(power))
    hi = np.clip(centres + round(.04 * rate), 0, len(power))
    if np.any(hi <= lo):
        raise ValueError('Audio does not cover animation')
    db = 10 * np.log10(np.maximum((cumulative[hi] - cumulative[lo]) / (hi - lo), 1e-12))
    low, high = np.percentile(db, [10, 85])
    if high - low < 1e-6:
        return np.full(frames, .5)
    return np.clip((db - low) / (high - low), 0, 1)


def blink_timing(weights, order, energy, begin, end, fps):
    """Deterministic acting rule from local acoustic and expression context."""
    centre = (begin + end - 1) // 2
    radius = max(1, round(.25 * fps))
    lo, hi = max(0, centre - radius), min(len(weights), centre + radius + 1)
    level = float(np.mean(energy[max(0, centre-round(.08*fps)):min(len(energy), centre+round(.08*fps)+1)]))
    preceding = float(np.mean(energy[lo:centre+1]))
    following = float(np.mean(energy[centre:hi]))
    rising = max(0., following - preceding)
    # The source already contains expressive lid lowering in some passages.
    # Read surrounding pose, not the forced full blink at its peak.
    def surrounding(names):
        available = [order.index(n) for n in names if n in order]
        return float(np.median(np.mean(weights[lo:hi, available], axis=1))) if available else 0.
    lid_context = surrounding(['EyeBlinkLeft', 'EyeBlinkRight'])
    expression = surrounding(['BrowInnerUp', 'MouthFrownLeft', 'MouthFrownRight'])
    widening = surrounding(['EyeWideLeft', 'EyeWideRight'])
    softness = float(np.clip(.7*lid_context + .5*expression - widening, 0., 1.))
    authored_hold = (end - begin) / fps
    long_gesture = float(np.clip((authored_hold - .07) / .20, 0., 1.))
    quiet = 1. - level
    # Natural-blink reference: approximately 100 ms down and 220 ms up.
    # Context varies these phases, without a long fully-closed waiting plateau.
    closing = float(np.clip(.09 + .010*quiet + .005*softness, .08, .11))
    plateau = 1/fps if quiet > .9 and softness > .15 else 0.
    opening = float(np.clip(.205 + .025*quiet + .020*softness + .010*long_gesture - .010*rising, .18, .26))
    return closing, plateau, opening, dict(audio_level=level, audio_rise=rising,
        surrounding_lid=lid_context, expression_softness=softness, source_hold_s=authored_hold)


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(z)) for a, z in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def select_blinks(weights, order, energy, fps, keep_fraction):
    """Thin shared events by acoustic/context importance and irregular spacing."""
    if type(keep_fraction) not in (int, float) or not math.isfinite(keep_fraction) or not 0 < keep_fraction <= 1:
        raise ValueError('Invalid blink keep fraction')
    channels = [order.index('EyeBlink'+side) for side in ('Left', 'Right')]
    candidates = []
    for begin, end in runs(np.max(weights[:, channels], axis=1) >= .99):
        _, _, _, context = blink_timing(weights, order, energy, begin, end, fps)
        windows = {}; protected = (end-begin)/fps > .10 or context['expression_softness'] > .4
        for side, j in zip(('Left', 'Right'), channels):
            if not np.any(weights[begin:end, j] >= .99):
                protected = True  # Do not discard an authored unilateral gesture.
                continue
            lo, hi = begin, end-1
            while lo > max(0, begin-round(.35*fps)) and weights[lo, j] > .2:
                lo -= 1
            while hi < min(len(weights)-1, end+round(.35*fps)) and weights[hi, j] > .2:
                hi += 1
            if weights[lo, j] > .2 or weights[hi, j] > .2:
                protected = True  # Blink sits inside an expressive sustained closure.
            windows[side] = (lo, hi)
        score = 1.4*(1-context['audio_level']) + context['expression_softness'] - .8*context['audio_rise']
        candidates.append(dict(begin=begin, end=end, windows=windows, protected=bool(protected),
                               context=context, importance=float(score)))
    selected = set(range(len(candidates)))
    target = max(1, round(len(candidates)*keep_fraction)) if candidates else 0
    protected = {i for i, c in enumerate(candidates) if c['protected']}
    if target >= 4 and candidates:
        protected.update((0, len(candidates)-1))
    target = max(target, len(protected))
    while len(selected) > target:
        ordered = sorted(selected); cost = {}
        for k, i in enumerate(ordered):
            if i in protected:
                continue
            gaps = []
            if k: gaps.append((candidates[i]['begin']-candidates[ordered[k-1]]['begin'])/fps)
            if k+1 < len(ordered): gaps.append((candidates[ordered[k+1]]['begin']-candidates[i]['begin'])/fps)
            cost[i] = candidates[i]['importance'] + .7*min(gaps, default=0.)
        selected.remove(min(cost, key=lambda i:(cost[i], candidates[i]['begin'])))
    for i, c in enumerate(candidates):
        c['selected'] = i in selected
    return candidates


def remove_ordinary_pulses(original, ids, weights, order, shapes, profile, candidates):
    """Recompose only lid columns, preserving their non-blink expression inputs."""
    edited = weights.copy(); changed = np.zeros(len(weights), dtype=bool)
    for c in candidates:
        if c['protected']:
            continue
        for side, (lo, hi) in c['windows'].items():
            j = order.index('EyeBlink'+side)
            edited[lo:hi+1, j] = np.linspace(weights[lo, j], weights[hi, j], hi-lo+1)
            changed[lo:hi+1] = True
    result = original.copy()
    if changed.any():
        eyelids = {5, 6, 20, 21}
        lid_shapes = {name:{sid:p for sid,p in positions.items() if sid in eyelids}
                      for name,positions in shapes.items()}
        mixing = dict(profile, mixing=profile.get('mixing', 'bounded_weighted_deltas'),
                      jaw_owner=profile.get('jaw_owner', 'all_calibrated'),
                      paired_eye_look_is_already_in_lids=profile.get('paired_eye_look_is_already_in_lids', True))
        lid_ids, positions, _ = compile_trajectory(edited, order, lid_shapes, mixing, 'face')
        for j, sid in enumerate(lid_ids):
            result[changed, ids.index(sid)] = positions[changed, j]
    return result


def adjust_blinks(payload, profile, weights, order, shapes, speed=1200, audio_energy=None, keep_fraction=1.):
    if type(speed) is not int or not 1 <= speed <= 32767:
        raise ValueError('Invalid eyelid speed')
    ids = payload['motor_ids']; original = np.asarray(payload['positions'], dtype=float)
    weights = np.asarray(weights, dtype=float)
    fps = payload['fps']['num'] / payload['fps']['den']
    if (not math.isfinite(fps) or fps <= 0 or original.ndim != 2
            or original.shape != (payload['frames'], len(ids)) or not len(original)
            or weights.shape != (len(original), len(order))
            or not np.isfinite(original).all() or not np.isfinite(weights).all()):
        raise ValueError('Invalid synchronized blink inputs')
    energy = np.full(len(original), .5) if audio_energy is None else np.asarray(audio_energy, dtype=float)
    if energy.shape != (len(original),) or not np.isfinite(energy).all() or np.any((energy < 0) | (energy > 1)):
        raise ValueError('Invalid audio energy context')
    candidates = select_blinks(weights, order, energy, fps, keep_fraction)
    result = remove_ordinary_pulses(original, ids, weights, order, shapes, profile, candidates)
    baseline = result.copy(); motors = {m['id']: m for m in profile['motors']}
    report = dict(speed_counts_s=speed, source_blink_threshold=.99,
                  timing_mode='natural_asymmetric_reduced', audio_context_used=audio_energy is not None,
                  timing_policy='Stronger/rising audio: shorter; quieter or lowered-lid expression: gentler release; original peak times retained',
                  fixed_endpoint_dwell=False,
                  reference=dict(closing_s=.10, opening_s=.22, total_s=.32,
                      source='https://pubmed.ncbi.nlm.nih.gov/19157026/'),
                  phase_profiles=dict(closing='front-loaded sine', opening='cosine with time exponent 1.4'),
                  frequency=dict(source_events=len(candidates), selected_events=sum(c['selected'] for c in candidates),
                      keep_fraction=keep_fraction, candidates=candidates), events=[])
    for side, expected in [('Left', {5, 6}), ('Right', {20, 21})]:
        name = 'EyeBlink' + side; endpoints = shapes[name]
        if set(endpoints) != expected or not expected <= set(ids):
            raise ValueError('Unexpected Mojia eyelid mapping')
        for sid, endpoint in endpoints.items():
            m = motors[sid]
            if not max(m['safe_min'], m['device_min']) <= endpoint <= min(m['safe_max'], m['device_max']):
                raise ValueError(f'ID{sid}: closure endpoint outside existing limits')
        blink = weights[:, order.index(name)]
        if np.any(blink < 0) or np.any(blink > 1):
            raise ValueError('Invalid blink weight')
        # Detect separate full-closure peaks. A sustained half-closed expression
        # between two blinks must not turn into one multi-second closed plateau.
        for begin, end in runs(blink >= .99):
            candidate = next(c for c in candidates if c['begin'] <= begin < c['end'])
            if not candidate['selected']:
                continue
            closed_start = int(begin); original_end = int(end)
            closing, plateau, opening, context = blink_timing(weights, order, energy, begin, end, fps)
            close_frames = max(1, round(closing * fps))
            plateau_frames = max(0, round(plateau * fps))
            open_frames = max(2, round(opening * fps))
            start = max(0, closed_start - close_frames)
            closed_end = (max(original_end-1, closed_start + plateau_frames) if candidate['protected']
                          else closed_start + plateau_frames)
            finish = closed_end + open_frames + 1
            if finish > len(original):
                raise ValueError('Insufficient song tail for complete blink and reopening')
            frames = np.arange(start, finish)
            closure = np.ones(len(frames))
            before = frames < closed_start
            if closed_start > start:
                closure[before] = np.sin(np.pi/2 * (frames[before] - start) / (closed_start - start))
            after = frames >= closed_end
            phase = (frames[after] - closed_end) / (finish - closed_end - 1)
            closure[after] = .5 + .5 * np.cos(np.pi * phase**1.4)
            for sid, endpoint in endpoints.items():
                j = ids.index(sid); base = baseline[frames, j]
                proposed = base + closure * (endpoint - base)
                # Preserve a stronger pre-existing closure, including overlaps.
                result[frames, j] = (np.maximum(result[frames, j], proposed)
                                     if endpoint > motors[sid]['neutral']
                                     else np.minimum(result[frames, j], proposed))
            report['events'].append(dict(side=side, source_peak_frame=closed_start,
                start_frame=start, closed_start_frame=closed_start, closed_end_frame=closed_end,
                finish_frame=finish, command_plateau_s=(closed_end - closed_start) / fps,
                closing_s=(closed_start-start)/fps, opening_s=(finish-closed_end-1)/fps,
                duration_s=(finish-start-1)/fps, context=context,
                kind='expressive' if candidate['protected'] else 'ordinary',
                endpoints={str(sid): endpoint for sid, endpoint in endpoints.items()}))
    output = copy.deepcopy(payload); output['positions'] = result.tolist()
    output['blink_adjustment'] = report
    if 'trajectory_report' in output:
        output['trajectory_report']['postprocessing'] = 'Blink closure timing; original raw ranges precede this adjustment'
        for j, sid in enumerate(ids):
            output['trajectory_report']['target_ranges'][str(sid)] = [float(result[:, j].min()), float(result[:, j].max())]
    for sid in (5, 6, 20, 21):
        column = result[:, ids.index(sid)]; m = motors[sid]
        if column.min() < m['safe_min'] or column.max() > m['safe_max']:
            raise ValueError('Adjusted blink exceeds limits')
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--audio', type=Path, default=ROOT/'exmple/0001.wav')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speed', type=int, default=1200)
    parser.add_argument('--keep-fraction', type=float, default=.7)
    args = parser.parse_args()
    profile = read_json(args.profile); payload = read_json(args.targets)
    load_targets(args.targets, profile, args.audio, ROOT/'mojia/墨甲9.23.json')
    if sha256(args.clip) != payload['source_clip_sha256']:
        raise ValueError('Source clip hash mismatch')
    weights, fps, order, shapes = inputs(args.clip, ROOT/'mojia/墨甲9.23.json',
        ROOT/'xiaobai/blendshape_channels_61.csv', allowed_motor_ids=MOJIA_IDS)
    if fps != payload['fps']['num']/payload['fps']['den']:
        raise ValueError('Source FPS mismatch')
    energy = audio_energy_from_wav(args.audio, len(weights), fps)
    output, report = adjust_blinks(payload, profile, weights, order, shapes, speed=args.speed, audio_energy=energy, keep_fraction=args.keep_fraction)
    report.update(source_targets_sha256=sha256(args.targets), source_profile_sha256=sha256(args.profile))
    result_profile = copy.deepcopy(profile)
    for m in result_profile['motors']:
        if m['id'] in (5, 6, 20, 21):
            m['max_speed_counts_s'] = args.speed
    result_profile['blink_trial'] = dict(targets=args.output.name, changed_ids=[5, 6, 20, 21],
                                        speed_counts_s=args.speed, timing_mode=report['timing_mode'], mechanical_acceptance='pending')
    profile_path = args.output.with_name(args.output.stem+'_PROFILE.json')
    if args.output.exists() or profile_path.exists():
        raise ValueError('Choose new output paths; preserve existing versions')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
    result_profile['targets_sha256'] = sha256(args.output)
    profile_path.write_text(json.dumps(result_profile, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(targets=str(args.output), profile=str(profile_path), events=len(report['events']),
                          changed_ids=[5, 6, 20, 21]), ensure_ascii=False))


if __name__ == '__main__':
    main()
