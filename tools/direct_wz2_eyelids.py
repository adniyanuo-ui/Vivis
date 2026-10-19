"""Art-directed emotional eyelid dynamics; no physiological inference is made."""
import json
import math
from pathlib import Path

import numpy as np
from facemotion52 import load_stream, validate_frame

root = Path(__file__).resolve().parent.parent
producer = root/'outputs/wz2_Expressive_claire_v3.0'
out = root/'outputs/wz2_Expressive_EyelidDynamics_FaceMotion52_v3.0'
out.mkdir(exist_ok=True)
frames = load_stream(producer/'facemotion52.jsonl')
metadata = json.loads((producer/'metadata.json').read_text())
clauses = json.loads((root/'outputs/wz2_content_alignment.json').read_text())['content_phrases']
schedule = metadata['emotion_schedule']
key_times = [k['timestamp_ms'] for k in schedule['keyframes']]
key_values = np.array([k['weights'] for k in schedule['keyframes']])
rng = np.random.default_rng(20261014)
events = []
# Concentrated anger suppresses blinking; clusters belong to the self-reproach passage.
slots = [
    (.85, '开口前的短眨', 'normal'),
    (clauses[0]['end']+.16, '回忆转指责前的短停顿', 'normal'),
    (clauses[3]['end']+.36, '连续质问结束，解除凝视', 'normal'),
    (clauses[6]['end']+.28, '失落停顿中的较慢眨眼', 'slow'),
    (clauses[9]['start']+.25, '转入自责时的快速短眨', 'quick'),
    (clauses[11]['start']+.25, '心软前的局促短眨', 'quick'),
    (clauses[11]['start']+.62, '同一情绪段的短连眨', 'quick'),
    (clauses[12]['end']+.12, '痛苦转入生活失控前的短眨', 'normal'),
    (clauses[-1]['end']+.38, '最后爆发后的缓慢回落', 'slow'),
]
for t, reason, kind in slots:
    ranges = {'normal': ((.07,.09),(.026,.04),(.14,.18)),
              'quick': ((.055,.075),(.018,.028),(.10,.14)),
              'slow': ((.08,.10),(.04,.065),(.20,.26))}[kind]
    c, h, o = [float(rng.uniform(*interval)) for interval in ranges]
    events.append({'peak_seconds': round(t*30)/30, 'closing_seconds': c,
                   'closed_seconds': h, 'opening_seconds': o, 'reason': reason})

def smooth_window(t, start, end, fade=.35):
    def smooth(x):
        x = np.clip(x, 0, 1)
        return float(x*x*(3-2*x))
    return smooth((t-start+fade)/fade)*smooth((end+fade-t)/fade)

def blink(t):
    value = 0.
    for e in events:
        p, c, h, o = (e[k] for k in ['peak_seconds','closing_seconds','closed_seconds','opening_seconds'])
        if p-c <= t < p:
            value = max(value, .5-.5*math.cos(math.pi*(t-p+c)/c))
        elif p <= t <= p+h:
            value = 1.
        elif p+h < t < p+h+o:
            value = max(value, .5+.5*math.cos(math.pi*(t-p-h)/o))
    return value

anger_windows = [(3.25,7.6), (13.65,16.05), (23.62,26.08)]
modified_channels = {prefix+side for prefix in ['eyeBlink','eyeWide','eyeSquint','browDown']
                     for side in ['Left','Right']} | {'browInnerUp'}
modified = []
for frame in frames:
    f = json.loads(json.dumps(frame))
    t = f['timestamp_ms']/1000
    emotions = {name: float(np.interp(f['timestamp_ms'], key_times, key_values[:, i]))
                for i, name in enumerate(schedule['emotion_names'])}
    focused = max(smooth_window(t, a, z) for a, z in anger_windows)
    self_reproach = smooth_window(t, 16.95, 22.5, .5)
    sorrow = emotions['sadness']+.5*emotions['grief']
    tension = emotions['anger']*focused
    # Low-amplitude, slow variation avoids rigid sustained expressions and eyelid jitter.
    variation = 1+.045*math.sin(2*math.pi*t/3.7)+.025*math.sin(2*math.pi*t/5.3+.8)
    gesture = blink(t)
    # Mild resting lid lowering in sorrow; this is not counted as a blink.
    resting_lid = .055*sorrow*(1-.8*focused)
    lid = resting_lid+(1-resting_lid)*gesture
    f['blendshapes']['browInnerUp'] = min(.6, f['blendshapes']['browInnerUp']+.22*sorrow)
    for side, asymmetry in [('Left', .98), ('Right', 1.02)]:
        f['blendshapes']['browDown'+side] = min(.78,
            f['blendshapes']['browDown'+side]*(1-.22*self_reproach)+.20*tension*asymmetry)
        # Concurrent bounded widening and squint express focused upper/lower lid tension.
        # Never use a large startled widening for this angry recording.
        wide = min(.18, f['blendshapes']['eyeWide'+side]+.16*tension*variation*asymmetry)
        squint = min(.30, f['blendshapes']['eyeSquint'+side]+.22*tension*variation*asymmetry)
        f['blendshapes']['eyeWide'+side] = wide*(1-lid)
        f['blendshapes']['eyeSquint'+side] = squint*(1-lid)
        f['blendshapes']['eyeBlink'+side] = lid
    assert all(f['blendshapes'][n] == frame['blendshapes'][n]
               for n in frame['blendshapes'] if n not in modified_channels)
    assert f['extensions'] == frame['extensions']
    validate_frame(f)
    modified.append(f)

times = np.array([f['timestamp_ms']/1000 for f in modified])
render_times = np.arange(metadata['video_frames'])/30
indices = np.abs(times[:,None]-render_times).argmin(axis=0)
peak_checks = []
for event in events:
    index = round(event['peak_seconds']*30)
    weight = modified[indices[index]]['blendshapes']['eyeBlinkLeft']
    assert weight > .99
    peak_checks.append({'time_seconds': index/30, 'weight': weight})
for a, z in anger_windows:
    assert not any(e['peak_seconds']-e['closing_seconds'] < z and
                   e['peak_seconds']+e['closed_seconds']+e['opening_seconds'] > a for e in events)

stream_path = out/'facemotion52_eyelid_dynamics.jsonl'
with stream_path.open('w') as stream:
    for f in modified:
        stream.write(json.dumps(f, separators=(',',':'), allow_nan=False)+'\n')
report = {'method': 'manual acting direction from user reference and estimated content timings; not a physiological model or emotion diagnosis',
          'events': events, 'anger_blink_suppression_windows': anger_windows,
          'self_reproach_window': [16.95,22.5], 'rendered_blink_peak_checks': peak_checks,
          'all_non_upper_face_channels_and_extensions_unchanged': True,
          'frames_schema_valid': len(modified), 'video_frames': metadata['video_frames'],
          'visual_acceptance': 'pending user review'}
(out/'blink_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
