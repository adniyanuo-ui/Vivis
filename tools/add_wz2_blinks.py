"""Add nonperiodic eyelid gestures at wz2's emotional transitions."""
import json
import math
from pathlib import Path

import numpy as np
from facemotion52 import load_stream, validate_frame

root = Path(__file__).resolve().parent.parent
producer = root/'outputs/wz2_Expressive_claire_v3.0'
out = root/'outputs/wz2_Expressive_ContentBlink_FaceMotion52_v3.0'
out.mkdir(exist_ok=True)
frames = load_stream(producer/'facemotion52.jsonl')
schedule = json.loads((producer/'emotion_schedule.json').read_text())
key_times = [k['timestamp_ms'] for k in schedule['keyframes']]
key_values = np.array([k['weights'] for k in schedule['keyframes']])
clauses = json.loads((root/'outputs/wz2_content_alignment.json').read_text())['content_phrases']
rng = np.random.default_rng(20261013)
events = []
slots = [
    (clauses[0]['end']+.2, '回忆结束，转入指责', False),
    (clauses[3]['end']+.24, '质问后停顿，转入失落', False),
    (clauses[6]['end']+.27, '失去自我后的停顿', True),
    (clauses[6]['end']+.67, '同一停顿中的短连眨', False),
    (clauses[8]['end']+.43, '愤怒指责后，转向自责', True),
    (clauses[11]['start']+.16, '连接词期间短暂闭眼，随后说出心软', False),
    (clauses[-1]['end']+.38, '最后重复强调后的疲惫回落', True),
]
for t, reason, slow in slots:
    events.append({'peak_seconds': round(t*30)/30, 'closing_seconds': float(rng.uniform(.065,.095)),
        'closed_seconds': float(rng.uniform(.035,.06) if slow else rng.uniform(.025,.04)),
        'opening_seconds': float(rng.uniform(.19,.25) if slow else rng.uniform(.13,.18)),
        'reason': reason})

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

modified_channels = {prefix+side for prefix in ['eyeBlink','eyeWide','eyeSquint','browDown'] for side in ['Left','Right']} | {'browInnerUp'}
modified = []
for frame in frames:
    f = json.loads(json.dumps(frame))
    b = blink(f['timestamp_ms']/1000)
    emotions = {name: float(np.interp(f['timestamp_ms'], key_times, key_values[:, i]))
                for i, name in enumerate(schedule['emotion_names'])}
    # Bounded upper-face polish improves readability on the untextured preview rig.
    # Native speech mouth shapes, tongue, and teeth are preserved exactly.
    sorrow = emotions['sadness']+.5*emotions['grief']
    f['blendshapes']['browInnerUp'] = min(.65, f['blendshapes']['browInnerUp']+.24*sorrow)
    for side in ['Left','Right']:
        asymmetry = .97 if side == 'Left' else 1.03
        f['blendshapes']['browDown'+side] = min(.78, f['blendshapes']['browDown'+side]+.16*emotions['anger']*asymmetry)
        f['blendshapes']['eyeSquint'+side] = min(.35, f['blendshapes']['eyeSquint'+side]+.10*emotions['anger']*asymmetry)
        f['blendshapes']['eyeBlink'+side] = b
        for prefix in ['eyeWide','eyeSquint']:
            f['blendshapes'][prefix+side] *= 1-b
    assert all(f['blendshapes'][n] == frame['blendshapes'][n] for n in frame['blendshapes'] if n not in modified_channels)
    assert f['extensions'] == frame['extensions']
    validate_frame(f)
    modified.append(f)
for event in events:
    f = min(modified, key=lambda f: abs(f['timestamp_ms']/1000-event['peak_seconds']))
    assert f['blendshapes']['eyeBlinkLeft'] > .99
stream_path = out/'facemotion52_content_blinks.jsonl'
with stream_path.open('w') as stream:
    for f in modified:
        stream.write(json.dumps(f, separators=(',',':'), allow_nan=False)+'\n')
report = {'method': 'manually directed emotional transition blinks from estimated text timing; variable closure and recovery',
    'seed': 20261013, 'events': events, 'intervals_seconds': np.diff([e['peak_seconds'] for e in events]).tolist(),
    'upper_face_polish': 'bounded brow and eye tension from semantic emotion timeline; slight bilateral variation',
    'all_mouth_channels_tongue_and_teeth_unchanged': True, 'schema_validation': 'passed',
    'blink_peak_checks': 'passed', 'visual_acceptance': 'pending user review'}
(out/'blink_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
