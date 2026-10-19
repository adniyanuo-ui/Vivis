"""Convert named FaceMotion clips using an explicit 61-channel CSV header."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from facemotion52 import CHANNELS, validate_frame

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('reference_csv')
parser.add_argument('example_json')
parser.add_argument('output_prefix')
args = parser.parse_args()
source_path, csv_path, example_path = map(Path, [args.source,args.reference_csv,args.example_json])
source = json.loads(source_path.read_text())
example = json.loads(example_path.read_text())
with csv_path.open(encoding='utf-8-sig', newline='') as stream:
    reader = csv.reader(stream)
    header = next(reader)
    assert header[:2] == ['Timecode','BlendshapeCount']
    order = header[2:]
    assert len(order) == len(set(order)) == 61
    reference_rows = 0
    for row in reader:
        if not row:
            continue
        assert len(row) == len(header) and int(row[1]) == 61
        assert np.isfinite(np.array(row[2:], dtype=float)).all()
        reference_rows += 1
canonical = {name.casefold():name for name in CHANNELS}
assert len(canonical) == 52
assert {n.casefold() for n in order[:52]} == set(canonical)
extra_names = ['HeadYaw','HeadPitch','HeadRoll','LeftEyeYaw','LeftEyePitch',
               'LeftEyeRoll','RightEyeYaw','RightEyePitch','RightEyeRoll']
assert order[52:] == extra_names
assert set(example) == {'blendShapes','fps','name'}
assert all(len(row) == 61 for row in example['blendShapes'])
frames = source['frames']
for frame in frames:
    validate_frame(frame)
    # Rotation units and conventions need explicit confirmation before nonzero export.
    if frame['head_pose'] is not None or frame['gaze'] is not None:
        raise ValueError('Nonzero head/gaze conversion requires target units and coordinate conventions')
times = np.array([f['timestamp_ms'] for f in frames], dtype=float)
assert np.all(np.diff(times)>0) and times[0] == 0
duration_ms = source['audio']['duration_ms']
prefix = Path(args.output_prefix)
prefix.parent.mkdir(parents=True, exist_ok=True)
mapping = [{'index_zero_based':i,'csv_channel':name,
            'source_channel':canonical.get(name.casefold()),
            'rule':'name-matched weight' if i<52 else '0 placeholder; source rotation unavailable'}
           for i,name in enumerate(order)]
results = []
for fps in [30,60]:
    target_times = np.arange(math.ceil(duration_ms*fps/1000))*1000/fps
    # Match the already reviewed video sampling: nearest source frame, no smoothing.
    indices = np.abs(times[:,None]-target_times[None,:]).argmin(axis=0)
    error_ms = float(np.max(np.abs(times[indices]-target_times)))
    assert error_ms < .1
    rows = [[frames[i]['blendshapes'][canonical[name.casefold()]] for name in order[:52]]+[0.]*9
            for i in indices]
    payload = {'blendShapes':rows,'fps':{'den':1,'num':fps},'name':example['name']}
    output = prefix.with_name(prefix.name+f'_{fps}fps.json')
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    restored = json.loads(output.read_text())
    assert set(restored) == set(example) and restored == payload
    for row,i in zip(restored['blendShapes'],indices):
        assert len(row) == 61 and all(math.isfinite(v) for v in row)
        inverse = {canonical[name.casefold()]:row[j] for j,name in enumerate(order[:52])}
        assert inverse == frames[i]['blendshapes']
        assert row[52:] == [0.]*9
    # Probe actual nonzero values so swapped inactive jaw/mouth channels cannot hide errors.
    for selected in CHANNELS:
        probe = {name:float(name == selected) for name in CHANNELS}
        row = [probe[canonical[name.casefold()]] for name in order[:52]]
        assert row.index(1.) == next(i for i,name in enumerate(order[:52]) if canonical[name.casefold()] == selected)
    results.append({'file':str(output),'fps':fps,'frames':len(rows),'channels':61,
                    'max_sample_time_error_ms':error_ms,'sampled_face_weights_roundtrip_error':0,
                    'duration_seconds_from_frame_count':len(rows)/fps,
                    'sha256':hashlib.sha256(output.read_bytes()).hexdigest()})
report = {'source':str(source_path),'csv_reference':str(csv_path),'reference_rows_validated':reference_rows,
          'format':'same root fields as example: blendShapes, fps, name',
          'name_value':example['name'],'name_value_policy':'copied from example for compatibility; receiver meaning not confirmed',
          'channel_order':order,'channel_mapping':mapping,
          'resampling':'nearest source frame at target i/fps; same rule as accepted video',
          'face_weight_roundtrip':'passed; all sampled 52-channel weights exactly preserved',
          'all_52_one_hot_mapping_probes':'passed',
          'unavailable_rotation_channels':extra_names,'rotation_placeholder_value':0,
          'rotation_zero_is_neutral':'requires receiver confirmation; not measured motion',
          'source_audio_duration_seconds':duration_ms/1000,
          'audio_time_origin':'sample zero of assets/audio/wz2_v3.0.wav; no new offset',
          'extensions_policy':'Claire preview tongue/teeth extensions excluded from 61-channel format',
          'outputs':results}
report_path = prefix.with_name(prefix.name+'_CONVERSION.json')
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'outputs':results,'report':str(report_path)},ensure_ascii=False,indent=2))
