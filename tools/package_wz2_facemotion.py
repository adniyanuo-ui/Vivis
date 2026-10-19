"""Audit A2F outputs and package native and directed FaceMotion-52 JSON clips."""
import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np
from facemotion52 import CHANNELS, VERSION, load_stream, map_weights, validate_frame

root = Path(__file__).resolve().parent.parent
producer = root/'outputs/wz2_Expressive_claire_v3.0'
directed = root/'outputs/wz2_Expressive_EyelidDynamics_FaceMotion52_v3.0'
audit = root/'outputs/wz2_FACEMOTION_52_AUDIT'
audit.mkdir(exist_ok=True)
info = json.loads((producer/'export_info.json').read_text())
bindings = json.loads((producer/'source_bindings.json').read_text())
metadata = json.loads((producer/'metadata.json').read_text())
count = info['frames']
with wave.open(metadata['audio']) as audio:
    sample_rate, audio_samples = audio.getframerate(), audio.getnframes()
duration = audio_samples/sample_rate
assert count == math.ceil(duration*60)
assert hashlib.sha256((root/'wz2.m4a').read_bytes()).hexdigest() == metadata['original_audio_sha256']
assert hashlib.sha256((root/'wz2.txt').read_bytes()).hexdigest() == metadata['transcript_sha256']
sizes = {'skin': count*info['skin_vertices']*3,
         'tongue': count*info['tongue_vertices']*3, 'jaw': count*16,
         'weights_skin': count*52, 'weights_tongue': count*16}
for name, size in sizes.items():
    path = producer/(name+'.f32')
    assert path.stat().st_size == size*4, name
    assert np.isfinite(np.memmap(path, dtype='float32', mode='r')).all(), name
stamps = np.loadtxt(producer/'timestamps.csv', delimiter=',', dtype=np.int64)
assert np.array_equal(stamps[:,0], np.arange(count))
assert stamps[0,1] == 0 and np.all(np.diff(stamps[:,1])>0)
assert stamps[-1,1] < audio_samples
grid_error_ms = float(np.max(np.abs(stamps[:,1]/sample_rate-np.arange(count)/60))*1000)
assert grid_error_ms <= 1000/sample_rate
native = load_stream(producer/'facemotion52.jsonl')
final = load_stream(directed/'facemotion52_eyelid_dynamics.jsonl')
assert len(native) == len(final) == count
names = bindings['skin']['names']
weights = np.fromfile(producer/'weights_skin.f32', dtype='float32').reshape(count,52)
assert set(names) == set(CHANNELS) and len(set(names)) == 52
modified_channels = {prefix+side for prefix in ['eyeBlink','eyeWide','eyeSquint','browDown']
                     for side in ['Left','Right']} | {'browInnerUp'}
mapping_max_error = 0.
for i, (a,b) in enumerate(zip(native,final)):
    mapped = map_weights(names, weights[i])
    assert a['blendshapes'] == mapped
    mapping_max_error = max(mapping_max_error, max(abs(a['blendshapes'][n]-float(weights[i,names.index(n)])) for n in CHANNELS))
    assert a['timestamp_ms'] == b['timestamp_ms'] == stamps[i,1]*1000/sample_rate
    assert a['extensions'] == b['extensions']
    assert all(a['blendshapes'][n] == b['blendshapes'][n] for n in CHANNELS if n not in modified_channels)
    validate_frame(b)

common = {'schema_version': VERSION, 'format': 'FaceMotion-52 clip; frames conform to per-frame schema',
          'channel_names': list(CHANNELS), 'frame_count': count,
          'audio': {'original_file': 'wz2.m4a', 'sha256': metadata['original_audio_sha256'],
                    'timeline_file': 'assets/audio/wz2_v3.0.wav',
                    'timeline_sha256': hashlib.sha256(Path(metadata['audio']).read_bytes()).hexdigest(),
                    'sample_rate_hz': sample_rate, 'sample_count': audio_samples,
                    'duration_ms': duration*1000, 'time_origin': 'sample zero of timeline_file'},
          'transcript': {'file': 'wz2.txt', 'sha256': metadata['transcript_sha256']},
          'model': {'name': metadata['model'], 'character': 'Claire', 'identity_index': 0},
          'timing': {'fps_numerator': 60, 'fps_denominator': 1, 'timestamp_units': 'milliseconds',
                     'playback_rule': 'display/interpolate frames against the same audio playback clock',
                     'final_frame_hold_until_ms': duration*1000},
          'semantics': {'mapping': 'exact channel-name matching; no index-only mapping',
                        'left_right': 'subject anatomical left/right', 'weights_range': [0,1],
                        'weights_are_independent': True, 'gaze_and_head_pose': 'null means unavailable',
                        'extensions': 'Claire preview tongue and teeth data; not robot servo commands'}}
files = []
for label, frames, filename in [
    ('native semantic adapter', native, 'wz2_FACEMOTION_52_NATIVE_v1.0_v3.0-Claire.json'),
    ('native semantic adapter plus accepted content-directed brow/eyelid animation', final, 'wz2_FACEMOTION_52_v1.0_v3.0-Claire.json'),
]:
    payload = dict(common, variant=label, frames=frames)
    path = root/'outputs'/filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    # Read the actual delivery JSON back and prepare independent-consumer input.
    restored = json.loads(path.read_text())
    assert restored['frames'] == frames
    stream_path = audit/('native.jsonl' if frames is native else 'final.jsonl')
    with stream_path.open('w') as stream:
        for f in restored['frames']:
            validate_frame(f)
            stream.write(json.dumps(f, separators=(',',':'), allow_nan=False)+'\n')
    files.append({'file': filename, 'bytes': path.stat().st_size,
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
report = {'source_audio_hash_matches': True, 'source_transcript_hash_matches': True,
          'a2f_float_outputs_correct_size_and_finite': True, 'frames': count,
          'channels_per_frame':52, 'native_semantic_mapping_max_weight_error': mapping_max_error,
          'native_inactive_channels': [n for n,active in zip(names,bindings['skin']['active']) if not active],
          'final_changed_channels_allowed': sorted(modified_channels),
          'final_all_mouth_jaw_tongue_teeth_data_unchanged':True,
          'all_frame_timestamps_unchanged':True, 'timestamp_grid_max_error_ms':grid_error_ms,
          'audio_duration_ms':duration*1000, 'last_frame_timestamp_ms':native[-1]['timestamp_ms'],
          'final_frame_hold_ms':duration*1000-native[-1]['timestamp_ms'],
          'json_roundtrip_and_frame_schema_validation':'passed', 'files':files,
          'scope':'digital interface and reconstruction validation; no robot hardware validation',
          'subjective_lip_sync':'previously reviewed video uses the same final stream; no new subjective acceptance inferred'}
(audit/'adapter_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
