"""Direct native Claire emotions from wz2's clause timing and vocal energy."""
import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
folder = root/'outputs/wz2_Expressive_claire_v3.0'
folder.mkdir(exist_ok=True)
audio = root/'assets/audio/wz2_v3.0.wav'
alignment = json.loads((root/'outputs/wz2_content_alignment.json').read_text())
clauses = alignment['content_phrases']
with wave.open(str(audio)) as wav:
    rate = wav.getframerate()
    samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')/32768
duration = len(samples)/rate
names = json.loads((root/'assets/models/a2f-v3.0/network_info.json').read_text())['params']['emotions']
# Mixed emotions stay restrained; anger never becomes a permanent maximal pose.
directions = [
    ('委屈回忆', {'sadness': .32, 'pain': .12, 'anger': .12}),
    ('愤怒指责', {'anger': .62, 'pain': .18}),
    ('质问', {'anger': .56, 'disgust': .10}),
    ('再次质问', {'anger': .65, 'disgust': .12}),
    ('失落', {'sadness': .40, 'grief': .14, 'anger': .18}),
    ('失落加深', {'sadness': .46, 'pain': .18, 'anger': .15}),
    ('失去自我', {'sadness': .54, 'grief': .22, 'pain': .20}),
    ('愤怒再起', {'anger': .68, 'disgust': .12, 'pain': .12}),
    ('指责高点', {'anger': .72, 'disgust': .20, 'pain': .12}),
    ('转向自责', {'sadness': .52, 'pain': .30, 'anger': .20}),
    ('清醒与厌恶', {'disgust': .30, 'sadness': .35, 'anger': .20}),
    ('心软与懊悔', {'sadness': .58, 'grief': .16, 'pain': .22}),
    ('被拖累的痛苦', {'sadness': .40, 'pain': .34, 'anger': .28}),
    ('生活失控', {'sadness': .35, 'anger': .35, 'pain': .22}),
    ('第一次爆发', {'anger': .62, 'grief': .24, 'pain': .28}),
    ('重复强调后回落', {'anger': .58, 'sadness': .32, 'pain': .30}),
]
assert len(clauses) == len(directions)
centers = np.array([(c['start']+c['end'])/2 for c in clauses])
targets = np.array([[d.get(name, 0) for name in names] for _, d in directions])
# Smoothstep interpolation makes transitions continuous with zero endpoint slope.
anchors = np.r_[0, centers, duration]
vectors = np.vstack([targets[0]*.65, targets, targets[-1]*.45])
step = rate//50
energy = np.sqrt(np.mean(samples[:len(samples)//step*step].reshape(-1, step)**2, axis=1))
energy = np.convolve(np.pad(energy, (5,5), mode='edge'), np.ones(11)/11, mode='valid')
energy_scale = max(float(np.percentile(energy, 85)), 1e-4)
keys = []
for t in sorted(set(np.arange(0, duration, .1).tolist()+[duration])):
    j = min(np.searchsorted(anchors, t, side='right')-1, len(anchors)-2)
    ratio = np.clip((t-anchors[j])/(anchors[j+1]-anchors[j]), 0, 1)
    ratio = ratio*ratio*(3-2*ratio)
    value = vectors[j]*(1-ratio)+vectors[j+1]*ratio
    vocal = np.clip(energy[min(round(t*50), len(energy)-1)]/energy_scale, 0, 1.3)
    # A bounded envelope adds vocal emphasis without twitching with each syllable.
    value *= .88+.12*min(vocal, 1)
    keys.append({'timestamp_ms': round(t*1000, 5), 'weights': value.tolist()})
schedule = {'emotion_names': names, 'keyframes': keys,
            'source': 'manual semantic direction using estimated clause timings and smoothed measured audio energy; not automatic emotion recognition'}
(folder/'emotion_schedule.json').write_text(json.dumps(schedule, ensure_ascii=False, indent=2)+'\n')
plan = [{'text': c['text'], 'start': c['start'], 'end': c['end'],
         'direction': label, 'target_weights': weights} for c, (label, weights) in zip(clauses, directions)]
(folder/'expression_direction.json').write_text(json.dumps({'clauses': plan, 'timing_source': alignment['method'],
    'head_and_gaze': 'not estimated', 'visual_acceptance': 'pending user review'}, ensure_ascii=False, indent=2)+'\n')
metadata = {'audio': str(audio), 'original_audio': 'wz2.m4a', 'transcript': 'wz2.txt',
    'duration_seconds': duration, 'sample_rate_hz': rate, 'channels': 1,
    'model': 'nvidia/Audio2Face-3D-v3.0', 'character': 'Claire', 'identity_index': 0,
    'constant_noise': True, 'video_playback_gain_multiplier': 1,
    'video_frames': math.ceil(duration*30), 'peak_dbfs': float(20*np.log10(abs(samples).max())),
    'rms_dbfs': float(20*np.log10(np.sqrt(np.mean(samples**2)))),
    'emotion_schedule': schedule, 'hide_emotion_label': True,
    'emotion_mode': 'native emotion conditioning, manual content direction with bounded audio-energy modulation',
    'content_alignment': str(root/'outputs/wz2_content_alignment.json')}
for field, path in [('original_audio_sha256', root/'wz2.m4a'), ('transcript_sha256', root/'wz2.txt')]:
    metadata[field] = hashlib.sha256(path.read_bytes()).hexdigest()
(folder/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'duration': duration, 'video_frames': metadata['video_frames'], 'plan': plan}, ensure_ascii=False, indent=2))
