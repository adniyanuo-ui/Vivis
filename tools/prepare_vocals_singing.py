"""Prepare a restrained singing performance from the actual vocal envelope."""
import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

root=Path(__file__).resolve().parent.parent
out=root/'outputs/vocals_Singing_claire_v3.0'
out.mkdir(exist_ok=True)
audio=root/'assets/audio/vocals_v3.0.wav'
with wave.open(str(audio)) as w:
    rate=w.getframerate()
    samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
duration=len(samples)/rate
features=json.loads((root/'outputs/vocals_acoustic_features.json').read_text())
times=np.array(features['time_seconds']);energy=np.array(features['smoothed_rms'])
scale=max(float(np.percentile(energy,85)),1e-5)
activity=gaussian_filter1d(np.clip(energy/(scale*.18),0,1),18)
intensity=np.clip(energy/scale,0,1)
names=json.loads((root/'assets/models/a2f-v3.0/network_info.json').read_text())['params']['emotions']
# The role is a warm, reflective performance. Phrase loudness modulates its strength.
# These are directed expression controls, not a claim of emotion recognition.
keys=[]
for t in sorted(set(np.arange(0,duration,.10).tolist()+[duration])):
    a=float(np.interp(t,times,activity));v=float(np.interp(t,times,intensity))
    reflective=.5+.5*math.sin(t*.087+.3)
    values={'joy':.07+a*(.11+.13*v),'sadness':a*(.025+.045*reflective),
            'amazement':a*.025*v,'outofbreath':0.0}
    keys.append({'timestamp_ms':round(t*1000,5),'weights':[values.get(n,0) for n in names]})
schedule={'emotion_names':names,'keyframes':keys,
          'source':'manually directed warm reflective singing; native emotions modulated by slowly smoothed measured vocal energy'}
(out/'emotion_schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
metadata={'audio':str(audio),'original_audio':'vocals.wav','duration_seconds':duration,
          'sample_rate_hz':rate,'channels':1,'model':'nvidia/Audio2Face-3D-v3.0',
          'character':'Claire','identity_index':0,'constant_noise':True,
          'video_playback_gain_multiplier':1,'video_frames':math.ceil(duration*30),
          'emotion_schedule':schedule,'hide_emotion_label':True,
          'face_appearance':'warm_skin_visible_brows',
          'original_audio_sha256':hashlib.sha256((root/'vocals.wav').read_bytes()).hexdigest(),
          'transcript_required_for_lip_sync':False,
          'transcript_status':'automatic Whisper draft; not ground truth',
          'performance_direction':'warm reflective singing, acoustic phrase timing; preserve native sung articulation',
          'peak_dbfs':float(20*np.log10(abs(samples).max())),
          'rms_dbfs':float(20*np.log10(np.sqrt(np.mean(samples*samples))))}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({'duration':duration,'video_frames':metadata['video_frames'],'emotion_keys':len(keys)}))
