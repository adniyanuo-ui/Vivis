"""New song-specific emotional direction; does not reuse wz2 emotion data."""
import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d, median_filter

root=Path(__file__).resolve().parent.parent
out=root/'outputs/0001_SingingNarrative_claire_v3.0';out.mkdir(exist_ok=True)
features=json.loads((root/'outputs/vocals_acoustic_features.json').read_text())
t=np.array(features['time_seconds']);rms=np.array(features['rms']);energy=np.array(features['smoothed_rms'])
voiced=np.array(features['voicing_strength']);pitch=np.array(features['pitch_hz'])
duration=features['duration_seconds'];rate=16000
with wave.open(str(root/'assets/audio/vocals_v3.0.wav')) as w:
    x=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
starts=np.rint(t*rate-400).astype(int);samples=x[starts[:,None]+np.arange(800)]
spectrum=np.abs(np.fft.rfft(samples*np.hanning(800),n=1024))**2
freq=np.fft.rfftfreq(1024,1/rate)
brightness=(spectrum*freq).sum(axis=1)/np.maximum(spectrum.sum(axis=1),1e-12)
reliable=(voiced>.72)&(rms>.025)
log_pitch=np.interp(t,t[reliable],np.log2(pitch[reliable]))
log_pitch=gaussian_filter1d(median_filter(log_pitch,size=15),14)
low,high=np.percentile(log_pitch[reliable],[20,80])
pitch_level=np.clip((log_pitch-low)/max(high-low,.1),0,1)
brightness=gaussian_filter1d(brightness,14)
low,high=np.percentile(brightness[reliable],[20,80]);bright=np.clip((brightness-low)/max(high-low,1),0,1)
power=np.clip(energy/np.percentile(energy,85),0,1)
activity=gaussian_filter1d(np.clip(energy/(np.percentile(energy,85)*.18),0,1),15)
# Stable fragments agreed across recognitions anchor the acting. Unclear words are omitted.
direction=[
 (0,'开头等待；依据音乐留白',{'joy':.035,'sadness':.06}),
 (2.4,'开口前期待',{'joy':.12,'amazement':.10}),
 (3.7,'回来啦：期待上扬',{'joy':.33,'amazement':.22}),
 (6.7,'识别作是风：暂按期待落空处理，文本未核实',{'sadness':.23,'joy':.025}),
 (9.6,'谁等他啦：轻微逞强；人称不确定',{'cheekiness':.27,'joy':.09,'sadness':.09}),
 (12.2,'停顿，收回表情',{'sadness':.12,'joy':.025}),
 (19.5,'进入叙述',{'joy':.12,'sadness':.06}),
 (23.0,'回忆段：按温和音色处理，细节歌词不确定',{'joy':.26,'sadness':.065}),
 (29.5,'回忆余韵',{'joy':.18,'sadness':.09}),
 (33.2,'天一黑他出门：关注与担心',{'sadness':.22,'amazement':.045}),
 (38.6,'今晚别等我：情绪收紧',{'sadness':.32,'pain':.07}),
 (40.8,'我冷冷回他一个：克制的逞强',{'cheekiness':.31,'joy':.07,'anger':.035}),
 (45.6,'叙述收束：不依据识别不清的词推断',{'cheekiness':.12,'sadness':.10}),
 (49.8,'出门后探看：关注与期待',{'amazement':.17,'joy':.16,'sadness':.05}),
 (54.9,'等他回来，我就跟他说：鼓起勇气',{'joy':.29,'amazement':.13,'sadness':.09}),
 (57.6,'我，我想：犹豫',{'sadness':.30,'pain':.09,'joy':.035}),
 (60.7,'算了：回落',{'sadness':.49,'grief':.11,'joy':.0}),
 (63.0,'副歌进入：温柔表达',{'joy':.25,'sadness':.17}),
 (66.0,'问你吃了没有：关心',{'joy':.34,'sadness':.12}),
 (69.5,'我不怕冷：柔和延续，后句不确定',{'joy':.22,'sadness':.20}),
 (74.9,'我不用睡：守候感',{'joy':.16,'sadness':.29}),
 (79.2,'我没有心：脆弱感',{'sadness':.37,'pain':.07,'joy':.04}),
 (82.5,'心疼：压住情绪的长音，尾词不确定',{'sadness':.44,'grief':.15,'pain':.08,'joy':.02}),
 (86.8,'长音后释放',{'sadness':.25,'grief':.035,'joy':.025}),
 (94.0,'间奏：延续余韵',{'sadness':.15,'joy':.035}),
 (101.0,'间奏后段：恢复关注',{'sadness':.08,'joy':.08,'amazement':.03}),
 (107.8,'人声重新进入：依据音色轻微抬眉，歌词不确定',{'joy':.12,'sadness':.12,'amazement':.075}),
 (duration,'片段结尾，保留未完的情绪',{'joy':.08,'sadness':.14})]
names=json.loads((root/'assets/models/a2f-v3.0/network_info.json').read_text())['params']['emotions']
knots=np.array([v[0] for v in direction]);values=np.array([[v[2].get(n,0) for n in names] for v in direction])
curves=PchipInterpolator(knots,values,axis=0)
keys=[]
for second in sorted(set(np.arange(0,duration,.05).tolist()+[duration])):
    a=float(np.interp(second,t,activity));p=float(np.interp(second,t,power));f=float(np.interp(second,t,pitch_level));b=float(np.interp(second,t,bright))
    # Acoustic cues change intensity within the directed state; they do not assign emotions.
    gain=.91+.09*p+.025*a*(f+b-1)
    weights=np.clip(curves(second)*gain,0,.7)
    keys.append({'timestamp_ms':round(second*1000,5),'weights':weights.tolist()})
schedule={'emotion_names':names,'keyframes':keys,'source':'fresh song-specific acting direction from tentative lyric fragments, measured vocal dynamics, pitch contour and spectral brightness; no wz2 emotion schedule'}
(out/'emotion_schedule.json').write_text(json.dumps(schedule,ensure_ascii=False,indent=2)+'\n')
(out/'direction_plan.json').write_text(json.dumps({'lyrics_status':'no verified lyrics supplied; tentative acting interpretation, not exact semantic alignment','anchors':[{'time_seconds':s,'direction':label,'weights':v} for s,label,v in direction]},ensure_ascii=False,indent=2)+'\n')
(out/'acoustic_direction.json').write_text(json.dumps({'time_seconds':t.tolist(),'activity':activity.tolist(),'power':power.tolist(),'pitch_level':pitch_level.tolist(),'brightness':bright.tolist(),'method':'coarse autocorrelation pitch and spectral brightness; bounded expressive intensity cues, not inferred psychological states'}))
meta={'audio':str(root/'assets/audio/vocals_v3.0.wav'),'playback_audio':str(root/'0001.wav'),'original_audio':'0001.wav','duration_seconds':duration,'video_frames':math.ceil(duration*30),'sample_rate_hz':16000,'channels':1,'model':'nvidia/Audio2Face-3D-v3.0','character':'Claire','identity_index':0,'constant_noise':True,'video_playback_gain_multiplier':1,'emotion_schedule':schedule,'hide_emotion_label':True,'face_appearance':'warm_skin_visible_brows','original_audio_sha256':hashlib.sha256((root/'0001.wav').read_bytes()).hexdigest(),'emotion_source':'new song-specific direction; reuse only Claire model and BROWS appearance','lyrics_status':'tentative ASR fragments; user has no verified text','preview_times_seconds':[3.7,9.6,40.8,54.9,60.7,66,79.2,82.5]}
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print('PREPARED_NEW_SONG_DIRECTION',len(keys),len(direction))
