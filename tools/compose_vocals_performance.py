"""Compose acoustic-phrase blinks and restrained singing eyes; preserve articulation."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from facemotion52 import load_stream,validate_frame

root=Path(__file__).resolve().parent.parent
producer=root/'outputs/vocals_Singing_claire_v3.0'
out=root/'outputs/vocals_Singing_VisibleBrows_FaceMotion52_v3.0'
out.mkdir(exist_ok=True)
frames=load_stream(producer/'facemotion52.jsonl')
meta=json.loads((producer/'metadata.json').read_text())
feat=json.loads((root/'outputs/vocals_acoustic_features.json').read_text())
times=np.array(feat['time_seconds']);rms=np.array(feat['rms']);slow=np.array(feat['smoothed_rms'])
duration=meta['duration_seconds'];scale=max(float(np.percentile(slow,85)),1e-5)
intensity=np.clip(slow/scale,0,1)
activity=gaussian_filter1d(np.clip(slow/(scale*.18),0,1),18)
rng=np.random.default_rng(20261015)
events=[];previous=.35
# Prefer actual acoustic valleys after phrases. Irregular intervals, not metronome timing.
while previous+3.0<duration-.4:
    desired=previous+float(rng.uniform(3.1,6.9))
    candidates=np.arange(math.ceil((previous+2.5)*30),min(math.floor((previous+8.2)*30),math.floor((duration-.45)*30))+1)/30
    if not len(candidates):break
    energy=np.interp(candidates,times,rms)
    score=energy/scale+.085*((candidates-desired)/2.0)**2
    score+=rng.uniform(0,.014,len(candidates))
    peak=float(candidates[np.argmin(score)])
    events.append({'peak_seconds':peak,'closing_seconds':float(rng.uniform(.07,.095)),
                   'closed_seconds':float(rng.uniform(.025,.045)),
                   'opening_seconds':float(rng.uniform(.14,.21)),
                   'reason':'acoustic phrase valley or quiet break', 'amplitude':1.0})
    previous=peak
# A few gentle, longer closures during sustained voiced phrases, widely separated.
voiced=np.array(feat['voicing_strength'])>.68
sustained=gaussian_filter1d(voiced.astype(float),18)>.86
options=np.where(sustained & (intensity>.6) & (times>12) & (times<duration-3))[0]
expressive=[]
for idx in sorted(options,key=lambda i:float(intensity[i]),reverse=True):
    peak=round(float(times[idx])*30)/30
    if any(abs(peak-e['peak_seconds'])<1.25 for e in events):continue
    if any(abs(peak-e['peak_seconds'])<24 for e in expressive):continue
    expressive.append({'peak_seconds':peak,'closing_seconds':.20,'closed_seconds':.12,
                       'opening_seconds':.32,'reason':'restrained sustained-note eyelid gesture','amplitude':.82})
    if len(expressive)==3:break
events=sorted(events+expressive,key=lambda e:e['peak_seconds'])

def blink(t):
    value=0.
    for e in events:
        p,c,h,o=(e[k] for k in ['peak_seconds','closing_seconds','closed_seconds','opening_seconds'])
        if p-c<=t<p:v=.5-.5*math.cos(math.pi*(t-p+c)/c)
        elif p<=t<=p+h:v=1.
        elif p+h<t<p+h+o:v=.5+.5*math.cos(math.pi*(t-p-h)/o)
        else:continue
        value=max(value,v*e['amplitude'])
    return value

changed={p+s for p in ['eyeBlink','eyeWide','eyeSquint'] for s in ['Left','Right']}|{'browInnerUp'}
modified=[]
for original in frames:
    f=json.loads(json.dumps(original));t=f['timestamp_ms']/1000
    a=float(np.interp(t,times,activity));v=float(np.interp(t,times,intensity));b=blink(t)
    f['blendshapes']['browInnerUp']=min(.45,f['blendshapes']['browInnerUp']+.055*a+.045*v)
    for side in ['Left','Right']:
        base=.012+.025*v
        f['blendshapes']['eyeBlink'+side]=base+(1-base)*b
        f['blendshapes']['eyeWide'+side]*=1-b
        f['blendshapes']['eyeSquint'+side]=min(.3,f['blendshapes']['eyeSquint'+side]+.065*v)*(1-b)
    assert f['extensions']==original['extensions']
    assert all(f['blendshapes'][n]==original['blendshapes'][n] for n in original['blendshapes'] if n not in changed)
    validate_frame(f);modified.append(f)
path=out/'facemotion52_singing.jsonl'
with path.open('w') as stream:
    for f in modified:stream.write(json.dumps(f,separators=(',',':'),allow_nan=False)+'\n')
poses=[]
for t in np.arange(0,duration+.2,.2):
    a=float(np.interp(t,times,activity));v=float(np.interp(t,times,intensity))
    # Blender-space Euler angles for the whole preview bust, not inferred head motion.
    poses.append({'time_seconds':float(t),'pitch_deg':float(.35*math.sin(t*.73)+.55*v),
                  'roll_deg':float((.5+.4*a)*math.sin(t*.29+.2)),
                  'yaw_deg':float((.4+.6*a)*math.sin(t*.19))})
report={'method':'audio-envelope phrase valleys plus few sustained-note gestures; seeded irregular schedule',
        'events':events,'seed':20261015,'native_mouth_jaw_tongue_teeth_unchanged':True,
        'frame_schema_validation':'passed','frames':len(modified),'visual_acceptance':'pending user review'}
(out/'blink_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(out/'performance_pose.json').write_text(json.dumps({'source':'art-directed low-amplitude preview bust pose, not measured or A2F-inferred head motion','keyframes':poses},indent=2)+'\n')
print(json.dumps({'frames':len(modified),'blinks':len(events)-len(expressive),'sustained_note_gestures':len(expressive),'stream':str(path)}))
