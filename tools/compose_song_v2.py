"""Song-specific upper-face and preview direction; keep native articulation intact."""
import copy
import json
import math
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from facemotion52 import load_stream, validate_frame

root=Path(__file__).resolve().parent.parent
producer=root/'outputs/0001_SingingNarrative_claire_v3.0'
out=root/'outputs/0001_SingingNarrative_VisibleBrows_FaceMotion52_v3.0'
out.mkdir(exist_ok=True)
frames=load_stream(producer/'facemotion52.jsonl')
meta=json.loads((producer/'metadata.json').read_text())
feat=json.loads((producer/'acoustic_direction.json').read_text())
schedule=json.loads((producer/'emotion_schedule.json').read_text())
st=np.array([k['timestamp_ms']/1000 for k in schedule['keyframes']])
sw=np.array([k['weights'] for k in schedule['keyframes']])
at=np.array(feat['time_seconds']);duration=meta['duration_seconds']
raw=json.loads((root/'outputs/vocals_acoustic_features.json').read_text())
rt=np.array(raw['time_seconds']);rms=np.array(raw['rms'])
# Each event has a phrase/story purpose, with an acoustic valley selected locally.
targets=[(1.4,'waiting'),(5.3,'expectation falls'),(8.65,'thought change'),
 (11.2,'spoken line release'),(16.1,'instrumental breath'),(19.0,'prepare entry'),
 (25.3,'recollection phrase'),(31.2,'phrase end'),(35.2,'departure thought'),
 (39.55,'withholding a response'),(44.4,'defensive phrase end'),(47.1,'new thought'),
 (52.5,'return attention'),(56.4,'hesitation'),(59.25,'hesitation release'),
 (67.05,'tender phrase end'),(72.65,'phrase end'),(77.65,'thought transition'),
 (85.4,'vocal release'),(90.2,'afterglow'),(96.5,'afterglow'),
 (100.1,'renew attention'),(104.8,'prepare reentry'),(111.7,'last phrase')]
rng=np.random.default_rng(20261023)
events=[]
for target,reason in targets:
    candidates=np.arange(target-.25,target+.26,1/30)
    score=np.interp(candidates,rt,rms)+.06*((candidates-target)/.25)**2
    peak=round(float(candidates[np.argmin(score)])*30)/30
    events.append(dict(peak_seconds=peak,closing_seconds=float(rng.uniform(.065,.10)),
        closed_seconds=float(rng.uniform(.025,.052)),opening_seconds=float(rng.uniform(.13,.22)),
        amplitude=1.,reason=reason))
# Different durations for hesitation, surrender, and sustained lyrical emotion.
for p,c,h,o,a,reason in [(56.83,.07,.025,.15,.92,'small second hesitation blink'),
 (61.2,.22,.10,.43,1.,'letting go after tentative 算了'),
 (70.7,.24,.16,.40,.88,'sustained tender note'),
 (83.05,.28,.22,.52,1.,'heartache phrase, inward attention')]:
    events.append(dict(peak_seconds=p,closing_seconds=c,closed_seconds=h,
                       opening_seconds=o,amplitude=a,reason=reason))
events.sort(key=lambda e:e['peak_seconds'])
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
changed={p+s for p in ['eyeBlink','eyeWide','eyeSquint','browDown','browOuterUp'] for s in ['Left','Right']}|{'browInnerUp'}
path=out/'facemotion52_singing.jsonl'
with path.open('w') as stream:
    for original in frames:
        f=copy.deepcopy(original);t=f['timestamp_ms']/1000;b=blink(t)
        emotion={name:float(np.interp(t,st,sw[:,i])) for i,name in enumerate(schedule['emotion_names'])}
        power=float(np.interp(t,at,feat['power']));activity=float(np.interp(t,at,feat['activity']))
        sad=emotion['sadness'];joy=emotion['joy'];cheek=emotion['cheekiness'];amaze=emotion['amazement']
        w=f['blendshapes']
        w['browInnerUp']=min(.56,w['browInnerUp']+.36*sad+.07*power*activity)
        for side,asym in [('Left',1.),('Right',.60)]:
            w['browOuterUp'+side]=min(.43,w['browOuterUp'+side]+.20*amaze+.20*cheek*asym)
            w['browDown'+side]=min(.36,w['browDown'+side]+.11*cheek*(1.6-asym)+.04*power*activity)
            resting=.012+.09*sad+.018*power*activity
            w['eyeBlink'+side]=resting+(1-resting)*b
            w['eyeWide'+side]=min(.40,w['eyeWide'+side]+.23*amaze)*(1-b)
            w['eyeSquint'+side]=min(.34,w['eyeSquint'+side]+.15*joy+.06*power*activity+.055*sad)*(1-b)
        assert f['extensions']==original['extensions']
        assert all(w[n]==original['blendshapes'][n] for n in w if n not in changed)
        validate_frame(f)
        stream.write(json.dumps(f,separators=(',',':'),allow_nan=False)+'\n')
# Art-directed preview-only motion, tied to phrases; no mechanical coordinate claim.
# Values are renderer Euler pitch, roll, yaw, followed by local eye yaw, pitch.
anchors=[(0,.3,0,0,0,0),(2.3,-.3,-.2,.4,0,-.5),(3.7,-1.2,.4,0,0,-1),
 (5.7,.5,.7,-1,-2,1),(7.2,1.1,.2,-1.3,-3,2),(9.6,-.5,-1.1,.6,1,0),
 (12,1,.2,.4,1,1),(16,.5,.6,-.6,-1,1),(20,-.3,-.3,0,0,0),
 (23,-.6,-.9,.8,0,0),(28,.2,.8,-.5,1,.3),(32.5,.5,1,-.8,-1,1),
 (37.8,1.1,.5,-1.3,-2,1.8),(40.8,-.6,-1.5,1.1,1.8,.2),
 (44.6,.3,-.4,.5,0,.5),(48.8,-.4,.8,-1.4,-2.8,-.3),
 (51.5,-.6,.4,-.4,0,0),(54.8,-1.1,-.5,0,0,-.4),
 (57.1,.3,-.1,.5,1.2,1),(58.8,1.3,.6,1.4,3.4,3.5),
 (61.1,2.2,.8,1.7,2.5,4),(63.6,.4,.4,.5,0,.2),
 (66,-.7,-.8,0,0,-.3),(68.8,.3,-.3,-.4,0,.5),
 (71.2,-.5,.8,-.7,-.5,0),(74,.4,.3,.1,0,.5),
 (76.2,.8,-.7,.6,.5,1),(79.2,1.2,-.5,.2,0,1.5),
 (82.7,1.8,.8,-.4,-.5,2.8),(85.8,1.5,.5,-.9,-2,2),
 (90.5,1.1,-.3,-1.5,-3,1.5),(96,.7,.4,-.8,-1,1),
 (101,.3,-.4,.2,0,.4),(105.5,-.2,.2,0,0,0),
 (109,-.4,-.7,.5,0,-.3),(duration,.4,.3,.1,0,.5)]
a=np.array(anchors);dense=np.unique(np.r_[np.arange(0,duration,.1),duration])
values=PchipInterpolator(a[:,0],a[:,1:],axis=0)(dense)
pose={'source':'song-phrase art direction of preview bust only; not measured head motion',
      'keyframes':[dict(time_seconds=float(t),**dict(zip(['pitch_deg','roll_deg','yaw_deg'],map(float,v[:3])))) for t,v in zip(dense,values)]}
gaze={'source':'song-phrase art direction of preview eyes only; not inferred gaze or servo commands',
      'keyframes':[dict(time_seconds=float(t),yaw_deg=float(v[3]),pitch_deg=float(v[4])) for t,v in zip(dense,values)]}
report={'frames':len(frames),'events':events,'frame_schema_validation':'passed',
 'native_mouth_jaw_tongue_teeth_unchanged':True,'unmodified_channels':[n for n in frames[0]['blendshapes'] if n not in changed],
 'method':'fresh song narrative plus measured vocal phrase valleys; uncertainty retained in direction_plan.json',
 'visual_acceptance':'pending user review'}
for name,data in [('blink_validation',report),('performance_pose',pose),('performance_gaze',gaze)]:
    (out/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
meta.update(audio=str(root/'0001.wav'),video_playback_gain_multiplier=1,
 interface_source=str(path),blink_schedule=report,performance_pose=pose,performance_gaze=gaze,
 preview_times_seconds=[3.7,9.6,40.8,54.9,60.7,61.23,66,79.2,82.5,83.1,94])
(out/'performance_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'frames':len(frames),'events':len(events),'output':str(out)}))
