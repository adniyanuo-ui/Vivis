"""Compose irregular, content-aware eyelid gestures using estimated speech timings."""
import json,math,wave
from pathlib import Path
import numpy as np
from facemotion52 import load_stream,validate_frame
root=Path(__file__).resolve().parent.parent
folder=root/'outputs/wz_Welcome_claire_v3.0'
frames=load_stream(folder/'facemotion52.jsonl');alignment=json.loads((root/'outputs/wz_content_alignment.json').read_text());meta=json.loads((folder/'metadata.json').read_text())
rng=np.random.default_rng(20261012)
with wave.open(meta['audio']) as wav:
 rate=wav.getframerate();samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
step=int(rate/50);rms=np.sqrt((samples[:len(samples)//step*step].reshape(-1,step)**2).mean(axis=1))
phrases=alignment['content_phrases'];protected=alignment['protected_keywords'];events=[]
def allowed(event):
 start=event['peak_seconds']-event['closing_seconds'];end=event['peak_seconds']+event['closed_seconds']+event['opening_seconds']
 if not 0.3<start<end<meta['duration_seconds']-0.2:return False
 if any(start<k['end']+0.08 and end>k['start']-0.08 for k in protected):return False
 if any(abs(event['peak_seconds']-e['peak_seconds'])<0.3 for e in events):return False
 return True
def add_between(a,b,reason,short=False):
 close=float(rng.uniform(0.065,0.105) if not short else rng.uniform(0.06,0.075))
 hold=float(rng.uniform(0.025,0.04));opening=float(rng.uniform(0.13,0.2) if not short else rng.uniform(0.115,0.145))
 options=[]
 for t in np.arange(math.ceil(a*30),math.floor(b*30)+1)/30:
  event={'peak_seconds':float(t),'closing_seconds':close,'closed_seconds':hold,'opening_seconds':opening,'reason':reason}
  if allowed(event):options.append(event)
 if not options:raise RuntimeError('No content-safe blink slot: '+reason)
 scores=np.array([1/math.sqrt(float(rms[min(round(e['peak_seconds']*50),len(rms)-1)])+0.02) for e in options]);scores/=scores.sum()
 event=options[int(rng.choice(len(options),p=scores))];events.append(event);return event
# Select only some discourse boundaries, not every sentence ending.
add_between(phrases[0]['end']+0.18,phrases[0]['end']+0.28,'company identity complete; brief greeting reset')
security=next(k for k in protected if k['text']=='仿生安全');embodied=next(k for k in protected if k['text']=='具身智能')
add_between(security['end']+0.2,embodied['start']-0.28,'light autonomous blink between the two focus areas')
first=add_between(phrases[2]['end']+0.14,phrases[2]['end']+0.25,'service positioning complete; transition to core capabilities')
add_between(first['peak_seconds']+0.4,first['peak_seconds']+0.5,'one brief second blink during the same discourse pause',short=True)
span=phrases[4]['end']-phrases[4]['start']
add_between(phrases[4]['start']+0.15*span,phrases[4]['start']+0.6*span,'autonomous blink during the lower-emphasis transition to industry sectors')
add_between(phrases[-1]['end']+0.2,phrases[-1]['end']+0.35,'solution statement complete; relaxed closing eye contact')
events.sort(key=lambda e:e['peak_seconds'])
def blink(t):
 value=0.0
 for e in events:
  p=e['peak_seconds'];c=e['closing_seconds'];h=e['closed_seconds'];o=e['opening_seconds']
  if p-c<=t<p:value=max(value,0.5-0.5*math.cos(math.pi*(t-p+c)/c))
  elif p<=t<=p+h:value=1.0
  elif p+h<t<p+h+o:value=max(value,0.5+0.5*math.cos(math.pi*(t-p-h)/o))
 return value
channels={'eyeBlinkLeft','eyeBlinkRight','eyeWideLeft','eyeWideRight','eyeSquintLeft','eyeSquintRight'}
modified=[]
for frame in frames:
 f=json.loads(json.dumps(frame));b=blink(f['timestamp_ms']/1000)
 for side in ['Left','Right']:
  f['blendshapes']['eyeBlink'+side]=b
  for prefix in ['eyeWide','eyeSquint']:f['blendshapes'][prefix+side]*=1-b
 assert all(f['blendshapes'][n]==frame['blendshapes'][n] for n in frame['blendshapes'] if n not in channels)
 assert f['extensions']==frame['extensions'];validate_frame(f);modified.append(f)
intervals=np.diff([e['peak_seconds'] for e in events])
assert intervals.min()<0.6 and intervals.max()>5.0 and float(intervals.std()/intervals.mean())>0.45
for e in events:
 f=min(modified,key=lambda f:abs(f['timestamp_ms']/1000-e['peak_seconds']));assert f['blendshapes']['eyeBlinkLeft']>0.99
out=root/'outputs/wz_WelcomeNatural_ContentBlink_FaceMotion52_v3.0';out.mkdir(exist_ok=True)
with (out/'facemotion52_content_blinks.jsonl').open('w') as stream:
 for f in modified:stream.write(json.dumps(f,separators=(',',':'),allow_nan=False)+'\n')
report={'seed':20261012,'method':'estimated text-aligned discourse cues plus limited spontaneous blinking; not a periodic timer','events':events,'intervals_seconds':intervals.tolist(),'interval_coefficient_of_variation':float(intervals.std()/intervals.mean()),'protected_keywords':protected,'protected_keywords_have_no_overlapping_blink':True,'same_audio_and_welcome_expression':True,'all_non_eyelid_channels_and_extensions_unchanged':True,'one_double_blink':True,'visual_acceptance':'pending user review'}
(out/'blink_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
