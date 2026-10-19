"""Arbitrate eyelids in FaceMotion-52 while preserving every speech channel."""
import argparse,json,math,wave
from pathlib import Path
import numpy as np
from facemotion52 import load_stream,validate_frame
parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args();folder=Path(args.folder)
frames=load_stream(folder/'facemotion52.jsonl');meta=json.loads((folder/'metadata.json').read_text())
with wave.open(meta['audio']) as audio:
 samples=np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2').astype(np.float32)/32768;rate=audio.getframerate()
# Prefer nearby low-energy moments, without exact periodic intervals.
window=round(rate*0.02);centers=np.arange(len(samples)//window)*0.02+0.01
rms=np.sqrt((samples[:len(centers)*window].reshape(-1,window)**2).mean(axis=1))
peaks=[]
for target in [3.2,7.5,11.8,16.3,20.8,24.7]:
 indices=np.flatnonzero(abs(centers-target)<=0.35)
 score=rms[indices]+0.01*abs(centers[indices]-target)
 peaks.append(round(float(centers[indices[np.argmin(score)]])*30)/30)
events=[{'peak_seconds':p,'closing_seconds':c,'closed_seconds':h,'opening_seconds':o} for p,c,h,o in zip(peaks,[0.08,0.075,0.085,0.08,0.075,0.085],[0.035,0.035,0.035,0.035,0.035,0.035],[0.155,0.17,0.16,0.15,0.165,0.155])]
def blink(t):
 weight=0.0
 for event in events:
  p=event['peak_seconds'];c=event['closing_seconds'];h=event['closed_seconds'];o=event['opening_seconds']
  if p-c<=t<p:weight=max(weight,0.5-0.5*math.cos(math.pi*(t-(p-c))/c))
  elif p<=t<=p+h:weight=1.0
  elif p+h<t<p+h+o:weight=max(weight,0.5+0.5*math.cos(math.pi*(t-p-h)/o))
 return weight
modified=[];eyelid_channels={'eyeBlinkLeft','eyeBlinkRight','eyeWideLeft','eyeWideRight','eyeSquintLeft','eyeSquintRight'}
for frame in frames:
 f=json.loads(json.dumps(frame));b=blink(f['timestamp_ms']/1000)
 f['blendshapes']['eyeBlinkLeft']=b;f['blendshapes']['eyeBlinkRight']=b
 for n in ['eyeWideLeft','eyeWideRight','eyeSquintLeft','eyeSquintRight']:f['blendshapes'][n]*=1-b
 assert all(f['blendshapes'][n]==frame['blendshapes'][n] for n in frame['blendshapes'] if n not in eyelid_channels)
 assert f['extensions']==frame['extensions']
 validate_frame(f);modified.append(f)
for event in events:
 near=[f['blendshapes']['eyeBlinkLeft'] for f in modified if abs(f['timestamp_ms']/1000-event['peak_seconds'])<0.035]
 assert max(near)>0.99,'Blink did not fully close on the sampled timeline'
assert len(events)==6 and len(set(round(events[i+1]['peak_seconds']-events[i]['peak_seconds'],2) for i in range(5)))>3
output=folder/'facemotion52_welcome_blinks.jsonl'
with output.open('w') as stream:
 for f in modified:stream.write(json.dumps(f,separators=(',',':'),allow_nan=False)+'\n')
report={'blink_count':6,'events':events,'controller':'single scheduled eyelid source; override native blink weights, suppress eye-wide and squint conflicts during closure','same_left_right_closure':True,'speech_and_emotion_channels_unchanged':True,'tongue_and_lower_jaw_extensions_unchanged':True,'timeline_unchanged':True,'blink_sampling_peak_greater_than_0_99':True,'visual_acceptance':'pending user review'}
(folder/'blink_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
