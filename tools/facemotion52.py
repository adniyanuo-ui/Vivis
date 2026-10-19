"""FaceMotion-52 v1: named ARKit-compatible weights, audio-relative milliseconds.
Left/Right refer to the subject's anatomy. Null gaze/head_pose mean unavailable.
Preview-only rig data belongs to extensions and must not be used as servo commands.
"""
import json,math
from pathlib import Path
from jsonschema import Draft202012Validator
CHANNELS=tuple('''eyeBlinkLeft eyeLookDownLeft eyeLookInLeft eyeLookOutLeft eyeLookUpLeft eyeSquintLeft eyeWideLeft eyeBlinkRight eyeLookDownRight eyeLookInRight eyeLookOutRight eyeLookUpRight eyeSquintRight eyeWideRight jawForward jawLeft jawRight jawOpen mouthClose mouthFunnel mouthPucker mouthLeft mouthRight mouthSmileLeft mouthSmileRight mouthFrownLeft mouthFrownRight mouthDimpleLeft mouthDimpleRight mouthStretchLeft mouthStretchRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper mouthPressLeft mouthPressRight mouthLowerDownLeft mouthLowerDownRight mouthUpperUpLeft mouthUpperUpRight browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight cheekPuff cheekSquintLeft cheekSquintRight noseSneerLeft noseSneerRight tongueOut'''.split())
VERSION='1.0.0'
NUMBER={'type':'number','minimum':0,'maximum':1}
SCHEMA={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'FaceMotion-52 frame v1.0.0','description':'52 named movement weights. Left/Right are anatomical. Weights are independent and do not sum to one. Null gaze/head_pose mean unavailable.','type':'object','additionalProperties':False,'required':['schema_version','timestamp_ms','blendshapes','gaze','head_pose'],'properties':{
 'schema_version':{'const':VERSION},
 'timestamp_ms':{'type':'number','minimum':0,'description':'Target display time relative to audio sample zero, in milliseconds.'},
 'blendshapes':{'type':'object','additionalProperties':False,'required':list(CHANNELS),'properties':{name:dict(NUMBER) for name in CHANNELS}},
 'gaze':{'anyOf':[{'type':'null'},{'type':'object','additionalProperties':False,'required':['direction'],'properties':{'direction':{'type':'array','items':{'type':'number','minimum':-1,'maximum':1},'minItems':3,'maxItems':3}}}],'description':'Optional unit direction in face-local X=subject left, Y=up, Z=forward; distinct from eye-look weights.'},
 'head_pose':{'anyOf':[{'type':'null'},{'type':'object','additionalProperties':False,'required':['yaw_deg','pitch_deg','roll_deg'],'properties':{name:{'type':'number','minimum':-180,'maximum':180} for name in ['yaw_deg','pitch_deg','roll_deg']}}],'description':'Optional right-handed intrinsic yaw(Y), pitch(X), roll(Z), degrees in the same face-local frame; mechanical limits are external.'},
 'extensions':{'type':'object','description':'Optional producer-specific data; core-only consumers may ignore it.'}}}
VALIDATOR=Draft202012Validator(SCHEMA)
def _finite(value):
 if isinstance(value,float) and not math.isfinite(value):raise ValueError('Non-finite number')
 if isinstance(value,dict):
  for item in value.values():_finite(item)
 elif isinstance(value,list):
  for item in value:_finite(item)
def validate_frame(frame):
 _finite(frame);VALIDATOR.validate(frame)
 if frame['gaze'] is not None:
  norm=math.sqrt(sum(v*v for v in frame['gaze']['direction']))
  if abs(norm-1)>1e-5:raise ValueError('Gaze direction must be a unit vector')
 return frame
def map_weights(names,values):
 if len(names)!=len(set(names)):raise ValueError('Duplicate source channel names')
 if set(names)!=set(CHANNELS):raise ValueError('Source channel set does not match FaceMotion-52')
 if len(values)!=len(names):raise ValueError('Wrong weight vector length')
 lookup=dict(zip(names,map(float,values)))
 if any(not math.isfinite(v) or not -1e-6<=v<=1+1e-6 for v in lookup.values()):raise ValueError('Invalid source weight')
 return {name:min(1.0,max(0.0,lookup[name])) for name in CHANNELS}
def _pairs(pairs):
 out={}
 for name,value in pairs:
  if name in out:raise ValueError('Duplicate JSON key: '+name)
  out[name]=value
 return out
def load_stream(path):
 frames=[];previous=-1
 with Path(path).open() as stream:
  for line in stream:
   frame=validate_frame(json.loads(line,object_pairs_hook=_pairs))
   if frame['timestamp_ms']<=previous:raise ValueError('Timestamps must increase strictly')
   previous=frame['timestamp_ms'];frames.append(frame)
 if not frames:raise ValueError('Empty stream')
 return frames
