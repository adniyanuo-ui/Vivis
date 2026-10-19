"""Export preview-authored gaze/pose and FaceMotion weights to the example array schema.

The example defines structure, not rotation semantics. This exporter explicitly
chooses radians, face-local X=subject left, Y=up, Z=forward, intrinsic Y-X-Z.
Do not assume that a receiver uses this convention without its specification.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from facemotion52 import CHANNELS, load_stream

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('folder',type=Path)
parser.add_argument('reference_csv',type=Path)
parser.add_argument('example_json',type=Path)
parser.add_argument('output_prefix',type=Path)
args=parser.parse_args()
meta=json.loads((args.folder/'metadata.json').read_text())
frames=load_stream(meta['interface_source'])
example=json.loads(args.example_json.read_text())
assert set(example)=={'blendShapes','fps','name'}
assert all(len(row)==61 for row in example['blendShapes'])
with args.reference_csv.open(encoding='utf-8-sig',newline='') as stream:
    header=next(csv.reader(stream))
assert header[:2]==['Timecode','BlendshapeCount']
order=header[2:]
assert len(order)==len(set(order))==61
canonical={name.casefold():name for name in CHANNELS}
assert {n.casefold() for n in order[:52]}==set(canonical)
assert order[52:]==['HeadYaw','HeadPitch','HeadRoll','LeftEyeYaw','LeftEyePitch','LeftEyeRoll','RightEyeYaw','RightEyePitch','RightEyeRoll']
weights=np.array([[f['blendshapes'][canonical[n.casefold()]] for n in order[:52]] for f in frames])
# Use exactly the audio-sample timestamps used by the accepted video renderer.
samples=np.loadtxt(args.folder/'timestamps.csv',delimiter=',')[:,1]
times=samples/16000
protocol_times=np.array([f['timestamp_ms']/1000 for f in frames])
assert len(times)==len(frames) and np.all(np.diff(times)>0) and times[0]==0
assert np.max(abs(times-protocol_times))<1/16000

def evaluate(field,axis_names):
    keys=meta[field]['keyframes'];kt=np.array([k['time_seconds'] for k in keys])
    assert np.all(np.diff(kt)>0)
    return np.column_stack([np.interp(times,kt,[k[n] for k in keys]) for n in axis_names])

pose=evaluate('performance_pose',['pitch_deg','roll_deg','yaw_deg'])
gaze=evaluate('performance_gaze',['yaw_deg','pitch_deg'])
# Blender's object Euler XYZ means Rz(yaw) @ Ry(roll) @ Rx(pitch).
blender_head=Rotation.from_euler('xyz',pose,degrees=True).as_matrix()
blender_eye=Rotation.from_euler('Z',gaze[:,0],degrees=True).as_matrix() @ Rotation.from_euler('X',gaze[:,1],degrees=True).as_matrix()
# Coordinate change only; these are orientations, not neck joint translations.
C=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
face_head=C @ blender_head @ C.T
face_eye=C @ blender_eye @ C.T
head_ypr=Rotation.from_matrix(face_head).as_euler('YXZ')
eye_ypr=Rotation.from_matrix(face_eye).as_euler('YXZ')
# Eliminate insignificant numerical residuals, never clamp signed rotations to [0,1].
head_ypr[abs(head_ypr)<1e-14]=0
eye_ypr[abs(eye_ypr)<1e-14]=0
rows=np.column_stack([weights,head_ypr,eye_ypr,eye_ypr])
assert rows.shape==(len(frames),61) and np.isfinite(rows).all()
assert np.min(rows[:,:52])>=0 and np.max(rows[:,:52])<=1
head_error=float(np.max(abs(Rotation.from_euler('YXZ',head_ypr).as_matrix()-face_head)))
eye_error=float(np.max(abs(Rotation.from_euler('YXZ',eye_ypr).as_matrix()-face_eye)))
assert max(head_error,eye_error)<1e-12
# Axis probes document and verify the chosen positive directions independently.
forward=np.array([0.,0,1])
assert (Rotation.from_euler('Y',.1).as_matrix()@forward)[0]>0
assert (Rotation.from_euler('X',.1).as_matrix()@forward)[1]<0
for selected in CHANNELS:
    probe={name:float(name==selected) for name in CHANNELS}
    mapped=[probe[canonical[n.casefold()]] for n in order[:52]]
    assert mapped.index(1.)==next(i for i,n in enumerate(order[:52]) if canonical[n.casefold()]==selected)

prefix=args.output_prefix;prefix.parent.mkdir(exist_ok=True,parents=True)
outputs=[]
for fps in [30,60]:
    target_times=np.arange(math.ceil(meta['duration_seconds']*fps))/fps
    indices=np.abs(times[:,None]-target_times[None,:]).argmin(axis=0)
    time_error=float(np.max(abs(times[indices]-target_times)))
    assert time_error<.001
    payload={'blendShapes':rows[indices].tolist(),'fps':{'den':1,'num':fps},'name':example['name']}
    path=prefix.with_name(prefix.name+f'_{fps}fps.json')
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    restored=json.loads(path.read_text());assert restored==payload
    out_rows=np.array(restored['blendShapes'])
    assert out_rows.shape==(len(target_times),61)
    assert np.array_equal(out_rows[:,:52],weights[indices])
    # Reconstruct original Blender eye/head rotations from the actual written fields.
    restored_head=C.T @ Rotation.from_euler('YXZ',out_rows[:,52:55]).as_matrix() @ C
    restored_eye=C.T @ Rotation.from_euler('YXZ',out_rows[:,55:58]).as_matrix() @ C
    h_error=float(np.max(abs(restored_head-blender_head[indices])))
    e_error=float(np.max(abs(restored_eye-blender_eye[indices])))
    assert max(h_error,e_error)<1e-12
    assert np.array_equal(out_rows[:,55:58],out_rows[:,58:61])
    outputs.append({'file':str(path.resolve()),'fps':fps,'frames':len(target_times),'channels':61,
      'duration_seconds':len(target_times)/fps,'max_sample_time_error_ms':time_error*1000,
      'face_weights_roundtrip_max_error':0,'head_rotation_matrix_roundtrip_max_error':h_error,
      'eye_rotation_matrix_roundtrip_max_error':e_error,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
rotation_convention={
 'status':'explicit export convention; example JSON and CSV do not specify receiver convention',
 'unit':'radians, signed; not degrees and not normalized weights',
 'basis':'right-handed face-local X=subject left (screen right), Y=up, Z=forward',
 'euler_order':'intrinsic Y-X-Z: R = Ry(yaw) @ Rx(pitch) @ Rz(roll)',
 'positive_yaw':'turn toward subject left / screen right',
 'positive_pitch':'look downward',
 'positive_roll':'right-handed around forward Z; subject left side rises',
 'head_reference':'relative to neutral preview bust orientation; pivot/translation are not represented',
 'eye_reference':'relative to head, with zero pointing along face-local +Z',
 'left_right_eye_policy':'same conjugate orientation as rendered; no independently authored vergence',
 'receiver_compatibility':'units, axis signs and Euler order require receiver specification for guaranteed matching playback'}
mapping=[]
for i,name in enumerate(order):
    mapping.append({'index_zero_based':i,'csv_channel':name,
      'source':canonical[name.casefold()] if i<52 else 'preview performance_pose' if i<55 else 'preview performance_gaze',
      'unit':'weight [0,1]' if i<52 else 'radians'})
report={'source_video':'0001_CLAIRE_SINGING_GAZE_v3.0-Claire.mp4',
 'source_folder':str(args.folder.resolve()),'source_stream':meta['interface_source'],
 'example_json':str(args.example_json.resolve()),'channel_order_csv':str(args.reference_csv.resolve()),
 'root_fields':list(example),'name':example['name'],'name_policy':'copied from example; receiver meaning unspecified',
 'audio':'0001.wav','audio_duration_seconds':meta['duration_seconds'],'audio_time_origin':'sample zero; no added offset',
 'resampling':'nearest source frame at i/fps; gaze and head evaluated at the same source timestamp used by render_variant.py',
 'face_weights':'all 52 named weights preserved exactly, including blink and downward gaze lid follow',
 'eyeLook_channels':'preserved at zero; eye rotation is carried by the six eye-angle fields, not duplicated as blendshape weights',
 'rotation_convention':rotation_convention,'channel_mapping':mapping,
 'all_52_one_hot_mapping_probes':'passed','rotation_roundtrip':'passed for every exported frame',
 'rotation_ranges_radians':{n:[float(rows[:,i].min()),float(rows[:,i].max())] for i,n in enumerate(order) if i>=52},
 'format_limitations':['Claire preview tongue/teeth rig extensions are absent from the example 61-channel schema.',
  'Head orientation is retained; original preview bust pivot and resulting translation are not part of this format.',
  'Receiver rig calibration and rotation convention are not supplied; no hardware playback validation is claimed.'],
 'outputs':outputs}
path=prefix.with_name(prefix.name+'_CONVERSION.json')
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'outputs':outputs,'report':str(path.resolve()),'rotation_convention':rotation_convention},ensure_ascii=False,indent=2))
