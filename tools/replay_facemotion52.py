"""Standalone FaceMotion-52 consumer: rebuild preview geometry from named weights.
Inputs are the interface stream, static rig bindings, static mesh, and audio path.
This consumer never reads the producer's raw animated geometry or jaw files.
"""
import argparse,json,shutil,wave
from pathlib import Path
import numpy as np
from facemotion52 import CHANNELS,load_stream
parser=argparse.ArgumentParser();parser.add_argument('stream');parser.add_argument('model');parser.add_argument('mesh');parser.add_argument('output');parser.add_argument('--audio',required=True);args=parser.parse_args()
frames=load_stream(args.stream);model=Path(args.model);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
suffix='_Claire' if (model/'bs_skin_Claire.npz').exists() else ''
shape_counts={}
for part in ['skin','tongue']:
 with np.load(model/('bs_'+part+suffix+'.npz')) as basis:
  names=[x.decode() if isinstance(x,bytes) else str(x) for x in basis['poseNames']][1:]
  neutral=basis['neutral'];vertices=len(neutral);shape_counts[part]=vertices
  deltas=np.stack([basis[n].reshape(-1) for n in names])
  weights=[]
  for frame in frames:
   values=frame['blendshapes'] if part=='skin' else frame.get('extensions',{}).get('claire_preview',{}).get('tongue_blendshapes',{n:0 for n in names})
   if set(values)!=set(names):raise ValueError('Rig channel names do not match interface: '+part)
   weights.append([values[n] for n in names])
  weights=np.asarray(weights,dtype=np.float32)
  result=np.memmap(out/(part+'.f32'),mode='w+',dtype=np.float32,shape=(len(frames),vertices,3))
  for start in range(0,len(frames),64):
   batch=weights[start:start+64]@deltas+neutral.reshape(-1)
   assert np.isfinite(batch).all();result[start:start+len(batch)]=batch.reshape(-1,vertices,3)
  result.flush();del result
jaw=np.asarray([frame.get('extensions',{}).get('claire_preview',{}).get('lower_teeth_transform',{}).get('matrix_column_major',np.eye(4).flatten(order='F').tolist()) for frame in frames],dtype=np.float32)
assert jaw.shape==(len(frames),16)
jaw.tofile(out/'jaw.f32')
with (out/'timestamps.csv').open('w') as stream:
 for i,f in enumerate(frames):stream.write(f'{i},{round(f["timestamp_ms"]*16)}\n')
shutil.copyfile(args.mesh,out/'mesh.json')
(out/'export_info.json').write_text(json.dumps({'frames':len(frames),'skin_vertices':shape_counts['skin'],'tongue_vertices':shape_counts['tongue'],'fps_numerator':60,'fps_denominator':1,'data_source':'FaceMotion-52 only'},indent=2)+'\n')
with wave.open(args.audio) as wav:duration=wav.getnframes()/wav.getframerate()
(out/'metadata.json').write_text(json.dumps({'audio':str(Path(args.audio).resolve()),'duration_seconds':duration,'interface_source':str(Path(args.stream).resolve()),'data_source':'FaceMotion-52 named weights and optional preview rig extensions','video_playback_gain_multiplier':1 if 'english' in args.audio else 4},indent=2)+'\n')
print('FACE_MOTION_REPLAY_READY',out,len(frames),shape_counts)
