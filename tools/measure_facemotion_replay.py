"""Measure animation reconstruction loss separately from protocol validity."""
import json,sys
from pathlib import Path
import numpy as np
source=Path(sys.argv[1]);replay=Path(sys.argv[2]);model=Path(sys.argv[3]);info=json.loads((source/'export_info.json').read_text())
f=info['frames'];v=info['skin_vertices'];a=np.memmap(source/'skin.f32',dtype='float32',mode='r',shape=(f,v,3));b=np.memmap(replay/'skin.f32',dtype='float32',mode='r',shape=(f,v,3))
file='bs_skin_Claire.npz' if (model/'bs_skin_Claire.npz').exists() else 'bs_skin.npz'
with np.load(model/file) as rig:
 neutral=rig['neutral'];mask=np.linalg.norm(rig['mouthPucker'],axis=1)>0.03
counts={'whole_face':0,'mouth_pucker_region':0};sums={k:0.0 for k in counts};maxima={k:0.0 for k in counts}
for i in range(0,f,64):
 distances=np.linalg.norm(a[i:i+64]-b[i:i+64],axis=2)
 for name,d in [('whole_face',distances),('mouth_pucker_region',distances[:,mask])]:
  sums[name]+=float(np.sum(d.astype(np.float64)**2));counts[name]+=d.size;maxima[name]=max(maxima[name],float(d.max()))
report={'meaning':'Geometry-to-blendshape approximation loss in the digital rig, not measured robot error','coordinate_units':'Maya centimeter (model units)','core_input':'52 named weights read back from JSONL','preview_extensions':'tongue blendshape weights and lower-teeth transform, read from the same JSONL','face_width_model_units':float(np.ptp(neutral[:,0])),'errors':{name:{'vertex_position_rms_model_units':(sums[name]/counts[name])**0.5,'max_vertex_distance_model_units':maxima[name]} for name in counts},'visual_acceptance':'pending user review'}
(replay/'reconstruction_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
