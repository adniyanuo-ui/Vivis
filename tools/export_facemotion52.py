"""Adapt native A2F solver output into validated FaceMotion-52 JSON Lines."""
import csv,json,sys
from pathlib import Path
import numpy as np
from facemotion52 import CHANNELS,VERSION,SCHEMA,map_weights,validate_frame,load_stream
root=Path(__file__).resolve().parent.parent
folder=Path(sys.argv[1]).resolve();info=json.loads((folder/'export_info.json').read_text());bindings=json.loads((folder/'source_bindings.json').read_text())
count=info['frames'];skin=np.fromfile(folder/'weights_skin.f32',dtype=np.float32).reshape(count,bindings['skin']['count']);tongue=np.fromfile(folder/'weights_tongue.f32',dtype=np.float32).reshape(count,bindings['tongue']['count']);jaw=np.fromfile(folder/'jaw.f32',dtype=np.float32).reshape(count,16)
stamps=np.loadtxt(folder/'timestamps.csv',delimiter=',',dtype=np.int64)
assert len(stamps)==count and np.array_equal(stamps[:,0],np.arange(count))
assert np.isfinite(tongue).all() and np.min(tongue)>=-1e-6 and np.max(tongue)<=1+1e-6
names=bindings['skin']['names'];out=folder/'facemotion52.jsonl'
with out.open('w') as stream:
 for i in range(count):
  frame={'schema_version':VERSION,'timestamp_ms':float(stamps[i,1])/16,'blendshapes':map_weights(names,skin[i]),'gaze':None,'head_pose':None,'extensions':{'claire_preview':{'tongue_blendshapes':dict(zip(bindings['tongue']['names'],map(float,tongue[i]))),'lower_teeth_transform':{'matrix_column_major':jaw[i].tolist(),'units':'centimeter','coordinate_frame':'maya_y_up'}}}}
  validate_frame(frame);stream.write(json.dumps(frame,separators=(',',':'),allow_nan=False)+'\n')
frames=load_stream(out)
(root/'FACEMOTION_52_SCHEMA_V1.0.json').write_text(json.dumps(SCHEMA,indent=2)+'\n')
with (folder/'channel_mapping.csv').open('w') as stream:
 writer=csv.writer(stream);writer.writerow(['channel','source_index','source_active','min_weight','max_weight'])
 for channel in CHANNELS:
  i=names.index(channel);writer.writerow([channel,i,bindings['skin']['active'][i],float(skin[:,i].min()),float(skin[:,i].max())])
report={'schema_version':VERSION,'frames':len(frames),'channels_per_frame':52,'source_names_unique':True,'mapping':'matched by exact semantic name; neutral excluded','weight_range':[float(skin.min()),float(skin.max())],'timestamp_units':'milliseconds relative to audio sample zero','inactive_source_channels':[n for n,a in zip(names,bindings['skin']['active']) if not a],'gaze':None,'head_pose':None,'preview_extensions':['16 tongue blendshape weights','lower teeth rigid transform in model coordinates'],'core_channels_do_not_sum_to_one':True}
(folder/'facemotion_validation.json').write_text(json.dumps(report,indent=2)+'\n')
(folder/'facemotion52_example_frame.json').write_text(json.dumps(frames[100],indent=2)+'\n')
print(json.dumps(report,indent=2))
