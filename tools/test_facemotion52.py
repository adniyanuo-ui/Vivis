"""Protocol behavior tests: semantic mapping, invalid inputs and timeline integrity."""
import copy,json,tempfile,unittest
from pathlib import Path
from facemotion52 import CHANNELS,map_weights,validate_frame,load_stream
ROOT=Path(__file__).resolve().parent.parent
def frame(t=0):return {'schema_version':'1.0.0','timestamp_ms':t,'blendshapes':{n:0.0 for n in CHANNELS},'gaze':None,'head_pose':None}
class ProtocolTests(unittest.TestCase):
 def test_all_channels_survive_permuted_source_order(self):
  names=list(reversed(CHANNELS));values=[(i+1)/53 for i in range(52)]
  result=map_weights(names,values)
  for n,v in zip(names,values):self.assertEqual(result[n],v)
 def test_one_channel_probes_keep_left_and_right_distinct(self):
  for name in CHANNELS:
   values=[1.0 if n==name else 0.0 for n in CHANNELS];result=map_weights(CHANNELS,values)
   self.assertEqual([n for n,v in result.items() if v], [name])
 def test_independent_weights_need_not_sum_to_one(self):
  f=frame();f['blendshapes']={n:0.5 for n in CHANNELS};validate_frame(f)
 def test_missing_extra_or_duplicate_source_channels_fail(self):
  for names in [CHANNELS[:-1],CHANNELS+('unknown',),CHANNELS[:-1]+(CHANNELS[0],)]:
   with self.assertRaises(ValueError):map_weights(names,[0.0]*len(names))
 def test_invalid_payloads_fail(self):
  cases=[]
  for value in [-0.1,1.1,float('nan'),float('inf')]:
   f=frame();f['blendshapes']['jawOpen']=value;cases.append(f)
  f=frame();del f['blendshapes']['jawOpen'];cases.append(f)
  f=frame();f['blendshapes']['unknown']=0;cases.append(f)
  f=frame();f['schema_version']='2.0.0';cases.append(f)
  f=frame();f['timestamp_ms']=-1;cases.append(f)
  f=frame();f['gaze']={'direction':[0,0,0]};cases.append(f)
  for f in cases:
   with self.assertRaises(Exception):validate_frame(f)
 def test_optional_motion_fields(self):
  f=frame();f['gaze']={'direction':[0,0,1]};f['head_pose']={'yaw_deg':0,'pitch_deg':0,'roll_deg':0};validate_frame(f)
 def test_out_of_order_timestamps_fail(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'stream.jsonl';p.write_text('\n'.join(json.dumps(frame(t)) for t in [0,33,16]))
   with self.assertRaises(ValueError):load_stream(p)
 def test_duplicate_json_keys_fail(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'stream.jsonl';p.write_text(json.dumps(frame()).replace('"schema_version": "1.0.0"','"schema_version": "1.0.0", "schema_version": "1.0.0"'))
   with self.assertRaises(ValueError):load_stream(p)
class BindingTests(unittest.TestCase):
 def test_real_rigs_anatomical_left_right(self):
  import numpy as np
  rigs=[('claire-v2.3.1','bs_skin.npz'),('a2f-v3.0','bs_skin_Claire.npz')]
  if any(not (ROOT/'assets/models'/model/file).is_file() for model,file in rigs):
   self.skipTest('Optional NVIDIA rig assets are not installed')
  for model,file in rigs:
   with np.load(ROOT/'assets/models'/model/file) as rig:
    x=rig['neutral'][:,0]
    for name in ['eyeBlinkLeft','eyeBlinkRight','mouthSmileLeft','mouthSmileRight']:
     energy=np.sum(rig[name]**2,axis=1)
     positive=float(energy[x>0].sum()/energy.sum())
     self.assertGreater(positive,0.95) if name.endswith('Left') else self.assertLess(positive,0.05)
if __name__=='__main__':unittest.main()
