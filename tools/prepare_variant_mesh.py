"""Extract neutral meshes and polygon topology from NVIDIA's Maya ASCII asset."""
import json,re,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent.parent
asset=Path(sys.argv[1]).resolve()
model_folder=Path(sys.argv[2]).resolve()
output_folder=Path(sys.argv[3]).resolve();output_folder.mkdir(parents=True,exist_ok=True)
text=asset.read_text()
parts={}
wanted={'c_headWatertight_hiShape':'skin','c_tongue_hiShape':'tongue','c_topDenture_hiShape':'upper_teeth','c_bottomDenture_hiShape':'lower_teeth','r_choroid_hiShape':'right_eye','l_choroid_hiShape':'left_eye'}
for block in re.split(r'(?=^createNode )',text,flags=re.M):
    match=re.match(r'createNode mesh -n "([^"]+)"',block)
    if not match or match[1] not in wanted:continue
    arrays={}
    for attr in ['vt','ed']:
        chunks=[]
        pattern=r'setAttr[^;]*?"\.'+attr+r'\[(\d+):(\d+)\]"\s+([^;]+);'
        for m in re.finditer(pattern,block):
            a,b=map(int,m.group(1,2))
            data=np.fromstring(m[3],sep=' ',dtype=np.float32 if attr=='vt' else np.int32).reshape(-1,3)
            assert len(data)==b-a+1
            chunks.append((a,b,data))
        assert chunks,(match[1],attr)
        array=np.empty((max(b for a,b,d in chunks)+1,3),dtype=chunks[0][2].dtype)
        covered=np.zeros(len(array),dtype=bool)
        for a,b,d in chunks:array[a:b+1]=d;covered[a:b+1]=True
        assert covered.all()
        arrays[attr]=array
    polygons=[]
    for m in re.finditer(r'\bf (\d+)\s+((?:-?\d+\s+)+)',block):
        count=int(m[1]);ids=list(map(int,m[2].split()))
        assert len(ids)==count
        face=[int(arrays['ed'][e,0] if e>=0 else arrays['ed'][-e-1,1]) for e in ids]
        assert min(face)>=0 and max(face)<len(arrays['vt'])
        polygons.append(face)
    assert polygons
    name=wanted[match[1]]
    parts[name]={'vertices':arrays['vt'].tolist(),'faces':polygons}
    print(name,len(arrays['vt']),len(polygons))
model=np.load(model_folder/('model_data_Claire.npz' if (model_folder/'model_data_Claire.npz').exists() else 'model_data.npz'))
for name in ['skin','tongue']:
    neutral=np.array(parts[name]['vertices'],dtype=np.float32)
    reference=model['neutral_'+name] if 'neutral_'+name in model else np.load(model_folder/('bs_'+name+'.npz'))['neutral']
    assert neutral.shape==reference.shape
    difference=float(np.max(np.abs(neutral-reference)))
    print(name,'neutral max coordinate difference:',difference)
    dist=np.linalg.norm(neutral-reference,axis=1)
    print(name,'mean/p95/max distance:',float(dist.mean()),float(np.percentile(dist,95)),float(dist.max()))
    assert np.percentile(dist,95)<0.5 and dist.max()<1.5,'Asset geometry does not match model closely enough'
    parts[name]['vertices']=reference.tolist()
(output_folder/'mesh.json').write_text(json.dumps(parts,separators=(',',':')))
