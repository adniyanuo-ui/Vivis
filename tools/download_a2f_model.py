import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent.parent
folder=root/'assets/models/mark-v2.3';folder.mkdir(parents=True,exist_ok=True)
for info in json.loads((root/'tools/a2f-model-files.json').read_text()):
    name=info['path']
    if name=='.gitattributes':continue
    dest=folder/name
    if not dest.exists() or dest.stat().st_size!=info['size']:
        print('Downloading/resuming',name,flush=True)
        subprocess.run(['curl','--http1.1','-fLsS','--retry','5','--retry-all-errors','-C','-', 'https://huggingface.co/nvidia/Audio2Face-3D-v2.3-Mark/resolve/main/'+name,'-o',str(dest)],check=True)
    assert dest.stat().st_size==info['size'],name
    if info.get('lfs',{}).get('oid'):
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==info['lfs']['oid'],name
print('Model download complete; sizes and LFS hashes verified',flush=True)
