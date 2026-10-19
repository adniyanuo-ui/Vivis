import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent.parent
manifest=json.loads((root/'tools/cuda-redistrib.json').read_text())
prefix=root/'tools/cuda-12.9'
prefix.mkdir(exist_ok=True)
for name in ['cuda_nvcc','cuda_cudart','cuda_cccl','cuda_nvrtc','libcurand','cuda_nvtx']:
    info=manifest[name]['linux-x86_64']
    archive=root/'tools'/Path(info['relative_path']).name
    print('Downloading',name,flush=True)
    subprocess.run(['curl','-fL','--retry','3','https://developer.download.nvidia.com/compute/cuda/redist/'+info['relative_path'],'-o',str(archive)],check=True)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==info['sha256']
    subprocess.run(['tar','-xJf',str(archive),'--strip-components=1','-C',str(prefix)],check=True)
    archive.unlink()
if not (prefix/'lib64').exists():(prefix/'lib64').symlink_to('lib')
print('CUDA components installed',flush=True)
