import concurrent.futures,json,hashlib,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent.parent
packages=json.loads((root/'tools/tensorrt-packages.json').read_text())
prefix=root/'tools/tensorrt-debs';prefix.mkdir(exist_ok=True)
(root/'tools/debs').mkdir(exist_ok=True)
for name in ['libnvinfer10','libnvinfer-plugin10','libnvonnxparsers10','libnvinfer-headers-dev','libnvinfer-headers-plugin-dev','python3-libnvinfer']:
    info=packages[name]
    dest=root/'tools/debs'/Path(info['Filename']).name
    size=int(info['Size']);count=8 if size>100000000 else 1
    parts=[]
    url='https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/'+info['Filename'].removeprefix('./')
    for i in range(count):parts.append((Path(str(dest)+f'.part{i}'),size*i//count,size*(i+1)//count-1))
    print('Downloading',name,flush=True)
    def fetch(p):
        path,a,b=p
        for attempt in range(6):
            present=path.stat().st_size if path.exists() else 0
            if present==b-a+1:return
            assert present<b-a+1
            piece=Path(str(path)+'.tail')
            result=subprocess.run(['curl','--http1.1','-fLsS','--range',f'{a+present}-{b}',url,'-o',str(piece)])
            if piece.exists():
                with path.open('ab') as out, piece.open('rb') as inp:
                    while chunk:=inp.read(8<<20):out.write(chunk)
                piece.unlink()
        assert path.stat().st_size==b-a+1
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:list(pool.map(fetch,parts))
    sha=hashlib.sha256()
    with dest.open('wb') as output:
        for part,_,_ in parts:
            with part.open('rb') as source:
                while data:=source.read(8<<20):output.write(data);sha.update(data)
            part.unlink()
    assert sha.hexdigest()==info['SHA256']
    subprocess.run(['dpkg-deb','-x',str(dest),str(prefix)],check=True)
    print('Installed and SHA256 verified:',name,flush=True)
print('TensorRT local package extraction complete',flush=True)
