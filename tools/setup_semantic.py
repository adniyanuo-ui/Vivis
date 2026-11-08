#!/usr/bin/env python3
"""Prepare the optional local semantic server; no audio/camera/robot access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as file:
        while chunk:=file.read(4*1024*1024):value.update(chunk)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--jobs',type=int,default=6);args=parser.parse_args()
    if not 1<=args.jobs<=16:parser.error('jobs must be 1..16')
    config=json.loads((ROOT/'configs/interaction/semantic.json').read_text())['runtime']
    sources=json.loads((ROOT/config['source_manifest']).read_text())
    directory=ROOT/config['directory'];directory.mkdir(parents=True,exist_ok=True)
    model=sources['model'];path=directory/model['file']
    if not (path.exists() and path.stat().st_size==model['size'] and digest(path)==model['sha256']):
        temporary=path.with_suffix('.gguf.tmp');offset=temporary.stat().st_size if temporary.exists() else 0
        if offset>model['size']:raise RuntimeError('Oversize partial model; inspect '+str(temporary))
        url=f'https://huggingface.co/{model["repository"]}/resolve/{model["revision"]}/{model["file"]}'
        request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
        if offset!=model['size']:
            with urllib.request.urlopen(request,timeout=90) as response:
                mode='ab' if offset and response.status==206 else 'wb'
                with temporary.open(mode) as file:
                    while chunk:=response.read(4*1024*1024):file.write(chunk)
        if temporary.stat().st_size!=model['size'] or digest(temporary)!=model['sha256']:
            raise RuntimeError('Model integrity check failed: '+str(temporary))
        temporary.replace(path)
    print('Verified Qwen model SHA256.',flush=True)
    engine=sources['engine'];source=directory/'llama.cpp'
    if not source.exists():
        subprocess.run(['git','clone','--depth','1','--branch',engine['tag'],engine['repository'],str(source)],check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if revision!=engine['revision']:raise RuntimeError('Unexpected llama.cpp revision; preserve and inspect '+str(source))
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=source,text=True).strip():
        raise RuntimeError('llama.cpp source has local edits; inspect before building')
    compiler=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    cuda=not args.cpu and Path(compiler).is_file()
    command=['cmake','-S',str(source),'-B',str(directory/'build'),'-DCMAKE_BUILD_TYPE=Release',
             '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_CURL=OFF','-DGGML_CUDA='+('ON' if cuda else 'OFF')]
    if cuda:command+=['-DCMAKE_CUDA_COMPILER='+compiler,'-DCMAKE_CUDA_ARCHITECTURES=87']
    subprocess.run(command,check=True)
    subprocess.run(['cmake','--build',str(directory/'build'),'--target','llama-server','-j',str(args.jobs)],check=True)
    print('Semantic server prepared; start with .venv/bin/python -m vivis.interaction.semantic --serve')


if __name__=='__main__':main()
