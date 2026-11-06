#!/usr/bin/env python3
"""Prepare optional P0 models/builds without changing system packages or moving servos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def git_source(entry, destination):
    if not destination.exists():
        subprocess.run(['git','clone',entry['url'],str(destination)],check=True)
        subprocess.run(['git','-C',str(destination),'checkout','--detach',entry['commit']],check=True)
    actual=subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip()
    if actual!=entry['commit'] or subprocess.check_output(['git','-C',str(destination),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Dependency checkout differs from lock: '+str(destination))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-source',type=Path,required=True,help='Original A2F-v3.0 model directory')
    parser.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    parser.add_argument('--skip-vision',action='store_true')
    args=parser.parse_args(argv)
    # Jetson shells often source ROS. Do not turn those environment packages
    # into P0 dependencies or import them into model preparation.
    os.environ.pop('PYTHONPATH',None)
    os.environ['PYTHONNOUSERSITE']='1'
    folder=ROOT/'outputs/a2f';folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'native/a2f/model_manifest.json').read_text())
    destination=folder/'model_bundle';destination.mkdir(exist_ok=True)
    for entry in manifest['files']:
        if entry['path']=='.gitattributes':continue
        source=args.model_source/entry['path'];data=source.read_bytes()
        digest=(hashlib.sha256(data).hexdigest() if 'lfs' in entry else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest())
        expected=entry['lfs']['oid'] if 'lfs' in entry else entry['oid']
        if digest!=expected or len(data)!=entry['size']:raise RuntimeError('Model mismatch: '+str(source))
        target=destination/source.name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(data).digest():
                raise RuntimeError('Existing model changed: '+str(target))
        else:
            try:os.link(source,target)
            except OSError:shutil.copyfile(source,target)
    lock=json.loads((ROOT/'native/a2f/sources.json').read_text())
    git_source(lock['sdk'],folder/'Audio2Face-3D-SDK')
    for entry in lock['dependencies']:git_source(entry,folder/'deps'/entry['name'])
    subprocess.run(['cmake','-S',str(ROOT/'native/a2f'),'-B',str(folder/'build'),
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=87',
        '-DCMAKE_CUDA_COMPILER='+str(args.cuda/'bin/nvcc')],check=True)
    subprocess.run(['cmake','--build',str(folder/'build'),'--target','a2f_chunk_probe','--parallel','2'],check=True)
    subprocess.run([sys.executable,str(ROOT/'native/a2f/build_model.py'),str(destination)],check=True)
    if not args.skip_vision:
        venv=folder/'vision_venv'
        subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
        subprocess.run([str(venv/'bin/python'),'-m','pip','install','-r',str(ROOT/'native/a2f/vision_requirements.txt')],check=True)
        model=folder/'face_landmarker.task'
        config=json.loads((ROOT/'configs/interaction/p0.json').read_text())['social_smile']
        if not model.exists():
            url='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            data=urllib.request.urlopen(url,timeout=60).read()
            if hashlib.sha256(data).hexdigest()!=config['sha256']:raise RuntimeError('Landmarker model mismatch')
            model.write_bytes(data)
        if hashlib.sha256(model.read_bytes()).hexdigest()!=config['sha256']:raise RuntimeError('Landmarker model changed')
    print('P0 dependencies prepared; no servo, camera or audio device opened.')


if __name__=='__main__':main()
