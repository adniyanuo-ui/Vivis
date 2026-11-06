#!/usr/bin/env python3
"""Prepare the shared optional Face Landmarker; no A2F or robot connection."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.environ.pop('PYTHONPATH',None)
    os.environ['PYTHONNOUSERSITE']='1'
    config=json.loads((ROOT/'configs/perception/face_landmarker.json').read_text())
    python=ROOT/config['python'];venv=python.parent.parent
    if not python.exists():subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'tools/requirements/face_landmarker.txt')],check=True)
    model=ROOT/config['model'];model.parent.mkdir(parents=True,exist_ok=True)
    if not model.exists():
        data=urllib.request.urlopen(config['url'],timeout=60).read()
        if len(data)!=config['bytes'] or hashlib.sha256(data).hexdigest()!=config['sha256']:
            raise RuntimeError('Face Landmarker download integrity check failed')
        model.write_bytes(data)
    if model.stat().st_size!=config['bytes'] or hashlib.sha256(model.read_bytes()).hexdigest()!=config['sha256']:
        raise RuntimeError('Face Landmarker model changed')
    print('Shared Face Landmarker prepared; no camera, audio or serial opened.')


if __name__=='__main__':main()
