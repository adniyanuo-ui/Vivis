#!/usr/bin/env python3
"""Prepare optional local speech assets and prompts; never open robot/audio devices."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def valid(data,entry):
    if len(data)!=entry['size']:return False
    if 'lfs' in entry:return hashlib.sha256(data).hexdigest()==entry['lfs']['sha256']
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==entry['blobId']


def main():
    os.environ.pop('PYTHONPATH',None);os.environ['PYTHONNOUSERSITE']='1'
    cfg=json.loads((ROOT/'configs/perception/speech.json').read_text());python=ROOT/cfg['python']
    if cfg.get('prompt_level',{}).get('method')=='r128' and shutil.which('sox') is None:
        raise RuntimeError('Prepare voice prompts first requires: sudo apt-get install sox')
    if not python.exists():subprocess.run([sys.executable,'-m','venv',str(python.parent.parent)],check=True)
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'tools/requirements/speech.txt')],check=True)
    sources=json.loads((ROOT/'native/speech/sources.json').read_text())
    for source in sources['sources']:
        folder=ROOT/cfg[source['kind']]['directory'];folder.mkdir(parents=True,exist_ok=True)
        for entry in source['files']:
            rel=entry['rfilename'];path=folder/rel
            if path.exists() and valid(path.read_bytes(),entry):continue
            url=f'https://huggingface.co/{source["repository"]}/resolve/{source["revision"]}/{urllib.parse.quote(rel)}'
            print('Download '+rel,flush=True)
            last=None
            for attempt in range(3):
                try:
                    data=urllib.request.urlopen(url,timeout=60).read()
                    if not valid(data,entry):raise RuntimeError('Speech asset integrity mismatch: '+rel)
                    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
                    temporary.write_bytes(data);temporary.replace(path);last=None;break
                except Exception as exc:last=exc
            if last is not None:raise last
    env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
    subprocess.run([str(python),'-m','vivis.speech.prompts','--prepare'],env=env,cwd=ROOT,check=True)
    print('P1 speech dependencies ready; no microphone, speaker, camera or servo opened.')


if __name__=='__main__':main()
