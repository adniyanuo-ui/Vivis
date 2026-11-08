#!/usr/bin/env python3
"""Prepare pinned Qwen3-TTS in its own environment; never open devices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as src:
        for data in iter(lambda:src.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def download(entry,repository,directory):
    relative=Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid model file path')
    path=directory/relative;path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and path.stat().st_size==entry['size'] and digest(path)==entry['sha256']:return
    temporary=path.with_suffix(path.suffix+'.part')
    url=('https://modelscope.cn/api/v1/models/'+repository+'/repo/?Revision=master&FilePath='
         +urllib.parse.quote(entry['path'],safe=''))
    for attempt in range(3):
        try:
            offset=temporary.stat().st_size if temporary.exists() else 0
            if offset==entry['size'] and digest(temporary)==entry['sha256']:
                temporary.replace(path);return
            headers={'Range':f'bytes={offset}-'} if 0<offset<entry['size'] else {}
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
                if response.status!=206:offset=0
                if offset and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Invalid resume range')
                with temporary.open('ab' if offset else 'wb') as dest:
                    for data in iter(lambda:response.read(1024*1024),b''):
                        dest.write(data)
                        if dest.tell()>entry['size']:raise ValueError('Model file exceeds expected size')
            if temporary.stat().st_size!=entry['size'] or digest(temporary)!=entry['sha256']:
                temporary.unlink(missing_ok=True);raise ValueError('Qwen model hash mismatch: '+entry['path'])
            temporary.replace(path);print('Verified '+entry['path'],flush=True);return
        except (OSError,ValueError):
            if attempt==2:raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/perception/speech.json')
    parser.add_argument('--skip-install',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text());t=config['tts']
    if t.get('engine')!='qwen3':parser.error('Select qwen3 in the speech configuration first')
    python=ROOT/t['python'];env={**os.environ,'PYTHONNOUSERSITE':'1'};env.pop('PYTHONPATH',None)
    if not args.skip_install:
        if not python.exists():subprocess.run([sys.executable,'-m','venv',str(python.parent.parent)],env=env,check=True)
        subprocess.run([str(python),'-m','pip','install','torch==2.10.0+cpu','torchaudio==2.10.0+cpu',
                        '--index-url','https://download.pytorch.org/whl/cpu'],env=env,check=True)
        subprocess.run([str(python),'-m','pip','install','qwen-tts==0.1.1','pyloudnorm==0.1.1'],env=env,check=True)
        subprocess.run([str(python),'-m','pip','check'],env=env,check=True)
    source=json.loads((ROOT/t['source_manifest']).read_text())
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda entry:download(entry,source['model'],ROOT/t['directory']),source['files']))
    print('Qwen3-TTS files verified; CPU prompt preparation only. No microphone, speaker or servo opened.')


if __name__=='__main__':main()
