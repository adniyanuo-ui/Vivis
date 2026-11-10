#!/usr/bin/env python3
"""Fetch pinned local control keyword models; never open audio or actuators."""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    c=json.loads((ROOT/'configs/perception/speech.json').read_text()).get('control_keywords',{})
    if not c.get('enabled'):return
    s=json.loads((ROOT/'native/speech/control_kws_sources.json').read_text());folder=ROOT/c['directory']
    def valid(path,row):return path.is_file() and path.stat().st_size==row['size'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    if all(valid(folder/r['name'],r) for r in s['files']):
        print('Local stop keyword models verified.');return
    archive=ROOT/'outputs/speech/control-keywords.tar.bz2';archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest()!=s['sha256']:
        temp=archive.with_suffix('.part')
        with urllib.request.urlopen(s['url'],timeout=60) as response,temp.open('wb') as target:
            while chunk:=response.read(1024*1024):
                if target.tell()+len(chunk)>64*1024*1024:raise ValueError('Oversize keyword model archive')
                target.write(chunk)
        if hashlib.sha256(temp.read_bytes()).hexdigest()!=s['sha256']:raise ValueError('Keyword archive hash mismatch')
        temp.replace(archive)
    folder.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        for row in s['files']:
            if Path(row['name']).name!=row['name']:raise ValueError('Invalid model path')
            matches=[m for m in tar.getmembers() if m.isfile() and Path(m.name).name==row['name']]
            if len(matches)!=1 or matches[0].size!=row['size']:raise ValueError('Unexpected keyword model archive member')
            data=tar.extractfile(matches[0]).read()
            if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Keyword model hash mismatch')
            temp=folder/(row['name']+'.part');temp.write_bytes(data);temp.replace(folder/row['name'])
    print('Local stop keyword models prepared; no device opened.')


if __name__=='__main__':main()
