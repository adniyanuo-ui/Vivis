#!/usr/bin/env python3
"""Download a pinned optional SFace model; no camera or robot opened."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'native/vision/sface_sources.json').read_text())
    folder=ROOT/'outputs/models/sface';folder.mkdir(parents=True,exist_ok=True)
    for row in manifest['files']:
        path=folder/row['name']
        def valid(p):return p.is_file() and p.stat().st_size==row['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        if valid(path):continue
        temporary=path.with_suffix('.part')
        with urllib.request.urlopen(row['url'],timeout=60) as response,temporary.open('wb') as stream:
            while chunk:=response.read(1024*1024):
                if stream.tell()+len(chunk)>row['size']:raise ValueError('Oversized SFace asset')
                stream.write(chunk)
        if not valid(temporary):raise ValueError('SFace asset integrity mismatch')
        temporary.replace(path)
    print('SFace assets verified; no image uploaded or device opened.')


if __name__=='__main__':main()
