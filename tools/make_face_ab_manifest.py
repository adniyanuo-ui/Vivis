#!/usr/bin/env python3
"""Assemble operator-labelled captures and unchanged registration photos, locally."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.paths import ROOT
from vivis.vision.face_benchmark import capture_clip
from vivis.vision.identity_config import backend_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,nargs='+',required=True,help='Explicit recording directories; do not mix old runs silently')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    gallery=[]
    for path in sorted((ROOT/'private/people').glob('*/profile.json')):
        photos=[f for f in path.parent.glob('photo.*') if f.suffix.lower() in ('.png','.jpg','.jpeg')]
        if len(photos)!=1:raise ValueError('Need exactly one original registration photo for '+path.parent.name)
        gallery.append(dict(person_id=path.parent.name,path=str(photos[0].resolve())))
    clips=[capture_clip(folder) for folder in args.capture]
    c=backend_config(json.loads((ROOT/'configs/perception/familiar_faces.json').read_text()),'sface')
    data=dict(schema_version='vivis-face-ab-1.0',kind='video',annotated_preview=False,sample_hz=2,
        baseline_gallery=str(ROOT/c['gallery']),
        thresholds=dict(sface=dict(score=.5,margin=.08),seeta6=dict(score=.6,margin=.08)),gallery=gallery,clips=clips)
    args.output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with args.output.open('x') as stream:json.dump(data,stream,ensure_ascii=False,indent=2)
    print(args.output)
if __name__=='__main__':main()
