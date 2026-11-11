#!/usr/bin/env python3
"""Compare local labelled media; no camera, robot or cloud connection."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.vision.face_benchmark import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=1,help='Warm measurement repetitions per sampled frame')
    args=p.parse_args();report=run(args.manifest,args.output,args.repeats)
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
