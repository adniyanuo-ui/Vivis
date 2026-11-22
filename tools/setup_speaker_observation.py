#!/usr/bin/env python3
"""Prepare/hash-check the optional Chinese speaker model; opens no devices."""
import argparse,hashlib,json,os,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.paths import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');args=p.parse_args()
    c=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text());path=ROOT/c['model']
    url='https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx'
    def valid():return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256']
    if not valid() and not args.check:
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.download')
        try:
            with urllib.request.urlopen(url,timeout=30) as source,tmp.open('wb') as out:
                total=0
                while block:=source.read(1024*1024):
                    total+=len(block)
                    if total>64*1024*1024:raise ValueError('Oversized model download')
                    out.write(block)
            if hashlib.sha256(tmp.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Speaker model hash mismatch')
            tmp.replace(path)
        finally:tmp.unlink(missing_ok=True)
    if not valid():raise SystemExit('声纹模型缺失或校验失败；运行 .venv/bin/python tools/setup_speaker_observation.py')
    print('声纹模型校验通过；仅观察，不赋予身份或业务权限。运行环境复用 outputs/speech/venv。')


if __name__=='__main__':main()
