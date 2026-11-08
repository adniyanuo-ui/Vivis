#!/usr/bin/env python3
"""Export same-text local voice auditions; no audio, camera or robot devices."""
import argparse
import json
from pathlib import Path
import sys
import os
import subprocess
import wave
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.paths import ROOT
from vivis.speech.prompts import create_engine,synthesize_prompt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/perception/speech.json')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--speaker',type=int,action='append')
    group.add_argument('--voice',action='append',help='Named Qwen voice, e.g. Dylan or Uncle_Fu')
    parser.add_argument('--text',default='我叫小帅，北京男孩，今年二十岁。唱歌、演戏、学表情，你来挑，我来演！')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());t=config['tts']
    if t.get('engine')=='qwen3' and Path(sys.prefix).resolve()!=(ROOT/t['python']).parent.parent.resolve():
        subprocess.run([str(ROOT/t['python']),str(Path(__file__).resolve()),*sys.argv[1:]],
            env={**os.environ,'PYTHONNOUSERSITE':'1','PYTHONPATH':str(ROOT/'src'),'HF_HUB_OFFLINE':'1'},check=True)
        return
    engine=create_engine(config)
    if args.voice and t.get('engine')!='qwen3':parser.error('Named voices require Qwen3-TTS')
    if args.speaker and any(not 0<=sid<engine.num_speakers for sid in args.speaker):parser.error('Unknown speaker ID')
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    for value in args.voice or args.speaker:
        if args.voice:engine.set_voice(value)
        sid=0 if args.voice else value
        samples,rate,_=synthesize_prompt(engine,args.text,sid,config['tts']['speed'],config.get('prompt_timing'))
        peak=float(np.max(np.abs(samples)));samples=samples/max(peak,.001)*.7
        path=args.output/f'voice_{value}.wav'
        with wave.open(str(path),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(rate)
            wav.writeframes(np.clip(samples*32767,-32768,32767).astype('<i2').tobytes())
        rows.append(dict(voice=value,file=path.name,duration_s=len(samples)/rate))
        print(path,flush=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(text=args.text,tts=config['tts'],
        level='peak-normalized for audition only; final prompts use saved R128',candidates=rows),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
