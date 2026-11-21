"""Prepare the current Qwen voice and A2F once; never open audio/camera/servos."""
import argparse
import asyncio
import csv
import io
import json
import os
from pathlib import Path
import time
import wave

import numpy as np

from vivis.paths import ROOT
from vivis.speech.fixed_replies import REPLIES,FORMAT,FixedReplyCache,digest,signature,validate_frames


def context(config,robot_path):
    from vivis.speech.managed import voice_configuration
    from vivis.runtime.robot import Robot
    settings=json.loads(config.read_text())
    speech=json.loads((ROOT/settings['speech']).read_text())
    speech['managed_voice']=settings['realtime_voice'];speech['programs']=str(ROOT/settings['programs'])
    voice,_,url,session=voice_configuration(speech,robot_path)
    session=dict(session,model=voice['model'])
    robot=Robot(robot_path)
    with robot.resolve('channels').open(encoding='utf-8-sig',newline='') as f:order=next(csv.reader(f))[2:54]
    return settings,voice,url,session,order


def write_blob(directory,name,suffix,data):
    sha=digest(data);path=directory/(name+'-'+sha[:12]+suffix)
    if not path.exists():
        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(data);os.replace(temporary,path)
    elif digest(path.read_bytes())!=sha:raise ValueError('Unexpected fixed reply hash collision')
    return path.name,sha


def generate_animation(mouth,name,pcm,timeout=60):
    rid='prepare_'+name;mouth.add(rid,pcm);mouth.finish(rid);deadline=time.monotonic()+timeout
    while True:
        if mouth.error:raise RuntimeError('Fixed reply A2F failed: '+mouth.error)
        with mouth.lock:
            row=mouth.responses.get(rid)
            if row and row['done']:
                frames=list(row['frames']);break
        if time.monotonic()>=deadline:raise TimeoutError('Fixed reply A2F preparation timed out')
        time.sleep(.02)
    indices=np.array([i for i,_ in frames],dtype='<i8');weights=np.array([w for _,w in frames],dtype='<f4')
    validate_frames(indices,weights,len(pcm)//2)
    raw=io.BytesIO();np.savez_compressed(raw,sample_indices=indices,weights=weights)
    return raw.getvalue(),len(frames)


async def prepare(settings,voice,url,session,order,directory,force=False):
    from vivis.speech.credentials import load_key
    from vivis.speech.trusted_reading import RealtimeReader
    from vivis.speech.realtime import Journal
    from vivis.speech.message_voice import pcm24
    from vivis.speech.mouth import SpeechMouth
    key=load_key(voice['api_key_env'])
    if not key:raise ValueError('请先配置本机百炼API Key；参见语音服务文档。')
    directory.mkdir(parents=True,exist_ok=True)
    cache=FixedReplyCache(directory);entries={};mouth=None
    stage=ROOT/'outputs/fixed_replies'/time.strftime('%Y%m%d_%H%M%S');stage.mkdir(parents=True,exist_ok=True)
    log=Journal(stage/'generation.jsonl',key)
    try:
        # The reader's model is in the URL, not its session.update body.
        reader=RealtimeReader(url,key,{k:v for k,v in session.items() if k!='model'},log)
        lip=settings['realtime_voice']['lip_sync']
        for name,text in REPLIES.items():
            if not force:
                try:existing=cache.get(text,session,order)
                except (OSError,ValueError,KeyError,TypeError,EOFError):existing=None
                if existing is not None:
                    entries[name]=cache.entries[name];print('Verified existing: '+name,flush=True);continue
            audio=await reader.generate(text)
            pcm=await asyncio.to_thread(pcm24,audio)
            if not 0<len(pcm)<=24000*30*2:raise ValueError('Fixed reply exceeds30seconds')
            if mouth is None:
                mouth=SpeechMouth(ROOT/lip['binary'],ROOT/lip['model'],stage/'mouth.json',order,
                    secret_env=(voice['api_key_env'],),buffer_seconds=30)
            animation,frames=await asyncio.to_thread(generate_animation,mouth,name,pcm)
            wav=io.BytesIO()
            with wave.open(wav,'wb') as output:
                output.setparams((1,2,24000,0,'NONE','not compressed'));output.writeframes(pcm)
            audio_name,audio_sha=write_blob(directory,name,'.wav',wav.getvalue())
            animation_name,animation_sha=write_blob(directory,name,'.npz',animation)
            entries[name]=dict(text=text,audio=audio_name,audio_sha256=audio_sha,
                animation=animation_name,animation_sha256=animation_sha,frames=frames,duration_s=len(pcm)/48000)
            print('Prepared: '+name+' '+text,flush=True)
        manifest=dict(schema_version=FORMAT,generation=signature(session),order=order,replies=entries,
            animation_source=dict(binary_sha256=digest((ROOT/lip['binary']).read_bytes()),
                model_config_sha256=digest((ROOT/lip['model']).read_bytes())),
            note='Original verified Qwen PCM; canonical52 from the existing SpeechMouth live resampler/A2F. No actuator positions or personal context.')
        temporary=directory/'manifest.json.tmp'
        temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        report=FixedReplyCache(directory,manifest_name=temporary.name).verify(session,order)
        os.replace(temporary,directory/'manifest.json')  # Publish only the completely verified bundle.
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:
        if mouth:mouth.close()
        log.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/interaction/p1.json')
    parser.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true',help='Verify files only; no cloud, GPU or devices')
    parser.add_argument('--list',action='store_true',help='List the fixed replies only')
    parser.add_argument('--force',action='store_true',help='Explicitly regenerate all replies')
    args=parser.parse_args(argv)
    if args.list:
        for name,text in REPLIES.items():print(name+'：'+text)
        return 0
    settings,voice,url,session,order=context(args.config,args.robot)
    directory=args.output or ROOT/settings['realtime_voice']['fixed_replies']
    if args.check:
        print(json.dumps(FixedReplyCache(directory).verify(session,order),ensure_ascii=False));return 0
    asyncio.run(prepare(settings,voice,url,session,order,directory,args.force));return 0


if __name__=='__main__':raise SystemExit(main())
