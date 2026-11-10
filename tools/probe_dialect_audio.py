#!/usr/bin/env python3
"""Generate bounded synthetic dialect WAVs; never open local audio or actuators."""
import argparse
import asyncio
import base64
from datetime import datetime
import json
from pathlib import Path
import sys
import wave

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vivis.interaction.speech_dialect import SpeechDialect,configuration as dialect_configuration
from vivis.speech.credentials import load_key
from vivis.speech.realtime import Journal,connect_session,websocket_connection


async def run(args):
    c=json.loads((ROOT/'configs/perception/realtime_voice.json').read_text())
    secret=load_key(c['api_key_env'])
    if not secret:raise ValueError('Configure API Key locally first')
    folder=ROOT/'outputs/dialect_audio'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True);log=Journal(folder/'events.jsonl',secret)
    preference=SpeechDialect();preference.name=args.dialect
    session=dict(modalities=['text','audio'],voice=args.voice,speech_rate=1.,
        instructions='这是合成文字语音测试。'+preference.instructions()+'使用目标方言的声调、韵母和节奏；不要把方言词用普通话读音念出来。',
        input_audio_format='pcm',output_audio_format='pcm',input_audio_transcription=dict(language='zh'),
        output_audio=dict(language='zh'),turn_detection=dict(type='smart_turn_v2',threshold=c['vad_threshold']),tools=[])
    pcm=bytearray();transcript='';path=folder/(args.dialect+'.wav')
    try:
        async with websocket_connection(c['endpoint']+'?model='+c['model'],secret) as ws:
            await connect_session(ws,session,log)
            await ws.send(json.dumps(dict(type='conversation.item.create',item=dict(type='message',role='user',
                content=[dict(type='input_text',text='请用目标方言说两句话，介绍你喜欢打篮球和唱歌，问朋友今天忙不忙。不要说明在做测试，不要介绍口音。')])),ensure_ascii=False))
            await ws.send(json.dumps(dict(type='response.create')))
            async def receive():
                nonlocal transcript
                async for raw in ws:
                    event=json.loads(raw);kind=event.get('type')
                    if kind=='response.audio.delta':
                        chunk=base64.b64decode(event['delta'],validate=True)
                        if len(pcm)+len(chunk)>60*48000:raise ValueError('Synthetic sample exceeds60seconds')
                        pcm.extend(chunk)
                    elif kind=='response.audio_transcript.done':transcript=event.get('transcript','')
                    elif kind=='error':raise RuntimeError('Cloud sample rejected: '+str(event.get('error',{})))
                    elif kind=='response.done':
                        if event['response'].get('status')!='completed':raise RuntimeError('Incomplete sample')
                        return
            await asyncio.wait_for(receive(),timeout=60)
        if not pcm:raise ValueError('No cloud audio returned')
        with wave.open(str(path),'wb') as f:f.setparams((1,2,24000,0,'NONE','not compressed'));f.writeframes(pcm)
        result=dict(voice=args.voice,dialect=args.dialect,audio_seconds=len(pcm)/48000,transcript=transcript,
            wav=str(path),no_devices_opened=True,pronunciation_verified=False)
        (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False))
    except Exception as exc:raise RuntimeError(str(exc).replace(secret,'[redacted]')) from None
    finally:log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voice',choices=('longanyang_v3.1','xunanchuan_v3.1'),default='xunanchuan_v3.1')
    parser.add_argument('--dialect',choices=dialect_configuration()['dialects'],default='四川话')
    asyncio.run(run(parser.parse_args()))


if __name__=='__main__':main()
