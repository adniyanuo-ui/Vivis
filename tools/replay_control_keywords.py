#!/usr/bin/env python3
"""Replay recorded uplink/reference through local stop detection; no devices/cloud."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import wave

import numpy as np

from vivis.paths import ROOT
from vivis.speech.control_keywords import KeywordControl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path);p.add_argument('--offset',type=float,default=0.)
    p.add_argument('--seconds',type=float,default=60.);p.add_argument('--output',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/perception/speech.json');a=p.parse_args()
    if not np.isfinite(a.offset) or not np.isfinite(a.seconds) or not 0<=a.offset<=86400 or not 0<a.seconds<=900:
        p.error('Invalid replay interval')
    folder=a.session/'voice';samples={}
    for name in ('uplink','reference'):
        with wave.open(str(folder/(name+'_16k.wav'))) as w:
            if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(16000,1,2):raise ValueError('Need aligned mono16k PCM16 recordings')
            samples[name]=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(np.float32)/32768
    if len(samples['uplink'])!=len(samples['reference']):raise ValueError('Recorded input/reference lengths differ')
    anchors=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
    if not anchors:raise ValueError('No recorded clock anchors')
    indices=np.array([r['sample'] for r in anchors]);stamps=np.array([r['adc_monotonic'] for r in anchors])
    start=int(a.offset*50)*320;end=min(len(samples['uplink']),start+round(a.seconds*16000))
    if end-start<320:raise ValueError('Requested interval is outside recording')
    hypotheses=[]
    for line in (a.session/'asr_events.jsonl').read_text().splitlines():
        event=json.loads(line)
        if event.get('kind') in ('asr_partial','asr_final'):hypotheses.append(event)
    hypotheses.sort(key=lambda e:e['captured_end_monotonic'])
    config=json.loads(a.config.read_text())['control_keywords'];events=[];watch=KeywordControl(config,events.append)
    cursor=0;text='';began=time.monotonic()
    for index in range(start,end-319,320):
        stamp=float(np.interp(index,indices,stamps))
        while cursor<len(hypotheses) and hypotheses[cursor]['captured_end_monotonic']<=stamp:
            h=hypotheses[cursor];text=h.get('text','') if h['kind']=='asr_partial' else '';cursor+=1
        watch.feed(samples['uplink'][index:index+320],samples['reference'][index:index+320],stamp,text)
    report=dict(session=str(a.session),config=config,events=events,wall_s=time.monotonic()-began,
        input_seconds=(end-start)/16000,offset_s=start/16000,hardware_opened=False,network_called=False,
        note='Replay of saved audio; detection timestamp is not measured end-of-utterance or onsite cancellation latency.')
    output=a.output or ROOT/'outputs/diagnostics'/('control_replay_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),events=events,wall_s=report['wall_s']),ensure_ascii=False))


if __name__=='__main__':main()
