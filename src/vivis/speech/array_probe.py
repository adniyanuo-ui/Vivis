"""Isolated USB array channel/reference experiment; no DSP control or robot writes."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
from pathlib import Path
import queue
import time
import wave

import numpy as np
from scipy.signal import resample_poly

from vivis.paths import ROOT
from vivis.speech.audio_clock import AudioStreamClock
from vivis.speech.voice_check import PHRASE,source_audio

RATE=48000
BLOCK=960


def make_plan(kind,seconds):
    conditions=([('far_A',False),('far_B',True),('far_A2',False)] if kind=='far-only'
                else [('near',False),('double_A',False),('double_B',True)])
    stages=[];offset=0.
    for name,reference in conditions:
        stages.append(dict(name='quiet',start_s=offset,duration_s=3,board_reference=False))
        offset+=3
        stages.append(dict(name=name,start_s=offset,duration_s=seconds,board_reference=reference))
        offset+=seconds
    stages.append(dict(name='quiet',start_s=offset,duration_s=3,board_reference=False))
    return stages


def waveforms(stages,gain):
    length=round((stages[-1]['start_s']+stages[-1]['duration_s'])*RATE)
    external=np.zeros(length,dtype=np.float32);board=np.zeros_like(external)
    for stage in stages:
        if stage['name']=='quiet' or stage['name']=='near':continue
        pcm=np.frombuffer(source_audio(stage['duration_s']),'<i2').astype(np.float32)/32768
        pcm=resample_poly(pcm,2,1).astype(np.float32)*gain
        # Bound stage edges rather than introducing a click into the path test.
        edge=min(1920,len(pcm)//2)
        pcm[:edge]*=np.linspace(0,1,edge);pcm[-edge:]*=np.linspace(1,0,edge)
        start=round(stage['start_s']*RATE);external[start:start+len(pcm)]=pcm
        if stage['board_reference']:board[start:start+len(pcm)]=pcm
    return external,board


def render(signal,stamp,start,frames):
    """Render against each stream's DAC clock, not callback arrival/order."""
    position=(stamp-start)*RATE+np.arange(frames)
    left=np.floor(position).astype(np.int64);fraction=position-left
    out=np.zeros(frames,dtype=np.float32)
    valid=(left>=0)&(left+1<len(signal))
    i=left[valid];out[valid]=signal[i]*(1-fraction[valid])+signal[i+1]*fraction[valid]
    return out


def run(folder,stages,gain,mic,speaker):
    import sounddevice as sd
    devices=sd.query_devices()
    def select(name,direction):
        hits=[i for i,d in enumerate(devices) if name in d['name'] and d[direction]>=2]
        if len(hits)!=1:raise RuntimeError('Expected one stereo device: '+name+'; found '+str(hits))
        return hits[0]
    capture=select(mic,'max_input_channels');output=select(speaker,'max_output_channels')
    if capture==output:raise ValueError('External speaker must differ from the microphone board')
    if devices[capture]['max_output_channels']<2:raise RuntimeError('Array has no stereo playback endpoint')
    external,board=waveforms(stages,gain)
    report=dict(kind='usb_array_reference_probe',stages=stages,output_gain=gain,
                host_aec=False,cloud_called=False,actuators_opened=False,
                board_speaker_disconnected_user_confirmed=True,
                dsp_controls_changed=False,error=None,callback_warnings=[],clipped={},clocks={},devices={})
    packets=queue.Queue(maxsize=300);faults=[];clocks={};start=[None]
    for key,index in [('capture',capture),('external',output),('board',capture)]:
        report['devices'][key]={**dict(devices[index]),'index':index}
    def record(key,stamp,data,status):
        if status and len(report['callback_warnings'])<128:
            report['callback_warnings'].append(dict(stream=key,status=str(status)))
        try:packets.put_nowait((key,stamp,data.copy()))
        except queue.Full:
            if not faults:faults.append('Audio recording queue overflow')
    def input_cb(data,frames,timing,status):
        record('capture',clocks['capture'].convert(timing.inputBufferAdcTime),data,status)
    def output_cb(key,signal):
        def callback(data,frames,timing,status):
            stamp=clocks[key].convert(timing.outputBufferDacTime)
            mono=np.zeros(frames,dtype=np.float32) if start[0] is None else render(signal,stamp,start[0],frames)
            data[:]=mono[:,None];record(key,stamp,data,status)
        return callback
    samples={key:0 for key in ('capture','external','board')}
    try:
        with ExitStack() as stack:
            writers={};indices={}
            for key in samples:
                writer=stack.enter_context(wave.open(str(folder/(key+'_48k.wav')),'wb'))
                writer.setparams((2,2,RATE,0,'NONE','not compressed'));writers[key]=writer
                indices[key]=stack.enter_context((folder/(key+'_frames.jsonl')).open('w'))
            streams={}
            for key,index,signal in [('external',output,external),('board',capture,board),('capture',capture,None)]:
                factory=sd.InputStream if key=='capture' else sd.OutputStream
                callback=input_cb if key=='capture' else output_cb(key,signal)
                streams[key]=factory(device=index,channels=2,samplerate=RATE,blocksize=BLOCK,
                    dtype='float32',latency='low',callback=callback)
                stack.callback(streams[key].close)
            for key,stream in streams.items():
                clocks[key]=AudioStreamClock.calibrate(stream);report['clocks'][key]=clocks[key].metadata()
            for stream in streams.values():stream.start();stack.callback(stream.stop)
            start[0]=time.monotonic()+.5;report['start_monotonic']=start[0]
            end=start[0]+len(external)/RATE+.3;announced=set();last={key:time.monotonic() for key in samples}
            while time.monotonic()<end:
                now=time.monotonic()
                for i,stage in enumerate(stages):
                    if i in announced or now<start[0]+stage['start_s']:continue
                    announced.add(i);name=stage['name']
                    text=('请保持安静，仅音箱播放。' if name.startswith('far') else
                          '请立即重复读：'+PHRASE if name.startswith('double') else
                          '请读两遍：'+PHRASE if name=='near' else '安静，等待下一阶段。')
                    print(name+'：'+text,flush=True)
                try:
                    key,stamp,data=packets.get(timeout=.02)
                except queue.Empty:pass
                else:
                    last[key]=now
                    indices[key].write(json.dumps(dict(sample=samples[key],time_monotonic=stamp,frames=len(data)))+'\n')
                    report['clipped'][key]=report['clipped'].get(key,0)+int(np.count_nonzero(np.abs(data)>=.999))
                    writers[key].writeframesraw((np.clip(data,-1,32767/32768)*32768).astype('<i2').tobytes())
                    samples[key]+=len(data)
                if faults:raise RuntimeError(faults[0])
                if any(now-value>1 for value in last.values()):raise RuntimeError('Audio callback watchdog')
            # Streams are stopped by ExitStack before files close. Final queued
            # tail is intentionally excluded, with per-stream sample counts logged.
    except KeyboardInterrupt:report['error']='Operator stop'
    except Exception as exc:report['error']=str(exc)
    finally:
        report['samples']=samples
        (folder/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('实验结束：'+str(folder),flush=True)
    if report['error']:print(report['error'],flush=True)
    return 1 if report['error'] else 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('far-only','speech'),default='far-only')
    p.add_argument('--seconds',type=int,default=16)
    p.add_argument('--output-gain',type=float,default=.5)
    p.add_argument('--mic',default='DOV:');p.add_argument('--speaker',default='USB2.0 Device:')
    p.add_argument('--board-speaker-disconnected',action='store_true',help='确认板卡自身未接喇叭，允许写入USB播放参考')
    p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true')
    args=p.parse_args(argv)
    if not 10<=args.seconds<=30:p.error('seconds须在10..30内')
    if not np.isfinite(args.output_gain) or not .1<=args.output_gain<=.5:p.error('本实验gain须在0.1..0.5内')
    stages=make_plan(args.stage,args.seconds)
    if args.check:
        print(json.dumps(dict(stages=stages,hardware_opened=False,cloud_called=False),ensure_ascii=False));return 0
    if not args.board_speaker_disconnected:p.error('先确认板卡自身未接喇叭，再加--board-speaker-disconnected')
    path=ROOT/'outputs/runtime/p1.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('P1或其他语音实验正在运行，请先退出')
        folder=args.output or ROOT/'outputs/array_probe'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        folder.mkdir(parents=True,exist_ok=False)
        print('USB阵列对照（仅本地录音，不控制机器头）：'+str(folder),flush=True)
        return run(folder,stages,args.output_gain,args.mic,args.speaker)


if __name__=='__main__':raise SystemExit(main())
