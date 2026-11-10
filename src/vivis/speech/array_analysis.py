"""Offline stereo/USB-reference comparison. Does not infer DSP firmware settings."""
import argparse
import json
from pathlib import Path
import wave

import numpy as np
from scipy.signal import resample_poly

from vivis.speech.array_probe import RATE
from vivis.paths import ROOT
from vivis.speech.voice_analysis import rms,correlation_peak,asr_engine,transcribe,replay


def channel_identity(capture):
    """Test digital duplication only; this cannot identify pre-DSP raw channels."""
    candidates=[]
    for delay in range(-4,5):
        a=capture[max(-delay,0):len(capture)-max(delay,0),0]
        b=capture[max(delay,0):len(capture)-max(-delay,0),1]
        candidates.append(dict(channel1_delay_samples=delay,equal_fraction=float(np.mean(a==b)),difference_rms=rms(a-b)))
    return max(candidates,key=lambda item:item['equal_fraction'])


def load(folder,key):
    with wave.open(str(folder/(key+'_48k.wav'))) as wav:
        if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(RATE,2,2):
            raise ValueError('Expected48k stereo PCM16')
        data=np.frombuffer(wav.readframes(wav.getnframes()),'<i2').reshape(-1,2).astype(np.float32)/32768
    frames=[json.loads(line) for line in (folder/(key+'_frames.jsonl')).read_text().splitlines()]
    stamps=np.empty(len(data),dtype=np.float64);expected=0
    for f in frames:
        first=f['sample'];count=f['frames']
        if first!=expected or first+count>len(data):raise ValueError('Noncontiguous recorded indices')
        stamps[first:first+count]=f['time_monotonic']+np.arange(count)/RATE;expected+=count
    if expected!=len(data) or not len(data):raise ValueError('Incomplete audio/time index')
    # Small timestamp jitter can produce overlaps between adjacent blocks.
    # Report gaps/overlaps; align per block in analysis rather than hiding them.
    gaps=np.diff([f['time_monotonic'] for f in frames])-np.array([f['frames']/RATE for f in frames[:-1]])
    timing=dict(min_gap_ms=float(np.min(gaps))*1000 if len(gaps) else 0,
                max_gap_ms=float(np.max(gaps))*1000 if len(gaps) else 0)
    return data,stamps,timing


def analyze(folder,output,recognize=False,host_aec=False):
    summary=json.loads((folder/'summary.json').read_text())
    if summary.get('error'):raise ValueError('Recording failed: '+summary['error'])
    capture,stamps,timing=load(folder,'capture');external,ext_stamps,ext_timing=load(folder,'external')
    board,board_stamps,board_timing=load(folder,'board')
    # Outputs are rendered from one timestamp-indexed source. Sort timestamp
    # samples before interpolation if adjacent host blocks slightly overlap.
    def align(data,times):
        order=np.argsort(times,kind='stable')
        return np.interp(stamps,times[order],data[order,0],left=0,right=0)
    reference=align(external,ext_stamps);mirror=align(board,board_stamps)
    down=lambda values:resample_poly(values,1,3,axis=0).astype(np.float32)
    channels=down(capture);reference=down(reference);mirror=down(mirror);times=stamps[::3]
    length=min(len(times),len(channels),len(reference));times=times[:length]
    channels=channels[:length];reference=reference[:length];mirror=mirror[:length]
    report=dict(source=str(folder),host_aec=False,recording_error=summary.get('error'),
        callback_warnings=summary.get('callback_warnings'),timing=dict(capture=timing,external=ext_timing,board=board_timing),
        digital_channel_identity=channel_identity(capture),
        scope='USB channel/level comparison; not isolated near-speech quality, ERLE or firmware identification',stages=[])
    engine=asr_engine() if recognize else None
    output.mkdir(parents=True,exist_ok=False)
    signals=dict(channel0=channels[:,0],channel1=channels[:,1],external=reference,board_reference=mirror)
    if host_aec:
        speech=json.loads((ROOT/'configs/perception/speech.json').read_text());count=length//320*320
        result,timings=replay(channels[:count,0],reference[:count],speech,{'backend':'aec3'})
        # The final partial frame has no AEC result, never pad it with fake data.
        signals['host_aec3']=result;report['offline_host_aec3']=dict(**timings,samples=count)
    def save(name,data):
        with wave.open(str(output/(name+'_16k.wav')),'wb') as wav:
            wav.setparams((1,2,16000,0,'NONE','not compressed'))
            wav.writeframes(np.clip(data*32768,-32768,32767).astype('<i2').tobytes())
    for key,data in signals.items():save(key,data)
    for s in summary['stages']:
        begin=summary['start_monotonic']+s['start_s'];end=begin+s['duration_s']
        use=(times>=begin+1)&(times<end-1);full=(times>=begin)&(times<end)
        entry=dict(**s,channels=[],external_rms=rms(reference[use]),board_reference_rms=rms(mirror[use]))
        a,b=channels[use,0],channels[use,1]
        entry['channel_difference_rms']=rms(a-b)
        entry['channel_correlation']=float(np.corrcoef(a,b)[0,1]) if rms(a)>1e-6 and rms(b)>1e-6 else None
        for i in range(2):
            values=channels[use,i]
            item=dict(rms=rms(values),reference_peak=correlation_peak(values,reference[use]),
                      peak_abs=float(np.max(np.abs(values))) if len(values) else 0)
            if engine:item['transcript']=transcribe(engine,channels[full,i]) if rms(channels[full,i])>=1e-4 else ''
            entry['channels'].append(item)
        if rms(reference[use])>1e-4 and rms(mirror[use])>1e-4:
            entry['digital_feed_difference_rms']=rms(reference[use]-mirror[use])
        if s['name']!='quiet':
            for name in ('channel0','host_aec3'):
                if name not in signals:continue
                data=signals[name];mask=full[:len(data)];save(s['name']+'_'+name,data[mask])
                if name=='host_aec3':
                    result=dict(rms=rms(data[use[:len(data)]]))
                    if engine:result['transcript']=transcribe(engine,data[mask]) if rms(data[mask])>=1e-4 else ''
                    entry['offline_host_aec3']=result
        report['stages'].append(entry)
    (output/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    parser.add_argument('--transcribe',action='store_true');parser.add_argument('--output',type=Path)
    parser.add_argument('--host-aec',action='store_true',help='仅离线复算AEC3并导出试听，不改变现场采集或默认前端')
    args=parser.parse_args();output=args.output or args.folder/'analysis'
    result=analyze(args.folder,output,args.transcribe,args.host_aec)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
