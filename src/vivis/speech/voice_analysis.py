"""Offline acoustic check/replay. Metrics are not a substitute for duplex acceptance."""
import argparse
import json
from pathlib import Path
import time
import wave

import numpy as np
from scipy.signal import correlate,correlation_lags

from vivis.paths import ROOT
from vivis.speech.voice_frontend import VoiceFrontEnd


def read_pcm(path):
    with wave.open(str(path)) as wav:
        if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(16000,1,2):
            raise ValueError('Need16k mono PCM16: '+str(path))
        return np.frombuffer(wav.readframes(wav.getnframes()),'<i2').astype(np.float32)/32768


def rms(data):return float(np.sqrt(np.mean(data.astype(np.float64)**2))) if len(data) else 0.


def attenuation(near,clean):
    if rms(near)<1e-4:return None  # Do not report huge "improvements" on quantized silence.
    return 20*float(np.log10(max(rms(near),1e-9)/max(rms(clean),1e-9)))


def correlation_peak(near,far,rate=16000,max_delay_ms=300):
    """Measured correlation peak only, not proven echo path / physical delay."""
    if rms(near)<1e-5 or rms(far)<1e-5 or len(near)!=len(far):return None
    near=near.astype(np.float64)-np.mean(near);far=far.astype(np.float64)-np.mean(far)
    values=correlate(near,far,method='fft');lags=correlation_lags(len(near),len(far))
    use=(lags>=0)&(lags<=max_delay_ms*rate/1000)
    pos=int(np.argmax(np.abs(values[use])));lag=int(lags[use][pos])
    magnitude=float(values[use][pos]/max(np.linalg.norm(near)*np.linalg.norm(far),1e-12))
    return dict(delay_ms=lag*1000/rate,normalized_correlation=magnitude)


def stage_ranges(folder,summary,length):
    frames_path=folder/'frames.jsonl'
    if not frames_path.exists():return [dict(name='unlabelled',start_sample=0,end_sample=length)]
    frames=[json.loads(line) for line in frames_path.read_text().splitlines()]
    result=[]
    for index,stage in enumerate(summary.get('stages',[])):
        # Exclude first and last1s from scores, leaving transitions/convergence out of comparison.
        begin=stage['start_monotonic']+1.
        end=(stage['end_monotonic'] if 'end_monotonic' in stage
             else stage['start_monotonic']+stage['duration_s'])-1.
        selected=[f['sample'] for f in frames if begin<=f['adc_monotonic']<end]
        if selected:result.append(dict(name=stage['name'],stage_index=index,start_sample=min(selected),
                                       end_sample=min(length,max(selected)+320)))
    return result


def replay(near,far,speech,frontend):
    if len(near)!=len(far) or len(near)%320:raise ValueError('Replay inputs must be equally sized complete20ms frames')
    processor=VoiceFrontEnd(speech,frontend);clean=np.empty_like(near);times=[]
    try:
        for i in range(0,len(near),320):
            began=time.perf_counter();clean[i:i+320]=processor.process16(near[i:i+320],far[i:i+320])
            times.append(time.perf_counter()-began)
    finally:processor.close()
    return clean,dict(process_p50_ms=float(np.median(times))*1000,
                      process_p95_ms=float(np.percentile(times,95))*1000,
                      process_max_ms=max(times)*1000,process_total_s=sum(times))


def asr_engine():
    import sherpa_onnx
    config=json.loads((ROOT/'configs/perception/speech.json').read_text())['asr_final'];folder=ROOT/config['directory']
    if not (folder/config['model']).is_file() or not (folder/config['tokens']).is_file():
        raise RuntimeError('Local SenseVoice assets missing; omit --transcribe for numerical analysis')
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(model=str(folder/config['model']),
        tokens=str(folder/config['tokens']),num_threads=config['threads'],language=config['language'],use_itn=False)


def transcribe(engine,audio):
    stream=engine.create_stream();stream.accept_waveform(16000,audio)
    engine.decode_stream(stream);return stream.result.text


def analyze(folder,output,replay_backends=(),recognize=False):
    summary=json.loads((folder/'summary.json').read_text())
    signal={name:read_pcm(folder/(name+'_16k.wav')) for name in ('microphone','reference','uplink')}
    length=len(signal['microphone'])
    if not length or any(len(v)!=length for v in signal.values()):raise ValueError('Empty or unaligned recordings')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(source=str(folder),duration_s=length/16000,recording_error=summary.get('error'),
                metrics_scope='RMS attenuation and correlation, not ERLE or ground-truth speech quality',
                recognition_scope='Optional local SenseVoice diagnostic, not cloud Qwen acceptance',
                stages=[],replays={},callback_warnings=summary.get('callback_warnings',[]))
    annotation_path=folder/'annotations.json'
    annotations=json.loads(annotation_path.read_text()) if annotation_path.exists() else {}
    report['operator_annotations']=annotations
    config=json.loads((ROOT/'configs/perception/realtime_voice.json').read_text())
    speech=json.loads((ROOT/config['speech']).read_text());near=signal['microphone'];far=signal['reference']
    for backend in replay_backends:
        options=dict(config.get('frontend',{}),backend=backend)
        result,timings=replay(near,far,speech,options);signal['replay_'+backend]=result;report['replays'][backend]=timings
        with wave.open(str(output/('replay_'+backend+'_16k.wav')),'wb') as wav:
            wav.setparams((1,2,16000,0,'NONE','not compressed'))
            wav.writeframes(np.clip(result*32768,-32768,32767).astype('<i2').tobytes())
    engine=asr_engine() if recognize else None
    for stage in stage_ranges(folder,summary,length):
        override=annotations.get('stage_overrides',{}).get(str(stage.get('stage_index')))
        if override:stage['operator_annotation']=override
        start,end=stage['start_sample'],stage['end_sample'];pieces={k:v[start:end] for k,v in signal.items()}
        stage.update(start_s=start/16000,end_s=end/16000,raw_rms=rms(pieces['microphone']),
                     reference_rms=rms(pieces['reference']),
                     microphone_reference_peak=correlation_peak(pieces['microphone'],pieces['reference']),backends={})
        for key,values in pieces.items():
            if key=='reference':continue
            result=dict(rms=rms(values),attenuation_db=attenuation(pieces['microphone'],values),
                        clipped_samples=int(np.count_nonzero(np.abs(values)>=.999)),
                        reference_peak=correlation_peak(values,pieces['reference']))
            if engine:result['local_transcript']=transcribe(engine,values)
            stage['backends'][key]=result
        report['stages'].append(stage)
    events_path=folder/'events.jsonl'
    if events_path.exists():
        events=[json.loads(line) for line in events_path.read_text().splitlines()]
        cuts=[e for e in events if e['kind']=='local_interrupt' and e.get('discarded_samples',0)>0]
        starts={}
        for e in events:
            if e['kind']=='playback_started':starts[e['response_id']]=starts.get(e['response_id'],0)+1
        report['playback']=dict(local_cuts=len(cuts),discarded_audio_s=sum(e['discarded_samples'] for e in cuts)/48000,
                                starts_by_response=starts,
                                responses_with_repeated_starts=[rid for rid,count in starts.items() if count>1])
    (output/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--replay',nargs='*',choices=('speex','aec3'),default=[])
    parser.add_argument('--transcribe',action='store_true',help='只用本地SenseVoice，无云请求')
    args=parser.parse_args(argv)
    output=args.output or args.folder/'analysis'
    result=analyze(args.folder.resolve(),output.resolve(),args.replay,args.transcribe)
    print('分析结果：'+str(output/'analysis.json'))
    for stage in result['stages']:
        print(stage['name'],json.dumps(stage['backends'],ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
