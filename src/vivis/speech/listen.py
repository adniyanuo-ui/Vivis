"""Streaming Chinese ASR in an isolated process; JSON events, no actuator access."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import queue
import signal
import socket
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import firwin,lfilter,resample_poly
from vivis.paths import ROOT
from vivis.speech.echo import ReferenceBuffer,EchoCanceller


def recognizer(config):
    import sherpa_onnx
    a=config['asr'];folder=ROOT/a['directory']
    extra={}
    if a.get('hotwords_score',0)>0:
        catalog=json.loads((ROOT/config.get('programs','configs/interaction/programs.json')).read_text())
        words=(set(catalog['standby_phrases'])|set(catalog.get('exit_phrases',[]))
               |set(config.get('wake_words',catalog.get('wake_words',[])))
               |{alias for p in catalog['programs'] for alias in p['aliases']})
        tokens={line.split()[0] for line in (folder/a['tokens']).read_text().splitlines() if line.strip()}
        path=ROOT/'outputs/speech/command_hotwords.txt';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('\n'.join(' '.join(w) for w in sorted(words) if all(char in tokens for char in w))+'\n')
        extra=dict(decoding_method='modified_beam_search',hotwords_file=str(path),hotwords_score=a['hotwords_score'])
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        **{name:str(folder/a[name]) for name in ('encoder','decoder','joiner','tokens')},
        num_threads=a['threads'],sample_rate=config['sample_rate_hz'],
        enable_endpoint_detection=False,**extra)


class Decoder:
    def __init__(self,config,emit,replay=False):
        self.config=config;self.engine=recognizer(config);self.stream=self.engine.create_stream()
        self.emit=emit;self.sequence=0;self.frames=0;self.last_text=''
        from vivis.speech.segment import Segmenter
        self.segmenter=Segmenter(config['segmentation'])
        self.last_changed=None;self.replay=replay
        self.finalizer=None;self.pool=None;self.slots=threading.BoundedSemaphore(2)
        if 'asr_final' in config:
            import sherpa_onnx
            final=config['asr_final'];folder=ROOT/final['directory']
            self.finalizer=sherpa_onnx.OfflineRecognizer.from_sense_voice(model=str(folder/final['model']),
                tokens=str(folder/final['tokens']),num_threads=final['threads'],language=final['language'],use_itn=False)
            self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='asr-final')
        from vivis.interaction.intents import IntentRouter
        self.router=IntentRouter(json.loads((ROOT/config.get('programs','configs/interaction/programs.json')).read_text()),
                                 wake_words=config.get('wake_words'))

    def reset(self):
        self.stream=self.engine.create_stream();self.last_text='';self.last_changed=None
        self.segmenter.reset()

    def close(self):
        if self.pool:self.pool.shutdown(wait=True,cancel_futures=False)

    def finalize(self,event,audio):
        try:
            stream=self.finalizer.create_stream();stream.accept_waveform(self.config['sample_rate_hz'],audio)
            self.finalizer.decode_stream(stream)
            final=stream.result.text.strip();streaming=event['streaming_text']
            # Both are complete-utterance hypotheses, never partial text.
            from vivis.interaction.intents import resolve_candidates
            text,source=resolve_candidates(self.router,final,streaming)
            event.update(text=text,final_text=final,recognizer=source,emitted_monotonic=time.monotonic())
            if event['text']:self.emit(event)
        except Exception as exc:self.emit(dict(kind='asr_error',error=str(exc)))
        finally:self.slots.release()

    def feed(self,pcm,stamp,metrics=None):
        self.stream.accept_waveform(self.config['sample_rate_hz'],pcm)
        while self.engine.is_ready(self.stream):self.engine.decode_stream(self.stream)
        self.frames+=len(pcm)
        text=self.engine.get_result(self.stream).strip()
        if text!=self.last_text:self.last_changed=stamp+len(pcm)/self.config['sample_rate_hz'];self.last_text=text
        segment=self.segmenter.feed(pcm,stamp,metrics)
        if segment is not None:
            self.stream.accept_waveform(self.config['sample_rate_hz'],np.zeros(8000,np.float32))
            self.stream.input_finished()
            while self.engine.is_ready(self.stream):self.engine.decode_stream(self.stream)
            text=self.engine.get_result(self.stream).strip()
            if text or self.finalizer is not None:
                summary=segment['metrics']
                self.sequence+=1;event=dict(kind='asr_final',utterance_id=self.sequence,text=text,
                    captured_end_monotonic=segment['end'],
                    speech_end_monotonic=segment['speech_end'],
                    emitted_monotonic=time.monotonic(),metrics=summary,streaming_text=text,recognizer='streaming_final')
                if self.finalizer is None:self.emit(event)
                elif self.slots.acquire(blocking=self.replay):self.pool.submit(self.finalize,event,segment['audio'])
                else:self.emit(dict(kind='asr_overload',utterance_id=self.sequence))
            self.reset()


def transcribe(path,config,emit):
    with wave.open(str(path)) as wav:
        if wav.getsampwidth()!=2:raise ValueError('Need PCM16 diagnostic audio')
        rate=wav.getframerate();channels=wav.getnchannels()
        data=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').reshape(-1,channels).astype(np.float32)/32768
    data=data[:,min(config['capture']['channel'],channels-1)]
    factor=math.gcd(rate,16000);data=resample_poly(data,16000//factor,rate//factor).astype(np.float32)
    def file_event(event):
        if event.get('kind')=='asr_final':
            event['audio_end_seconds']=event.pop('captured_end_monotonic')
            event['audio_speech_end_seconds']=event.pop('speech_end_monotonic')
        emit(event)
    input_seconds=len(data)/16000;load_started=time.monotonic()
    data=np.r_[data,np.zeros(24000,np.float32)];decoder=Decoder(config,file_event,replay=True);start=time.monotonic()
    for begin in range(0,len(data)-319,320):decoder.feed(data[begin:begin+320],begin/16000)
    decoder.close()
    return dict(seconds=input_seconds,padding_seconds=1.5,model_load_seconds=start-load_started,
                wall_seconds=time.monotonic()-start,utterances=decoder.sequence)


def live(config,emit,seconds=None):
    import sounddevice as sd
    c=config['capture'];rate=c['sample_rate_hz'];size=int(rate*c['frame_ms']/1000)
    if rate!=48000 or size!=960 or config['sample_rate_hz']!=16000:raise ValueError('Current capture contract:48k/20ms ->16k')
    devices=[i for i,d in enumerate(sd.query_devices()) if c['device'] in d['name'] and d['max_input_channels']>=c['channels']]
    if len(devices)!=1:raise RuntimeError('Select one microphone by a unique device name; matches='+str(devices))
    frames=queue.Queue(maxsize=c['queue_frames']);stop=threading.Event();dropped=[0];callback_faults=[]
    previous={s:signal.signal(s,lambda *_:stop.set()) for s in (signal.SIGINT,signal.SIGTERM)}
    def callback(data,n,timing,status):
        if status:callback_faults.append(str(status))
        stamp=time.monotonic()+float(timing.inputBufferAdcTime)-float(timing.currentTime)
        item=(stamp,data[:,c['channel']].copy())
        try:frames.put_nowait(item)
        except queue.Full:
            dropped[0]+=1
            try:frames.get_nowait()
            except queue.Empty:pass
            try:frames.put_nowait(item)
            except queue.Full:pass
    buffer=ReferenceBuffer();decoder=Decoder(config,emit);fir=firwin(63,1/3-.015)
    near_zi=np.zeros(62);far_zi=np.zeros(62);last_stamp=None;started=time.monotonic()
    try:
        with ExitStack() as stack:
            udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);stack.callback(udp.close)
            udp.bind(('127.0.0.1',config['aec']['reference_port']));udp.setblocking(False)
            echo=EchoCanceller(config['aec']) if config['aec']['enabled'] else None
            if echo:stack.callback(echo.close)
            stream=stack.enter_context(sd.InputStream(device=devices[0],samplerate=rate,
                channels=c['channels'],dtype='float32',blocksize=size,callback=callback))
            emit(dict(kind='ready',device=sd.query_devices(devices[0])['name'],aec=bool(echo),sample_rate_hz=rate))
            last_received=time.monotonic()
            while not stop.is_set() and (seconds is None or time.monotonic()-started<seconds):
                try:stamp,near=frames.get(timeout=.1)
                except queue.Empty:
                    if time.monotonic()-last_received>1.:raise RuntimeError('Microphone watchdog: no fresh capture frames')
                    continue
                now=time.monotonic()
                if now-stamp>.5:decoder.reset();continue
                last_received=now
                for _ in range(64):
                    try:data,source=udp.recvfrom(20000)
                    except BlockingIOError:break
                    if source[0]=='127.0.0.1':buffer.accept(data,now)
                if last_stamp is not None and stamp-last_stamp>1.5*size/rate:
                    decoder.reset();near_zi.fill(0);far_zi.fill(0)
                last_stamp=stamp
                far,coverage=buffer.sample(stamp-config['aec']['reference_delay_ms']/1000,size,rate)
                near,near_zi=lfilter(fir,1,near,zi=near_zi);far,far_zi=lfilter(fir,1,far,zi=far_zi)
                near=near[::3].astype(np.float32);far=far[::3].astype(np.float32)
                cleaned=echo.process(near,far) if echo else near
                rms=lambda a:float(np.sqrt(np.mean(a*a)))
                metrics=dict(raw_rms=rms(near),clean_rms=rms(cleaned),reference_rms=rms(far),reference_coverage=coverage)
                decoder.feed(cleaned,stamp,metrics)
    finally:
        decoder.close()
        for s,handler in previous.items():signal.signal(s,handler)
    return dict(dropped_capture_frames=dropped[0],callback_faults=callback_faults,utterances=decoder.sequence)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'configs/perception/speech.json')
    parser.add_argument('--transcribe',type=Path);parser.add_argument('--seconds',type=float)
    args=parser.parse_args(argv);config=json.loads(args.config.read_text())
    def emit(event):print(json.dumps(event,ensure_ascii=False,allow_nan=False),flush=True)
    result=transcribe(args.transcribe,config,emit) if args.transcribe else live(config,emit,args.seconds)
    emit(dict(kind='summary',**result))


if __name__=='__main__':main()
