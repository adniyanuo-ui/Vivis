"""Bounded streaming voice playback and the existing microphone/AEC front end."""
from collections import deque
from contextlib import ExitStack
import queue
import socket
import threading
import time

import numpy as np
from scipy.signal import firwin,lfilter

from vivis.speech.echo import ReferencePublisher,ReferenceBuffer
from vivis.speech.voice_frontend import VoiceFrontEnd


class VoiceBuffer:
    """24k PCM -> continuous48k PCM; cancellation rejects late old-response audio."""
    def __init__(self, gain=1., fade_ms=40, max_seconds=30):
        self.gain=gain;self.fade=int(fade_ms*48);self.maximum=int(max_seconds*48000)
        self.blocks=deque();self.count=0;self.cancelled=deque(maxlen=128);self.lock=threading.Lock()
        self.filters={};self.coefficients=firwin(63,.5)*2
        self.notices=queue.SimpleQueue();self.last_audible=None;self.clipped=0
        self.started_responses=set()

    def add(self, response_id, pcm):
        if not isinstance(response_id,str) or not response_id or len(pcm)%2:raise ValueError('Invalid audio response packet')
        with self.lock:
            if response_id in self.cancelled:return False
            samples=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
            up=np.zeros(len(samples)*2,np.float32);up[::2]=samples
            values,state=lfilter(self.coefficients,1,up,zi=self.filters.get(response_id,np.zeros(62)))
            values*=self.gain
            if self.count+len(values)>self.maximum:raise RuntimeError('Playback backlog exceeded its bounded buffer')
            self.filters[response_id]=state
            self.clipped+=int(np.count_nonzero(np.abs(values)>1.))
            self.blocks.append((response_id,np.clip(values,-1,1).astype(np.float32)))
            self.count+=len(values);return True

    def finish(self, response_id):
        # End of synthesis does not discard audio still waiting to reach the DAC.
        with self.lock:self.filters.pop(response_id,None)

    def cancel(self, response_id=None):
        with self.lock:
            ids=({sid for sid,_ in self.blocks}|set(self.filters)) if response_id is None else {response_id}
            fade_left=self.fade;kept=deque();dropped=0
            for sid,values in self.blocks:
                if sid not in ids:kept.append((sid,values));continue
                take=min(fade_left,len(values)) if sid in self.started_responses else 0
                if take:
                    ramp=np.arange(fade_left,fade_left-take,-1,dtype=np.float32)/max(self.fade,1)
                    kept.append((sid,values[:take]*ramp));fade_left-=take
                dropped+=len(values)-take
            self.blocks=kept;self.count=sum(len(v) for _,v in kept)
            for sid in ids:
                if sid not in self.cancelled:self.cancelled.append(sid)
                self.filters.pop(sid,None)
            return dict(responses=sorted(ids),discarded_samples=dropped,fade_samples=self.fade-fade_left)

    def render(self, count, dac_time):
        out=np.zeros(count,np.float32);cursor=0
        with self.lock:
            while self.blocks and cursor<count:
                sid,values=self.blocks.popleft();take=min(len(values),count-cursor)
                out[cursor:cursor+take]=values[:take];self.count-=take
                if len(values)>take:self.blocks.appendleft((sid,values[take:]))
                if np.any(np.abs(values[:take])>1e-6):
                    self.started_responses.add(sid)
                    if self.last_audible!=sid:
                        self.notices.put(dict(kind='playback_started',response_id=sid,dac_monotonic=dac_time+cursor/48000))
                    self.last_audible=sid
                cursor+=take
            if self.count==0 and self.last_audible is not None:
                self.notices.put(dict(kind='playback_buffer_empty',response_id=self.last_audible,
                                      dac_monotonic=dac_time+cursor/48000))
                self.last_audible=None
        return out


class AudioIO:
    def __init__(self, speech, output_device, player, aec=True,frontend=None):
        self.speech=speech;self.output_device=output_device;self.player=player;self.aec=aec
        self.frontend=frontend or {}
        self.capture=queue.Queue(maxsize=25);self.uplink=queue.Queue(maxsize=25)
        self.stop=threading.Event();self.fault=None;self.stack=None
        self.stats=dict(captured_frames=0,processed_frames=0,callback_warnings=[],reference_covered_frames=0,
                        processing_total_s=0.,processing_max_ms=0.,capture_age_max_ms=0.,
                        raw_clipped_samples=0,uplink_clipped_samples=0,reference_missing_frames=0,
                        callback_warning_count=0,callback_warning_events=[])
        self.last_capture=None;self.last_output=None
        self.meters=queue.Queue(maxsize=2)

    def open(self):
        import sounddevice as sd
        c=self.speech['capture']
        if c['sample_rate_hz']!=48000 or c['frame_ms']!=20:raise ValueError('Voice test capture requires48kHz/20ms')
        devices=sd.query_devices()
        def unique(name,key,channels):
            matches=[i for i,d in enumerate(devices) if name in d['name'] and d[key]>=channels]
            if len(matches)!=1:raise RuntimeError('Audio device must match exactly once: '+name+'; matches='+str(matches))
            return matches[0]
        input_device=unique(c['device'],'max_input_channels',c['channels'])
        output_device=unique(self.output_device,'max_output_channels',1)
        self.stack=ExitStack()
        try:
            ref=self.speech['aec'];udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);self.stack.callback(udp.close)
            udp.bind(('127.0.0.1',ref['reference_port']));udp.setblocking(False)
            publisher=ReferencePublisher(dict(host='127.0.0.1',port=ref['reference_port'],session_id='realtime-voice'),48000)
            self.stack.callback(publisher.close)
            front=VoiceFrontEnd(self.speech,self.frontend,self.aec)
            self.stack.callback(front.close);self.stats['aec_backend']=front.backend
            def warning(stream,status):
                self.stats['callback_warning_count']+=1
                if len(self.stats['callback_warnings'])<128:
                    self.stats['callback_warnings'].append(stream+':'+str(status))
                    self.stats['callback_warning_events'].append(dict(stream=stream,status=str(status),monotonic=time.monotonic()))
            def output(data,n,timing,status):
                self.last_output=time.monotonic()
                try:
                    stamp=self.last_output+float(timing.outputBufferDacTime)-float(timing.currentTime)
                    data[:]=self.player.render(n,stamp)[:,None]
                    publisher.send(data,timing)
                except Exception as exc:data.fill(0);self.fault='Output callback: '+str(exc)
                if status:warning('output',status)
            def capture(data,n,timing,status):
                self.last_capture=time.monotonic()
                stamp=self.last_capture+float(timing.inputBufferAdcTime)-float(timing.currentTime)
                self.stats['captured_frames']+=1
                if status:warning('input',status)
                try:self.capture.put_nowait((stamp,data[:,c['channel']].copy()))
                except queue.Full:self.fault='Microphone processing backlog exceeded500ms'
            # Streams close before publisher/UDP/DSP. No serial or camera modules are imported.
            self.stack.enter_context(sd.OutputStream(device=output_device,samplerate=48000,
                channels=min(2,devices[output_device]['max_output_channels']),dtype='float32',blocksize=960,latency='low',callback=output))
            self.stack.enter_context(sd.InputStream(device=input_device,samplerate=48000,channels=c['channels'],
                dtype='float32',blocksize=960,latency='low',callback=capture))
            self.thread=threading.Thread(target=self.process,args=(udp,front),daemon=True);self.thread.start()
            return dict(microphone=devices[input_device]['name'],speaker=devices[output_device]['name'])
        except BaseException:self.close();raise

    def process(self, udp, front):
        c=self.speech['aec'];reference=ReferenceBuffer();previous=None
        try:
            while not self.stop.is_set():
                try:stamp,near=self.capture.get(timeout=.2)
                except queue.Empty:continue
                began=time.monotonic()
                self.stats['capture_age_max_ms']=max(self.stats['capture_age_max_ms'],(began-stamp)*1000)
                if time.monotonic()-stamp>.5:raise RuntimeError('Stale microphone frames exceeded500ms')
                if previous is not None and stamp-previous>.035:raise RuntimeError('Gap in microphone capture clock')
                previous=stamp
                for _ in range(64):
                    try:data,source=udp.recvfrom(20000)
                    except BlockingIOError:break
                    if source[0]=='127.0.0.1':reference.accept(data,time.monotonic())
                far,coverage=reference.sample(stamp-c['reference_delay_ms']/1000,960,48000)
                self.stats['raw_clipped_samples']+=int(np.count_nonzero(np.abs(near)>=.999))
                near,far,cleaned=front.process48(near,far)
                self.stats['uplink_clipped_samples']+=int(np.count_nonzero(np.abs(cleaned)>=.999))
                pcm=lambda x:np.clip(x*32768,-32768,32767).astype('<i2').tobytes()
                try:self.uplink.put_nowait((stamp,pcm(cleaned),pcm(near),pcm(far)))
                except queue.Full:raise RuntimeError('Network uplink backlog exceeded500ms')
                self.stats['processed_frames']+=1
                if coverage>.95:self.stats['reference_covered_frames']+=1
                else:self.stats['reference_missing_frames']+=1
                elapsed=time.monotonic()-began;self.stats['processing_total_s']+=elapsed
                self.stats['processing_max_ms']=max(self.stats['processing_max_ms'],elapsed*1000)
                if self.stats['processed_frames']%50==0:
                    rms=lambda x:float(np.sqrt(np.mean(x*x)))
                    try:self.meters.put_nowait(dict(kind='audio_meter',raw_rms=rms(near),uplink_rms=rms(cleaned),
                                                   reference_rms=rms(far),reference_coverage=coverage,
                                                   processing_ms=elapsed*1000))
                    except queue.Full:pass
        except Exception as exc:self.fault=str(exc)

    def close(self):
        self.stop.set()
        if getattr(self,'thread',None):self.thread.join(timeout=1.)
        if self.stack:self.stack.close();self.stack=None
