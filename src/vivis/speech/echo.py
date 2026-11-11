"""Timestamped DAC reference and SpeexDSP processing on the microphone clock."""
from collections import deque
import ctypes as C
import ctypes.util
import json
import math
import socket
import time

import numpy as np


class ReferencePublisher:
    """Nonblocking loopback reference from the ACTUAL output callback buffer."""
    def __init__(self,config,rate):
        if config['host']!='127.0.0.1' or not 1024<=config['port']<=65535 or not config['session_id']:
            raise ValueError('Invalid playback reference destination')
        self.address=(config['host'],config['port']);self.session=config['session_id'];self.rate=rate
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);self.socket.setblocking(False)
        self.dropped=0

    def send(self,pcm,timing,*,timestamp=None):
        stamp=(time.monotonic()+float(timing.outputBufferDacTime)-float(timing.currentTime)
               if timestamp is None else float(timestamp))
        if not math.isfinite(stamp):raise ValueError('Nonfinite DAC reference timestamp')
        mono=np.asarray(pcm,dtype=np.float32).mean(axis=1).astype('<f4')
        for start in range(0,len(mono),4096):
            part=mono[start:start+4096]
            header=dict(version=1,session_id=self.session,rate=self.rate,timestamp=stamp+start/self.rate,n=len(part))
            raw=json.dumps(header,separators=(',',':')).encode()+b'\n'+part.tobytes()
            try:self.socket.sendto(raw,self.address)
            except (OSError,BlockingIOError):self.dropped+=1

    def close(self):
        try:self.socket.sendto(json.dumps(dict(version=1,end=True,session_id=self.session,timestamp=time.monotonic())).encode()+b'\n',self.address)
        except OSError:pass
        self.socket.close()


class ReferenceBuffer:
    """Interpolate actual DAC timestamps onto ADC timestamps, including clock drift."""
    def __init__(self,seconds=3.):
        self.blocks=deque();self.seconds=seconds;self.rejected=0

    def accept(self,data,now):
        try:
            header,body=data.split(b'\n',1);h=json.loads(header)
            if h.get('end') is True:
                cutoff=h['timestamp'];session=h['session_id']
                if not math.isfinite(cutoff) or abs(now-cutoff)>self.seconds:raise ValueError('Invalid reference end')
                self.blocks=deque((begin,rate,pcm[:max(0,math.ceil((cutoff-begin)*rate-1e-7))] if sid==session else pcm,sid)
                                  for begin,rate,pcm,sid in self.blocks)
                return
            stamp,rate,n=h['timestamp'],h['rate'],h['n']
            if (h['version']!=1 or not isinstance(h['session_id'],str) or not h['session_id']
                    or type(n) is not int or not 1<=n<=4096 or len(body)!=n*4
                    or type(rate) not in (float,int) or not 8000<=rate<=192000
                    or not math.isfinite(stamp) or not -self.seconds<=now-stamp<=self.seconds):
                raise ValueError('Invalid PCM reference')
            pcm=np.frombuffer(body,dtype='<f4').copy()
            if not np.isfinite(pcm).all() or np.max(np.abs(pcm))>1.01:raise ValueError('Invalid DAC samples')
            self.append(stamp,rate,pcm,h['session_id'],now)
        except (ValueError,KeyError,TypeError,AttributeError):self.rejected+=1

    def append(self,stamp,rate,pcm,session,now):
        self.blocks.append((stamp,rate,pcm,session))
        while self.blocks and self.blocks[0][0]<now-self.seconds:self.blocks.popleft()

    def sample(self,stamp,count,rate):
        times=stamp+np.arange(count)/rate;result=np.zeros(count,np.float32);covered=np.zeros(count,bool)
        blocks=[b for b in self.blocks if len(b[2])]
        for i,(begin,sr,data,session) in enumerate(blocks):
            period=1/sr;extension=None
            if i+1<len(blocks):
                end,_,after,next_session=blocks[i+1]
                if (next_session==session and end>begin
                        and abs(end-(begin+len(data)/sr))<.002):
                    # The next DAC anchor estimates the actual device clock.
                    # Include its first PCM sample; never hold the last sample
                    # through a packet seam or accumulate nominal-rate drift.
                    period=(end-begin)/len(data);extension=after[0]
            limit=len(data) if extension is not None else len(data)-1
            # Only overlapping blocks need a full per-sample index/mask. A3s
            # history otherwise allocates ~150 unnecessary arrays every20ms.
            if count and (times[-1]<begin-period*1e-7 or times[0]>begin+limit*period+period*1e-7):
                continue
            index=(times-begin)/period;mask=(index>=-1e-7)&(index<limit-1e-7 if extension is not None else index<=limit+1e-7)
            if mask.any():
                values=np.r_[data,extension] if extension is not None else data
                result[mask]=np.interp(index[mask],np.arange(len(values)),values)
                covered[mask]=True
        return result,float(covered.mean())


class EchoCanceller:
    def __init__(self,config,frame_size=320):
        path=ctypes.util.find_library('speexdsp')
        if path is None:raise RuntimeError('AEC requires libspeexdsp1: sudo apt-get install libspeexdsp1')
        self.lib=C.CDLL(path);self.size=frame_size;rate=config['sample_rate_hz']
        self.lib.speex_echo_state_init.argtypes=[C.c_int,C.c_int];self.lib.speex_echo_state_init.restype=C.c_void_p
        self.lib.speex_echo_state_destroy.argtypes=[C.c_void_p]
        self.lib.speex_echo_ctl.argtypes=[C.c_void_p,C.c_int,C.c_void_p]
        self.lib.speex_echo_cancellation.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p]
        self.lib.speex_preprocess_state_init.argtypes=[C.c_int,C.c_int];self.lib.speex_preprocess_state_init.restype=C.c_void_p
        self.lib.speex_preprocess_state_destroy.argtypes=[C.c_void_p]
        self.lib.speex_preprocess_ctl.argtypes=[C.c_void_p,C.c_int,C.c_void_p]
        self.lib.speex_preprocess_run.argtypes=[C.c_void_p,C.c_void_p]
        self.echo=self.lib.speex_echo_state_init(frame_size,int(rate*config['tail_ms']/1000))
        self.pre=self.lib.speex_preprocess_state_init(frame_size,rate)
        if not self.echo or not self.pre:raise RuntimeError('SpeexDSP initialization failed')
        value=C.c_int(rate);self.lib.speex_echo_ctl(self.echo,24,C.byref(value))
        self.lib.speex_preprocess_ctl(self.pre,24,self.echo)
        value=C.c_int(config['noise_suppress_db']);self.lib.speex_preprocess_ctl(self.pre,18,C.byref(value))
        value=C.c_int(0);self.lib.speex_preprocess_ctl(self.pre,2,C.byref(value))  # Disable AGC.

    def process(self,near,far):
        if len(near)!=self.size or len(far)!=self.size:raise ValueError('AEC needs complete20ms frames')
        n=np.clip(np.asarray(near)*32768,-32768,32767).astype(np.int16)
        f=np.clip(np.asarray(far)*32768,-32768,32767).astype(np.int16);out=np.empty(self.size,np.int16)
        self.lib.speex_echo_cancellation(self.echo,n.ctypes.data,f.ctypes.data,out.ctypes.data)
        self.lib.speex_preprocess_run(self.pre,out.ctypes.data)
        return out.astype(np.float32)/32768

    def close(self):
        if self.pre:self.lib.speex_preprocess_state_destroy(self.pre);self.pre=None
        if self.echo:self.lib.speex_echo_state_destroy(self.echo);self.echo=None
