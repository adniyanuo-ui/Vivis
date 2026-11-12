#!/usr/bin/env python3
"""Offline UVR Voc_FT inference. No microphone, speaker, camera or actuator I/O.

Optional build dependency: Jetson TensorRT Python bindings and a locally built
Voc_FT engine. STFT conventions follow the UVR MDX model (periodic Hann,
centered reflection padding, unnormalised FFT). Playback keeps the original mix.
"""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import sys
import wave

import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import resample_poly

NFFT, HOP, BINS, FRAMES = 7680, 1024, 3072, 256
CHUNK = HOP * (FRAMES - 1)
WINDOW = (.5 - .5*np.cos(2*np.pi*np.arange(NFFT)/NFFT)).astype(np.float32)


def stft(samples):
    padded=np.pad(samples,((0,0),(NFFT//2,NFFT//2)),mode='reflect')
    blocks=np.lib.stride_tricks.sliding_window_view(padded,NFFT,axis=-1)[:,::HOP,:]
    spectrum=rfft(blocks*WINDOW,axis=-1).transpose(0,2,1)
    return np.stack((spectrum.real,spectrum.imag),axis=1).reshape(1,4,NFFT//2+1,FRAMES)[:,:,:BINS].copy()


def istft(spectrum):
    pairs=spectrum.reshape(2,2,BINS,FRAMES)
    full=np.zeros((2,NFFT//2+1,FRAMES),np.complex64)
    full[:,:BINS]=pairs[:,0]+1j*pairs[:,1]
    blocks=irfft(full.transpose(0,2,1),n=NFFT,axis=-1)*WINDOW
    out=np.zeros((2,CHUNK+NFFT),np.float32);denom=np.zeros(CHUNK+NFFT,np.float32)
    for i in range(FRAMES):
        start=i*HOP;out[:,start:start+NFFT]+=blocks[:,i];denom[start:start+NFFT]+=WINDOW**2
    out/=np.maximum(denom,1e-8)
    return out[:,NFFT//2:-NFFT//2]


class Engine:
    def __init__(self,path):
        # Optional system bindings; never add them to the runtime environment.
        sys.path.append('/usr/lib/python3.12/dist-packages')
        import tensorrt as trt
        self.logger=trt.Logger(trt.Logger.ERROR);self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:raise RuntimeError('Cannot load local TensorRT engine')
        self.context=self.engine.create_execution_context()
        self.context.set_input_shape('input',(1,4,BINS,FRAMES))
        self.cuda=ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
        self.cuda.cudaMalloc.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]
        self.cuda.cudaMemcpy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
        self.cuda.cudaFree.argtypes=[ctypes.c_void_p]
        self.buffers={};self.output=None
        for i in range(self.engine.num_io_tensors):
            name=self.engine.get_tensor_name(i);shape=tuple(self.context.get_tensor_shape(name))
            if shape!=(1,4,BINS,FRAMES) or self.engine.get_tensor_dtype(name)!=trt.float32:
                raise ValueError('Unexpected Voc_FT tensor contract')
            ptr=ctypes.c_void_p();self.check(self.cuda.cudaMalloc(ctypes.byref(ptr),int(np.prod(shape))*4))
            self.buffers[name]=ptr;self.context.set_tensor_address(name,ptr.value)
            if self.engine.get_tensor_mode(name)==trt.TensorIOMode.OUTPUT:self.output=name

    @staticmethod
    def check(code):
        if code:raise RuntimeError('CUDA error '+str(code))

    def __call__(self,value):
        value=np.ascontiguousarray(value,dtype=np.float32);out=np.empty_like(value)
        self.check(self.cuda.cudaMemcpy(self.buffers['input'],value.ctypes.data,value.nbytes,1))
        if not self.context.execute_async_v3(0):raise RuntimeError('Voc_FT inference failed')
        self.check(self.cuda.cudaDeviceSynchronize())
        self.check(self.cuda.cudaMemcpy(out.ctypes.data,self.buffers[self.output],out.nbytes,2))
        if not np.isfinite(out).all():raise ValueError('Non-finite separated spectrum')
        return out

    def close(self):
        for ptr in self.buffers.values():self.check(self.cuda.cudaFree(ptr))
        self.buffers.clear()


def read_wav(path):
    with wave.open(str(path)) as f:
        if f.getsampwidth()!=2:raise ValueError('Need PCM16 WAV')
        rate=f.getframerate();channels=f.getnchannels()
        data=np.frombuffer(f.readframes(f.getnframes()),'<i2').reshape(-1,channels).T.astype(np.float32)/32768
    if channels==1:data=np.repeat(data,2,axis=0)
    if data.shape[0]!=2:raise ValueError('Need mono/stereo source')
    return data,rate


def write_wav(path,data,rate):
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate)
        f.writeframes(np.clip(np.rint(data*32768),-32768,32767).astype('<i2').tobytes())


def separate(engine,source,destination):
    mix,rate=read_wav(source);g=math.gcd(rate,44100)
    mix=resample_poly(mix,44100//g,rate//g,axis=1).astype(np.float32)
    trim=NFFT//2;usable=CHUNK-2*trim;length=mix.shape[1]
    padded=np.pad(mix,((0,0),(trim,CHUNK)))
    result=np.zeros_like(mix)
    for start in range(0,length,usable):
        value=stft(padded[:,start:start+CHUNK])
        # Polarity ensemble reduces residual model bias in quiet passages.
        pred=(engine(value)-engine(-value))*.5
        piece=istft(pred)[:,trim:-trim]*1.021
        count=min(usable,length-start);result[:,start:start+count]=piece[:,:count]
    mono=resample_poly(result.mean(axis=0),160,441)
    original,original_rate=read_wav(source)
    expected=round(original.shape[1]/original_rate*16000)
    mono=mono[:expected]
    if len(mono)!=expected:raise ValueError('Separated sample count mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True);write_wav(destination,mono,16000)
    return dict(sample_rate_hz=16000,sample_count=expected,time_origin='sample_zero',
                peak=float(np.max(np.abs(mono))),clipped_samples=int(np.sum(np.abs(mono)>1)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--audio',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();engine=Engine(a.engine)
    try:report=separate(engine,a.audio,a.output)
    finally:engine.close()
    report.update(source_sha256=hashlib.sha256(a.audio.read_bytes()).hexdigest(),
                  inference_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
