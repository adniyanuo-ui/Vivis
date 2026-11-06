"""Streaming20ms speech segmentation with prefix retention; no ASR or device I/O."""
from collections import deque
import numpy as np


class Segmenter:
    def __init__(self,config,detector=None):
        if detector is None:
            import webrtcvad
            vad=webrtcvad.Vad(config['aggressiveness'])
            detector=lambda pcm:vad.is_speech(np.clip(pcm*32768,-32768,32767).astype('<i2').tobytes(),16000)
        self.detector=detector;self.c=config;self.prefix=deque(maxlen=config['prefix_frames'])
        self.reset()

    def reset(self):
        self.prefix.clear();self.frames=[];self.voiced=self.silent=0;self.started=False;self.last_voice=None

    def feed(self,pcm,stamp,metrics=None):
        if len(pcm)!=320:raise ValueError('Speech segmenter requires16kHz20ms frames')
        speech=self.detector(pcm);frame=(pcm.copy(),stamp,metrics or {})
        if not self.started:
            self.prefix.append(frame)
            self.voiced=self.voiced+1 if speech else 0
            if self.voiced>=self.c['min_speech_frames']:
                self.started=True;self.frames=list(self.prefix);self.prefix.clear();self.last_voice=stamp+.02
            return None
        self.frames.append(frame)
        if speech:self.last_voice=stamp+.02;self.silent=0
        else:self.silent+=1
        if self.silent<self.c['silence_frames'] and len(self.frames)<self.c['max_frames']:return None
        samples=self.frames;keys=samples[-1][2].keys()
        result=dict(audio=np.concatenate([p for p,_,_ in samples]),start=samples[0][1],end=stamp+.02,
                    speech_end=self.last_voice,metrics={k:float(np.mean([m.get(k,0.) for _,_,m in samples])) for k in keys})
        self.reset();return result
