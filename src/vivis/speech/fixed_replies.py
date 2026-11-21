"""Verified, portable short replies: original PCM and canonical52 animation."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np


REPLIES={
    'song_preparation':'好的，歌曲准备中。',
    'script_preparation':'好的，剧本准备中。',
    'expression_preparation':'好的，表情跟随准备中。',
    'program_preparation':'好的，节目准备中。',
    'stop':'好的，先停下。',
    'standby':'好，您忙，我这就休息。需要时再叫我。',
    'standby_fault':'机器头待机出现故障，暂时不能运动，请检查设备。系统会定期复查。',
}
FORMAT='vivis-fixed-replies-1.0'
MAX_SECONDS=30


def signature(session):
    # Never include conversation instructions, personal context, credentials or
    # the robot's calibration. The isolated reader replaces its instructions.
    return dict(model=session['model'],voice=session['voice'],
        speech_rate=session.get('speech_rate',1.),enable_speech_emotion=session.get('enable_speech_emotion',True),
        reader_version='trusted-reading-v1',animation_version='speech-mouth-live-v1')


def digest(data):return hashlib.sha256(data).hexdigest()


def validate_frames(indices,weights,samples):
    end_sample=samples*2/3  # Cached audio is24kHz; native animation timestamps are16kHz.
    if (indices.ndim!=1 or not 1<=len(indices)<=MAX_SECONDS*61 or indices.dtype.kind not in 'iu'
            or weights.shape!=(len(indices),52) or weights.dtype.kind!='f'
            or not np.isfinite(weights).all() or np.any(weights<0) or np.any(weights>1)
            or indices[0]<0 or indices[0]>266 or np.any(np.diff(indices)<=0)
            or np.any(np.diff(indices)>800) or indices[-1]>end_sample+800
            or indices[-1]<end_sample-800):
        raise ValueError('Invalid fixed reply animation or audio coverage')


class FixedReplyCache:
    def __init__(self,directory,manifest_name='manifest.json'):
        self.directory=Path(directory);self.entries={};self.loaded={};self.error=None
        try:
            path=self.directory/manifest_name
            if path.stat().st_size>65536:raise ValueError('Oversized fixed reply manifest')
            data=json.loads(path.read_text())
            if data.get('schema_version')!=FORMAT:raise ValueError('Invalid fixed reply manifest')
            order=data['order']
            if (not isinstance(order,list) or len(order)!=52 or len(set(order))!=52
                    or any(not isinstance(n,str) or not n for n in order)):
                raise ValueError('Invalid fixed reply channel order')
            self.order=order;self.generation=data['generation'];self.entries=data['replies']
            if (not isinstance(self.entries,dict) or not set(self.entries)<=set(REPLIES)
                    or any(not isinstance(row,dict) for row in self.entries.values())):
                raise ValueError('Invalid fixed reply catalogue')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.error=type(exc).__name__;self.entries={}

    def blob(self,entry,key,maximum):
        name=entry[key]
        if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
            raise ValueError('Invalid fixed reply asset path')
        path=self.directory/name
        if not 0<path.stat().st_size<=maximum:raise ValueError('Invalid fixed reply asset size')
        raw=path.read_bytes()
        if digest(raw)!=entry[key+'_sha256']:raise ValueError('Fixed reply asset hash mismatch')
        return raw

    def get(self,text,session,order=None):
        name=next((name for name,words in REPLIES.items() if text==words),None)
        if (name not in self.entries or self.entries[name].get('text')!=text
                or self.generation!=signature(session) or order is not None and list(order)!=self.order):return None
        if name in self.loaded:return self.loaded[name]
        entry=self.entries[name]
        with wave.open(io.BytesIO(self.blob(entry,'audio',24000*MAX_SECONDS*2+4096))) as wav:
            if (wav.getnchannels()!=1 or wav.getsampwidth()!=2 or wav.getframerate()!=24000
                    or not 0<wav.getnframes()<=24000*MAX_SECONDS):raise ValueError('Invalid fixed reply WAV')
            count=wav.getnframes();pcm=wav.readframes(count)
            if len(pcm)!=count*2:raise ValueError('Truncated fixed reply WAV')
        with np.load(io.BytesIO(self.blob(entry,'animation',4*1024*1024)),allow_pickle=False) as clip:
            indices=np.asarray(clip['sample_indices']);weights=np.asarray(clip['weights'])
        validate_frames(indices,weights,count)
        samples=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
        if not np.any(samples):raise ValueError('Silent fixed reply WAV')
        audio=SimpleNamespace(samples=samples,sample_rate=24000,pcm24_bytes=pcm,mouth_order=self.order,
            mouth_frames=tuple((int(i),w.tolist()) for i,w in zip(indices,weights)))
        self.loaded[name]=audio
        return audio

    def verify(self,session,order):
        for name,text in REPLIES.items():
            if self.get(text,session,order) is None:raise ValueError('Missing/mismatched fixed reply: '+name)
        return dict(replies=len(REPLIES),voice=session['voice'],audio_s=sum(len(a.samples)/24000 for a in self.loaded.values()))
