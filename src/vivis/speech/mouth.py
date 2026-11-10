"""Bounded cloud-PCM A2F worker. No microphone, camera or actuator access."""
from collections import deque
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np
from scipy.signal import firwin,lfilter

from vivis.interaction.animation import A2FAdapter


class SpeechMouth:
    def __init__(self,binary,model,path,order,secret_env=(),buffer_seconds=30):
        self.path=Path(path);self.order=order[:52];self.lock=threading.Lock()
        self.stop=threading.Event();self.commands=queue.Queue(maxsize=256)
        self.responses={};self.adapter=None;self.current=None;self.heads=deque(maxlen=128)
        self.error=None;self.warning=None;self.prebuffer_timeouts=0;self.frames=0;self.published=0;self.gated=0
        self.filters={};self.received={};self.coefficients=firwin(63,1/3)*2
        self.frame_limit=int(buffer_seconds*60)+122
        self.log=self.path.with_suffix('.native.log').open('w')
        self.process=subprocess.Popen([str(binary),str(model),'--live',str(self.path.with_suffix('.native'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,
            env={k:v for k,v in os.environ.items() if k not in secret_env})
        self.threads=[threading.Thread(target=f,daemon=True) for f in (self.write,self.read,self.publish)]
        for thread in self.threads:thread.start()

    def enqueue(self,record):
        if self.error:return
        try:self.commands.put_nowait(record)
        except queue.Full:self.error='A2F input queue full; mouth disabled, voice/control continue'

    def add(self,rid,pcm):
        if self.error:return
        if rid!=self.current:
            self.current=rid;self.filters.clear();self.received.clear()
            with self.lock:
                # Keep current and previous DAC owner, each covering the PCM horizon.
                if len(self.responses)>=2:self.responses.pop(next(iter(self.responses)))
                self.responses[rid]=dict(frames=deque(maxlen=self.frame_limit),done=False,created=time.monotonic())
            self.enqueue(dict(kind='start',playback_id=rid))
        values=np.frombuffer(pcm,'<i2').astype(float)/32768
        up=np.zeros(len(values)*2);up[::2]=values
        state,offset=self.filters.get(rid,(np.zeros(62),0))
        filtered,state=lfilter(self.coefficients,1,up,zi=state)
        output=np.clip(filtered[(-offset)%3::3],-1,1)
        self.filters[rid]=(state,(offset+len(up))%3)
        self.received[rid]=self.received.get(rid,0)+len(output)
        for start in range(0,len(output),4000):
            self.enqueue(dict(kind='audio',playback_id=rid,samples=output[start:start+4000].tolist()))

    def finish(self,rid):
        if rid==self.current:self.enqueue(dict(kind='finish',playback_id=rid))

    def ready(self,rid,end_sample):
        if self.error:return True  # Voice remains usable when optional animation fails.
        with self.lock:
            r=self.responses.get(rid)
            if r and r.get('prebuffer_error'):return True
            if r and r['frames'] and (r['done'] or r['frames'][-1][0]>=end_sample/3+266):
                r['wait_since']=None;return True
            if r:
                now=time.monotonic();frontier=r['frames'][-1][0] if r['frames'] else -1
                if r.get('wait_since') is None or frontier!=r.get('wait_frontier'):
                    r['wait_since']=now;r['wait_frontier']=frontier
                elif now-r['wait_since']>8:
                    r['prebuffer_error']='A2F prebuffer stalled for8seconds'
                    self.prebuffer_timeouts+=1
                    self.warning='A2F本句口型数据连续停滞超过8秒；本句不等待口型，后续句继续重试。response_id='+rid
                    return True
        self.gated+=1;return False

    def played(self,rid,start,end,dac):
        # Audio callback: only a tiny in-memory clock record, no pipe/disk/model.
        with self.lock:self.heads.append((rid,start,end,dac))

    def deactivate(self):
        with self.lock:self.heads.clear()

    def weights(self,now):
        if self.error:return None
        with self.lock:
            if not self.heads:return None
            while len(self.heads)>1 and self.heads[1][3]<=now:self.heads.popleft()
            rid,start,end,dac=self.heads[0]
            if now<dac or now>dac+(end-start)/48000+.06:return None
            r=self.responses.get(rid)
            if not r or r.get('prebuffer_error') or not r['frames']:return None
            sample=min(end,max(start,start+(now-dac)*48000))/3
            frames=r['frames']
            while len(frames)>2 and frames[1][0]<=sample:frames.popleft()
            a=frames[0];b=frames[1] if len(frames)>1 else a
            fraction=max(0.,min(1.,(sample-a[0])/max(1,b[0]-a[0])))
            return [x+(y-x)*fraction for x,y in zip(a[1],b[1])]

    def write(self):
        try:
            while not self.stop.is_set():
                try:record=self.commands.get(timeout=.1)
                except queue.Empty:continue
                self.process.stdin.write(json.dumps(record,separators=(',',':'))+'\n');self.process.stdin.flush()
        except Exception as exc:
            if not self.stop.is_set():self.error='A2F input: '+str(exc)

    def read(self):
        try:
            for line in self.process.stdout:
                if not line.startswith('{'):continue
                packet=json.loads(line);rid=packet.get('playback_id')
                if packet.get('schema_version')=='vivis-a2f-bindings-1.0':
                    self.adapter=A2FAdapter(self.order,rid);self.adapter.bind(packet)
                elif packet.get('kind')=='finished':
                    with self.lock:
                        if rid in self.responses:self.responses[rid]['done']=True
                else:
                    frame=self.adapter.adapt(packet)
                    with self.lock:
                        if rid in self.responses:
                            self.responses[rid]['frames'].append((frame['audio_sample_index'],frame['weights']))
                            self.frames+=1
            if not self.stop.is_set():self.error='A2F process ended; see '+str(self.path.with_suffix('.native.log'))
        except Exception as exc:
            if not self.stop.is_set():self.error='A2F output: '+str(exc)

    def publish(self):
        while not self.stop.wait(1/60):
            try:
                now=time.monotonic();weights=self.weights(now)
                packet=dict(schema_version='vivis-speech-mouth-1.0',timestamp_monotonic=now,
                    order=self.order,weights=weights,error=self.error,warning=self.warning,prebuffer_timeouts=self.prebuffer_timeouts)
                temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(packet));os.replace(temp,self.path)
                self.published+=1
            except Exception as exc:self.error='Mouth publish: '+str(exc)

    def close(self):
        self.deactivate();self.stop.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
        for thread in self.threads:thread.join(timeout=2)
        self.process.stdin.close();self.process.stdout.close();self.log.close()


class MouthReceiver:
    """Runtime-side validated coefficients; the existing loop alone sends goals."""
    def __init__(self,path,order):self.path=Path(path);self.order=order[:52];self.active_ticks=0

    def sample(self,now):
        try:
            raw=self.path.read_bytes()
            if len(raw)>16384:return None
            packet=json.loads(raw);weights=packet['weights']
            if (packet['schema_version']!='vivis-speech-mouth-1.0' or packet['order']!=self.order or packet.get('error')
                    or not 0<=now-packet['timestamp_monotonic']<=.2
                    or not isinstance(weights,list) or len(weights)!=52
                    or any(type(w) not in (int,float) or not np.isfinite(w) or not 0<=w<=1 for w in weights)):return None
            self.active_ticks+=1;return weights
        except (OSError,ValueError,KeyError,TypeError):return None
