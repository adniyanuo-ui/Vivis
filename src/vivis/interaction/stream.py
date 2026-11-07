"""Bounded native-A2F pipe; model work never runs in the servo/audio thread."""
from collections import deque
import json
import math
import os
import queue
from pathlib import Path
import subprocess
import threading
import time
import uuid

from vivis.interaction.animation import A2FAdapter


class LiveFrames:
    def __init__(self, binary, model, audio, duration, output, mapper, social=None,
                 capacity=180, prebuffer=120, underflow_grace_s=.2, capture=False):
        if (type(capacity) is not int or type(prebuffer) is not int
                or not 1<=prebuffer<=capacity<=600 or not .02<=underflow_grace_s<=.5):
            raise ValueError('Invalid animation queue policy')
        self.mapper,self.social=mapper,social
        self.started=time.monotonic();self.first_window_ready=None
        self.capacity,self.prebuffer,self.grace=capacity,prebuffer,underflow_grace_s
        self.playback_id=str(uuid.uuid4())
        self.adapter=A2FAdapter(mapper.order[:52],self.playback_id)
        self.condition=threading.Condition();self.stop=threading.Event()
        self.frames=deque();self.current=None;self.error=None;self.done=False
        self.generated=self.discarded=self.underflows=0
        self.last_time=None;self.smile=0.;self.max_queued=0
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.log=(self.output/'native.log').open('w')
        self.capture=(self.output/'animation.ndjson').open('w') if capture else None
        self.capture_queue=queue.Queue(maxsize=capacity)
        self.capture_dropped=0
        self.capture_error=None
        self.capture_thread=None
        self.smile_ticks=0;self.smile_peak=0.;self.social_delta_peak=0.
        try:
            self.process=subprocess.Popen([str(binary),str(model),str(audio),str(self.output/'native'),
                str(duration),'--stream',self.playback_id],stdout=subprocess.PIPE,stderr=self.log,
                text=True,bufsize=1,start_new_session=os.environ.get('VIVIS_MANAGED_SESSION')!='1')
        except BaseException:
            self.log.close()
            if self.capture:self.capture.close()
            raise
        self.thread=threading.Thread(target=self._read,name='a2f-animation-reader',daemon=True)
        if self.capture:
            self.capture_thread=threading.Thread(target=self._capture,name='animation-log-writer',daemon=True)
            self.capture_thread.start()
        self.thread.start()

    def _capture(self):
        try:
            while not self.stop.is_set() or not self.capture_queue.empty():
                try:record=self.capture_queue.get(timeout=.1)
                except queue.Empty:continue
                self.capture.write(json.dumps(record)+'\n')
        except Exception as exc:self.capture_error=str(exc)

    def _read(self):
        try:
            while not self.stop.is_set():
                line=self.process.stdout.readline(65537)
                if not line:break
                if len(line)>65536:raise ValueError('Oversize A2F pipe record')
                if not line.startswith('{'):
                    self.log.write(line)
                    continue
                packet=json.loads(line)
                if packet.get('schema_version')=='vivis-a2f-bindings-1.0':
                    self.adapter.bind(packet)
                    continue
                frame=self.adapter.adapt(packet)
                with self.condition:
                    while len(self.frames)>=self.capacity and not self.stop.is_set():
                        self.condition.wait(.1)
                    if self.stop.is_set():break
                    self.frames.append(frame);self.generated+=1
                    if self.first_window_ready is None and len(self.frames)>=self.prebuffer:self.first_window_ready=time.monotonic()
                    self.max_queued=max(self.max_queued,len(self.frames))
                    self.condition.notify_all()
            if not self.stop.is_set() and self.process.wait(timeout=5)!=0:
                raise RuntimeError('A2F process failed; see '+str(self.output/'native.log'))
        except BaseException as exc:
            with self.condition:self.error=str(exc)
        finally:
            with self.condition:self.done=True;self.condition.notify_all()

    def wait_ready(self, guard=lambda:None, timeout=30):
        deadline=time.monotonic()+timeout
        while True:
            guard()
            with self.condition:
                if self.error:raise RuntimeError(self.error)
                if len(self.frames)>=self.prebuffer or self.done and self.frames:return
                if self.done:raise RuntimeError('A2F finished without frames')
                if time.monotonic()>=deadline:raise RuntimeError('A2F first-window timeout')
                self.condition.wait(.05)

    def sample(self, audio_seconds, now):
        if not math.isfinite(audio_seconds) or audio_seconds<0 or not math.isfinite(now):
            raise ValueError('Invalid playback clock')
        used=0
        with self.condition:
            if self.error:raise RuntimeError(self.error)
            while self.frames and self.frames[0]['audio_sample_index']/16000<=audio_seconds+1e-6:
                self.current=self.frames.popleft();used+=1
            self.discarded+=max(0,used-1)
            self.condition.notify_all()
            frame=self.current
        if frame is None:raise RuntimeError('A2F has no frame at audio start')
        age=audio_seconds-frame['audio_sample_index']/16000
        if age>1/60+.002:self.underflows+=1
        if age>self.grace:raise RuntimeError('A2F animation underflow exceeds bounded hold')
        event=self.social(now) if self.social else None
        desired=0.
        if (isinstance(event,dict) and event.get('schema_version')=='vivis-smile-event-1.0'
                and type(event.get('timestamp_monotonic')) in (int,float)
                and 0<=now-event['timestamp_monotonic']<=.3):
            strength=event.get('strength')
            if type(strength) in (int,float) and math.isfinite(strength) and 0<=strength<=1:
                desired=strength
        dt=1/60 if self.last_time is None else min(.1,max(0,now-self.last_time))
        self.last_time=now
        self.smile+=(desired-self.smile)*(1-math.exp(-dt/.08))
        target=self.mapper.map_weights(frame['weights'],self.smile)
        self.smile_ticks+=int(self.smile>.01);self.smile_peak=max(self.smile_peak,self.smile)
        self.social_delta_peak=max(self.social_delta_peak,float(max((abs(v) for v in getattr(self.mapper,'last_social_delta',[])),default=0)))
        if self.capture:
            record={**frame,'submitted_audio_seconds':audio_seconds,'social_smile_strength':self.smile,
                    'fused_weights':self.mapper.last_fused,'protected_ids':self.mapper.protected_ids,
                    'motor_ids':self.mapper.ids,'nominal_positions':target.tolist()}
            try:self.capture_queue.put_nowait(record)
            except queue.Full:self.capture_dropped+=1
        return target

    def summary(self):
        return dict(kind='runtime_a2f',playback_id=self.playback_id,generated_frames=self.generated,
                    discarded_past_frames=self.discarded,underflow_ticks=self.underflows,
                    queue_capacity=self.capacity,max_queued=self.max_queued,error=self.error,
                    prebuffer_frames=self.prebuffer,first_window_ready_s=(None if self.first_window_ready is None else self.first_window_ready-self.started),
                    channel_mapping_version='a2f-skin52-v1',raw_animation_capture=bool(self.capture),
                    social_smile_channel_gain=getattr(self.mapper,'smile_strength',None),
                    source_channel_gains=dict(getattr(self.mapper,'channel_gains',{})),social_response_ticks=self.smile_ticks,
                    social_envelope_peak=self.smile_peak,social_target_delta_peak_counts=self.social_delta_peak,
                    capture_dropped=self.capture_dropped,capture_error=self.capture_error)

    def close(self):
        self.stop.set()
        with self.condition:self.condition.notify_all()
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
        self.thread.join(timeout=3)
        if self.capture_thread:self.capture_thread.join(timeout=3)
        self.process.stdout.close();self.log.close()
        if self.capture and not self.capture_thread.is_alive():self.capture.close()
