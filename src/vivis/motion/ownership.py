"""Single-owner Mojia music and live eye/neck arbitration; no serial ownership."""
import copy
import json
import time
import threading
from pathlib import Path
import numpy as np

from vivis.motion.mapping import ROOT
from vivis.vision.pursuit import validate_mojia, MojiaPursuit, NeckCommandScheduler,neck_command_limits


class PreviewPublisher:
    """Latest-only asynchronous preview: filesystem latency cannot block motion."""
    def __init__(self,path):
        self.path=path;self.pending=None;self.closing=False;self.error=None
        self.wake=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True,name='mojia-preview-status')
        self.thread.start()

    def submit(self,data):
        self.pending=data
        self.wake.set()

    def run(self):
        while True:
            self.wake.wait();self.wake.clear()
            data=self.pending
            try:
                temporary=self.path.with_suffix('.tmp')
                temporary.write_text(json.dumps(data)+'\n')
                temporary.replace(self.path)
            except Exception as exc:self.error=str(exc)
            if self.closing and self.pending is data:return

    def finish(self,data):
        self.pending=data;self.closing=True;self.wake.set()
        self.thread.join(timeout=2)
        return self.error or ('Preview writer did not finish within 2 seconds' if self.thread.is_alive() else None)


def tracking_profile(config, profile, ids, neck=True):
    validate_mojia(config,profile,Path(profile.get('calibration_path',ROOT/'configs/robots/mojia/calibration.json')),True,True)
    p=copy.deepcopy(profile)
    motors={m['id']:m for m in p['motors']}
    owned=[a['id'] for a in config['control']['axes']]+([32] if neck else [])
    if not set(owned)<=set(ids):raise ValueError('Missing Mojia tracking axes')
    for a in config['control']['axes']:
        motors[a['id']].update(safe_min=min(a['negative'],a['positive']),
                              safe_max=max(a['negative'],a['positive']),
                              max_speed_counts_s=a.get('speed_counts_s',config['control']['speed_counts_s']))
    n=config['neck']
    low,high=neck_command_limits(config,profile)
    motors[32].update(safe_min=low,safe_max=high,max_speed_counts_s=n['speed_counts_s'])
    held=set(config['control'].get('held_eye_ids',[]))
    if not held<={7,22} or held&set(owned):raise ValueError('Invalid held eye ownership')
    p['diagnostic_command_ids']=sorted((set(profile['diagnostic_command_ids'])|({32} if neck else set()))-held)
    p['held_eye_ids']=sorted(held)
    p['held_neck_ids']=[33,34] if neck else [32,33,34]
    p['gaze_owner_ids']=owned
    return p


class MojiaPlaybackGaze:
    """Own only IDs8/23/7/22/32; preserve every other prerecorded target."""
    def __init__(self, config, profile, ids, receiver, status_path=None, round_number=1, repeat=1, neck=True):
        validate_mojia(config,profile,Path(profile.get('calibration_path',ROOT/'configs/robots/mojia/calibration.json')),True,True)
        self.config=config;self.ids=list(ids);self.receiver=receiver
        self.command_ids=list(profile['diagnostic_command_ids'])
        self.neck=neck
        self.axes=list(config['control']['axes'])+([dict(id=32)] if neck else [])
        self.owned=[a['id'] for a in self.axes]
        self.columns=[self.ids.index(sid) for sid in self.owned]
        motors={m['id']:m for m in profile['motors']}
        expected=tracking_profile(config,profile,ids,neck=neck)
        expected_motors={m['id']:m for m in expected['motors']}
        for sid in self.owned:
            for key in ('safe_min','safe_max','max_speed_counts_s'):
                if motors[sid][key]!=expected_motors[sid][key]:
                    raise ValueError('Mojia tracking limits/speeds must match the saved preset')
        self.speeds=[motors[sid]['max_speed_counts_s'] for sid in self.owned]
        self.controller=MojiaPursuit(config,self.owned,self.speeds,neck,
                                     neck_bounds=(motors[32]['safe_min'],motors[32]['safe_max']))
        self.camera_eye_id=self.controller.camera_eye_id
        self.previous=None;self.scheduler=None;self.last_diagnostic={};self.speed_overrides={}
        self.status_path=status_path;self.last_publish=0.;self.round=round_number;self.repeat=repeat
        self.publisher=PreviewPublisher(status_path) if status_path is not None else None
        self.audio_seconds=0.;self.duration=0.;self.feedback=None
        self.returning=False;self.telemetry_warnings=[]

    def prepare(self, trajectory, measured, snapshots, now):
        for s in snapshots:
            if s['id'] not in self.owned:continue
            raw=bytes.fromhex(s['config_0_39_hex'])
            if not raw[18]&4:raise RuntimeError('Tracking speed unit changed')
            if s['id']==32:
                ram=bytes.fromhex(s['state_40_70_hex']);n=self.config['neck']
                if (list(raw[21:24])!=n['observed_pid_raw'] or list(raw[26:28])!=n['observed_deadband_counts']
                        or ram[1]!=n['observed_acceleration_raw']):
                    raise RuntimeError('ID32 no longer matches saved hardware preset')
        self.previous=np.asarray(measured,dtype=float)[self.columns].copy()
        self.controller.observe_eye(self.previous[self.owned.index(self.camera_eye_id)],now)
        if self.neck:
            self.scheduler=NeckCommandScheduler(self.previous[self.owned.index(32)],now,self.speeds[-1],
                adaptive_speed=self.config['neck']['adaptive_speed'],endpoints=self.controller.neck_bounds,
                filter_tau_s=self.config['neck'].get('target_filter_tau_s',0.))
        # Approach the song's first face pose while holding the tracked axes at
        # their measured starting pose; no fast recenter jump between rounds.
        result=np.array(trajectory,copy=True)
        result[:,self.columns]=self.previous
        return result

    def observe(self,sid,state,now):
        self.feedback=dict(id=sid,**state)
        if sid==self.camera_eye_id:self.controller.observe_eye(state['position'],now)

    def set_audio_time(self,seconds,duration):
        self.audio_seconds,self.duration=seconds,duration

    def apply(self,base,last,dt,now):
        self.returning=self.returning or self.audio_seconds>=max(0.,self.duration-self.controller.return_window(self.previous[-1]))
        returning=self.returning
        applied_neck=self.scheduler.filtered_position if self.neck and self.scheduler is not None else None
        self.previous,d=self.controller.step(self.receiver.poll(now),self.previous,dt,now,
                                             returning=returning,applied_neck=applied_neck)
        result=np.array(base,copy=True);result[self.columns]=self.previous
        speed=None
        if self.neck:
            neck_column=self.ids.index(32)
            d['planned_neck_goal']=float(self.previous[-1])
            goal,speed=self.scheduler.step(self.previous[-1],now,returning=d['returning_to_center'])
            result[neck_column]=goal
            d['smoothed_neck_goal']=self.scheduler.filtered_position
        d['neck_speed_command']=speed
        self.speed_overrides={self.config['neck']['id']:speed} if self.neck else {}
        self.last_diagnostic=d
        if now-self.last_publish>=.2:
            self.publish(dict(active=True,state=d['state'],command_ids=self.command_ids,
                goals={sid:int(round(result[j])) for j,sid in enumerate(self.ids) if sid in self.command_ids},feedback=self.feedback,
                remaining_seconds=max(0.,self.duration-self.audio_seconds),round_number=self.round,repeat=self.repeat,
                neck_speed_command=speed,neck_assist_reason=d['neck_assist_reason'],image_error_x=d['image_error_x']))
            self.last_publish=now
        return result

    def publish(self,data):
        if self.publisher is not None:
            self.publisher.submit(dict(updated_monotonic=time.monotonic(),telemetry_warnings=self.telemetry_warnings,**data))

    def finish(self,completed,error):
        if self.publisher is not None:
            return self.publisher.finish(dict(updated_monotonic=time.monotonic(),active=False,
                state='completed' if completed else 'stopped',error=error,command_ids=self.command_ids,
                round_number=self.round,repeat=self.repeat))
