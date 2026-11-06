"""Low-frequency blinking and small wandering gaze, without camera/person tracking."""
import argparse
import csv
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import random
import signal
from types import SimpleNamespace

import numpy as np
from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.show import prepare_round,recovery_clearance_plan,unload_tongue
from vivis.runtime.homing import recovery_policy
from vivis.runtime.playback import play
from vivis.runtime.cancellation import operator_cancelled
from vivis.interaction.follow import MonotonicClock
from vivis.interaction.animation import FrameMapper
from vivis.motion.mapping import compile_trajectory,read_json,live_preflight


class IdleFrames:
    def __init__(self,mapper,settings,seed=None):
        self.mapper,self.c=mapper,settings;self.rng=random.Random(seed)
        for key in ('blink_interval_s','look_interval_s'):
            values=settings[key]
            if len(values)!=2 or not all(type(v) in (int,float) and math.isfinite(v) for v in values) or not 2<=values[0]<=values[1]<=120:
                raise ValueError('Invalid standby interval')
        if (not .2<=settings['blink_duration_s']<=2 or not 0<=settings['blink_peak']<=1
                or not 2<=settings['look_duration_s']<=10 or not 0<=settings['eye_yaw_span_counts']<=60
                or not 0<=settings['neck_yaw_span_counts']<=32):raise ValueError('Invalid standby motion')
        self.next_blink=self.rng.uniform(*settings['blink_interval_s']);self.blink=None
        self.next_look=self.rng.uniform(*settings['look_interval_s']);self.look=None;self.direction=1
        self.blinks=self.looks=0

    @staticmethod
    def smooth(value):return value**3*(10+value*(-15+6*value))

    def sample(self,t,now):
        c=self.c;weights=[0.]*52
        if t>=self.next_blink:
            self.blink=t;self.next_blink=t+self.rng.uniform(*c['blink_interval_s']);self.blinks+=1
        if self.blink is not None:
            u=(t-self.blink)/c['blink_duration_s']
            if u<1:
                phase=u/.4 if u<.4 else (1-u)/.6
                value=c['blink_peak']*self.smooth(max(0.,min(1.,phase)))
                for name in ('EyeBlinkLeft','EyeBlinkRight'):weights[self.mapper.order.index(name)]=value
        target=self.mapper.map_weights(weights)
        if t>=self.next_look:
            self.look=t;self.next_look=t+self.rng.uniform(*c['look_interval_s'])
            self.direction=self.rng.choice((-1,1));self.looks+=1
        blend=0.
        if self.look is not None:
            u=(t-self.look)/c['look_duration_s']
            if u<1:blend=self.smooth(2*u if u<.5 else 2*(1-u))*self.direction
        motors={m['id']:m for m in self.mapper.profile['motors']}
        for sid,span in ((8,c['eye_yaw_span_counts']),(23,c['eye_yaw_span_counts']),(32,c['neck_yaw_span_counts'])):
            if sid in self.mapper.ids:
                m=motors[sid];target[self.mapper.ids.index(sid)]=np.clip(m['neutral']+span*blend,m['safe_min'],m['safe_max'])
        return target


def plan(robot,settings,seconds):
    if settings.get('schema_version')!='vivis-standby-1.0' or not 1<=seconds<=3600:raise ValueError('Invalid standby configuration/duration')
    profile=robot.motion_profile()
    with robot.resolve('channels').open(encoding='utf-8-sig',newline='') as f:order=next(csv.reader(f))[2:]
    shapes={s['shape']:{p['id']:p['position'] for p in s['positions']} for s in read_json(robot.calibration)['blendShapes']}
    face,_,_=compile_trajectory(np.zeros((1,61)),order,shapes,profile,scope='face')
    eligible=set(face)|{7,8,22,23,32,33,34};home=robot.home('eyes-neck')
    home['axes']=[a for a in home['axes'] if a['id'] in eligible]
    ids=sorted(a['id'] for a in home['axes']);profile['motors']=[m for m in profile['motors'] if m['id'] in ids]
    if set(ids)!={m['id'] for m in profile['motors']}:raise ValueError('Standby ownership mismatch')
    speed=settings['home_speed_counts_s']
    if type(speed) is not int or not 1<=speed<=min(m['max_speed_counts_s'] for m in profile['motors']):
        raise ValueError('Standby home speed exceeds canonical motor caps')
    profile.update(diagnostic_command_ids=ids,held_eye_ids=[],held_neck_ids=[],gaze_owner_ids=[],
        developer_mode='face',scheduled_feedback=True,post_goal_quiet_ms=1,snapshot_chunk_bytes=2,
        face_update_hz=60,control_hz=60,capture_serial_trace=False,retain_event_limit=600)
    live_preflight(profile,ids,audio_required=False)
    mapper=FrameMapper(order,shapes,profile,ids,smile_strength=0.,scope='face')
    source=IdleFrames(mapper,settings)
    neutral=np.repeat(mapper.neutral[None,:],math.ceil(seconds*60),axis=0)
    return profile,ids,home,mapper,source,neutral


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/interaction/standby.json')
    parser.add_argument('--seconds',type=float,default=600);parser.add_argument('--session-stop-file',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv);robot=Robot(args.robot);settings=read_json(args.config)
    profile,ids,home,mapper,source,neutral=plan(robot,settings,args.seconds)
    if args.session_stop_file is not None:profile['session_stop_file']=str(args.session_stop_file.resolve())
    if args.check:
        print(json.dumps(dict(offline_valid=True,ids=ids,camera_opened=False,hardware_opened=False),ensure_ascii=False));return 0
    output=args.output or ROOT/'outputs/standby'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True,exist_ok=False)
    stopped=False;stop_file=ROOT/'outputs/STOP'
    def stop(*_):
        nonlocal stopped
        stopped=True
    def guard():
        if stopped or stop_file.exists() or args.session_stop_file is not None and args.session_stop_file.exists():raise RuntimeError('Operator stop')
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    motors={m['id']:m for m in profile['motors']};authorize=robot.definition['defaults'].get('authorized_tongue_observation',False)
    startup=robot.startup(active_ids=motors,inventory_ids=ids)
    home_args=SimpleNamespace(home_speed=settings['home_speed_counts_s'],home_only=True,usb_volume_times_three=False,
        authorized_tongue_observation=authorize,home_position_recovery=recovery_policy(home.get('disabled_position_recovery')),
        home_sensor_fault_confirmation=home.get('sensor_fault_confirmation',False))
    report={};result=dict(completed=False,camera_opened=False,program='standby')
    lock_path=ROOT/'outputs/developer/control.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                prepare_round(home_args,profile,ids,home['axes'],startup,recovery_clearance_plan(home,motors),robot.telemetry(),
                    guard,report,output,lambda *_:None)
                guard();print('STANDBY START: low-frequency blink, small glances, no person tracking',flush=True)
                playback=play(profile,ids,neutral,60,None,output/'playback_log.json',stop_file,
                    frame_source=source,clock_factory=lambda:MonotonicClock(args.seconds))
                result['completed']=playback['completed']
            finally:
                if authorize and 'hardware_restore' in report:unload_tongue(profile,report)
    except Exception as exc:
        result['error']=str(exc)
        if operator_cancelled(exc):return 130
        raise
    finally:
        for sig,h in handlers.items():signal.signal(sig,h)
        result.update(blinks=source.blinks,glances=source.looks)
        (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        (output/'preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
