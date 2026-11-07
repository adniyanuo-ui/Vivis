"""待机：有人时眼颈跟随与微笑回应，无人时低频眨眼和轻微游视。"""
import argparse
import copy
import csv
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import numpy as np
from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.show import prepare,prepare_round,recovery_clearance_plan,unload_tongue
from vivis.runtime.developer import check_competitors,vision_service
from vivis.runtime.homing import recovery_policy
from vivis.runtime.playback import play
from vivis.runtime.cancellation import operator_cancelled
from vivis.interaction.follow import MonotonicClock
from vivis.interaction.animation import FrameMapper
from vivis.motion.mapping import compile_trajectory,read_json,live_preflight,direct_mouth_ids
from vivis.motion.ownership import MojiaPlaybackGaze,PreviewPublisher
from vivis.motion.upstream import configure as configure_trajectory
from vivis.vision.parallel_follow import configure as configure_parallel
from vivis.vision.independent_follow import configure as configure_independent
from vivis.vision.policy import TargetReceiver,validate_packet
from vivis.interaction.smile import SmileGate,selected_face_allowed


class IdleFrames:
    def __init__(self,mapper,settings,seed=None,tracking=None,perception=None):
        self.mapper,self.c=mapper,settings;self.rng=random.Random(seed)
        self.tracking,self.perception=tracking,perception
        self.receiver=self.publisher=None;self.smile=0.;self.last_time=None;self.last_publish=-math.inf
        self.missing_since=None;self.missing_origin=0.
        self.smile_ticks=0;self.smile_peak=0.;self.social_delta_by_id={}
        self.response=settings.get('social_response',{'enabled':False})
        self.tracking_enabled=settings.get('tracking_enabled',False)
        if type(self.tracking_enabled) is not bool or type(self.response.get('enabled')) is not bool:
            raise ValueError('Invalid standby vision switch')
        if self.response['enabled']:
            SmileGate(self.response['gate'])
            selected_face_allowed(self.response,0)
            tau=self.response['filter_tau_s']
            if type(tau) not in (int,float) or not math.isfinite(tau) or not .01<=tau<=1:
                raise ValueError('Invalid standby response filter')
        self.response_tau=self.response['filter_tau_s'] if self.response['enabled'] else .08
        self.missing_hold=self.response.get('missing_hold_s',0.)
        self.missing_release=self.response.get('missing_release_s',.08)
        if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (self.missing_hold,self.missing_release))
                or not 0<=self.missing_hold<=1 or not .05<=self.missing_release<=3):
            raise ValueError('Invalid standby missing-smile release')
        self.glance_tau=settings.get('glance_blend_tau_s',.25)
        if type(self.glance_tau) not in (int,float) or not math.isfinite(self.glance_tau) or not .05<=self.glance_tau<=2:
            raise ValueError('Invalid standby glance blend')
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
        desired=0.;observed=False
        if self.receiver is not None and self.response['enabled']:
            packet=self.receiver.poll(now)
            try:
                validate_packet(packet,now,self.receiver.max_age)
                event=packet['social_smile'];strength=event['strength'];stamp=event['timestamp_monotonic']
                if (packet['target'] is not None and event.get('schema_version')=='vivis-smile-event-1.0'
                        and event.get('valid') is True and type(strength) in (int,float)
                        and math.isfinite(strength) and 0<=strength<=1
                        and type(stamp) in (int,float) and stamp==packet['timestamp_monotonic']):
                    desired=strength;observed=True
            except (TypeError,ValueError,KeyError,AttributeError):pass
        if observed:self.missing_since=None
        else:
            if self.missing_since is None:self.missing_since=now;self.missing_origin=self.smile
            u=max(0.,min(1.,(now-self.missing_since-self.missing_hold)/self.missing_release))
            desired=self.missing_origin*(1-self.smooth(u))
        dt=1/60 if self.last_time is None else max(0.,min(.1,now-self.last_time));self.last_time=now
        self.smile+=(desired-self.smile)*(-math.expm1(-dt/self.response_tau))
        target=self.mapper.map_weights(weights,self.smile)
        self.smile_ticks+=int(self.smile>.01);self.smile_peak=max(self.smile_peak,self.smile)
        for sid,delta in zip(self.mapper.ids,self.mapper.last_social_delta):
            self.social_delta_by_id[sid]=max(self.social_delta_by_id.get(sid,0.),abs(float(delta)))
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
        if not self.tracking_enabled:self.publish(now,target,dict(state='idle',tracking_active=False))
        return target

    def publish(self,now,target,diagnostic):
        if self.publisher is not None and now-self.last_publish>=.2:
            self.publisher.submit(dict(active=True,program='standby',updated_monotonic=now,
                **diagnostic,command_ids=self.mapper.profile['diagnostic_command_ids'],social_smile_strength=self.smile,
                social_response_ticks=self.smile_ticks,nominal_goals=dict(zip(self.mapper.ids,map(float,target)))))
            self.last_publish=now

    def summary(self):
        return dict(blinks=self.blinks,glances=self.looks,tracking_enabled=self.tracking_enabled,
            smile_enabled=self.response['enabled'],smile_channel_gain=self.mapper.smile_strength,
            social_response_ticks=self.smile_ticks,social_envelope_peak=self.smile_peak,
            social_target_delta_peak_by_id=self.social_delta_by_id)


class StandbyGaze(MojiaPlaybackGaze):
    """Existing pursuit owns eyes/neck; quiet glances fade in after neutral return.

    Small idle offsets are intents, not synthetic camera/encoder observations.
    They are bounded by the same command envelope and sent by the one runtime.
    """
    def __init__(self,config,profile,ids,receiver,source):
        super().__init__(config,profile,ids,receiver,neck=True)
        self.source=source;self.glance_blend=0.
        self.neutral=source.mapper.neutral
        self.low=np.array([m['safe_min'] for sid in ids for m in profile['motors'] if m['id']==sid])
        self.high=np.array([m['safe_max'] for sid in ids for m in profile['motors'] if m['id']==sid])

    def apply(self,base,last,dt,now):
        result=super().apply(base,last,dt,now)
        if not np.isfinite(result).all() or np.any(result<self.low) or np.any(result>self.high):
            raise ValueError('Standby pursuit target exceeds command limits')
        d=self.last_diagnostic
        centered=np.max(abs(result[self.columns]-self.neutral[self.columns]))<1.
        idle=d['state']=='idle' and centered and not self.returning
        self.glance_blend+=(float(idle)-self.glance_blend)*(-math.expm1(-dt/self.source.glance_tau))
        # Only the already bounded horizontal idle intents differ from neutral.
        result[self.columns]+=(np.asarray(base)[self.columns]-self.neutral[self.columns])*self.glance_blend
        result[self.columns]=np.clip(result[self.columns],self.low[self.columns],self.high[self.columns])
        d.update(standby_glance_blend=self.glance_blend,tracking_active=d['state'] in ('tracking','holding'))
        self.source.publish(now,result,d)
        return result


def plan(robot,settings,seconds):
    if (settings.get('schema_version')!='vivis-standby-1.0' or type(seconds) not in (int,float)
            or not math.isfinite(seconds) or not 1<=seconds<=3600):raise ValueError('Invalid standby configuration/duration')
    profile=robot.motion_profile()
    with robot.resolve('channels').open(encoding='utf-8-sig',newline='') as f:order=next(csv.reader(f))[2:]
    shapes={s['shape']:{p['id']:p['position'] for p in s['positions']} for s in read_json(robot.calibration)['blendShapes']}
    tracking_enabled=settings.get('tracking_enabled',False)
    response=settings.get('social_response',{'enabled':False})
    if (type(tracking_enabled) is not bool or not isinstance(response,dict)
            or type(response.get('enabled')) is not bool):raise ValueError('Invalid standby vision switch')
    needs_camera=tracking_enabled or response.get('enabled',False)
    config=robot.tracking(device=None if needs_camera else 'unused') if needs_camera else None
    perception=None
    if config is not None:
        if robot.definition['defaults'].get('hold_vertical_eyes'):
            held={a['id'] for a in config['control']['axes'] if a['coordinate']=='y'}
            config['control'].update(vertical_enabled=False,held_eye_ids=sorted(held))
            config['control']['axes']=[a for a in config['control']['axes'] if a['id'] not in held]
        config['social_smile']={'enabled':False};config['face_expression']={'enabled':False}
        if response.get('enabled'):
            perception=read_json(ROOT/response['perception'])
            config['social_smile']=dict(enabled=True,gate=copy.deepcopy(response['gate']),face_selection=response.get('face_selection','single'),
                **{k:perception[k] for k in ('model','sha256')})
        mode='eyes-neck' if tracking_enabled else 'face'
        configure_trajectory(config,mode);configure_parallel(config,mode);configure_independent(config,mode)
    face,zero,_=compile_trajectory(np.zeros((1,61)),order,shapes,profile,scope='face_eyes' if tracking_enabled else 'face')
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
    profile['mouth_command_mode']=profile.get('default_mouth_command_mode','limited')
    profile['direct_position_ids']=direct_mouth_ids(profile) if profile['mouth_command_mode']=='lips-direct' else []
    if tracking_enabled:
        profile['diagnostic_command_ids']=face
        profile,ids,_=prepare(config,profile,face,zero,False,1.,robot.telemetry(),'eyes-neck',
            profile['mouth_command_mode'],audio_required=False)
        home['axes']=[a for a in home['axes'] if a['id'] in ids]
    if set(ids)!={a['id'] for a in home['axes']}:raise ValueError('Standby home ownership mismatch')
    live_preflight(profile,ids,audio_required=False)
    mapper=FrameMapper(order,shapes,profile,ids,smile_strength=response.get('strength',0.) if response.get('enabled') else 0.,
        social_channels=response.get('channels'),scope='face',reserve_speech_resources=False)
    source=IdleFrames(mapper,settings,tracking=config,perception=perception)
    neutral=np.repeat(mapper.neutral[None,:],math.ceil(seconds*60),axis=0)
    return profile,ids,home,mapper,source,neutral


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/interaction/standby.json')
    parser.add_argument('--seconds',type=float,default=600);parser.add_argument('--session-stop-file',type=Path)
    parser.add_argument('--tracking',choices=('on','off'),help='默认按待机配置开启眼颈跟随')
    parser.add_argument('--smile',choices=('on','off'),help='默认按待机配置开启微笑回应')
    parser.add_argument('--smile-strength',type=float,help='单次微笑增量0..0.5，不保存配置')
    parser.add_argument('--overlap-warmup',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--output',type=Path);parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv);robot=Robot(args.robot);settings=read_json(args.config)
    if args.smile_strength is not None and (not math.isfinite(args.smile_strength) or not 0<=args.smile_strength<=.5):
        parser.error('微笑增量须为有限数0..0.5')
    if args.tracking is not None:settings['tracking_enabled']=args.tracking=='on'
    if args.smile is not None:settings.setdefault('social_response',{})['enabled']=args.smile=='on'
    if args.smile_strength is not None:settings.setdefault('social_response',{})['strength']=args.smile_strength
    profile,ids,home,mapper,source,neutral=plan(robot,settings,args.seconds)
    needs_camera=source.tracking_enabled or source.response['enabled']
    required=[ROOT/source.perception[k] for k in ('python','model')] if source.perception else []
    if args.session_stop_file is not None:profile['session_stop_file']=str(args.session_stop_file.resolve())
    if args.check:
        print(json.dumps(dict(offline_valid=True,ids=ids,camera_opened=False,hardware_opened=False,
            tracking_enabled=source.tracking_enabled,smile_enabled=source.response['enabled'],
            smile_channel_gain=mapper.smile_strength,a2f_used=False,audio_opened=False,
            missing_dependencies=[str(x) for x in required if not x.is_file()]),ensure_ascii=False));return 0
    for path in required:
        if not path.is_file():parser.error('缺少待机视觉依赖：'+str(path)+'；运行python3 tools/setup_face_landmarker.py')
    check_competitors(needs_camera)
    output=args.output or ROOT/'outputs/standby'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True,exist_ok=False)
    for name,data in [('settings',settings),('tracking',source.tracking),('profile',profile),('home',home)]:
        (output/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
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
    report={};result=dict(completed=False,camera_opened=False,program='standby',audio_opened=False,a2f_used=False)
    receiver=publisher=None
    lock_path=ROOT/'outputs/developer/control.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status=ROOT/'outputs/runtime/control_status.json';status.parent.mkdir(parents=True,exist_ok=True)
            publisher=PreviewPublisher(status);source.publisher=publisher
            try:
                options={'wait_ready':False} if args.overlap_warmup else {}
                python=str(ROOT/source.perception['python']) if source.perception else None
                with vision_service(output/'tracking.json',output,needs_camera,python=python,**options) as vision:
                    if needs_camera:
                        receiver=TargetReceiver(source.tracking);source.receiver=receiver
                        result['camera_opened']=True
                    def preparing_guard():
                        guard()
                        if vision is not None:vision.check()
                    if receiver is not None and not args.overlap_warmup:receiver.wait_ready()
                    prepare_round(home_args,profile,ids,home['axes'],startup,recovery_clearance_plan(home,motors),robot.telemetry(),
                        preparing_guard,report,output,lambda *_:None)
                    if receiver is not None and args.overlap_warmup:
                        if vision is not None:vision.wait_ready(guard)
                        receiver.wait_ready();guard()
                    gaze=StandbyGaze(source.tracking,profile,ids,receiver,source) if source.tracking_enabled else None
                    guard();print(f'STANDBY START: tracking={source.tracking_enabled}, smile={source.response["enabled"]}, gain={mapper.smile_strength:g}',flush=True)
                    print('待机：有人时眼颈跟随、微笑回应；无人时回中后低频眨眼和轻微游视。',flush=True)
                    playback=play(profile,ids,neutral,60,None,output/'playback_log.json',stop_file,
                        frame_source=source,gaze_override=gaze,clock_factory=lambda:MonotonicClock(args.seconds))
                    result['completed']=playback['completed']
            finally:
                if receiver is not None:receiver.close()
                if authorize and 'hardware_restore' in report:unload_tongue(profile,report)
                error=publisher.finish(dict(active=False,program='standby',state='stopped',updated_monotonic=time.monotonic()))
                if error:result['preview_error']=error
    except Exception as exc:
        result['error']=str(exc)
        if operator_cancelled(exc):return 130
        raise
    finally:
        for sig,h in handlers.items():signal.signal(sig,h)
        result.update(source.summary())
        (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        (output/'preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
