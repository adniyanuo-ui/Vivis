"""实时表情跟随；可选眼球、颈部联合跟踪，不使用音频或A2F。"""
import argparse
import csv
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.developer import check_competitors,vision_service
from vivis.runtime.show import prepare,prepare_round,recovery_clearance_plan,unload_tongue
from vivis.runtime.homing import recovery_policy
from vivis.runtime.playback import play
from vivis.runtime.cancellation import operator_cancelled
from vivis.motion.mapping import compile_trajectory,read_json
from vivis.motion.ownership import PreviewPublisher,MojiaPlaybackGaze
from vivis.motion.upstream import configure as configure_trajectory
from vivis.vision.parallel_follow import configure as configure_parallel
from vivis.vision.independent_follow import configure as configure_independent
from vivis.vision.policy import TargetReceiver
from vivis.interaction.animation import FrameMapper
from vivis.interaction.expression import ExpressionFrames


class MonotonicClock:
    """The existing control loop's clock contract, with no sound device opened."""
    def __init__(self,duration):
        self.duration=duration;self.fault=None;self.started=None
    def start(self):self.started=time.monotonic()
    def time(self):return None if self.started is None else time.monotonic()-self.started
    def close(self):pass


def plan(robot,settings,seconds):
    if type(seconds) not in (float,int) or not math.isfinite(seconds) or not 1<=seconds<=600:
        raise ValueError('Duration must be finite1..600 seconds')
    tracking_enabled=settings.get('tracking_enabled',False)
    if (settings.get('schema_version')!='vivis-expression-follow-1.0' or settings.get('control_hz')!=60
            or settings.get('face_selection')!='largest' or type(tracking_enabled) is not bool):
        raise ValueError('Unsupported expression-follow configuration')
    mode='eyes-neck' if tracking_enabled else 'face'
    robot.require_mode(mode)
    profile=robot.motion_profile()
    profile['exclude_held_axes_from_session']=True
    with robot.resolve('channels').open(encoding='utf-8-sig',newline='') as f:
        order=next(csv.reader(f))[2:]
    shapes={s['shape']:{p['id']:p['position'] for p in s['positions']}
            for s in read_json(robot.calibration)['blendShapes']}
    face_ids,zero,_=compile_trajectory(np.zeros((1,61)),order,shapes,profile,scope='face')
    config=robot.tracking()
    if robot.definition['defaults'].get('hold_vertical_eyes'):
        held={a['id'] for a in config['control']['axes'] if a['coordinate']=='y'}
        config['control']['vertical_enabled']=False
        config['control']['held_eye_ids']=sorted(held)
        config['control']['axes']=[a for a in config['control']['axes'] if a['id'] not in held]
    # Perception owns only face/eyelid mapping. Add neutral eye slots for the
    # existing gaze owner, which overrides those slots in the ONE control loop.
    input_ids=list(face_ids)
    if mode!='face':
        motors={m['id']:m for m in profile['motors']}
        eye_ids=[a['id'] for a in config['control']['axes']]
        input_ids+=eye_ids
        zero=np.column_stack([zero,*[np.full(len(zero),motors[sid]['neutral']) for sid in eye_ids]])
        indices=np.argsort(input_ids);input_ids=sorted(input_ids);zero=zero[:,indices]
    profile['diagnostic_command_ids']=list(input_ids)
    neutral=np.repeat(zero,math.ceil(seconds*60),axis=0)
    config['social_smile']={'enabled':False}
    perception=read_json(ROOT/settings['perception'])
    config['face_expression']={k:perception[k] for k in ('model','sha256')}
    config['face_expression'].update(enabled=True,face_selection=settings['face_selection'])
    configure_trajectory(config,mode)
    configure_parallel(config,mode)
    configure_independent(config,mode)
    authorize=robot.definition['defaults'].get('authorized_tongue_observation',False)
    profile,ids,neutral=prepare(config,profile,input_ids,neutral,authorize,1.,robot.telemetry(),
                              mode,'lips-direct',audio_required=False)
    profile.update(face_update_hz=60,control_hz=60,snapshot_chunk_bytes=2)
    home=robot.home(mode)
    home['axes']=[a for a in home['axes'] if a['id'] in ids]
    if set(a['id'] for a in home['axes'])!=set(ids) or set(ids)!=set(face_ids)|set(profile['gaze_owner_ids']):
        raise ValueError('Expression follow must prepare, monitor and command exactly its owned inventory')
    mapper=FrameMapper(order,shapes,profile,ids,smile_strength=0.,scope='face')
    ExpressionFrames(SimpleNamespace(max_age=config['transport']['max_age_s']),mapper,settings)
    return dict(profile=profile,ids=ids,neutral=neutral,tracking=config,perception=perception,
                home=home,mapper=mapper,authorize=authorize,mode=mode)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/interaction/expression_follow.json')
    parser.add_argument('--seconds',type=float,default=60,help='回中后跟随时长1..600秒')
    parser.add_argument('--tracking',choices=('on','off'),help='on叠加眼球和颈部位置跟随；默认off保持纯表情模式')
    parser.add_argument('--gain',type=float,help='全部表情通道的统一增益，默认1.0')
    parser.add_argument('--mirror-expression',choices=('on','off'),help='on交换左右表情通道；默认off为同侧解剖映射')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--session-stop-file',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--compact-logs',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--overlap-warmup',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--home-speed',type=int,default=60,help='Single-session homing cap in counts/s')
    parser.add_argument('--check',action='store_true',help='离线检查，不打开相机、串口、音频或模型')
    args=parser.parse_args(argv)
    if not 1<=args.home_speed<=32767:parser.error('Invalid home speed')
    settings=read_json(args.config)
    if args.gain is not None:settings['gain']=args.gain
    if args.mirror_expression is not None:settings['mirror_expression']=args.mirror_expression=='on'
    if args.tracking is not None:settings['tracking_enabled']=args.tracking=='on'
    robot=Robot(args.robot)
    p=plan(robot,settings,args.seconds)
    profile,ids,home,mode=p['profile'],p['ids'],p['home'],p['mode']
    if args.session_stop_file is not None:profile['session_stop_file']=str(args.session_stop_file.resolve())
    if args.compact_logs:profile.update(retain_event_limit=600,capture_serial_trace=False)
    required=[ROOT/p['perception'][k] for k in ('python','model')]
    if args.check:
        print(json.dumps(dict(offline_valid=True,program='expression-follow',robot=robot.definition['name'],tracking_mode=mode,
            command_ids=ids,home_ids=[a['id'] for a in home['axes']],
            held_eye_ids=profile['held_eye_ids'],held_neck_ids=profile['held_neck_ids'],
            audio_opened=False,a2f_used=False,hardware_opened=False,
            missing_dependencies=[str(x) for x in required if not x.is_file()]),ensure_ascii=False))
        return 0
    for path in required:
        if not path.is_file():parser.error('缺少共享视觉依赖：'+str(path)+'；运行python3 tools/setup_face_landmarker.py')
    check_competitors(True)
    output=args.output or ROOT/'outputs/expression_follow'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    for name,data in [('tracking',p['tracking']),('settings',settings),('profile',profile),('home',home)]:
        (output/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    motors={m['id']:m for m in profile['motors']}
    startup=robot.startup(active_ids=motors,inventory_ids=ids)
    home_args=SimpleNamespace(home_speed=args.home_speed,home_only=False,usb_volume_times_three=False,
        authorized_tongue_observation=p['authorize'],home_position_recovery=recovery_policy(home.get('disabled_position_recovery')),
        home_sensor_fault_confirmation=home.get('sensor_fault_confirmation',False))
    clearance=recovery_clearance_plan(home,motors)
    stop=False;stop_file=ROOT/'outputs/STOP'
    def request_stop(*_):
        nonlocal stop
        stop=True
    def guard():
        if stop or stop_file.exists() or args.session_stop_file is not None and args.session_stop_file.exists():
            raise RuntimeError('Operator stop')
    handlers={sig:signal.signal(sig,request_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    summary=dict(completed=False,program='expression-follow',robot=robot.definition['name'],audio_opened=False,
                 a2f_used=False,command_ids=ids,eye_neck_held=mode=='face',tracking_mode=mode,overlap_warmup=args.overlap_warmup)
    status=ROOT/'outputs/runtime/control_status.json';status.parent.mkdir(parents=True,exist_ok=True)
    publisher=receiver=source=None;report={}
    def publish(state,error=None):
        publisher.submit(dict(active=False,program='expression-follow',state=state,error=error,
            updated_monotonic=time.monotonic(),command_ids=ids,eye_neck_held=mode=='face',tracking_mode=mode))
    lock_path=ROOT/'outputs/developer/control.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    print('节目：实时表情跟随（面部＋眼睑）；'+('叠加眼球、颈部跟随' if mode!='face' else '眼球、颈部保持原位')+'；不播放音频。',flush=True)
    print('日志：'+str(output),flush=True)
    try:
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            publisher=PreviewPublisher(status)
            try:
                options={'wait_ready':False} if args.overlap_warmup else {}
                with vision_service(output/'tracking.json',output,True,python=str(ROOT/p['perception']['python']),**options) as vision:
                    receiver=TargetReceiver(p['tracking'])
                    def preparing_guard():
                        guard()
                        if vision is not None:vision.check()
                    if not args.overlap_warmup:receiver.wait_ready()
                    else:print('WARMUP START 视觉预热与自检回中并行；回中后才启用跟随',flush=True)
                    guard();print('PREPARE '+mode,flush=True)
                    prepare_round(home_args,profile,ids,home['axes'],startup,clearance,robot.telemetry(),
                                  preparing_guard if args.overlap_warmup else guard,report,output,publish)
                    if args.overlap_warmup:
                        if vision is not None:vision.wait_ready(guard)
                        receiver.wait_ready();guard()
                        print('WARMUP READY 视觉与回中均已就绪',flush=True)
                    # One preview publisher and one receiver are shared by both
                    # sources. Gaze uses the existing reviewed controller only.
                    gaze=MojiaPlaybackGaze(p['tracking'],profile,ids,receiver,neck=True) if mode!='face' else None
                    source=ExpressionFrames(receiver,p['mapper'],settings,publisher,args.seconds,tracking_mode=mode)
                    guard();print(f'EXPRESSION FOLLOW START {args.seconds:g}s',flush=True)
                    result=play(profile,ids,p['neutral'],60,None,output/'playback_log.json',stop_file,
                                frame_source=source,gaze_override=gaze,clock_factory=lambda:MonotonicClock(args.seconds))
                    summary['completed']=result['completed']
                    print('EXPRESSION FOLLOW COMPLETE',flush=True)
            finally:
                if receiver:receiver.close()
                if p['authorize'] and 'hardware_restore' in report:unload_tongue(profile,report)
                error=publisher.finish(dict(active=False,program='expression-follow',state='stopped',
                    updated_monotonic=time.monotonic(),command_ids=ids,eye_neck_held=mode=='face',tracking_mode=mode))
                if error:summary['preview_error']=error
    except BaseException as exc:
        summary['error']=str(exc)
        if operator_cancelled(exc):return 130
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        if source:summary['expression']=source.summary()
        (output/'preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
