"""P0: runtime PCM-to-face and observed smile, using the existing motion owner."""
import argparse
import copy
from contextlib import ExitStack
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
from types import SimpleNamespace
import time
import wave

import numpy as np

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.developer import check_competitors, vision_service
from vivis.runtime.show import prepare,prepare_round,recovery_clearance_plan,unload_tongue
from vivis.runtime.homing import recovery_policy
from vivis.runtime.playback import play
from vivis.runtime.cancellation import operator_cancelled
from vivis.motion.mapping import compile_trajectory,read_json,live_preflight
from vivis.motion.ownership import MojiaPlaybackGaze
from vivis.vision.policy import TargetReceiver
from vivis.motion.upstream import configure as configure_trajectory
from vivis.vision.parallel_follow import configure as configure_parallel
from vivis.vision.independent_follow import configure as configure_independent
from vivis.interaction.animation import FrameMapper,speech_channel_gains
from vivis.interaction.stream import LiveFrames


def audio_info(path):
    with wave.open(str(path)) as f:
        if f.getsampwidth()!=2:raise ValueError('P0 audio must be PCM16 WAV')
        return f.getnframes(),f.getframerate(),f.getnchannels()


def model_audio(path, output):
    frames,rate,channels=audio_info(path)
    if rate==16000 and channels==1:return Path(path)
    from scipy.signal import resample_poly
    with wave.open(str(path)) as f:
        data=np.frombuffer(f.readframes(frames),dtype='<i2').reshape(-1,channels).astype(np.float64).mean(axis=1)
    factor=math.gcd(rate,16000)
    data=resample_poly(data,16000//factor,rate//factor)
    destination=Path(output)/'inference_pcm16_16000.wav'
    with wave.open(str(destination),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
        f.writeframes(np.clip(np.rint(data),-32768,32767).astype('<i2').tobytes())
    return destination


def inference_source(work):
    """Choose verified, sample-aligned model audio; never replace playback PCM."""
    playback=work['audio'];path=work.get('inference_audio',playback)
    if 'inference_audio' not in work:return path
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=work.get('inference_audio_sha256'):raise ValueError('Inference audio hash mismatch')
    report_path=work.get('inference_alignment')
    if report_path is None or hashlib.sha256(report_path.read_bytes()).hexdigest()!=work.get('inference_alignment_sha256'):
        raise ValueError('Missing or changed inference alignment report')
    report=read_json(report_path)
    n,rate,_=audio_info(path);play_n,play_rate,_=audio_info(playback)
    if (report.get('schema_version')!='vivis-inference-audio-alignment-1.0'
            or report.get('time_origin')!='sample_zero' or report.get('sample_rate_hz')!=16000
            or report.get('sample_count')!=round(n/rate*16000)
            or abs(n/rate-play_n/play_rate)>1/16000
            or report.get('inference_sha256')!=digest or report.get('playback_sha256')!=work['audio_sha256']):
        raise ValueError('Inference/playback alignment does not match these audio files')
    return path


def plan(robot, work_name, mode, settings, smile=False):
    work=robot.work(work_name)
    audio=work['audio']
    if hashlib.sha256(audio.read_bytes()).hexdigest()!=work['audio_sha256']:
        raise ValueError('Playback audio hash mismatch')
    inference=inference_source(work)
    frames,rate,_=audio_info(audio);duration=frames/rate
    profile=robot.motion_profile()
    with robot.resolve('channels').open(encoding='utf-8-sig',newline='') as f:
        order=next(csv.reader(f))[2:]
    source_gains=speech_channel_gains(settings,order)
    calibration=read_json(robot.calibration)
    shapes={s['shape']:{p['id']:p['position'] for p in s['positions']} for s in calibration['blendShapes']}
    mapping_scope='face' if mode=='face' else 'face_eyes'
    face_ids,zero,_=compile_trajectory(np.zeros((1,61)),order,shapes,profile,scope=mapping_scope)
    # A neutral scaffold defines duration/ownership only; every song target
    # comes from runtime A2F. No expression or target JSON is loaded.
    neutral=np.repeat(zero,max(1,math.ceil(duration*60)),axis=0)
    config=robot.tracking(device='unused' if mode=='face' else None)
    if robot.definition['defaults'].get('hold_vertical_eyes'):
        held={a['id'] for a in config['control']['axes'] if a['coordinate']=='y'}
        config['control']['vertical_enabled']=False
        config['control']['held_eye_ids']=sorted(held)
        config['control']['axes']=[a for a in config['control']['axes'] if a['id'] not in held]
    config['social_smile']=copy.deepcopy(settings['social_smile'])
    perception=read_json(ROOT/settings['perception'])
    config['social_smile'].update(model=perception['model'],sha256=perception['sha256'])
    config['social_smile']['enabled']=smile
    configure_trajectory(config,mode)
    configure_parallel(config,mode,None)
    configure_independent(config,mode)
    authorize=robot.definition['defaults'].get('authorized_tongue_observation',False)
    profile,ids,neutral=prepare(config,profile,face_ids,neutral,authorize,1.,robot.telemetry(),mode,'lips-direct')
    profile.update(face_update_hz=60,control_hz=60,snapshot_chunk_bytes=2)
    live_preflight(profile,ids)
    # Validate social configuration during --check, before opening any device.
    FrameMapper(order,shapes,profile,ids,settings['smile_strength'],
        channel_gains=source_gains,social_channels=settings.get('social_channels'),scope=mapping_scope)
    return dict(work=work,audio=audio,inference_audio=inference,duration=duration,profile=profile,ids=ids,neutral=neutral,
                tracking=config,order=order,shapes=shapes,authorize=authorize,perception=perception,source_gains=source_gains,
                mapping_scope=mapping_scope)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json')
    parser.add_argument('--work',default='0001')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/interaction/p0.json')
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--mode',choices=('face','eyes','eyes-neck'),default='eyes-neck')
    parser.add_argument('--smile',choices=('on','off'),default='off',help='节目默认关闭实时微笑叠加；on仅作显式对照')
    parser.add_argument('--smile-strength',type=float,help='One-run social smile increment0..0.5; saved default unchanged')
    parser.add_argument('--mouth-speed',type=int)
    parser.add_argument('--jaw-open-gain',type=float,help='One-run absolute JawOpen source gain; default saved mouth/jaw gain')
    parser.add_argument('--round-lip-gain',type=float,help='One-run absolute Funnel/Pucker source gain; default saved round-lip gain')
    parser.add_argument('--inference-audio',type=Path,help='Optional sample-aligned dry voice; caller must verify alignment')
    parser.add_argument('--capture-animation',action='store_true',help='Opt-in bounded asynchronous animation trace')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--session-stop-file',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--audio-reference-port',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--session-id',default='',help=argparse.SUPPRESS)
    parser.add_argument('--soft-audio-cancel-ms',type=int,default=0,help=argparse.SUPPRESS)
    parser.add_argument('--compact-logs',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--overlap-warmup',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--check',action='store_true',help='Offline configuration checks; no devices/models opened')
    args=parser.parse_args(argv)
    if not 1<=args.repeat<=20:parser.error('Repeat must be1..20')
    if args.mode=='face' and args.smile=='on':parser.error('Smile requires the single camera/target service; use eyes or eyes-neck')
    settings=read_json(args.config)
    if settings.get('schema_version')!='vivis-interactive-singing-1.0':parser.error('Unsupported P0 configuration')
    for argument,key in (('jaw_open_gain','jaw_open_channel_gain'),('round_lip_gain','round_lip_channel_gain')):
        value=getattr(args,argument)
        if value is not None:
            if not math.isfinite(value) or not 0<value<=3:parser.error(argument.replace('_','-')+' must be finite in (0,3]')
            settings[key]=value
    if args.smile_strength is not None:
        if not math.isfinite(args.smile_strength) or not 0<=args.smile_strength<=.5:parser.error('Smile strength must be finite0..0.5')
        settings['smile_strength']=args.smile_strength
    robot=Robot(args.robot);robot.require_mode(args.mode)
    if args.mouth_speed is not None:robot.set_mouth_speed(args.mouth_speed)
    p=plan(robot,args.work,args.mode,settings,args.smile=='on')
    profile,ids,config=p['profile'],p['ids'],p['tracking']
    if not 0<=args.soft_audio_cancel_ms<=100:parser.error('Invalid audio cancellation fade')
    if args.soft_audio_cancel_ms:profile['soft_audio_cancel_ms']=args.soft_audio_cancel_ms
    if args.compact_logs:profile.update(retain_event_limit=600,capture_serial_trace=False)
    if args.session_stop_file is not None:profile['session_stop_file']=str(args.session_stop_file.resolve())
    if args.audio_reference_port is not None:
        if not 1024<=args.audio_reference_port<=65535 or not args.session_id:parser.error('Invalid audio reference session')
        profile['audio_reference']=dict(host='127.0.0.1',port=args.audio_reference_port,session_id=args.session_id)
    inference=args.inference_audio or p['inference_audio']
    n,rate,_=audio_info(inference)
    if abs(n/rate-p['duration'])>1/16000:parser.error('Inference/playback durations differ; need sample-aligned sources')
    required=[ROOT/settings['binary'],ROOT/settings['model']]
    if args.smile=='on':required += [ROOT/p['perception']['python'],ROOT/p['perception']['model']]
    if args.check:
        print(json.dumps(dict(offline_valid=True,source='runtime_a2f',prerecorded_animation_read=False,
            robot=robot.definition['name'],audio=str(p['audio']),duration_seconds=p['duration'],
            inference_audio=str(inference),inference_audio_override=args.inference_audio is not None,
            commanded=profile['diagnostic_command_ids'],source_channel_gains=p['source_gains'],social_smile_enabled=args.smile=='on',social_smile_strength=settings['smile_strength'],
            missing_dependencies=[str(x) for x in required if not x.is_file()],
            motion_hardware_opened=False),ensure_ascii=False))
        return 0
    for path in required:
        if not path.is_file():parser.error('Missing P0 dependency: '+str(path)+'; run tools/setup_p0.py')
    check_competitors(args.mode!='face')
    print(f'TUNING 实时歌曲：其余口部通道×{settings.get("mouth_channel_gain",1.):g}；张嘴绝对×{p["source_gains"]["JawOpen"]:g}；圆唇绝对×{p["source_gains"]["MouthFunnel"]:g}（不叠乘）；微笑回应={args.smile}（增量{settings["smile_strength"]:g}）',flush=True)
    print(f'AUDIO INPUT A2F：{inference}；播放：{p["audio"]}',flush=True)
    output=args.output or ROOT/'outputs/p0'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    config_path=output/'tracking.json';config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    (output/'interaction_settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n')
    (output/'profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
    inference=model_audio(inference,output)
    home=robot.home(args.mode);motors={m['id']:m for m in profile['motors']}
    if profile.get('exclude_held_axes_from_session'):
        held=set(profile.get('held_eye_ids',[]))|set(profile.get('held_neck_ids',[]))
        home['axes']=[a for a in home['axes'] if a['id'] not in held]
    home_ids={a['id'] for a in home['axes']}
    if not set(ids)<=home_ids:raise ValueError('P0 home/session inventory mismatch')
    startup=robot.startup(active_ids=motors,inventory_ids=home_ids)
    home_args=SimpleNamespace(home_speed=60,home_only=False,
        usb_volume_times_three=robot.definition['defaults'].get('usb_volume_times_three',False),
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
    summary=dict(completed=False,source='runtime_a2f',prerecorded_animation_read=False,rounds=[],overlap_warmup=args.overlap_warmup,
        inference_audio=str(inference),inference_audio_sha256=hashlib.sha256(inference.read_bytes()).hexdigest(),playback_audio=str(p['audio']))
    status=ROOT/'outputs/runtime/control_status.json';status.parent.mkdir(parents=True,exist_ok=True)
    def publish(state,error=None):
        temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(dict(active=False,state=state,error=error,
            updated_monotonic=time.monotonic(),command_ids=profile['diagnostic_command_ids']))+'\n');temporary.replace(status)
    lock_path=ROOT/'outputs/developer/control.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with ExitStack() as resources:
            lock=resources.enter_context(lock_path.open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            resources.push(lambda _,exc,__:publish('stopped',str(exc) if exc is not None else None))
            vision_python=str(ROOT/p['perception']['python']) if args.smile=='on' else None
            options={'wait_ready':False} if args.overlap_warmup else {}
            with vision_service(config_path,output,args.mode!='face',python=vision_python,**options) as vision:
                for lap in range(1,args.repeat+1):
                    guard();directory=output/f'round_{lap}';directory.mkdir()
                    receiver=TargetReceiver(config) if args.mode!='face' else None
                    source=None
                    report={}
                    try:
                        if receiver and not args.overlap_warmup:receiver.wait_ready()
                        def social(now):
                            if receiver is None:return None
                            packet=receiver.poll(now)
                            return packet.get('social_smile') if packet else None
                        mapper=FrameMapper(p['order'],p['shapes'],profile,ids,settings['smile_strength'],
                            channel_gains=p['source_gains'],social_channels=settings.get('social_channels'),scope=p['mapping_scope'])
                        source=LiveFrames(ROOT/settings['binary'],ROOT/settings['model'],inference,p['duration'],
                            directory/'a2f',mapper,social if args.smile=='on' else None,
                            settings['buffer_frames'],settings['prebuffer_frames'],settings['underflow_grace_s'],args.capture_animation)
                        def preparing_guard():
                            guard()
                            if vision is not None:vision.check()
                            if source.error:raise RuntimeError(source.error)
                        if not args.overlap_warmup:source.wait_ready(guard)
                        else:print('WARMUP START 视觉/A2F预热与自检回中并行；回中后才播放',flush=True)
                        print(f'ROUND {lap}/{args.repeat} PREPARE runtime-A2F',flush=True)
                        prepare_round(home_args,profile,ids,home['axes'],startup,clearance,robot.telemetry(),
                                      preparing_guard if args.overlap_warmup else guard,report,directory,publish)
                        if args.overlap_warmup:
                            if vision is not None:vision.wait_ready(guard)
                            if receiver:receiver.wait_ready()
                            source.wait_ready(guard);guard()
                            print('WARMUP READY 视觉、A2F与回中均已就绪',flush=True)
                        gaze=MojiaPlaybackGaze(config,profile,ids,receiver,status,lap,args.repeat,
                            neck=args.mode=='eyes-neck') if receiver else None
                        guard();print(f'ROUND {lap}/{args.repeat} START runtime-A2F + smile={args.smile}',flush=True)
                        result=play(profile,ids,p['neutral'],60,p['audio'],directory/'playback_log.json',stop_file,
                                    gaze_override=gaze,frame_source=source)
                        summary['rounds'].append(dict(completed=result['completed'],animation=source.summary()))
                    finally:
                        if source:
                            source.close()
                            (directory/'animation_summary.json').write_text(json.dumps(source.summary(),indent=2)+'\n')
                        if receiver:receiver.close()
                        if p['authorize'] and 'hardware_restore' in report:unload_tongue(profile,report)
                        (directory/'preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                summary['completed']=True
    except BaseException as exc:
        summary['error']=str(exc)
        if operator_cancelled(exc):return 130
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
