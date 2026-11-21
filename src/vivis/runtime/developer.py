"""Three developer examples sharing one motion runtime and one serial owner."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.motion.mapping import direct_mouth_ids
from vivis.vision.policy import VERSION


def compatible_vision(state, config, digest):
    """A matching file hash alone does not prove a usable target stream."""
    if not isinstance(state,dict):return False
    service_age=state.get('service_age_s')
    age=state.get('frame_age_ms')
    maximum=config['transport']['max_age_s']
    features=state.get('target_features',[])
    if config.get('parallel_neck',{}).get('enabled') and (not isinstance(features,list) or 'eye_line_roll' not in features):
        return False
    for setting,feature in [('social_smile','social_smile'),('face_expression','face_expression')]:
        if config.get(setting,{}).get('enabled') and (not isinstance(features,list) or feature not in features):
            return False
    return (state.get('config_sha256')==digest and state.get('target_protocol_version')==VERSION
        and state.get('camera_transform')=={k:config['camera'][k] for k in ('rotation_deg','mirror_x')}
        and state.get('camera_device')==config['camera']['device']
        and state.get('camera_eye')==config['control'].get('camera_eye')
        and type(service_age) in (int,float) and 0<=service_age<=maximum
        and type(age) in (int,float) and 0<=age<=maximum*1000
        and age/1000+service_age<=maximum
        and state.get('status') not in ('camera_stale','inference_stale'))


def preview_state():
    try:
        with urlopen('http://127.0.0.1:8765/state', timeout=.5) as response:
            return json.load(response)
    except (OSError, URLError, ValueError):
        return None


class VisionReadiness:
    """Non-actuating worker startup, optionally overlapped with bounded homing."""
    def __init__(self,child,config,digest,output,ready=False,managed=None):
        self.child,self.config,self.digest,self.output=child,config,digest,output
        self.managed=managed
        self.started=time.monotonic();self.deadline=self.started+20;self.ready=ready;self.ready_at=None

    def check(self):
        if self.managed is not None:self.managed.check()
        if self.child is not None and self.child.poll() is not None:
            raise RuntimeError(self.failure('视觉服务启动失败或已退出'))

    def failure(self,message):
        diagnostics=self.managed.path.parent if self.managed is not None else self.output
        try:
            status=json.loads((diagnostics/'vision_startup.json').read_text())
            if self.managed is not None or status.get('config_sha256')==self.digest:
                stage=status.get('stage','unknown')
                message+='；阶段：'+str(stage)
                if status.get('property'):message+=' / '+str(status['property'])
                if status.get('error'):message+='；'+str(status['error'])
        except (OSError,ValueError,AttributeError):pass
        return message+'，查看 '+str(diagnostics/'vision.log')

    def wait_ready(self,guard=lambda:None):
        while True:
            guard();self.check()
            state=preview_state()
            if self.ready or (compatible_vision(state,self.config,self.digest)
                              and (self.managed is None or self.managed.matches(state))):
                self.ready=True;self.ready_at=time.monotonic();return
            if time.monotonic()>=self.deadline:raise RuntimeError(self.failure('视觉服务20秒内未就绪'))
            time.sleep(.1)


@contextmanager
def vision_service(config_path, output, required, python=None,wait_ready=True):
    """Stop only our own child. Reuse an external service only after hash match."""
    child = log = readiness = None
    previous_request=os.environ.get('VIVIS_VISION_REQUEST')
    primary_error=None
    try:
        if required:
            wanted = hashlib.sha256(config_path.read_bytes()).hexdigest()
            config=json.loads(config_path.read_text())
            managed=None;current=None
            if os.environ.get('VIVIS_VISION_OWNER'):
                from vivis.vision.persistent import PersistentClient
                managed=PersistentClient(os.environ['VIVIS_VISION_OWNER'],os.environ.get('VIVIS_VISION_TOKEN'))
                managed.request(config_path);os.environ['VIVIS_VISION_REQUEST']=managed.request_id
            else:current=preview_state()
            if managed is not None:pass
            elif current is not None:
                if os.environ.get('VIVIS_MANAGED_SESSION')=='1':
                    raise RuntimeError('P1受管会话需要独立持有相机；请先关闭已有独立视觉服务')
                if not compatible_vision(current,config,wanted):
                    raise RuntimeError('127.0.0.1:8765已有不同配置、旧协议或画面过期的视觉服务；请先关闭旧视觉服务再启动本示例')
            else:
                log = (output / 'vision.log').open('w')
                child = subprocess.Popen([python or sys.executable, '-m', 'vivis.vision.service',
                                          '--config', str(config_path), '--output', str(output / 'vision')],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'),'PYTHONNOUSERSITE':'1'},
                                         start_new_session=os.environ.get('VIVIS_MANAGED_SESSION')!='1')
            readiness=VisionReadiness(child,config,wanted,output,ready=current is not None,managed=managed)
            if wait_ready:readiness.wait_ready()
            print('监控：http://127.0.0.1:8765/', flush=True)
        yield readiness
    except BaseException as exc:
        primary_error=exc
        raise
    finally:
        try:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    child.kill()
                    try:child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        message='视觉子进程强制停止后仍未退出；PID='+str(child.pid)+'；相机驱动操作可能阻塞，不能确认设备已释放'
                        if primary_error is not None:
                            note=getattr(primary_error,'add_note',None)
                            if note is not None:note(message)
                            print(message,flush=True)
                        else:raise RuntimeError(message) from None
        finally:
            if log is not None:
                log.close()
            if previous_request is None:os.environ.pop('VIVIS_VISION_REQUEST',None)
            else:os.environ['VIVIS_VISION_REQUEST']=previous_request


def check_competitors(camera_required):
    for unit in ['dms.service'] + (['ies.service'] if camera_required else []):
        try:
            result = subprocess.run(['systemctl', 'is-active', '--quiet', unit], check=False,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            return
        if result.returncode == 0:
            raise RuntimeError(f'{unit}正在运行并可能占用设备，请先执行 sudo systemctl stop {unit}')


def run_example(mode, argv=None, default_work='wz2', default_repeat=1):
    """Used by all three readable main.py files; --check never opens devices."""
    parser = argparse.ArgumentParser(description='墨甲开发者示例：' + mode)
    parser.add_argument('--robot', type=Path, default=ROOT / 'configs/robots/mojia/robot.json')
    parser.add_argument('--work', default=default_work)
    parser.add_argument('--repeat', type=int, default=default_repeat)
    parser.add_argument('--camera', choices=('left', 'right'), default='right')
    parser.add_argument('--camera-device')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--home-only', action='store_true')
    parser.add_argument('--mouth-gain', type=float)
    parser.add_argument('--round-lip-gain',type=float,default=1.,help='Runtime MouthFunnel/MouthPucker channel multiplier')
    parser.add_argument('--mouth-speed',type=int,help='Task RAM speed for calibrated mouth axes; default saved hardware speed')
    parser.add_argument('--mouth-command-mode',choices=('limited','lips-direct'),
                        help='Default from robot; direct mouth group is configured per robot')
    parser.add_argument('--home-speed', type=int)
    parser.add_argument('--control-fps', type=int, choices=(30, 60))
    parser.add_argument('--face-fps', type=int, choices=(30, 60))
    parser.add_argument('--baud', type=int, choices=(250000, 500000, 1000000))
    parser.add_argument('--upstream-trajectory', choices=('on','off'),
                        help='Optional ID32 jerk-limited upstream trajectory planning')
    parser.add_argument('--parallel-neck',choices=('on','off'),help='ID33/34 coupled pitch and light roll')
    parser.add_argument('--session-stop-file',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--audio-reference-port',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--board-reference-device',help=argparse.SUPPRESS)
    parser.add_argument('--session-id',help=argparse.SUPPRESS)
    parser.add_argument('--soft-audio-cancel-ms',type=int,default=0,help=argparse.SUPPRESS)
    parser.add_argument('--compact-logs',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--overlap-warmup',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if mode not in ('face', 'eyes', 'eyes-neck'):
        parser.error('Unknown example mode')
    if args.upstream_trajectory=='on' and (mode!='eyes-neck' or args.home_only):
        parser.error('上游轨迹规划只能用于示例3的眼颈跟踪播放')
    robot = Robot(args.robot)
    robot.require_mode(mode)
    if args.mouth_speed is not None:robot.set_mouth_speed(args.mouth_speed)
    if mode != 'face' and not args.home_only:
        from vivis.runtime.assets import verify_model
        verify_model()
    defaults = robot.definition['defaults']
    if args.mouth_command_mode is None:args.mouth_command_mode=defaults.get('mouth_command_mode','limited')
    output = args.output or ROOT / 'outputs/developer' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + mode)
    command = ['--robot', str(robot.path), '--work', args.work, '--mode', mode,
               '--repeat', str(args.repeat), '--output', str(output / 'performance'),
               '--camera', args.camera,
               '--mouth-gain', str(args.mouth_gain if args.mouth_gain is not None else defaults['mouth_gain']),
               '--mouth-command-mode',args.mouth_command_mode,
               '--round-lip-gain',str(args.round_lip_gain),
               '--home-speed', str(args.home_speed if args.home_speed is not None else
                                    json.loads(robot.resolve('homing').read_text())['home_speed_counts_s']),
               '--control-fps', str(args.control_fps or defaults['control_hz']),
               '--face-fps', str(args.face_fps or defaults['face_hz'])]
    if defaults.get('hold_vertical_eyes',True):command+=['--hold-vertical-eyes']
    if defaults['authorized_tongue_observation']:
        command += ['--authorized-tongue-observation']
    if defaults['usb_volume_times_three']:
        command += ['--usb-volume-times-three']
    if args.camera_device:
        command += ['--camera-device', args.camera_device]
    if args.baud is not None:
        command += ['--baud', str(args.baud)]
    if args.mouth_speed is not None:
        command += ['--mouth-speed',str(args.mouth_speed)]
    if args.upstream_trajectory is not None:
        command += ['--upstream-trajectory',args.upstream_trajectory]
    if args.parallel_neck is not None:command+=['--parallel-neck',args.parallel_neck]
    if args.home_only:
        command += ['--home-only']
    for key in ('session_stop_file','audio_reference_port','board_reference_device','session_id','soft_audio_cancel_ms'):
        value=getattr(args,key)
        if value is not None:command+=['--'+key.replace('_','-'),str(value)]
    if args.compact_logs:command+=['--compact-logs']
    from vivis.runtime.show import main
    # Validate asset hashes, limits and ownership before any hardware access.
    if args.check:
        return main(command + ['--check'])
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        main(command + ['--check'])
    required = mode != 'face' and not args.home_only
    check_competitors(required)
    output.mkdir(parents=True, exist_ok=False)
    lock_path = ROOT / 'outputs/developer/control.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('另一份开发者示例正在运行，请先停止当前任务') from exc
        config = robot.tracking(args.camera, args.camera_device or ('unused' if not required else None))
        from vivis.motion.upstream import configure
        trajectory=configure(config,mode,args.upstream_trajectory,args.home_only)
        from vivis.vision.parallel_follow import configure as configure_parallel
        parallel=configure_parallel(config,mode,args.parallel_neck,args.home_only)
        from vivis.vision.independent_follow import configure as configure_independent
        independent=configure_independent(config,mode,args.home_only)
        if defaults.get('hold_vertical_eyes',True):
            config['control']['vertical_enabled'] = False
            config['control']['axes'] = [a for a in config['control']['axes'] if a['id'] not in (7, 22)]
            config['control']['held_eye_ids'] = [7, 22]
        config_path = output / 'tracking_effective.json'
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        managed_resume=os.environ.get('VIVIS_MANAGED_SESSION')=='1' and os.environ.get('VIVIS_MOTION_ACTION')=='resume'
        print(f'模式：{mode}；作品：{args.work}；'+('复用本次启动自检，不重复回中。' if managed_resume else
              f'每遍独立准备、{len(robot.home(mode)["axes"])}轴缓慢回中。'),flush=True)
        if args.mouth_speed is not None:
            print(f'口部速度对照：{args.mouth_speed}计数/秒；通过原启动流程写入并核验SRAM，默认配置不改。',flush=True)
        if args.round_lip_gain!=1.:
            print(f'圆唇通道增益：MouthFunnel/MouthPucker ×{args.round_lip_gain}；原素材及其他通道保持。',flush=True)
        if args.mouth_command_mode=='lips-direct' and not args.home_only:
            direct=direct_mouth_ids(robot.motion_profile())
            print(f'口型对照：歌曲阶段直接发送{len(direct)}个口型轴的映射目标；启动避让、回中和硬件限速保留。',flush=True)
        if mode=='eyes-neck' and not args.home_only:
            print('上游轨迹规划：'+('开启（仅ID32）' if trajectory['enabled'] else '关闭（使用原150ms位置低通）'),flush=True)
            if independent['enabled']:print('独立颈部：ID32左右、ID33俯仰、ID34轻微侧倾；眉心固定相机。',flush=True)
            else:print('并联颈部：'+('开启（ID33/34俯仰及轻微侧倾）' if parallel['enabled'] else '关闭（ID33/34保持）'),flush=True)
        print('日志：' + str(output), flush=True)
        def interrupt(*_):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            with vision_service(config_path, output, required,wait_ready=not args.overlap_warmup) as vision:
                try:
                    if args.overlap_warmup:return main(command,vision_readiness=vision)
                    return main(command)
                except RuntimeError as exc:
                    if str(exc).startswith(('Operator stop','Stopped during')):
                        print('已停止位置发送，正在清理资源；颈部保持扭矩。', flush=True)
                        return 130
                    raise
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    try:
        raise SystemExit(run_example('eyes-neck'))
    except KeyboardInterrupt:
        raise SystemExit(130)
