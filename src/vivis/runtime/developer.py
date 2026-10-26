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


def preview_state():
    try:
        with urlopen('http://127.0.0.1:8765/state', timeout=.5) as response:
            return json.load(response)
    except (OSError, URLError, ValueError):
        return None


@contextmanager
def vision_service(config_path, output, required):
    """Stop only our own child. Reuse an external service only after hash match."""
    child = log = None
    try:
        if required:
            wanted = hashlib.sha256(config_path.read_bytes()).hexdigest()
            current = preview_state()
            if current is not None:
                if current.get('config_sha256') != wanted:
                    raise RuntimeError('127.0.0.1:8765已有不同配置的视觉服务；请先关闭旧视觉服务再启动本示例')
            else:
                log = (output / 'vision.log').open('w')
                child = subprocess.Popen([sys.executable, '-m', 'vivis.vision.service',
                                          '--config', str(config_path), '--output', str(output / 'vision')],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, 'PYTHONPATH': str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')},
                                         start_new_session=True)
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if child.poll() is not None:
                        raise RuntimeError('视觉服务启动失败，查看 ' + str(output / 'vision.log'))
                    current = preview_state()
                    if current and current.get('config_sha256') == wanted:
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError('视觉服务20秒内未就绪，查看 ' + str(output / 'vision.log'))
            print('监控：http://127.0.0.1:8765/', flush=True)
        yield
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=4)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)
        if log is not None:
            log.close()


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
    parser.add_argument('--home-speed', type=int)
    parser.add_argument('--control-fps', type=int, choices=(30, 60))
    parser.add_argument('--face-fps', type=int, choices=(30, 60))
    parser.add_argument('--baud', type=int, choices=(250000, 500000, 1000000))
    args = parser.parse_args(argv)
    if mode not in ('face', 'eyes', 'eyes-neck'):
        parser.error('Unknown example mode')
    robot = Robot(args.robot)
    defaults = robot.definition['defaults']
    output = args.output or ROOT / 'outputs/developer' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + mode)
    command = ['--robot', str(robot.path), '--work', args.work, '--mode', mode,
               '--repeat', str(args.repeat), '--output', str(output / 'performance'),
               '--hold-vertical-eyes', '--camera', args.camera,
               '--mouth-gain', str(args.mouth_gain if args.mouth_gain is not None else defaults['mouth_gain']),
               '--home-speed', str(args.home_speed if args.home_speed is not None else
                                    json.loads(robot.resolve('homing').read_text())['home_speed_counts_s']),
               '--control-fps', str(args.control_fps or defaults['control_hz']),
               '--face-fps', str(args.face_fps or defaults['face_hz'])]
    if defaults['authorized_tongue_observation']:
        command += ['--authorized-tongue-observation']
    if defaults['usb_volume_times_three']:
        command += ['--usb-volume-times-three']
    if args.camera_device:
        command += ['--camera-device', args.camera_device]
    if args.baud is not None:
        command += ['--baud', str(args.baud)]
    if args.home_only:
        command += ['--home-only']
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
        config['control']['vertical_enabled'] = False
        config['control']['axes'] = [a for a in config['control']['axes'] if a['id'] not in (7, 22)]
        config['control']['held_eye_ids'] = [7, 22]
        config_path = output / 'tracking_effective.json'
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        print(f'模式：{mode}；作品：{args.work}；每遍独立准备、35轴缓慢回中。', flush=True)
        print('日志：' + str(output), flush=True)
        def interrupt(*_):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            with vision_service(config_path, output, required):
                try:
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
