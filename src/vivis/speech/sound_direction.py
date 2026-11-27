"""Independent, bounded sound-direction observation; no audio or motion owner."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
import threading
import sys
from importlib.machinery import PathFinder
from importlib import invalidate_caches

from vivis.paths import ROOT
from vivis.hardware.haokai import DirectionParser, DirectionState, open_receiver, select_port


CONFIG = ROOT / 'configs/perception/sound_direction.json'


def runtime_dependencies(python):
    """Inspect the SELECTED venv, not the supervisor's imports; open no device."""
    invalidate_caches()
    python=Path(python).absolute();prefix=python.parent.parent
    paths=[str(p) for p in prefix.glob('lib/python*/site-packages')]
    if python==Path(sys.executable).absolute():paths=list(sys.path)
    if not paths:return dict(available=None,missing_modules=[],python=str(python),check='environment_not_inspectable')
    missing=[name for name in ('serial','webrtcvad') if PathFinder.find_spec(name,paths) is None]
    return dict(available=not missing,missing_modules=missing,python=str(python),
        check='module_presence_only_no_device_open')


class DirectionObserver:
    """One optional read-only UART owner, detached from audio/control scheduling."""
    def __init__(self,c,protected_port):
        self.c=c;self.protected_port=protected_port;self.latest=None;self.error=None
        self.port=None;self.opened=False;self.generation=0;self.statistics={}
        self.stop=threading.Event();self.thread=None

    def start(self):
        self.thread=threading.Thread(target=self.run,daemon=True,name='sound-direction-uart')
        self.thread.start();return self

    def run(self):
        while not self.stop.is_set():
            receiver=None
            try:
                from serial.tools.list_ports import comports
                port=select_port(list(comports()),self.c['serial'],self.protected_port)
                receiver=open_receiver(port,self.c['serial'])
                self.latest=None;self.statistics={}
                self.port=port;self.opened=True;self.generation+=1
                verify=self.c.get('verify_checksum',True)
                parser=DirectionParser(verify)
                state=DirectionState(self.c['coordinates'],self.c['monitor']['max_age_s'],verify,
                    self.c['monitor']['update_mode'])
                self.error=None
                while not self.stop.is_set():
                    data=receiver.read(min(4096,max(1,receiver.in_waiting)))
                    now=time.monotonic()
                    for frame in parser.feed(data):state.observe(frame,now)
                    self.latest=dict(state.snapshot(now),updated_monotonic=state.received,
                        generation=self.generation,serial_open=True,
                        observer_lease_s=self.c['monitor']['observer_lease_s'])
                    self.statistics=parser.statistics()
            except Exception as exc:
                self.latest=None;self.error=type(exc).__name__+': '+str(exc)[:160]
            finally:
                self.opened=False;self.latest=None
                if receiver is not None:
                    try:receiver.close()
                    except OSError:pass
            self.stop.wait(10.)  # Optional observer failure never restarts the head.

    def status(self,now=None):
        now=time.monotonic() if now is None else now
        packet=self.latest or {};stamp=packet.get('received_monotonic')
        age=now-stamp if type(stamp) in (int,float) else None
        fresh=age is not None and 0<=age<=self.c['monitor']['max_age_s'] and packet.get('frame_valid') is True
        seen=packet.get('observed_monotonic')
        responsive=(type(seen) in (int,float) and 0<=now-seen<=self.c['monitor']['observer_lease_s'])
        held=(self.c['monitor']['update_mode']=='on_change' and not fresh and age is not None
            and age>=0 and packet.get('frame_valid') is True)
        state=('error' if self.error else 'off' if not self.opened else 'fresh' if fresh else
            'waiting' if stamp is None else 'held' if held else 'stale')
        if self.opened and stamp is not None and not responsive:state='unresponsive'
        usable=state in ('fresh','held')
        return dict(status=state,port=self.port,serial_open=self.opened,error=self.error,
            generation=self.generation,sector=packet.get('sector') if usable else None,
            sequence=packet.get('sequence',0),age_s=age,statistics=dict(self.statistics),
            event_fresh=usable and fresh,last_known_direction_available=usable,
            update_mode=self.c['monitor']['update_mode'],observer_responsive=responsive,
            checksum_verified=usable and self.c.get('verify_checksum',True))

    def attention_snapshot(self,now=None):
        now=time.monotonic() if now is None else now
        status=self.status(now);packet=self.latest
        if (not status['last_known_direction_available'] or not isinstance(packet,dict)
                or packet.get('generation')!=self.generation):return None
        return dict(packet,observer_status=status['status'])

    def close(self):
        self.stop.set()
        self.opened=False;self.latest=None
        if self.thread:self.thread.join(timeout=.5)


def configuration(path=CONFIG):
    c = json.loads(Path(path).read_text())
    if (c.get('schema_version') != 'vivis-sound-direction-1.0'
            or c.get('protocol') != 'hk-qxmic-v3-sector12' or c.get('motion_enabled') is not False):
        raise ValueError('Invalid sound-direction protocol or motion permission')
    s, angles, monitor = c['serial'], c['coordinates'], c['monitor']
    if type(c.get('verify_checksum',True)) is not bool:raise ValueError('Invalid checksum policy')
    if monitor.get('update_mode') not in ('periodic','on_change'):raise ValueError('Invalid coordinate update mode')
    if 'attention' in c:
        from vivis.interaction.sound_attention import configuration as attention_configuration
        attention_configuration(path)
    document = c.get('protocol_document')
    if (not isinstance(document, dict) or not isinstance(document.get('name'), str)
            or not document['name'] or not isinstance(document.get('sha256'), str)
            or len(document['sha256']) != 64
            or any(v not in '0123456789abcdef' for v in document['sha256'])):
        raise ValueError('Invalid direction protocol provenance')
    if (s.get('baud_rate') != 1000000 or s.get('data_bits') != 8 or s.get('parity') != 'N'
            or s.get('stop_bits') != 1
            or (s.get('device') is not None and (not isinstance(s['device'], str) or not s['device']))):
        raise ValueError('HK-QXMIC requires the documented 1000000/8N1 host setting')
    if any(type(s.get(k)) is not int or not 0 <= s[k] <= 65535 for k in ('usb_vid', 'usb_pid')):
        raise ValueError('Invalid direction USB identity')
    if angles != dict(sector_count=12, sector_step_deg=30, sector_zero_nominal_deg=180,
                      reference='manufacturer_board_diagram'):
        raise ValueError('Direction indices must retain the vendor coordinate convention')
    limits = ((s, 'read_timeout_s', .05, .3), (monitor, 'publish_interval_s', .2, 1.),
              (monitor, 'observer_lease_s', .3, 3.),
              (monitor, 'max_age_s', .3, 3.))
    if any(type(obj.get(key)) not in (int, float) or not math.isfinite(obj[key])
           or not low <= obj[key] <= high for obj, key, low, high in limits):
        raise ValueError('Invalid direction observation timing')
    if monitor['max_age_s'] < monitor['publish_interval_s']:
        raise ValueError('Direction freshness must cover publication interval')
    if any(type(monitor.get(k)) is not int or not 1024 <= monitor[k] <= 16*1024*1024
           for k in ('max_raw_bytes', 'max_event_bytes')):
        raise ValueError('Invalid direction evidence quota')
    return c


class Evidence:
    def __init__(self, directory, settings):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.raw = (self.directory/'raw.bin').open('wb')
        try:
            self.events = (self.directory/'events.jsonl').open('wb')
        except BaseException:
            self.raw.close()
            raise
        self.settings = settings
        self.raw_size = self.event_size = 0
        self.raw_truncated = self.events_truncated = False

    def raw_bytes(self, data):
        room = self.settings['max_raw_bytes']-self.raw_size
        self.raw.write(data[:room])
        self.raw_size += min(len(data), room)
        self.raw_truncated |= len(data) > room

    def event(self, value):
        data = (json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n').encode()
        if self.event_size+len(data) <= self.settings['max_event_bytes']:
            self.events.write(data)
            self.event_size += len(data)
        else:
            self.events_truncated = True

    def finish(self, summary):
        self.raw.close()
        self.events.close()
        summary.update(raw_bytes_saved=self.raw_size, raw_truncated=self.raw_truncated,
                       events_truncated=self.events_truncated)
        (self.directory/'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def monitor_receiver(receiver, c, seconds, evidence, emit=print, clock=time.monotonic):
    verify = c.get('verify_checksum',True)
    parser = DirectionParser(verify)
    state = DirectionState(c['coordinates'], c['monitor']['max_age_s'],verify,c['monitor']['update_mode'])
    start = clock()
    next_publish = start
    outcome, error, failure_stage = 'complete', None, None
    stage = 'serial_read'
    try:
        while clock()-start < seconds:
            stage = 'serial_read'
            data = receiver.read(min(4096, max(1, receiver.in_waiting)))
            now = clock()
            if data:
                stage = 'evidence_write'
                evidence.raw_bytes(data)
                stage = 'frame_decode'
                for frame in parser.feed(data):
                    state.observe(frame, now)
            if now >= next_publish:
                packet = state.snapshot(now)
                packet.update(timestamp=datetime.now().astimezone().isoformat(),
                              timestamp_monotonic=now, statistics=parser.statistics())
                stage = 'evidence_write'
                evidence.event(packet)
                if packet['frame_valid']:
                    text = (f"{'保持上次坐标' if packet['status']=='held' else '坐标'}{packet['sector']:02d}，板卡名义角度"
                            f"{packet['board_nominal_azimuth_deg']}°；非说话人身份")
                else:
                    text = '等待有效方位帧' if packet['status'] == 'waiting' else '方位数据已过期'
                emit(f"声源方位［{packet['status']}］：{text}；语音有效性未知，不联动运动")
                next_publish = now+c['monitor']['publish_interval_s']
    except KeyboardInterrupt:
        outcome = 'cancelled'
    except (OSError, ValueError) as exc:
        outcome = 'device_error' if stage == 'serial_read' else 'observer_error'
        error, failure_stage = f'{type(exc).__name__}: {exc}', stage
    finally:
        try:
            receiver.close()
        except OSError as exc:
            if error is None:
                outcome, error, failure_stage = 'device_error', f'{type(exc).__name__}: {exc}', 'serial_close'
    serial_port = getattr(receiver, 'port', None)
    summary = dict(schema_version='vivis-sound-direction-summary-1.0',
                   outcome=outcome, error=error, elapsed_s=clock()-start,
                   failure_stage=failure_stage, port=serial_port if isinstance(serial_port, str) else None,
                   protocol_document=c['protocol_document'],
                   serial_protocol=c['serial'], last_observation=state.snapshot(clock()),
                   statistics=parser.statistics(), serial_protocol_bytes_sent=0,
                   audio_opened=False, actuator_opened=False, motion_enabled=False)
    evidence.finish(summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=CONFIG)
    ap.add_argument('--robot', type=Path, default=ROOT/'configs/robots/xiaoshuai/robot.json')
    ap.add_argument('--port', help='独立CH340串口或其稳定by-id路径；不接受舵机串口')
    ap.add_argument('--seconds', type=float, default=30.)
    ap.add_argument('--output', type=Path)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='只检查配置和协议，不枚举或打开设备')
    modes.add_argument('--list', action='store_true', help='只列USB串口，不打开')
    args = ap.parse_args(argv)
    try:
        c = configuration(args.config)
        if not math.isfinite(args.seconds) or not 0 < args.seconds <= 300:
            raise ValueError('--seconds必须大于0且不超过300秒')
        robot = json.loads(args.robot.read_text())
        hardware = json.loads((args.robot.parent/robot['hardware']).read_text())
        protected = hardware['runtime']['port']
        if args.check:
            parser = DirectionParser(c.get('verify_checksum',True))
            if [f.sector for f in parser.feed(bytes.fromhex('55 AA 05 FA 00 E2 7E'))] != [5]:
                raise ValueError('Vendor direction example did not pass replay')
            print('声源方位离线检查通过：1000000/8N1，12方位，'+
                  ('校验FF－坐标' if c.get('verify_checksum',True) else '按用户设置忽略校验码，仍检查帧头及坐标范围')+'；未打开设备')
            return 0
        from serial.tools.list_ports import comports
        ports = [p for p in comports() if p.vid is not None]
        if args.list:
            for port in ports:
                match = port.vid == c['serial']['usb_vid'] and port.pid == c['serial']['usb_pid']
                print(f'{port.device}  {port.description}  {port.hwid}  '
                      f'{"匹配声源CH340" if match else "其他设备，不打开"}')
            if not ports:
                print('未枚举到USB串口节点；请先用lsusb核对转接板。USB可见但无串口节点时，请检查串口驱动；CH340需要ch341驱动。')
            return 0
        port = select_port(ports, c['serial'], protected, args.port)
        receiver = open_receiver(port, c['serial'])
        directory = args.output or ROOT/'outputs/sound_direction'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        try:
            evidence = Evidence(directory, c['monitor'])
        except BaseException:
            receiver.close()
            raise
        print(f'只接收声源方位，不采音、不控制机器头：{port}；记录：{directory}')
        result = monitor_receiver(receiver, c, args.seconds, evidence)
        print('结果：'+json.dumps(result['statistics'], ensure_ascii=False))
        if result['error']:
            print(f"声源方位观察异常［{result['failure_stage']}］："+result['error'])
            return 1
        if not result['statistics']['valid_frames']:
            print('未收到有效帧，不能宣称方位读取成功；请核对TX→RX、GND及串口')
            return 3
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'声源方位失败：{exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
