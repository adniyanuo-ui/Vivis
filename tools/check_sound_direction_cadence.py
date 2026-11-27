#!/usr/bin/env python3
"""60-second operator-guided UART receipt test. No microphone, GPU or motion."""
import argparse
from contextlib import contextmanager,nullcontext
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from vivis.hardware.haokai import DirectionParser,open_receiver,select_port
from vivis.speech.sound_direction import Evidence,configuration,monitor_receiver


PHASES=((0.,10.,'quiet_before','保持安静10秒'),
        (10.,30.,'fixed_voice','在正前方固定位置，连续自然说话20秒'),
        (30.,50.,'moving_voice','继续说话，缓慢移到一侧，再移到另一侧，共20秒'),
        (50.,60.,'quiet_after','停止说话，保持安静10秒'))


def phase_report(name,p,max_age):
    prefix=f"{name}：原始字节{p['raw_bytes']}，有效坐标{p['valid_frames']}个，分{p['valid_frame_batches']}批到达；"
    if not p['gap_count']:
        detail='有效批次不足两批，无法计算相邻收包间隔（不代表没有过期）'
    else:
        detail=(f"最长批间隔{p['gap_max_s']:.3f}秒，超过{max_age}秒{p['gaps_above_max_age']}次")
    tail=p['tail_without_coordinate_s']
    return prefix+detail+(f"；本阶段末尾{tail:.2f}秒没有新坐标。" if tail is not None else '。')


@contextmanager
def active_usb_capture(evidence,capture):
    """Same configured input endpoint/rate/channels; discard all audio samples."""
    import numpy as np
    import sounddevice as sd
    devices=sd.query_devices()
    matches=[i for i,d in enumerate(devices) if capture['device'] in d['name']
        and d['max_input_channels']>=capture['channels']]
    if len(matches)!=1:raise ValueError('USB采音设备必须唯一匹配：'+capture['device'])
    state=dict(device=devices[matches[0]]['name'],sample_rate_hz=capture['sample_rate_hz'],
        channels=capture['channels'],callback_count=0,status_count=0,samples=0,closed=False,
        levels={name:dict(samples=0,squared_sum=0.,peak_abs=0.) for _,_,name,_ in PHASES})
    def callback(data,frames,timing,status):
        state['callback_count']+=1;state['samples']+=frames
        if status:state['status_count']+=1
        if evidence.begin is None:return
        name=evidence.phase(time.monotonic()-evidence.begin)
        if name:
            # Scalars only; no recording, network, disk or blocking work here.
            x=data[:,capture['channel']];p=state['levels'][name]
            p['samples']+=frames;p['squared_sum']+=float(np.dot(x,x))
            if frames:p['peak_abs']=max(p['peak_abs'],float(np.max(np.abs(x))))
    stream=sd.InputStream(device=matches[0],samplerate=capture['sample_rate_hz'],
        channels=capture['channels'],dtype='float32',
        blocksize=round(capture['sample_rate_hz']*capture['frame_ms']/1000),
        latency=capture['input_latency_ms']/1000,callback=callback)
    try:
        stream.start();yield state
    finally:
        stream.close();state['closed']=True


class CadenceEvidence(Evidence):
    """Timestamp before disk writes; distinguish receipt batches from UI refreshes.

    All frames decoded in one host read share its timestamp. Those frames do
    not prove zero board-side intervals. No audio is collected: phase labels
    describe the operator's planned action, not a measured VAD result.
    """
    def __init__(self,directory,c,*,clock=time.monotonic):
        super().__init__(directory,c['monitor'])
        try:self.reads=(self.directory/'receipts.jsonl').open('w')
        except BaseException:
            self.raw.close();self.events.close();raise
        self.clock=clock;self.begin=None;self.shown=None;self.receipt_size=0;self.receipt_truncated=False
        self.parser=DirectionParser(c.get('verify_checksum',True));self.max_age=c['monitor']['max_age_s']
        self.stream_bytes=0;self.last_sector=None
        self.audio_stats=None
        self.phases={name:dict(planned_start_s=a,planned_end_s=b,raw_read_batches=0,raw_bytes=0,
            valid_frames=0,valid_frame_batches=0,sector_changes=0,same_sector_frames=0,
            gap_count=0,gap_sum_s=0.,gap_max_s=None,gaps_above_max_age=0,last_valid_s=None,
            first_valid_s=None,tail_without_coordinate_s=None) for a,b,name,_ in PHASES}

    def phase(self,elapsed):
        return next((name for a,b,name,_ in PHASES if a<=elapsed<b),None)

    def raw_bytes(self,data):
        stamp=self.clock();elapsed=stamp-self.begin;name=self.phase(elapsed)
        frames=self.parser.feed(data)
        record=dict(received_monotonic=stamp,elapsed_s=elapsed,planned_phase=name,
            stream_offset=self.stream_bytes,byte_count=len(data),sectors=[f.sector for f in frames],
            prefix_hex=data[:24].hex(' '))
        self.stream_bytes+=len(data)
        if name:
            p=self.phases[name];p['raw_read_batches']+=1;p['raw_bytes']+=len(data)
            p['valid_frames']+=len(frames)
            if frames:
                p['valid_frame_batches']+=1
                if p['first_valid_s'] is None:p['first_valid_s']=elapsed
                if p['last_valid_s'] is not None:
                    gap=elapsed-p['last_valid_s'];p['gap_count']+=1;p['gap_sum_s']+=gap
                    p['gap_max_s']=max(p['gap_max_s'] or 0.,gap)
                    p['gaps_above_max_age']+=int(gap>self.max_age)
                p['last_valid_s']=elapsed
        for frame in frames:
            if name and self.last_sector is not None:
                self.phases[name]['same_sector_frames' if self.last_sector==frame.sector else 'sector_changes']+=1
            self.last_sector=frame.sector
        line=json.dumps(record,allow_nan=False)+'\n';size=len(line.encode())
        if self.receipt_size+size<=self.settings['max_event_bytes']:
            self.reads.write(line);self.receipt_size+=size
        else:self.receipt_truncated=True
        super().raw_bytes(data)

    def emit_phase(self,_status):
        elapsed=self.clock()-self.begin;name=self.phase(elapsed)
        if name!=self.shown:
            self.shown=name
            for a,b,key,words in PHASES:
                if key==name:print(f'【{a:.0f}—{b:.0f}秒】{words}',flush=True)

    def finish(self,summary):
        elapsed=self.clock()-self.begin
        try:
            for p in self.phases.values():
                covered=max(0.,min(elapsed,p['planned_end_s'])-p['planned_start_s'])
                p['observed_phase_s']=covered
                p['gap_mean_s']=p['gap_sum_s']/p['gap_count'] if p['gap_count'] else None
                if covered:
                    end=p['planned_start_s']+covered
                    p['tail_without_coordinate_s']=end-(p['last_valid_s'] if p['last_valid_s'] is not None else p['planned_start_s'])
            summary['cadence']=dict(timing_basis='host_receipt_before_evidence_write_not_board_emission',
                phase_labels='operator_schedule_not_audio_verified',max_age_s=self.max_age,
                receipts_truncated=self.receipt_truncated,raw_bytes_received=self.stream_bytes,
                replay_statistics=self.parser.statistics(),phases=self.phases)
            summary['audio_opened']=self.audio_stats is not None
            summary['audio_saved']=False;summary['audio_sent_to_cloud']=False
            if self.audio_stats is not None:
                a=dict(self.audio_stats)
                a['levels']={name:dict(p,rms=(p['squared_sum']/p['samples'])**.5 if p['samples'] else None)
                    for name,p in self.audio_stats['levels'].items()}
                summary['capture_only']=a
            super().finish(summary)
        finally:self.reads.close()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true',help='仅检查配置，不枚举或打开设备')
    ap.add_argument('--port',help='可选声源USB串口；禁止舵机串口')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--audio-active',action='store_true',help='同时打开正常聊天配置的USB采音；不保存、不播放、不联网')
    args=ap.parse_args(argv);c=configuration()
    if args.check:
        print('60秒收包测试配置通过；--audio-active可对照USB采音启停；本次未打开设备。');return 0
    active=subprocess.run(['systemctl','is-active','vivis-p1.service'],capture_output=True,text=True)
    if active.stdout.strip() not in ('inactive','failed'):
        print('请先停止vivis-p1.service，避免两个进程争用声源串口。');return 1
    from serial.tools.list_ports import comports
    robot=ROOT/'configs/robots/xiaoshuai/robot.json';definition=json.loads(robot.read_text())
    hardware=json.loads((robot.parent/definition['hardware']).read_text())
    try:
        port=select_port(list(comports()),c['serial'],hardware['runtime']['port'],args.port)
        directory=args.output or ROOT/'outputs/sound_direction_cadence'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        evidence=CadenceEvidence(directory,c)
        try:receiver=open_receiver(port,c['serial'])
        except BaseException:
            evidence.raw.close();evidence.events.close();evidence.reads.close();raise
        print(f'只读测试：{port}；证据目录：{directory}',flush=True)
        print('5秒后开始；请跟随终端提示。不会播放提示音，也不会驱动机器头。',flush=True)
        try:
            capture=json.loads((ROOT/'configs/perception/speech.json').read_text())['capture']
            context=active_usb_capture(evidence,capture) if args.audio_active else nullcontext(None)
            with context as stats:
                evidence.audio_stats=stats
                if stats:print('USB采音已打开：'+stats['device']+'；仅保存电平统计，音频不保存、不播放、不联网。',flush=True)
                # Drain during preparation too; no five-second receive backlog.
                evidence.begin=time.monotonic()+5.
                result=monitor_receiver(receiver,c,65.,evidence,emit=evidence.emit_phase)
            if stats is not None:
                result['capture_only']['closed']=stats['closed']
                result['capture_only']['input_confirmed']=stats['callback_count']>0
                if not result['capture_only']['input_confirmed']:
                    result.update(outcome='audio_capture_unavailable',error='USB采音没有回调，本轮开流对照无效')
                (directory/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        except BaseException:
            receiver.close()
            for f in (evidence.raw,evidence.events,evidence.reads):f.close()
            raise
        print('测试结束，'+('完整完成。' if result['outcome']=='complete' else result['outcome']),flush=True)
        for name in ('fixed_voice','moving_voice'):
            p=result['cadence']['phases'][name]
            print(phase_report(name,p,result['cadence']['max_age_s']),flush=True)
        if stats is not None:
            print(f"USB采音回调{stats['callback_count']}次，状态告警{stats['status_count']}次；音频未保存。",flush=True)
        print(f'报告：{directory}/summary.json',flush=True)
        return 0 if result['outcome']=='complete' and result['statistics']['valid_frames'] else 1
    except Exception as exc:
        print('收包测试未完成：'+str(exc));return 1


if __name__=='__main__':raise SystemExit(main())
