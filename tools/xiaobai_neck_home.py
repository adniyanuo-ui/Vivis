#!/usr/bin/env python3
"""Supported-head, slow neck-only homing; no calibration/EEPROM/reset writes."""
import argparse
import json
import signal
import struct
import time
from pathlib import Path
from xiaobai_runtime.feetech_sdk import OfficialSDKBus, COMM_SUCCESS
from xiaobai_runtime.mapping import ROOT, read_json
from xiaobai_runtime.playback import check_feedback


def neck_limits(state, motor):
    if state['id'] not in (32,33,34) or state['model_raw']!=11272:
        raise ValueError('Expected known SM85BLHV neck ID')
    if state['mode']!=0 or state['torque_enable'] not in (0,1):
        raise ValueError('Expected position mode and ordinary torque state')
    if not state['configured_min']<=state['position_raw']<=state['configured_max']:
        raise ValueError('Support head until measured neck position is inside device limits')
    if not state['configured_min']<=2047<=state['configured_max']:
        raise ValueError('Home target is outside existing device limits')
    phase=bytes.fromhex(state['config_0_39_hex'])[18]
    if phase!=44 or bytes.fromhex(state['config_0_39_hex'])[:3]!=bytes([2,53,0]):
        raise ValueError('Neck firmware/phase changed; review speed and register layout')
    return {**motor,'safe_min':state['configured_min'],'safe_max':state['configured_max']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--port',default='/dev/ttyCH343USB0')
    parser.add_argument('--supported',action='store_true',help='Onsite operator has confirmed supporting the head')
    args=parser.parse_args()
    if not args.supported:raise SystemExit('Requires onsite head-support confirmation')
    args.output.mkdir(parents=True,exist_ok=False)
    motors={m['id']:m for m in read_json(ROOT/'xiaobai/playback_profile_raw_trace.json')['motors']}
    report=dict(completed=False,target=2047,ids=[32,33,34],speed_raw=60,
                command_rate_counts_s=50,events=[],snapshots=[],stop_policy='Keep last bounded target; never unload neck on exit')
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        if stopped or (ROOT/'outputs/xiaobai/STOP').exists():raise RuntimeError('Operator stop')
    bus=OfficialSDKBus(args.port,trace_path=args.output/'serial_trace.json')
    limits={}
    def feedback(sid):
        guard();s=bus.feedback(sid);check_feedback(sid,s,limits[sid]);return s
    def write(sid,position):
        guard()
        if not limits[sid]['safe_min']<=position<=limits[sid]['safe_max']:raise ValueError('Target outside device limits')
        event=dict(id=sid,target=position);report['events'].append(event)
        result,status=bus.sdk.WritePosEx(sid,position,60,bus.acceleration[sid])
        event.update(result=result,status=status)
        if result!=COMM_SUCCESS or status:raise RuntimeError('Neck write failed; no automatic retry')
        if bus.read(sid,42,6)!=struct.pack('<HHH',position,0,60):raise RuntimeError('Neck goal readback mismatch')
    try:
        # Validate all neck axes before any motion or torque write.
        for sid in [32,33,34]:
            guard();state=bus.snapshot(sid);limits[sid]=neck_limits(state,motors[sid])
            report['snapshots'].append(state);feedback(sid)
        for sid in [32,33,34]:
            current=feedback(sid)['position'];write(sid,current)
            if bus.read(sid,40,1)==b'\x00':
                guard();result,status=bus.sdk.write1ByteTxRx(sid,40,1)
                if result!=COMM_SUCCESS or status:raise RuntimeError('Neck enable failed')
            if bus.read(sid,40,1)!=b'\x01':raise RuntimeError('Neck enable readback mismatch')
            deadline=time.monotonic()+15
            while abs(current-2047)>0:
                if time.monotonic()>deadline:raise RuntimeError('Neck approach timed out')
                target=current+max(-5,min(5,2047-current));write(sid,target)
                time.sleep(.1)
                actual=feedback(sid)['position']
                if abs(actual-target)>20:raise RuntimeError('Neck did not follow bounded target')
                current=target
        # Require repeated near-home positions while preserving holding torque.
        for _ in range(10):
            for sid in [32,33,34]:
                s=feedback(sid)
                if abs(s['position']-2047)>10:raise RuntimeError('Neck failed to settle at home')
            time.sleep(.1)
        report['final']=[bus.snapshot(sid) for sid in [32,33,34]]
        report['completed']=True
    except BaseException as exc:
        report['error']=str(exc);raise
    finally:
        bus.close();(args.output/'neck_home.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['completed','target','final']},ensure_ascii=False))


if __name__=='__main__':main()
