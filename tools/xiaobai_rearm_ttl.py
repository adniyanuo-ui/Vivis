#!/usr/bin/env python3
"""Explicit restart preparation after power interruption; no EEPROM writes.

Offline decision helper and hardware --apply path. A disabled axis receives a
nearby bounded goal before any explicit enable; some firmware enables on a goal
write itself. Neck and unmapped 29 excluded.
"""
import argparse
import json
import signal
import struct
import time
from pathlib import Path
from xiaobai_runtime.mapping import ROOT,read_json,live_preflight
from xiaobai_runtime.playback import check_feedback


def rearm_target(state,motor,*,include_jaw=False):
    expected=11272 if include_jaw and state.get('id')==31 else 521
    if state.get('id') in [29,32,33,34]:raise ValueError('Unmapped or neck axis is excluded')
    if state['model_raw']!=expected or state['mode']!=0 or state['torque_enable'] not in [0,1]:
        raise ValueError('Expected verified model, position mode and torque 0 or 1')
    position=state['position_raw']
    if not state['configured_min']<=position<=state['configured_max']:
        raise ValueError('Measured position outside device range')
    if state['configured_min']>motor['safe_min'] or state['configured_max']<motor['safe_max']:
        raise ValueError('Device/software range mismatch')
    target=position
    if position<motor['safe_min']:target=min(motor['safe_max'],motor['safe_min']+2)
    if position>motor['safe_max']:target=max(motor['safe_min'],motor['safe_max']-2)
    if abs(target-position)>10 or (state['torque_enable']==1 and target!=position):
        raise ValueError('Restart recovery would exceed 10 counts or change an enabled axis')
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--scope',choices=['ttl','face_eyes'],default='ttl',
                        help='ttl requires RS485 disconnected; face_eyes requires both branches and includes jaw31')
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Use a new output directory')
    args.output.mkdir(parents=True)
    profile=read_json(ROOT/'xiaobai/playback_profile_raw_trace.json')
    ids=list(range(2,29))+[30]+([31] if args.scope=='face_eyes' else [])
    motors={m['id']:m for m in profile['motors']}
    live_preflight(profile,ids)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        if stopped or (ROOT/'outputs/xiaobai/STOP').exists():raise RuntimeError('Operator stop')
    from xiaobai_branch_trial import probe_branch
    guard();probe_branch(profile,args.output/'branch_probe','both' if args.scope=='face_eyes' else 'ttl')
    from xiaobai_runtime.feetech_sdk import OfficialSDKBus,COMM_SUCCESS
    bus=OfficialSDKBus(profile['port'],profile['baud'],trace_path=args.output/'serial_trace.json')
    report=dict(completed=False,apply=args.apply,plan=[],events=[],speed=60,
                scope=args.scope,command_ids=ids,
                policy='Mapped face axes only; existing enabled axes unchanged; RAM40 and41..47 only')
    def feedback(sid,state,motor):
        return {'position':state['position_raw'],**{k:state[k] for k in ['temperature_c','load_raw','status_register','voltage_raw']}}
    try:
        # Validate every axis before the first mutation.
        for sid in ids:
            guard();state=bus.snapshot(sid);motor=motors[sid]
            target=rearm_target(state,motor,include_jaw=args.scope=='face_eyes')
            if motor['max_temperature_c']>state['configured_max_temperature']:
                raise RuntimeError('Temperature limit exceeds device protection')
            recovery={**motor,'safe_min':max(state['configured_min'],min(state['position_raw'],target)-2),
                      'safe_max':min(state['configured_max'],max(state['position_raw'],target)+2)}
            check_feedback(sid,feedback(sid,state,motor),recovery)
            report['plan'].append(dict(id=sid,before=state,target=target,recovery_limits=recovery))
        if not args.apply:return
        for row in report['plan']:
            sid=row['id'];before=row['before'];target=row['target'];motor=motors[sid]
            if before['torque_enable']==1:continue
            guard();current=bus.snapshot(sid)
            if current['torque_enable']!=0 or abs(current['position_raw']-before['position_raw'])>2:
                raise RuntimeError(f'{sid}: state changed since restart plan')
            check_feedback(sid,feedback(sid,current,motor),row['recovery_limits'])
            event=dict(id=sid,primed_target=target,goal_write_attempted=True,
                       torque_enable_attempted=False,feedback=[])
            report['events'].append(event)
            bus.goals({sid:target},{sid:60})
            actual_goal=bus.read(sid,42,6)
            event['goal_readback_hex']=actual_goal.hex()
            if actual_goal!=struct.pack('<HHH',target,0,60):
                raise RuntimeError(f'{sid}: priming verification failed')
            torque=bus.read(sid,40,1)
            event['torque_after_goal']=torque[0]
            if torque not in [b'\x00',b'\x01']:raise RuntimeError(f'{sid}: unexpected torque mode after goal')
            guard()
            if torque==b'\x00':
                event['torque_enable_attempted']=True
                result,status=bus.sdk.write1ByteTxRx(sid,40,1)
                event.update(communication_result=result,servo_status=status)
                if result!=COMM_SUCCESS or status:raise RuntimeError(f'{sid}: enable acknowledgement failed; no write retry')
            if bus.read(sid,40,1)!=b'\x01':raise RuntimeError(f'{sid}: torque enable readback failed')
            deadline=time.monotonic()+2
            while True:
                guard();state=bus.feedback(sid);event['feedback'].append(state)
                check_feedback(sid,state,row['recovery_limits'])
                if motor['safe_min']<=state['position']<=motor['safe_max'] and abs(state['position']-target)<=3:
                    check_feedback(sid,state,motor);break
                if time.monotonic()>deadline:raise RuntimeError(f'{sid}: bounded recovery did not settle')
                time.sleep(.02)
        report['completed']=True
    except BaseException as exc:
        report['error']=str(exc);raise
    finally:
        bus.close()
        (args.output/'rearm_log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
