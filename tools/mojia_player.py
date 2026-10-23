#!/usr/bin/env python3
"""Play exported Mojia targets with separate model/limit checks and explicit arming."""
import argparse
import json
import signal
import struct
import time
from pathlib import Path

import numpy as np
from prepare_mojia import sha256, MOJIA_IDS
from xiaobai_runtime.mapping import ROOT, read_json, live_preflight
from xiaobai_runtime.playback import play, check_feedback


def startup_target(state,motor,neck=False,recover_disabled=False):
    """Explicit disabled-axis recovery may start outside limits; every goal stays bounded."""
    sid=motor['id']
    if state['model_raw']!=motor['model_raw_expected'] or state['mode']!=0 or state['torque_enable'] not in (0,1):
        raise ValueError(f'ID{sid}: unexpected model, mode or torque state')
    if (state['configured_min'],state['configured_max'])!=(motor['device_min'],motor['device_max']):
        raise ValueError(f'ID{sid}: device limits changed; regenerate profile')
    position=state['position_raw']
    if not 0<=position<=4095:raise ValueError(f'ID{sid}: invalid single-turn position')
    if recover_disabled:
        if sid not in (29,35,36) or state['torque_enable']!=0 or neck:
            raise ValueError(f'ID{sid}: disabled recovery is restricted to unpowered face axes 29/35/36')
        if not motor['device_min']<=2047<=motor['device_max'] or not motor['safe_min']<=2047<=motor['safe_max']:
            raise ValueError(f'ID{sid}: recovery target is outside reviewed limits')
        return 2047
    if not motor['device_min']<=position<=motor['device_max']:
        raise ValueError(f'ID{sid}: measured {position} outside device [{motor["device_min"]},{motor["device_max"]}]')
    if neck:return position
    target=max(motor['safe_min'],min(motor['safe_max'],position))
    if state['torque_enable']==1 and 'feedback_min' in motor:
        return target  # No priming write on enabled axes; check_feedback enforces device limits.
    tolerance=motor.get('position_feedback_tolerance_counts',0)
    if abs(target-position)>10 or state['torque_enable']==1 and abs(target-position)>tolerance:
        raise ValueError(f'ID{sid}: restart recovery exceeds bounded policy; measured {position}, allowed [{motor["safe_min"]},{motor["safe_max"]}]')
    return target


def feedback(state):
    return dict(position=state['position_raw'],**{k:state[k] for k in
                ['load_raw','temperature_c','voltage_raw','status_register']})


def load_targets(targets_path,profile,audio,calibration):
    if (sha256(targets_path)!=profile['targets_sha256'] or sha256(audio)!=profile['audio_sha256']
            or sha256(calibration)!=profile['calibration_sha256']):
        raise ValueError('Targets, audio or Mojia calibration differs from reviewed profile')
    payload=read_json(targets_path)
    if payload.get('schema_version')!='mojia-motor-targets-0.1' or payload.get('device_limits_applied') is not True:
        raise ValueError('Regenerate targets using a complete Mojia hardware scan')
    if payload['calibration_sha256']!=profile['calibration_sha256'] or payload['audio_sha256']!=profile['audio_sha256']:
        raise ValueError('Target provenance mismatch')
    ids=payload['motor_ids'];positions=np.asarray(payload['positions'],dtype=float)
    if (not ids or len(set(ids))!=len(ids) or any(type(sid) is not int or sid not in MOJIA_IDS for sid in ids)
            or positions.shape!=(payload['frames'],len(ids)) or not len(positions) or not np.isfinite(positions).all()):
        raise ValueError('Invalid Mojia target matrix')
    if set(ids)!=set(profile['diagnostic_command_ids']):raise ValueError('Command IDs differ from profile')
    fps=payload['fps']['num']/payload['fps']['den']
    if not np.isfinite(fps) or not 1<=fps<=120:raise ValueError('Invalid FPS')
    motors={m['id']:m for m in profile['motors']}
    for column,sid in enumerate(ids):
        if np.any(positions[:,column]<motors[sid]['safe_min']) or np.any(positions[:,column]>motors[sid]['safe_max']):
            raise ValueError(f'ID{sid}: target exceeds reviewed limits')
    return ids,positions,fps


def arm(bus,profile,output,guard,recover_disabled_ids=()):
    motors={m['id']:m for m in profile['motors']}
    plan=[];errors=[];events=[];completed=False
    try:
        # Review every axis before the first write, including held neck axes.
        for sid,motor in motors.items():
            guard()
            try:
                state=bus.snapshot(sid);neck=sid in profile['held_neck_ids']
                recovery=sid in recover_disabled_ids and state['torque_enable']==0
                target=startup_target(state,motor,neck,recovery)
                # Recovery only expands the measured startup corridor, never goal/device limits.
                envelope={**motor,'safe_min':min(motor['device_min'],state['position_raw']) if recovery else motor['device_min'],
                                  'safe_max':max(motor['device_max'],state['position_raw']) if recovery else motor['device_max']}
                if recovery and 'feedback_min' in envelope:
                    envelope.update(feedback_min=envelope['safe_min'],feedback_max=envelope['safe_max'])
                check_feedback(sid,feedback(state),envelope)
                plan.append(dict(id=sid,before=state,target=target,recovery=recovery,recovery_envelope=envelope))
            except Exception as exc:errors.append(str(exc))
        if errors:raise RuntimeError('Arming blocked: '+'; '.join(errors))
        for row in plan:
            sid=row['id'];motor=motors[sid];guard()
            state=bus.snapshot(sid)
            if state['torque_enable']!=row['before']['torque_enable'] or abs(state['position_raw']-row['before']['position_raw'])>2:
                raise RuntimeError(f'ID{sid}: startup state changed')
            startup_target(state,motor,sid in profile['held_neck_ids'],row['recovery'])
            check_feedback(sid,feedback(state),row['recovery_envelope'])
            if state['torque_enable']==0:
                # Prime a nearby position before enabling, preserving acceleration.
                speed=30 if row['recovery'] else 60
                bus.goals({sid:row['target']},{sid:speed})
                if bus.read(sid,42,6)!=struct.pack('<HHH',row['target'],0,speed):
                    raise RuntimeError(f'ID{sid}: goal RAM readback mismatch')
                guard()
                event=dict(id=sid,initial_target=row['target'],enabled=False)
                events.append(event)
                # Some firmware enables on the priming goal. Do not issue a redundant write.
                torque=bus.read(sid,40,1)
                if torque==b'\x00':
                    result,status=bus.sdk.write1ByteTxRx(sid,40,1)
                    event.update(enable_communication_result=result,enable_status=status)
                    torque=bus.read(sid,40,1)  # Read back even if the acknowledgement was lost; never retry the write.
                    if status:raise RuntimeError(f'ID{sid}: enable returned servo status {status}')
                if torque!=b'\x01':
                    raise RuntimeError(f'ID{sid}: torque enable verification failed')
                event['enabled']=True
                if row['recovery']:
                    # Re-establish the bounded recovery command in the enabled state.
                    # STS firmware may retain a primed RAM goal without beginning motion.
                    guard();bus.goals({sid:row['target']},{sid:speed})
                    if bus.read(sid,42,6)!=struct.pack('<HHH',row['target'],0,speed):
                        raise RuntimeError(f'ID{sid}: enabled-state goal readback mismatch')
                    event['recovery_goal_after_enable']=True
                    deadline=time.monotonic()+20;healthy=0
                    while healthy<2:
                        guard();measured=bus.feedback(sid)
                        check_feedback(sid,measured,row['recovery_envelope'])
                        events.append(dict(id=sid,recovery=True,target=row['target'],**measured))
                        low=motor.get('feedback_min',motor['safe_min']);high=motor.get('feedback_max',motor['safe_max'])
                        healthy=healthy+1 if abs(measured['position']-row['target'])<=20 and low<=measured['position']<=high else 0
                        if time.monotonic()>deadline:raise RuntimeError(f'ID{sid}: low-speed recovery timed out')
                        time.sleep(.05)
                    check_feedback(sid,measured,motor)
        for sid in profile['held_neck_ids']:
            motor=motors[sid]
            envelope={**motor,'safe_min':motor['device_min'],'safe_max':motor['device_max']}
            current=bus.feedback(sid)['position'];deadline=time.monotonic()+20
            while current!=2047:
                guard()
                if time.monotonic()>deadline:raise RuntimeError(f'ID{sid}: neck approach timeout')
                target=current+max(-5,min(5,2047-current))
                bus.goals({sid:target},{sid:60})
                time.sleep(.1)
                measured=bus.feedback(sid);check_feedback(sid,measured,envelope)
                if abs(measured['position']-target)>20:raise RuntimeError(f'ID{sid}: neck did not follow')
                events.append(dict(id=sid,target=target,position=measured['position']))
                current=target
            # Always normalize the held goal, including axes already at 2047.
            guard();bus.goals({sid:2047},{sid:60})
        for _ in range(5):
            for sid in profile['held_neck_ids']:
                guard();state=bus.snapshot(sid);check_feedback(sid,feedback(state),motors[sid])
                if abs(state['position_raw']-2047)>10 or state['goal_position_raw']!=2047:
                    raise RuntimeError(f'ID{sid}: neck not settled at hold target')
            time.sleep(.1)
        completed=True
    except BaseException as exc:
        if str(exc) not in errors:errors.append(str(exc))
        raise
    finally:
        (output/'arm_report.json').write_text(json.dumps(dict(completed=completed,plan=plan,errors=errors,events=events),ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','arm','play'])
    parser.add_argument('--targets',required=True,type=Path)
    parser.add_argument('--profile',required=True,type=Path)
    parser.add_argument('--audio',type=Path,default=ROOT/'exmple/0001.wav')
    parser.add_argument('--port',help='Override port after matching the adapter serial')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--supported-neck',action='store_true',help='Onsite operator confirms supporting Mojia neck during arming')
    parser.add_argument('--recover-disabled-ids',type=int,nargs='+',default=[],
                        help='Onsite-authorized slow recovery to 2047 for disabled axes 29/35/36 only')
    parser.add_argument('--max-seconds',type=float)
    args=parser.parse_args()
    profile=read_json(args.profile)
    if args.port:profile['port']=args.port
    ids,positions,fps=load_targets(args.targets,profile,args.audio,ROOT/'mojia/墨甲9.23.json')
    monitored=ids+profile['held_neck_ids'];live_preflight(profile,monitored)
    stop_file=ROOT/'outputs/mojia/STOP'
    if args.command=='arm' and not args.supported_neck:raise SystemExit('Arming requires onsite neck support confirmation')
    if args.recover_disabled_ids and (args.command!='arm' or not set(args.recover_disabled_ids)<={29,35,36}):
        raise SystemExit('Disabled recovery is available only during arm for IDs 29/35/36')
    if stop_file.exists():raise SystemExit('Operator STOP file exists')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
    if args.command=='play':
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        bus=OfficialSDKBus(profile['port'],profile['baud'],transaction_timeout_ms=profile['sdk_transaction_timeout_ms'])
        try:
            motors={m['id']:m for m in profile['motors']}
            for sid in profile['held_neck_ids']:
                state=bus.snapshot(sid)
                startup_target(state,motors[sid],True)
                check_feedback(sid,feedback(state),motors[sid])
                if state['torque_enable']!=1 or state['goal_position_raw']!=2047 or abs(state['position_raw']-2047)>10:
                    raise RuntimeError(f'ID{sid}: neck must already be homed and enabled')
        finally:bus.close()
        trajectory=np.column_stack([positions,np.full((len(positions),3),2047.)])
        result=play(profile,monitored,trajectory,fps,args.audio,args.output/'playback_log.json',stop_file,args.max_seconds)
        print('Playback completed:',result['completed'])
        return
    from xiaobai_runtime.feetech_sdk import OfficialSDKBus
    bus=OfficialSDKBus(profile['port'],profile['baud'],transaction_timeout_ms=profile['sdk_transaction_timeout_ms'])
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    def guard():
        if stopped or stop_file.exists():raise RuntimeError('Operator stop')
    previous={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        if args.command=='arm':arm(bus,profile,args.output,guard,args.recover_disabled_ids)
        else:
            motors={m['id']:m for m in profile['motors']};rows=[];errors=[]
            for sid in monitored:
                guard()
                try:
                    state=bus.snapshot(sid);rows.append(state)
                    startup_target(state,motors[sid],sid in profile['held_neck_ids'])
                    check_feedback(sid,feedback(state),motors[sid])
                    if state['torque_enable']!=1:raise ValueError(f'ID{sid}: torque is disabled')
                    if sid in profile['held_neck_ids'] and (state['goal_position_raw']!=2047 or abs(state['position_raw']-2047)>10):
                        raise ValueError(f'ID{sid}: neck is not homed')
                except Exception as exc:errors.append(str(exc))
            (args.output/'check.json').write_text(json.dumps(dict(ready=not errors,errors=errors,servos=rows),ensure_ascii=False,indent=2)+'\n')
            if errors:raise RuntimeError('Readiness blocked: '+'; '.join(errors))
        print(args.command,'completed')
    finally:
        bus.close()
        for sig,handler in previous.items():signal.signal(sig,handler)


if __name__=='__main__':main()
