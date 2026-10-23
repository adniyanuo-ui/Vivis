#!/usr/bin/env python3
"""Repeat the complete Mojia song with one bus owner and live eye/neck tracking."""
import argparse
import json
import signal
import time
from pathlib import Path
import numpy as np

from mojia_player import load_targets, startup_target, feedback
from xiaobai_runtime.mapping import ROOT, read_json, live_preflight
from xiaobai_runtime.playback import play, check_feedback
from xiaobai_runtime.person_gaze import TargetReceiver
from xiaobai_runtime.mojia_performance import tracking_profile, MojiaPlaybackGaze


def prepare(config,profile,ids,positions,authorize_tongue=False):
    monitored=list(ids)+[32,33,34]
    if len(set(monitored))!=len(monitored):raise ValueError('Repeated motor ownership')
    p=tracking_profile(config,profile,monitored)
    motors={m['id']:m for m in p['motors']}
    for sid in [12,13,27,28,29,31]:motors[sid]['max_speed_counts_s']=900
    if authorize_tongue:
        # Existing operator authorization for ID29 only; preserve hardware
        # alarms and device temperature ceiling. Never globally disable guards.
        motors[29].update(max_load_raw=1023,max_temperature_c=80,initial_settle_tolerance_counts=32)
        p['acceptance_exception']=dict(id=29,load_threshold_raw=1023,temperature_ceiling_c=80,
                                      basis='Previously authorized ID29 full-song observation')
    trajectory=np.column_stack([positions,np.full((len(positions),3),2047.)])
    for sid in p['gaze_owner_ids']:trajectory[:,monitored.index(sid)]=2047.
    live_preflight(p,monitored)
    return p,monitored,trajectory


def recover_tongue(bus,motor,guard,events):
    s=bus.snapshot(29)
    if s['torque_enable']==1:return
    if s['model_raw']!=521 or s['mode']!=0 or not 0<=s['position_raw']<=4095:
        raise RuntimeError('ID29 startup hardware mismatch')
    if (s['configured_min'],s['configured_max'])!=(motor['device_min'],motor['device_max']):
        raise RuntimeError('ID29 device limits changed')
    envelope={**motor,'feedback_min':min(s['position_raw'],motor['device_min']),
              'feedback_max':max(s['position_raw'],motor['device_max'])}
    check_feedback(29,feedback(s),envelope)
    guard();bus.goals({29:2047},{29:900})
    if bus.read(29,40,1)==b'\x00':
        result,status=bus.sdk.write1ByteTxRx(29,40,1)
        enabled=bus.read(29,40,1)
        if result or status or enabled!=b'\x01':raise RuntimeError('ID29 enable failed')
    guard();bus.goals({29:2047},{29:900});deadline=time.monotonic()+10;healthy=0
    while time.monotonic()<deadline:
        guard();f=bus.feedback(29);events.append(f);check_feedback(29,f,envelope)
        healthy=healthy+1 if abs(f['position']-2047)<=32 else 0
        if healthy>=2:
            check_feedback(29,f,motor)
            return
        time.sleep(.1)
    raise RuntimeError('ID29 did not reach its authorized startup corridor')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'mojia/person_gaze.json')
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--targets',type=Path,required=True)
    parser.add_argument('--audio',type=Path,default=ROOT/'exmple/0001.wav')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--authorized-tongue-observation',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if not 1<=args.repeat<=10:parser.error('Repeat must be 1..10')
    config,original=read_json(args.config),read_json(args.profile)
    ids,positions,fps=load_targets(args.targets,original,args.audio,ROOT/'mojia/墨甲9.23.json')
    profile,ids,trajectory=prepare(config,original,ids,positions,args.authorized_tongue_observation)
    if args.check:
        print(json.dumps(dict(offline_valid=True,monitored=ids,commanded=profile['diagnostic_command_ids'],
                             gaze_owner_ids=profile['gaze_owner_ids'],fps=fps,repeat=args.repeat)))
        return
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in [('config',config),('profile',profile)]:
        (args.output/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    stop_file=ROOT/'outputs/mojia/STOP';stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    def guard():
        if stopped or stop_file.exists():raise RuntimeError('Operator stop')
    summary=dict(completed=False,rounds=[],error=None,startup_events=[],tongue_disabled_after=False)
    receiver=bus=None
    motors={m['id']:m for m in profile['motors']}
    try:
        guard();receiver=TargetReceiver(config);receiver.wait_ready()
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        bus=OfficialSDKBus(profile['port'],profile['baud'],transaction_timeout_ms=20)
        snapshots=[bus.snapshot(sid) for sid in ids]
        summary['startup_snapshots']=snapshots
        # Review every axis before making any startup write.
        for s in snapshots:
            sid=s['id'];m=motors[sid]
            recovery=sid==29 and s['torque_enable']==0 and args.authorized_tongue_observation
            startup_target(s,m,sid in [32,33,34],recovery)
            if not recovery:
                if s['torque_enable']!=1:raise RuntimeError(f'ID{sid}: torque disabled')
                check_feedback(sid,feedback(s),m)
            if sid in [12,13,27,28,29,31]:
                ram=bytes.fromhex(s['state_40_70_hex'])
                if ram[1]!=0 or int.from_bytes(ram[6:8],'little')!=900:
                    raise RuntimeError(f'ID{sid}: expected saved mouth acceleration0/speed900')
        if args.authorized_tongue_observation:
            recover_tongue(bus,motors[29],guard,summary['startup_events'])
        bus.close();bus=None
        for lap in range(1,args.repeat+1):
            guard();directory=args.output/f'round_{lap}';directory.mkdir()
            override=MojiaPlaybackGaze(config,profile,ids,receiver,
                ROOT/'outputs/mojia/right_camera_tracking/control_status.json',lap,args.repeat)
            print(f'ROUND {lap}/{args.repeat} START',flush=True)
            result=play(profile,ids,trajectory,fps,args.audio,directory/'playback_log.json',stop_file,gaze_override=override)
            summary['rounds'].append(dict(round=lap,completed=result['completed'],
                full_clip_requested=result['full_clip_requested'],audio_seconds_at_stop=result['audio_seconds_at_stop'],
                sent_frames=result['sent_frames'],skipped_source_frames=result['skipped_source_frames']))
            (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(f'ROUND {lap}/{args.repeat} COMPLETE',flush=True)
        summary['completed']=True
    except BaseException as exc:
        summary['error']=str(exc)
        raise
    finally:
        if bus is not None:bus.close()
        if receiver is not None:receiver.close()
        # Established tongue stop policy: unload only ID29 after the show;
        # preserve the supported neck's torque and its final bounded goal.
        if args.authorized_tongue_observation:
            try:
                from xiaobai_runtime.feetech_sdk import OfficialSDKBus
                cleanup=OfficialSDKBus(profile['port'],profile['baud'],transaction_timeout_ms=20)
                try:
                    result,status=cleanup.sdk.write1ByteTxRx(29,40,0)
                    summary['tongue_disabled_after']=cleanup.read(29,40,1)==b'\x00'
                    summary['tongue_disable_result']=[result,status]
                finally:cleanup.close()
            except Exception as exc:summary['cleanup_error']=str(exc)
        for sig,handler in handlers.items():signal.signal(sig,handler)
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
