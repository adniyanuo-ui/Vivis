#!/usr/bin/env python3
"""Replay a reviewed observation profile at 30/60fps, optionally monitoring a held neck."""
import argparse
import hashlib
import json
import wave
from pathlib import Path
import numpy as np
from xiaobai_runtime.mapping import ROOT,read_json,inputs,compile_trajectory
from xiaobai_runtime.playback import play


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--fps',type=int,choices=[30,60],default=30)
    parser.add_argument('--monitor-neck-hold',action='store_true')
    parser.add_argument('--max-seconds',type=float)
    parser.add_argument('--feedback-polls-per-frame',type=int,choices=[1,2,3,4])
    parser.add_argument('--check',action='store_true',help='Read-only readiness check; no audio playback or motion')
    parser.add_argument('--clip',type=Path,default=ROOT/'outputs/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json')
    parser.add_argument('--audio',type=Path,default=ROOT/'0001.wav')
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Use a new output directory to preserve evidence')
    profile=read_json(args.profile)
    if args.feedback_polls_per_frame is not None:
        profile['feedback_polls_per_frame']=args.feedback_polls_per_frame
    calibration=ROOT/'xiaobai/xiaobai标定文件.json'
    if profile.get('calibration_sha256')!=hashlib.sha256(calibration.read_bytes()).hexdigest():
        raise SystemExit('Calibration differs from the reviewed profile')
    values,source_fps,order,shapes=inputs(args.clip,calibration,ROOT/'xiaobai/blendshape_channels_61.csv')
    if source_fps<args.fps or source_fps%args.fps:
        raise SystemExit('Requested fps must evenly divide source fps')
    ids,trajectory,report=compile_trajectory(values,order,shapes,profile,'face_eyes')
    trajectory=trajectory[::int(source_fps/args.fps)]
    with wave.open(str(args.audio),'rb') as audio:
        if audio.getsampwidth()!=2:raise ValueError('Audio must be PCM16 WAV')
        duration=audio.getnframes()/audio.getframerate()
    if abs(len(trajectory)/args.fps-duration)>1/args.fps:
        raise ValueError('Clip/audio duration mismatch')
    if args.monitor_neck_hold:
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        bus=OfficialSDKBus(profile['port'],profile['baud'],transaction_timeout_ms=profile.get('sdk_transaction_timeout_ms'))
        try:
            for sid in [32,33,34]:
                s=bus.snapshot(sid)
                if (s['model_raw']!=11272 or s['mode']!=0 or s['torque_enable']!=1
                        or s['goal_position_raw']!=2047 or abs(s['position_raw']-2047)>10):
                    raise RuntimeError('Neck must already be homed and enabled')
        finally:bus.close()
        # Only face axes are commanded; neck axes retain their existing goal.
        profile.setdefault('diagnostic_command_ids',list(ids))
        if not set(profile['diagnostic_command_ids'])<=set(ids):raise ValueError('Unexpected command IDs')
        ids=ids+[32,33,34]
        trajectory=np.column_stack([trajectory,np.full((len(trajectory),3),2047.)])
    if args.check:
        from xiaobai_runtime.mapping import live_preflight
        live_preflight(profile,ids)
        print(json.dumps(dict(ready=True,clip=str(args.clip),audio=str(args.audio),
            duration_seconds=duration,playback_fps=args.fps,motion_gain=profile['motion_gain'],
            feedback_policy=profile.get('feedback_policy','stop_on_fault'),
            neck_hold_checked=args.monitor_neck_hold),ensure_ascii=False,indent=2))
        return
    args.output.mkdir(parents=True)
    (args.output/'profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
    (args.output/'plan.json').write_text(json.dumps(dict(source_fps=source_fps,
        playback_fps=args.fps,frames=len(trajectory),motion_gain=profile['motion_gain'],
        feedback_policy=profile.get('feedback_policy','stop_on_fault'),
        neck_hold_monitored=args.monitor_neck_hold,trajectory_report=report),ensure_ascii=False,indent=2)+'\n')
    result=play(profile,ids,trajectory,args.fps,args.audio,args.output/'playback_log.json',
                ROOT/'outputs/xiaobai/STOP',args.max_seconds)
    print('Playback completed:',result['completed'])


if __name__=='__main__':main()
