#!/usr/bin/env python3
"""Replay a reviewed observation profile at 30/60fps, optionally monitoring a held neck."""
import argparse
import hashlib
import json
import math
import wave
from contextlib import ExitStack
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
    parser.add_argument('--audio-latency',type=float,help='Requested audio buffer latency in seconds (0.01..1)')
    parser.add_argument('--port',help='Serial port matched to the confirmed USB adapter')
    parser.add_argument('--audio-device',help='PortAudio output device name on this machine')
    parser.add_argument('--check',action='store_true',help='Read-only readiness check; no audio playback or motion')
    parser.add_argument('--track-person',action='store_true',help='Override horizontal eyes using the local camera service')
    parser.add_argument('--track-neck',action='store_true',help='Full-clip eyes plus confirmed ID32 smooth neck following')
    parser.add_argument('--neck-follow-trial',type=int,choices=[32],help='Horizontal neck trial ID32; ID34 is confirmed lateral tilt')
    parser.add_argument('--neck-follow-sign',type=int,choices=[-1,1],default=1)
    parser.add_argument('--neck-follow-offset-counts',type=int,choices=[20,80,160])
    parser.add_argument('--neck-speed-multiplier',type=float,choices=[1.,2.,3.],default=1.)
    parser.add_argument('--gaze-direction-trial',action='store_true',help='Direction test: bounded eyes, authored eye motion held neutral')
    parser.add_argument('--gaze-trial-offset-counts',type=int,choices=[40,80],default=40,
                        help='Direction trial range about neutral: +/-40 (60 counts/s) or +/-80 (120 counts/s)')
    parser.add_argument('--tracking-config',type=Path,default=ROOT/'xiaobai/person_gaze.json')
    parser.add_argument('--clip',type=Path,default=ROOT/'outputs/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json')
    parser.add_argument('--audio',type=Path,default=ROOT/'0001.wav')
    args=parser.parse_args()
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds<=0):
        parser.error('--max-seconds must be finite and positive')
    if args.track_neck:
        if args.neck_follow_trial or args.gaze_direction_trial:
            raise ValueError('Full-clip neck following cannot be combined with direction trials')
        args.track_person=True
        args.monitor_neck_hold=True
    if args.neck_follow_trial:
        if args.gaze_direction_trial:raise ValueError('Do not combine eye direction and neck direction trials')
        args.track_person=True
        args.monitor_neck_hold=True
        if args.max_seconds is None:args.max_seconds=30
        if not 5<=args.max_seconds<=60:raise ValueError('Neck trial must be 5..60 seconds')
    if args.gaze_direction_trial:
        args.track_person=True
        if args.max_seconds is None:args.max_seconds=20
        if not 0<args.max_seconds<=30:raise ValueError('Direction trial must be at most 30 seconds')
    if args.output.exists():raise SystemExit('Use a new output directory to preserve evidence')
    profile=read_json(args.profile)
    if args.audio_latency is not None:
        if not .01<=args.audio_latency<=1:raise SystemExit('Audio latency must be 0.01..1 seconds')
        profile['audio_latency_s']=args.audio_latency
    if args.port is not None:
        profile['port']=args.port
    if args.audio_device is not None:
        profile['audio_device']=args.audio_device
    if args.feedback_polls_per_frame is not None:
        profile['feedback_polls_per_frame']=args.feedback_polls_per_frame
    calibration=ROOT/'xiaobai/xiaobai标定文件.json'
    if profile.get('calibration_sha256')!=hashlib.sha256(calibration.read_bytes()).hexdigest():
        raise SystemExit('Calibration differs from the reviewed profile')
    values,source_fps,order,shapes=inputs(args.clip,calibration,ROOT/'xiaobai/blendshape_channels_61.csv')
    if source_fps<args.fps or source_fps%args.fps:
        raise SystemExit('Requested fps must evenly divide source fps')
    tracking_config=read_json(args.tracking_config) if args.track_person else None
    neck_settings=None
    if args.neck_follow_trial or args.track_neck:
        from xiaobai_runtime.person_gaze import prepare_neck_follow_trial
        offset=args.neck_follow_offset_counts
        if offset is None:offset=160 if args.track_neck else 20
        profile,neck_settings=prepare_neck_follow_trial(profile,32,args.neck_follow_sign,offset,
                                                       args.neck_speed_multiplier)
    if args.gaze_direction_trial:
        from xiaobai_runtime.person_gaze import prepare_direction_trial
        tracking_config,profile=prepare_direction_trial(tracking_config,profile,args.gaze_trial_offset_counts)
    ids,trajectory,report=compile_trajectory(values,order,shapes,profile,'face_eyes')
    trajectory=trajectory[::int(source_fps/args.fps)]
    if args.gaze_direction_trial:
        for axis in tracking_config['control']['axes']:
            trajectory[:,ids.index(axis['id'])]=axis['neutral']
        report['authored_horizontal_eyes_held_neutral']=True
    if tracking_config is not None:
        from xiaobai_runtime.person_gaze import validate_config
        # Reject incomplete calibration before even the read-only neck serial check.
        validate_config(tracking_config,live=True,profile=profile,ids=ids,direction_trial=args.gaze_direction_trial)
    with wave.open(str(args.audio),'rb') as audio:
        if audio.getsampwidth()!=2:raise ValueError('Audio must be PCM16 WAV')
        duration=audio.getnframes()/audio.getframerate()
    if abs(len(trajectory)/args.fps-duration)>1/args.fps:
        raise ValueError('Clip/audio duration mismatch')
    if args.monitor_neck_hold:
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        bus=OfficialSDKBus(profile['port'],profile['baud'],transaction_timeout_ms=profile.get('sdk_transaction_timeout_ms'),
                          servo_status_policy='record_only' if profile.get('feedback_policy')=='record_only' else 'raise')
        try:
            for sid in [32,33,34]:
                s=bus.snapshot(sid)
                if (s['model_raw']!=11272 or s['mode']!=0 or s['torque_enable']!=1
                        or s['goal_position_raw']!=2047 or abs(s['position_raw']-2047)>10):
                    raise RuntimeError('Neck must already be homed and enabled')
                if neck_settings is not None and sid==neck_settings['id']:
                    if (s['configured_min']>neck_settings['safe_min']
                            or s['configured_max']<neck_settings['safe_max']):
                        raise ValueError('Neck trial exceeds the current read-only device limits')
        finally:bus.close()
        # Other neck axes retain their existing goals; a trial owns one selected axis.
        profile.setdefault('diagnostic_command_ids',list(ids))
        allowed=set(ids)|({32} if neck_settings is not None else set())
        if not set(profile['diagnostic_command_ids'])<=allowed:raise ValueError('Unexpected command IDs')
        ids=ids+[32,33,34]
        trajectory=np.column_stack([trajectory,np.full((len(trajectory),3),2047.)])
    with ExitStack() as resources:
        gaze=None
        if tracking_config is not None:
            from xiaobai_runtime.person_gaze import GazeOverride
            gaze=GazeOverride(tracking_config,ids,profile,direction_trial=args.gaze_direction_trial)
            resources.callback(gaze.close)
            if neck_settings is not None:
                from xiaobai_runtime.person_gaze import NeckFollowTrial
                run_duration=min(duration,args.max_seconds) if args.max_seconds is not None else duration
                gaze=NeckFollowTrial(gaze,profile,ids,neck_settings,run_duration,full_song=args.track_neck)
            gaze.receiver.wait_ready()
        if args.check:
            from xiaobai_runtime.mapping import live_preflight
            live_preflight(profile,ids)
            print(json.dumps(dict(ready=True,clip=str(args.clip),audio=str(args.audio),
                duration_seconds=duration,playback_fps=args.fps,motion_gain=profile['motion_gain'],
                feedback_policy=profile.get('feedback_policy','stop_on_fault'),
                neck_hold_checked=args.monitor_neck_hold,person_gaze_checked=args.track_person,
                gaze_direction_trial=args.gaze_direction_trial,track_neck=args.track_neck,
                neck_follow_settings=neck_settings),ensure_ascii=False,indent=2))
            return
        args.output.mkdir(parents=True)
        (args.output/'profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
        if tracking_config is not None:
            (args.output/'person_gaze.json').write_text(json.dumps(tracking_config,ensure_ascii=False,indent=2)+'\n')
        (args.output/'plan.json').write_text(json.dumps(dict(source_fps=source_fps,
            playback_fps=args.fps,frames=len(trajectory),motion_gain=profile['motion_gain'],
            feedback_policy=profile.get('feedback_policy','stop_on_fault'),person_gaze_enabled=args.track_person,
            gaze_direction_trial=args.gaze_direction_trial,
            gaze_trial_offset_counts=args.gaze_trial_offset_counts if args.gaze_direction_trial else None,
            neck_follow_trial=neck_settings,
            track_neck=args.track_neck,
            neck_hold_monitored=args.monitor_neck_hold,trajectory_report=report),ensure_ascii=False,indent=2)+'\n')
        result=play(profile,ids,trajectory,args.fps,args.audio,args.output/'playback_log.json',
                    ROOT/'outputs/xiaobai/STOP',args.max_seconds,gaze_override=gaze)
        print('Playback completed:',result['completed'])


if __name__=='__main__':main()
