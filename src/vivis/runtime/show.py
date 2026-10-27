#!/usr/bin/env python3
"""Repeat the complete Mojia song with one bus owner and live eye/neck tracking."""
import argparse
import copy
import json
import signal
import time
import subprocess
import re
from pathlib import Path
import numpy as np

from vivis.runtime.trajectory import load_targets, startup_target, feedback
from vivis.motion.mapping import ROOT, read_json, live_preflight
from vivis.runtime.playback import play, check_feedback,startup_feedback
from vivis.vision.policy import TargetReceiver
from vivis.motion.ownership import tracking_profile, MojiaPlaybackGaze
from vivis.runtime.homing import (home_all,validate_home_axes,home_check,homing_feedback_motor,
                                 write_ram_verified,confirm_home_recoveries,recovery_policy)
from vivis.runtime.startup import startup_plan,restore_startup_settings
from vivis.runtime.telemetry import apply_policy,WarningRecorder,verify_sample

MOUTH_IDS=(12,13,27,28,29,31)


def mouth_gain_targets(profile,ids,positions,gain):
    """Scale mouth displacement from each calibrated neutral, retaining limits."""
    if type(gain) not in (int,float) or not np.isfinite(gain) or gain<=0:
        raise ValueError('Mouth gain must be finite and positive')
    result=np.array(positions,dtype=float,copy=True)
    if result.ndim!=2 or result.shape[1]!=len(ids) or not len(result) or not np.isfinite(result).all():
        raise ValueError('Invalid mouth target matrix')
    motors={m['id']:m for m in profile['motors']};report=dict(gain=float(gain),axes=[])
    gain_ids=profile.get('mouth_gain_ids',MOUTH_IDS)
    if not gain_ids or len(set(gain_ids))!=len(gain_ids):
        raise ValueError('Invalid mouth gain axis group')
    for sid in gain_ids:
        if sid not in ids:raise ValueError(f'Missing mouth axis ID{sid}')
        j=ids.index(sid);m=motors[sid];neutral=m['neutral'];before=result[:,j].copy()
        low=max(m['safe_min'],m['device_min']);high=min(m['safe_max'],m['device_max'])
        if not low<=neutral<=high:raise ValueError(f'ID{sid}: neutral outside mouth command limits')
        with np.errstate(over='ignore',invalid='ignore'):
            scaled=neutral+(before-neutral)*gain
        if not np.isfinite(scaled).all():raise ValueError('Mouth gain overflow')
        limited=np.clip(scaled,low,high);result[:,j]=limited
        report['axes'].append(dict(id=sid,neutral=neutral,limits=[low,high],
            original_range=[float(before.min()),float(before.max())],
            requested_range=[float(scaled.min()),float(scaled.max())],
            output_range=[float(limited.min()),float(limited.max())],
            clipped_frames=int(np.count_nonzero(scaled!=limited)),total_frames=len(result)))
    return result,report


def configure_show_volume(device):
    """Three times the recorded original output amplitude, without PCM clipping.

    Fixed reference: USB Speaker step15=-22.5dB. Step23=-10.5dB gives +12dB;
    a 0.7536 digital trim makes the net amplitude factor exactly 3. Repeated
    invocations set the same value rather than multiplying the current volume.
    """
    import sounddevice as sd
    info=sd.query_devices(device,'output')
    match=re.search(r'\(hw:(\d+),\d+\)',info['name'])
    if 'USB Audio Device' not in info['name'] or match is None:
        raise RuntimeError('Expected the reviewed USB Audio Device ALSA output')
    card=match.group(1)
    before=subprocess.run(['amixer','-c',card,'sget','Speaker'],check=True,capture_output=True,text=True).stdout
    if 'Limits: Playback 0 - 30' not in before:
        raise RuntimeError('USB speaker scale changed; review volume before playback')
    after=subprocess.run(['amixer','-c',card,'sset','Speaker','23','unmute'],check=True,capture_output=True,text=True).stdout
    levels=re.findall(r'Playback 23 \[\d+%\] \[(-?[\d.]+)dB\] \[on\]',after)
    if len(levels)!=2 or any(abs(float(v)+10.5)>.01 for v in levels):
        raise RuntimeError('USB speaker volume readback does not match -10.5dB')
    gain=3./10**(12/20)
    return gain,dict(reference_step=15,reference_db=-22.5,speaker_step=23,speaker_db=-10.5,
                     digital_trim=gain,output_amplitude_factor=3.,before=before,after=after)


def prepare(config,profile,ids,positions,authorize_tongue=False,mouth_gain=1.,telemetry_policy=None,mode='eyes-neck'):
    if mode not in ('face','eyes','eyes-neck'):
        raise ValueError('Unknown developer mode')
    monitored=list(ids)+[32,33,34]
    if len(set(monitored))!=len(monitored):raise ValueError('Repeated motor ownership')
    if mode=='face':
        p=copy.deepcopy(profile)
        p['diagnostic_command_ids']=[sid for sid in profile['diagnostic_command_ids'] if sid not in (7,8,22,23)]
        p.update(gaze_owner_ids=[],held_eye_ids=[7,8,22,23],held_neck_ids=[32,33,34])
    else:
        p=tracking_profile(config,profile,monitored,neck=mode=='eyes-neck')
    p['developer_mode']=mode
    # Allow UART transmission and servo broadcast parsing to finish before a
    # feedback query. The SDK accounts for wire time in addition to this gap.
    p['post_goal_quiet_ms']=1
    p['scheduled_feedback']=True
    motors={m['id']:m for m in p['motors']}
    if authorize_tongue:
        # Existing operator authorization for ID29 only; preserve hardware
        # alarms and device temperature ceiling. Never globally disable guards.
        motors[29].update(max_load_raw=1023,max_temperature_c=80,initial_settle_tolerance_counts=40)
        p['acceptance_exception']=dict(id=29,load_threshold_raw=1023,temperature_ceiling_c=80,
                                      initial_settle_tolerance_counts=40,
                                      basis='Authorized ID29 observation; measured 36-count startup residual within device feedback range')
    if telemetry_policy is not None:
        apply_policy(p,telemetry_policy)
        if authorize_tongue:
            p['acceptance_exception']=dict(id=29,initial_settle_tolerance_counts=40,
                basis='Existing authorized startup residual; load/temperature use the common telemetry policy')
    trajectory=np.column_stack([positions,np.full((len(positions),3),2047.)])
    for sid in p['gaze_owner_ids']:trajectory[:,monitored.index(sid)]=2047.
    for sid in p.get('held_eye_ids',[]):trajectory[:,monitored.index(sid)]=2047.
    trajectory,p['mouth_gain_report']=mouth_gain_targets(p,monitored,trajectory,mouth_gain)
    p['mouth_motion_gain']=mouth_gain
    live_preflight(p,monitored)
    return p,monitored,trajectory


def recover_tongue(bus,motor,guard,events,speed=1000):
    s=bus.snapshot(29)
    if s['torque_enable']==1:return
    if s['model_raw']!=521 or s['mode']!=0 or not 0<=s['position_raw']<=4095:
        raise RuntimeError('ID29 startup hardware mismatch')
    if (s['configured_min'],s['configured_max'])!=(motor['device_min'],motor['device_max']):
        raise RuntimeError('ID29 device limits changed')
    envelope={**motor,'feedback_min':min(s['position_raw'],motor['device_min']),
              'feedback_max':max(s['position_raw'],motor['device_max'])}
    warnings=WarningRecorder(events)
    def checked(state):
        fresh,alerts=verify_sample(bus.feedback,29,state,lambda sample:check_feedback(29,sample,envelope),
            motor.get('telemetry_policy'),events,should_stop=lambda:guard() or False)
        warnings.observe(29,fresh,alerts)
        return fresh
    checked(feedback(s))
    guard();bus.goals({29:2047},{29:speed})
    if bus.read(29,40,1)==b'\x00':
        events.append(dict(kind='enable',**write_ram_verified(bus,29,40,1)))
    guard();bus.goals({29:2047},{29:speed});deadline=time.monotonic()+10;healthy=0;f=feedback(s)
    while time.monotonic()<deadline:
        guard();f=checked(bus.feedback(29));events.append(f)
        healthy=healthy+1 if abs(f['position']-2047)<=motor['initial_settle_tolerance_counts'] else 0
        if healthy>=2:
            check_feedback(29,f,motor)
            return
        time.sleep(.1)
    raise RuntimeError(f'ID29 did not reach its authorized startup corridor: '
        f'target=2047 position={f["position"]} tolerance={motor["initial_settle_tolerance_counts"]} '
        f'load={f["load_raw"]&1023} temperature={f["temperature_c"]} status={f["status_register"]}')


def recovery_clearance_plan(inventory,motors):
    """Machine-specific coupling is configured with the homing inventory."""
    c=inventory.get('recovery_clearance')
    if c is None:return None
    sid=c['moving_id'];recovering=c['recovering_id']
    if sid==recovering or sid not in motors or recovering not in motors:
        raise ValueError('Invalid recovery clearance ownership')
    position=c['position'];speed=c['max_speed_counts_s'];tolerance=c['settle_tolerance_counts']
    m=motors[sid]
    if (type(position) is not int or not max(m['safe_min'],m['device_min'])<=position<=min(m['safe_max'],m['device_max'])
            or type(speed) is not int or not 1<=speed<=m['max_speed_counts_s']
            or type(tolerance) is not int or not 1<=tolerance<=40):
        raise ValueError('Recovery clearance exceeds reviewed position/speed/tolerance limits')
    axis=dict(id=sid,neutral=position,neutral_confirmed=True,model_raw_expected=m['model_raw_expected'])
    return c,axis,{**m,'neutral':position,'initial_settle_tolerance_counts':tolerance}


def prepare_recovery_clearance(bus,plan,snapshots,motors,guard,report,home_speed,position_recovery=None):
    if plan is None:return
    c,axis,motor=plan;states={s['id']:s for s in snapshots}
    recovering=states[c['recovering_id']];m=motors[c['recovering_id']]
    if recovering['torque_enable']!=0 or abs(recovering['position_raw']-m['neutral'])<=m.get('initial_settle_tolerance_counts',20):
        report.update(required=False)
        return
    report.update(required=True,reason='Disabled recovery axis requires the configured mechanical clearance')
    print(f'RECOVERY CLEARANCE ID{axis["id"]} -> {axis["neutral"]} before ID{c["recovering_id"]} recovery',flush=True)
    home_all(bus,[axis],{axis['id']:motor},guard,report,min(home_speed,c['max_speed_counts_s']),
             keep_enabled=[axis['id']],telemetry_policy=motor.get('telemetry_policy'),position_recovery=position_recovery)


def prepare_round(args,profile,ids,home_axes,hardware_preset,clearance_plan,telemetry_policy,guard,report,directory,publish):
    """Fresh full-inventory checks, tuning restoration and homing for one round."""
    home_ids=[a['id'] for a in home_axes];home_neutrals={a['id']:a['neutral'] for a in home_axes}
    motors={m['id']:m for m in profile['motors']}
    report.update(startup_events=[],homing={})
    warning_recorder=WarningRecorder(report.setdefault('telemetry_warnings',[]))
    bus=None
    try:
        guard()
        print(f'TELEMETRY load>{telemetry_policy["load_warning_raw"]}: warning; '
              f'temperature>{telemetry_policy["temperature_warning_c"]}C: warning; '
              f'temperature>{telemetry_policy["temperature_stop_c"]}C: stop after '
              f'{telemetry_policy.get("temperature_stop_consecutive_samples",1)} consecutive samples; hardware protection preserved',flush=True)
        if args.usb_volume_times_three and not args.home_only:
            profile['audio_gain'],report['audio_volume']=configure_show_volume(profile['audio_device'])
            (directory/'profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
        from vivis.hardware.feetech.sdk import OfficialSDKBus
        bus=OfficialSDKBus(profile['port'],profile['baud'],transaction_timeout_ms=20,
                           trace_path=directory/'homing_serial_trace.json',
                           snapshot_chunk_bytes=profile.get('snapshot_chunk_bytes'))
        snapshot_deadline=time.monotonic()+10
        snapshots=[startup_feedback(bus.snapshot,sid,snapshot_deadline,
                    report.setdefault('startup_communication_recoveries',[]),should_stop=lambda:guard() or False)
                   for sid in home_ids]
        validate_home_axes(home_axes,snapshots)
        original_speeds={s['id']:int.from_bytes(bytes.fromhex(s['state_40_70_hex'])[6:8],'little') for s in snapshots}
        report['startup_snapshots']=snapshots
        position_policy=getattr(args,'home_position_recovery',None)
        position_plans,confirmed=confirm_home_recoveries(bus,snapshots,position_policy,guard,
            report.setdefault('disabled_position_recoveries',[]),telemetry_policy)
        # Review every axis before making any startup write.
        for s in snapshots:
            sid=s['id']
            if sid not in motors:
                fresh,alerts=verify_sample(bus.feedback,sid,confirmed.get(sid,feedback(s)),
                    lambda sample:home_check(sample,s,home_neutrals[sid],20,telemetry_policy,position_plans.get(sid)),telemetry_policy,
                    report.setdefault('sensor_confirmations',[]),should_stop=lambda:guard() or False)
                warning_recorder.observe(sid,fresh,alerts)
                continue
            m=motors[sid]
            recovery=sid==29 and s['torque_enable']==0 and args.authorized_tongue_observation
            if (s['configured_min'],s['configured_max'])!=(m['device_min'],m['device_max']):
                raise RuntimeError(f'ID{sid}: device limits changed; '
                    f'actual=[{s["configured_min"]},{s["configured_max"]}] '
                    f'expected=[{m["device_min"]},{m["device_max"]}]; review home/task configuration')
            if recovery:startup_target(s,m,False,True)
            if not recovery:
                # The explicitly requested all-axis home primes disabled axes
                # at measured positions before enabling; playback still requires torque.
                def validate(sample):
                    home_check(sample,s,m['neutral'],m.get('initial_settle_tolerance_counts',20),telemetry_policy,position_plans.get(sid))
                    return check_feedback(sid,sample,homing_feedback_motor(m,s,position_plans.get(sid)))
                fresh,alerts=verify_sample(bus.feedback,sid,confirmed.get(sid,feedback(s)),validate,telemetry_policy,
                    report.setdefault('sensor_confirmations',[]),should_stop=lambda:guard() or False)
                warning_recorder.observe(sid,fresh,alerts)
        if hardware_preset is not None:
            # Validate the complete inventory before the first parameter write.
            startup_plan(hardware_preset,snapshots,args.home_speed)
            original_speeds=restore_startup_settings(bus,hardware_preset,snapshots,args.home_speed,
                guard,report.setdefault('hardware_restore',{}))
            print('STARTUP hardware tuning restored and verified; homing speed remains capped',flush=True)
        if args.authorized_tongue_observation:
            prepare_recovery_clearance(bus,clearance_plan,snapshots,motors,guard,
                report.setdefault('recovery_clearance',{}),args.home_speed,position_policy)
            recover_tongue(bus,motors[29],guard,report['startup_events'],speed=min(args.home_speed,motors[29]['max_speed_counts_s']))
        publish('homing')
        print(f'HOME {len(home_axes)} AXES START speed_cap={args.home_speed}',flush=True)
        home_all(bus,home_axes,motors,guard,report['homing'],args.home_speed,original_speeds,keep_enabled=ids,
                 telemetry_policy=telemetry_policy,position_recovery=position_policy)
        print('HOME COMPLETE; original speed settings restored',flush=True)
        bus.close();bus=None
    finally:
        if bus is not None:bus.close()


def unload_tongue(profile,report):
    """Established per-run ending: unload ID29, retain the neck's holding torque."""
    from vivis.hardware.feetech.sdk import OfficialSDKBus
    bus=OfficialSDKBus(profile['port'],profile['baud'],transaction_timeout_ms=20)
    try:
        result,status=bus.sdk.write1ByteTxRx(29,40,0)
        report['tongue_disabled_after']=bus.read(29,40,1)==b'\x00'
        report['tongue_disable_result']=[result,status]
        if not report['tongue_disabled_after'] or status:
            raise RuntimeError('ID29 end-of-round unload could not be verified')
    finally:bus.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',type=Path,help='Developer robot definition; resolves canonical hardware and work assets')
    parser.add_argument('--work',default='wz2')
    parser.add_argument('--mode',choices=('face','eyes','eyes-neck'),default='eyes-neck')
    parser.add_argument('--camera',choices=('left','right'),default='right')
    parser.add_argument('--camera-device',help='Explicit video capture node, required for an unconfigured camera')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--mouth-gain',type=float,default=1.,help='Mouth displacement multiplier, clipped to existing command limits')
    parser.add_argument('--home-speed',type=int,default=60,help='Temporary neutral homing speed cap in counts/s')
    parser.add_argument('--home-only',action='store_true',help='Home all inventory axes; no camera, audio or task execution')
    parser.add_argument('--face-fps',type=int,choices=(30,60),default=60,
                        help='Facial song sampling; eye/neck control remains at the source 60 Hz')
    parser.add_argument('--control-fps',type=int,choices=(30,60),default=60,
                        help='Actual combined control rate; source timing and audio stay unchanged')
    parser.add_argument('--baud',type=int,choices=(250000,500000,1000000),
                        help='Host baud; all servos must already be verified at this baud')
    parser.add_argument('--upstream-trajectory',choices=('on','off'),help='Optional ID32 upstream trajectory planning')
    parser.add_argument('--authorized-tongue-observation',action='store_true')
    parser.add_argument('--usb-volume-times-three',action='store_true',help='Use the saved original USB volume reference, never cumulative gain')
    parser.add_argument('--hold-vertical-eyes',action='store_true',help='Hold ID7/22: neither song nor tracking may send them position commands')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=args.repeat<=20:parser.error('Repeat must be 1..20')
    if not 1<=args.home_speed<=32767:parser.error('Home speed must be within 1..32767')
    from vivis.runtime.robot import Robot
    robot=Robot(args.robot)
    work=robot.work(args.work)
    args.targets,args.audio=work['targets'],work['audio']
    original=robot.profile(args.work)
    config=robot.tracking(args.camera,args.camera_device or ('unused' if args.mode=='face' else None))
    from vivis.motion.upstream import configure
    trajectory_settings=configure(config,args.mode,args.upstream_trajectory,args.home_only)
    telemetry_policy=robot.telemetry()
    calibration=robot.calibration
    if args.hold_vertical_eyes:
        config['control']['vertical_enabled']=False
        config['control']['axes']=[a for a in config['control']['axes'] if a['id'] not in (7,22)]
        config['control']['held_eye_ids']=[7,22]
    ids,positions,fps=load_targets(args.targets,original,args.audio,calibration)
    profile,ids,trajectory=prepare(config,original,ids,positions,args.authorized_tongue_observation,args.mouth_gain,telemetry_policy,args.mode)
    if fps!=60:parser.error('The reviewed combined controller requires the 60 FPS source')
    profile['face_update_hz']=min(args.face_fps,args.control_fps)
    profile['control_hz']=args.control_fps
    profile['snapshot_chunk_bytes']=2
    if args.baud is not None:profile['baud']=args.baud
    home_inventory=robot.home()
    args.home_position_recovery=recovery_policy(home_inventory.get('disabled_position_recovery'))
    home_axes=home_inventory['axes'];home_ids=[a['id'] for a in home_axes]
    if (len(set(home_ids))!=len(home_ids) or not set(ids)<=set(home_ids)
            or any(type(a.get('neutral')) is not int or a.get('neutral_confirmed') is not True for a in home_axes)):
        parser.error('Home inventory must cover every task axis with confirmed neutrals')
    motors={m['id']:m for m in profile['motors']}
    clearance_plan=recovery_clearance_plan(home_inventory,motors)
    hardware_preset=robot.startup() if not args.home_only else None
    if hardware_preset is not None:
        expected={a['id']:a for a in hardware_preset['axes']}
        if set(expected)!=set(motors) or any(expected[sid]['speed_raw']!=m['max_speed_counts_s'] for sid,m in motors.items()):
            parser.error('Startup hardware inventory/speeds differ from task profile')
    if any(a['neutral']!=motors[a['id']]['neutral'] for a in home_axes if a['id'] in motors):
        parser.error('Home inventory and task neutrals differ')
    if args.check:
        print(json.dumps(dict(offline_valid=True,monitored=ids,commanded=profile['diagnostic_command_ids'],
                             gaze_owner_ids=profile['gaze_owner_ids'],fps=fps,repeat=args.repeat,
                             face_fps=profile['face_update_hz'],control_fps=args.control_fps,baud=profile['baud'],
                             snapshot_chunk_bytes=profile['snapshot_chunk_bytes'],
                             scheduled_feedback=profile['scheduled_feedback'],
                             home_ids=home_ids,home_speed=args.home_speed,home_only=args.home_only,
                             disabled_position_recovery=args.home_position_recovery,
                             hardware_preset=str(robot.resolve('hardware')) if hardware_preset is not None else None,
                             telemetry_policy=telemetry_policy,
                             repeat_mode='independent_full_cycle',preparation_each_round=True,
                             developer_mode=args.mode,camera_required=args.mode!='face' and not args.home_only,
                             mouth_gain_report=profile['mouth_gain_report'],
                             acceleration_control='servo_register_41',
                             upstream_trajectory=trajectory_settings,
                             volume_times_three=args.usb_volume_times_three and not args.home_only)))
        return
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in [('config',config),('profile',profile),('home_inventory',home_inventory)]:
        (args.output/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    if hardware_preset is not None:
        (args.output/'hardware_preset.json').write_text(json.dumps(hardware_preset,ensure_ascii=False,indent=2)+'\n')
    (args.output/'mouth_gain_report.json').write_text(json.dumps(profile['mouth_gain_report'],indent=2)+'\n')
    stop_file=ROOT/'outputs/STOP';stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    def guard():
        if stopped or stop_file.exists():raise RuntimeError('Operator stop')
    summary=dict(completed=False,rounds=[],error=None,startup_events=[],tongue_disabled_after=False,mouth_motion_gain=args.mouth_gain)
    summary.update(repeat_mode='independent_full_cycle',requested_rounds=args.repeat,cleanup_attempted=False,
                   control_hz=args.control_fps,baud=profile['baud'],source_fps=fps,developer_mode=args.mode,
                   upstream_trajectory=trajectory_settings)
    summary['homing']={}
    status_path=ROOT/'outputs/runtime/control_status.json'
    status_path.parent.mkdir(parents=True,exist_ok=True)
    def publish(state, error=None):
        temporary=status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(updated_monotonic=time.monotonic(),active=False,
            state=state,error=error,command_ids=profile['diagnostic_command_ids']))+'\n')
        temporary.replace(status_path)
    publish('starting')
    receiver=None
    try:
        if args.home_only:
            prepare_round(args,profile,ids,home_axes,hardware_preset,clearance_plan,telemetry_policy,
                          guard,summary,args.output,publish)
            summary['completed']=True
            return
        for lap in range(1,args.repeat+1):
            guard();directory=args.output/f'round_{lap}';directory.mkdir()
            startup=dict(completed=False,error=None,tongue_disabled_after=False)
            summary['active_round']=lap
            summary['tongue_disabled_after']=False;summary['cleanup_attempted']=False
            publish('preparing')
            print(f'ROUND {lap}/{args.repeat} PREPARE',flush=True)
            try:
                prepare_round(args,profile,ids,home_axes,hardware_preset,clearance_plan,telemetry_policy,
                              guard,startup,directory,publish)
                # Preserve top-level startup diagnostics for existing consumers.
                summary['homing']=startup['homing']
                guard();override=None
                if args.mode!='face':
                    receiver=TargetReceiver(config);receiver.wait_ready()
                    override=MojiaPlaybackGaze(config,profile,ids,receiver,
                        ROOT/'outputs/runtime/control_status.json',lap,args.repeat,
                        neck=args.mode=='eyes-neck')
                print(f'ROUND {lap}/{args.repeat} START',flush=True)
                result=play(profile,ids,trajectory,fps,args.audio,directory/'playback_log.json',stop_file,gaze_override=override)
                if not result['completed'] or not result['full_clip_requested']:
                    raise RuntimeError('Independent round did not complete the full clip')
                round_result=dict(round=lap,completed=result['completed'],
                    full_clip_requested=result['full_clip_requested'],audio_seconds_at_stop=result['audio_seconds_at_stop'],
                    sent_frames=result['sent_frames'],skipped_source_frames=result['skipped_source_frames'],
                    preparation=str(directory/'preparation.json'))
                startup['playback_completed']=True
                del result
            except BaseException as exc:
                startup['error']=str(exc)
                raise
            finally:
                if receiver is not None:
                    try:receiver.close()
                    except Exception as exc:startup['cleanup_error']='Vision receiver: '+str(exc)
                    finally:receiver=None
                if args.authorized_tongue_observation:
                    summary['cleanup_attempted']=True
                    try:unload_tongue(profile,startup)
                    except Exception as exc:
                        startup['cleanup_error']=str(exc)
                summary['tongue_disabled_after']=startup['tongue_disabled_after']
                startup['completed']=startup.get('playback_completed',False) and not startup.get('error') and not startup.get('cleanup_error')
                (directory/'preparation.json').write_text(json.dumps(startup,ensure_ascii=False,indent=2)+'\n')
            if startup.get('cleanup_error'):raise RuntimeError(startup['cleanup_error'])
            summary['rounds'].append(round_result)
            (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(f'ROUND {lap}/{args.repeat} COMPLETE',flush=True)
        summary['completed']=True
    except BaseException as exc:
        summary['error']=str(exc)
        raise
    finally:
        if receiver is not None:receiver.close()
        # Established tongue stop policy: unload only ID29 after the show;
        # preserve the supported neck's torque and its final bounded goal.
        if args.authorized_tongue_observation and not summary.get('cleanup_attempted'):
            try:unload_tongue(profile,summary)
            except Exception as exc:summary['cleanup_error']=str(exc)
        for sig,handler in handlers.items():signal.signal(sig,handler)
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        publish('completed' if summary['completed'] else 'stopped',summary['error'])


if __name__=='__main__':main()
