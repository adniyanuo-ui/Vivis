#!/usr/bin/env python3
"""Commission Mojia horizontal eyes8/23, optional vertical7/22 and neck32."""
import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
from xiaobai_runtime.mapping import ROOT, read_json, live_preflight
from xiaobai_runtime.mojia_gaze import validate_mojia, MojiaPursuit, NeckCommandScheduler,neck_command_limits
from xiaobai_runtime.person_gaze import TargetReceiver
from xiaobai_runtime.playback import check_feedback


def tracking_start_position(state, motor=None):
    """Resume eyes from measured position; never recenter uncommanded neck axes."""
    sid=state['id']
    position,goal=state['position_raw'],state['goal_position_raw']
    if motor is None:
        if not state['configured_min']<=position<=state['configured_max'] or not state['configured_min']<=goal<=state['configured_max']:
            raise RuntimeError(f'ID{sid}: held neck outside device limits')
        return None
    if sid in (7,8,22,23,32):
        if not motor['safe_min']<=position<=motor['safe_max'] or not motor['safe_min']<=goal<=motor['safe_max']:
            raise RuntimeError(f'ID{sid}: resume position/goal exceeds command limits')
        return float(position)
    if abs(position-2047)>10 or goal!=2047:
        raise RuntimeError(f'ID{sid}: commanded neck must start from neutral')
    return float(position)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'mojia/person_gaze.json')
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--port', default='/dev/ttyCH343USB0')
    parser.add_argument('--neck', action='store_true')
    parser.add_argument('--hold-vertical-eyes',action='store_true',help='Send no vertical eye commands during this trial')
    parser.add_argument('--direction-trial', action='store_true')
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--check', action='store_true', help='Offline configuration validation; no hardware opened')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 10 <= args.seconds <= 600:
        parser.error('Commissioning duration must be 10..600 seconds')
    config, profile = read_json(args.config), read_json(args.profile)
    if args.hold_vertical_eyes:
        config['control']['vertical_enabled']=False
        config['control']['axes']=[a for a in config['control']['axes'] if a['coordinate']!='y']
    validate_mojia(config, profile, ROOT/'mojia/墨甲9.23.json', args.neck, args.direction_trial)
    ids = [a['id'] for a in config['control']['axes']]+([32] if args.neck else [])
    motors = {m['id']: dict(m) for m in profile['motors'] if m['id'] in ids}
    for axis in config['control']['axes']:
        m = motors[axis['id']]
        m.update(safe_min=min(axis['negative'], axis['positive']),
                 safe_max=max(axis['negative'], axis['positive']),
                 max_speed_counts_s=axis.get('speed_counts_s',config['control'].get('speed_counts_s',60)))
    if args.neck:
        n = config['neck']
        low,high=neck_command_limits(config,profile)
        motors[32].update(safe_min=low, safe_max=high,
                          max_speed_counts_s=n['speed_counts_s'],
                          max_load_raw=n.get('max_load_raw',motors[32]['max_load_raw']))
    selected = {**profile, 'port': args.port, 'motors': list(motors.values())}
    live_preflight(selected, ids)
    if args.check:
        print(json.dumps(dict(offline_valid=True, ids=ids, camera=config['camera'],
                             command_limits={sid:[m['safe_min'],m['safe_max']] for sid,m in motors.items()},
                             software_load_limits={sid:m['max_load_raw'] for sid,m in motors.items()})))
        return
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    stop_file = ROOT/'outputs/mojia/STOP'
    if stop_file.exists():
        raise RuntimeError('Operator STOP file exists')
    args.output.mkdir(parents=True)
    (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    (args.output/'profile.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    handlers = {sig: signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    receiver = bus = None
    live_status = ROOT/'outputs/mojia/right_camera_tracking/control_status.json'
    live_status.parent.mkdir(parents=True,exist_ok=True)
    def publish(data):
        temporary=live_status.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(updated_monotonic=time.monotonic(),**data))+'\n')
        temporary.replace(live_status)
    publish(dict(active=False,state='starting',command_ids=ids))
    result = dict(completed=False, command_ids=ids, events=[], snapshots=[], error=None,
                  stop_policy='Stop sending; preserve bounded goals and neck torque')
    try:
        receiver = TargetReceiver(config)
        receiver.wait_ready()
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus
        bus = OfficialSDKBus(args.port, profile['baud'], feedback_mode='short_registers',
                             changed_goals_only=True, transaction_timeout_ms=20,
                             trace_path=args.output/'serial_trace.json')
        initial_positions={}
        # Read held neck state, but never move it to satisfy eye-only preflight.
        for sid in sorted(set(ids)|{32,33,34}):
            s = bus.snapshot(sid);result['snapshots'].append(s)
            expected = 521 if sid in (7,8,22,23) else 10760
            if s['model_raw'] != expected or s['mode'] != 0 or s['torque_enable'] != 1:
                raise RuntimeError(f'ID{sid}: wrong hardware/mode or disabled torque')
            initial_positions[sid]=tracking_start_position(s,motors.get(sid))
            if sid in motors:
                m = motors[sid]
                if (s['configured_min'],s['configured_max']) != (m['device_min'],m['device_max']):
                    raise RuntimeError(f'ID{sid}: device limits changed')
                if not bytes.fromhex(s['config_0_39_hex'])[18]&4:
                    raise RuntimeError(f'ID{sid}: unexpected speed unit; do not write raw counts/s')
                if m['max_temperature_c']>s['configured_max_temperature']:
                    raise RuntimeError(f'ID{sid}: temperature ceiling exceeds device protection')
                check_feedback(sid,dict(position=s['position_raw'],**{k:s[k] for k in ['load_raw','temperature_c','voltage_raw','status_register']}),m)
        speeds = {sid:int(motors[sid]['max_speed_counts_s']) for sid in ids}
        controller = MojiaPursuit(config, ids, [speeds[sid] for sid in ids], args.neck,
                                 neck_bounds=neck_command_limits(config,profile) if args.neck else None)
        previous = np.array([initial_positions[sid] for sid in ids],dtype=float)
        start = last = time.monotonic();next_tick = start+1/60;poll = 0
        controller.observe_eye(initial_positions[controller.camera_eye_id],start)
        neck_scheduler=NeckCommandScheduler(initial_positions[32],start,speeds[32],
                                            adaptive_speed=config['neck'].get('adaptive_speed',True),endpoints=controller.neck_bounds) if args.neck else None
        returning=False
        while time.monotonic()-start < args.seconds:
            if stopped or stop_file.exists():
                raise RuntimeError('Operator stop')
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(.002,next_tick-now));continue
            dt = now-last
            if dt>.1:
                raise RuntimeError('Tracking watchdog stalled >100ms')
            packet = receiver.poll(now)
            returning=returning or now-start>=args.seconds-controller.return_window(previous[ids.index(32)] if args.neck else 2047)
            previous, diagnostic = controller.step(packet,previous,dt,now,returning=returning)
            goals = {sid:int(round(previous[i])) for i,sid in enumerate(ids)}
            if neck_scheduler is not None:
                diagnostic['planned_neck_goal']=float(previous[ids.index(32)])
                goals[32],speeds[32]=neck_scheduler.step(previous[ids.index(32)],now,returning=returning)
                diagnostic['neck_speed_command']=speeds[32]
            bus.goals(goals,speeds)
            sid = ids[poll%len(ids)];poll+=1
            feedback = bus.feedback(sid);check_feedback(sid,feedback,motors[sid])
            if sid == controller.camera_eye_id:
                controller.observe_eye(feedback['position'],time.monotonic())
            result['events'].append(dict(seconds=now-start,goals=goals,feedback=dict(id=sid,**feedback),**diagnostic))
            if poll%12==0:
                publish(dict(active=True,state=diagnostic['state'],command_ids=ids,goals=goals,
                             feedback=dict(id=sid,**feedback),
                             neck_speed_command=speeds.get(32),
                             neck_assist_reason=diagnostic['neck_assist_reason'],
                             image_error_x=diagnostic['image_error_x'],
                             camera_eye=config['control']['camera_eye'],camera_eye_id=controller.camera_eye_id,
                             neck_requested_velocity=diagnostic['neck_requested_velocity'],
                             remaining_seconds=max(0,args.seconds-(now-start))))
            last = now;next_tick=max(next_tick+1/60,now+1/60)
        for sid in ids:
            measured=bus.feedback(sid);check_feedback(sid,measured,motors[sid])
            if abs(measured['position']-2047)>20 or abs(previous[ids.index(sid)]-2047)>1:
                raise RuntimeError(f'ID{sid}: normal return home not confirmed')
        result['completed']=True
    except BaseException as exc:
        result['error']=str(exc)
        raise
    finally:
        publish(dict(active=False,state='completed' if result['completed'] else 'stopped',error=result['error'],command_ids=ids))
        if bus is not None:
            bus.close()
        if receiver is not None:
            receiver.close()
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        (args.output/'tracking_log.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Tracking completed:',result['completed'])


if __name__=='__main__':
    main()
