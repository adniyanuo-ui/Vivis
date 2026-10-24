"""Configured neutral poses and a shared smooth startup move; no axis IDs here."""
import math
import time


def write_ram_verified(bus,sid,address,value,width=1):
    """Write once; a missing acknowledgement requires independent readback.

    Never replay a write, accept a servo alarm, or trust a damaged read payload.
    """
    if width not in (1,2):raise ValueError('Unsupported RAM field width')
    result,status=(bus.sdk.write1ByteTxRx if width==1 else bus.sdk.write2ByteTxRx)(sid,address,value)
    checks=[int.from_bytes(bus.read(sid,address,width),'little') for _ in range(2 if result else 1)]
    record=dict(id=sid,address=address,target=value,width=width,result=result,status=status,
                readbacks=checks,write_retried=False)
    if status or any(actual!=value for actual in checks):
        raise RuntimeError(f'ID{sid}: RAM{address} write could not be verified: {record}')
    return record


def validate_home_axes(axes,snapshots):
    states={s['id']:s for s in snapshots}
    if not axes or len({a['id'] for a in axes})!=len(axes):raise ValueError('Invalid home axes')
    for axis in axes:
        sid=axis['id'];s=states[sid];neutral=axis['neutral']
        if axis.get('neutral_confirmed') is not True or type(neutral) is not int:
            raise ValueError(f'ID{sid}: home neutral is unconfirmed')
        if s['model_raw']!=axis['model_raw_expected'] or s['mode']!=0 or s['torque_enable'] not in (0,1):
            raise RuntimeError(f'ID{sid}: home model/mode/torque mismatch')
        if not s['configured_min']<=neutral<=s['configured_max']:
            raise RuntimeError(f'ID{sid}: home neutral outside device limits')
        # Counts/s commands require the already commissioned fine-speed mode.
        # Inspect this setting; homing never changes the phase register.
        if not bytes.fromhex(s['config_0_39_hex'])[18]&4:
            raise RuntimeError(f'ID{sid}: home speed unit has not been commissioned')


def home_check(state,snapshot,neutral=None,tolerance=0):
    sid=snapshot['id']
    position=state['position']
    near_home=neutral is not None and abs(position-neutral)<=tolerance
    if not 0<=position<=4095 or (not snapshot['configured_min']<=position<=snapshot['configured_max'] and not near_home):
        raise RuntimeError(f'ID{sid}: home position outside device limits')
    if state['status_register']:raise RuntimeError(f'ID{sid}: home servo alarm {state["status_register"]}')
    if state['temperature_c']>=snapshot['configured_max_temperature']:
        raise RuntimeError(f'ID{sid}: home hardware temperature threshold reached')
    if not snapshot['configured_min_voltage_raw']<=state['voltage_raw']<=snapshot['configured_max_voltage_raw']:
        raise RuntimeError(f'ID{sid}: home voltage outside device limits')


def homing_feedback_motor(motor,snapshot):
    """Neutral recovery has a device-wide corridor, independent of tracking stroke.

    Keep task temperature/load checks. This changes only the homing feedback
    position domain; the playback/following command and feedback limits stay intact.
    """
    tolerance=motor.get('initial_settle_tolerance_counts',20);neutral=motor['neutral']
    return {**motor,'safe_min':snapshot['configured_min'],'safe_max':snapshot['configured_max'],
            'feedback_min':max(0,min(snapshot['configured_min'],neutral-tolerance)),
            'feedback_max':min(4095,max(snapshot['configured_max'],neutral+tolerance))}


def smooth_home_targets(start,neutral,speed,accelerations=None):
    if type(speed) not in (int,float) or not math.isfinite(speed) or not 1<=speed<=32767:
        raise ValueError('Home speed must be finite and within 1..32767')
    # Quintic smoothstep has peak derivative 1.875 and zero endpoint velocity
    # and acceleration. Use a common duration so every configured axis arrives together.
    duration=max(1.,1.875*max(abs(neutral[sid]-p) for sid,p in start.items())/speed)
    for sid,a in (accelerations or {}).items():
        if sid not in start or type(a) not in (int,float) or not math.isfinite(a) or a<=0:
            raise ValueError('Invalid home software acceleration')
        # Quintic smoothstep's peak second derivative is10/sqrt(3).
        duration=max(duration,math.sqrt(10/math.sqrt(3)*abs(neutral[sid]-start[sid])/a))
    def at(elapsed):
        u=max(0.,min(1.,elapsed/duration));blend=u*u*u*(10+u*(-15+6*u))
        return {sid:p+(neutral[sid]-p)*blend for sid,p in start.items()}
    return duration,at


def home_all(bus,axes,task_motors,guard,report,max_speed=60,restore_speeds=None,keep_enabled=()):
    """Home the configured inventory; restore speed RAM and temporary enables.

    Hardware acceleration/PID/deadband are preserved. Existing task protection
    is supplied by the caller for task axes. Other inventory axes use their live
    device thresholds, with no new per-axis software tuning.
    """
    from .playback import check_feedback,startup_feedback
    recoveries=report.setdefault('communication_recoveries',[])
    guard();snapshot_deadline=time.monotonic()+10
    snapshots=[startup_feedback(bus.snapshot,a['id'],snapshot_deadline,recoveries,
                should_stop=lambda:guard() or False) for a in axes]
    validate_home_axes(axes,snapshots)
    states={s['id']:s for s in snapshots};neutral={a['id']:a['neutral'] for a in axes}
    speed=int(max_speed)
    if speed!=max_speed or not 1<=speed<=32767:raise ValueError('Invalid integer home speed')
    start={s['id']:s['position_raw'] for s in snapshots}
    for sid,motor in task_motors.items():
        if sid in start:
            if not motor['safe_min']<=neutral[sid]<=motor['safe_max']:
                raise RuntimeError(f'ID{sid}: home neutral outside task command limits')
    accelerations={sid:m['max_acceleration_counts_s2'] for sid,m in task_motors.items()
                   if sid in start and m.get('max_acceleration_counts_s2') is not None}
    duration,trajectory=smooth_home_targets(start,neutral,speed,accelerations)
    original={s['id']:int.from_bytes(bytes.fromhex(s['state_40_70_hex'])[6:8],'little') for s in snapshots}
    if restore_speeds is not None:
        if set(restore_speeds)!=set(original):raise ValueError('Home speed restore inventory differs')
        original.update(restore_speeds)
    if any(type(value) is not int or not 0<=value<=32767 for value in original.values()):
        raise ValueError('Invalid original speed RAM')
    disabled={s['id'] for s in snapshots if not s['torque_enable']}
    temporary=disabled-set(keep_enabled)
    report.update(completed=False,ids=list(neutral),max_speed_counts_s=speed,
                  software_acceleration_counts_s2=accelerations,
                  planned_duration_s=duration,original_speeds=original,temporary_enable_ids=sorted(temporary),
                  neutral_targets=neutral,initial_positions={s['id']:s['position_raw'] for s in snapshots},
                  samples=[],communication_recoveries=recoveries,ram_writes=[],restored_speeds=False)
    def check(sid,state):
        tolerance=task_motors.get(sid,{}).get('initial_settle_tolerance_counts',20)
        home_check(state,states[sid],neutral[sid],tolerance)
        if sid in task_motors:check_feedback(sid,state,homing_feedback_motor(task_motors[sid],states[sid]))
        return state
    def read(sid):
        guard()
        return check(sid,bus.feedback(sid))
    for s in snapshots:
        check(s['id'],dict(position=s['position_raw'],**{k:s[k] for k in ('load_raw','temperature_c','voltage_raw','status_register')}))
    deadline=time.monotonic()+duration+10
    restored=[];motion_complete=False;begin=None
    try:
        # Prime each disabled axis at its measured position before enabling it.
        # This prevents an old goal from becoming active on torque enable.
        for sid in sorted(disabled):
            initial=max(states[sid]['configured_min'],min(states[sid]['configured_max'],start[sid]))
            guard();bus.goals({sid:initial},{sid:speed})
            if bus.read(sid,40,1)==b'\x00':
                report['ram_writes'].append(write_ram_verified(bus,sid,40,1))
            if bus.read(sid,40,1)!=b'\x01':raise RuntimeError(f'ID{sid}: home torque readback failed')
        begin=last_tick=time.monotonic();poll=0;progress=0.
        while True:
            guard();now=time.monotonic();elapsed=now-begin
            if now>deadline:raise RuntimeError('Neutral homing timeout')
            if now-last_tick<1/60:time.sleep(min(.002,1/60-(now-last_tick)));continue
            # Pause trajectory progress during slow reads; never catch up with a
            # larger jump after recovery. Preserve the smooth endpoint shape.
            progress+=min(now-last_tick,1/60)
            planned=trajectory(progress)
            goals={sid:max(states[sid]['configured_min'],min(states[sid]['configured_max'],int(round(p))))
                   for sid,p in planned.items()}
            bus.goals(goals,{sid:speed for sid in goals});last_tick=now
            sid=axes[poll%len(axes)]['id'];poll+=1
            state=startup_feedback(read,sid,deadline,report['communication_recoveries'])
            report['samples'].append(dict(id=sid,elapsed_s=elapsed,**state))
            if progress>=duration:break
        # Confirm two whole passes at neutral, using the existing per-axis task
        # settle criteria (their removal is a separate user review).
        for _ in range(2):
            for axis in axes:
                sid=axis['id'];state=startup_feedback(read,sid,deadline,report['communication_recoveries'])
                if int.from_bytes(bus.read(sid,42,2),'little')!=neutral[sid]:
                    raise RuntimeError(f'ID{sid}: neutral goal RAM readback differs')
                tolerance=task_motors.get(sid,{}).get('initial_settle_tolerance_counts',20)
                if abs(state['position']-neutral[sid])>tolerance:
                    raise RuntimeError(f'ID{sid}: home did not settle at neutral')
                report['samples'].append(dict(id=sid,settle=True,**state))
        motion_complete=True
    finally:
        if begin is not None:report['actual_duration_s']=time.monotonic()-begin
        # Inventory axes unused by the task return to their original disabled state.
        for sid in sorted(temporary):
            report['ram_writes'].append(write_ram_verified(bus,sid,40,0))
        if motion_complete:
            for sid,value in original.items():
                report['ram_writes'].append(write_ram_verified(bus,sid,46,value,2))
                restored.append(sid)
        report['restored_speeds']=len(restored)==len(original)
        report['completed']=motion_complete and report['restored_speeds']
