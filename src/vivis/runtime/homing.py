"""Configured neutral poses and a shared smooth startup move; no axis IDs here."""
import math
import time
from vivis.runtime.telemetry import evaluate,WarningRecorder,verify_sample


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


def recovery_policy(policy):
    if policy is None:return None
    if (not isinstance(policy,dict) or type(policy.get('max_outside_counts')) is not int or not 0<=policy['max_outside_counts']<=20
            or type(policy.get('confirmation_samples')) is not int or not 2<=policy['confirmation_samples']<=3
            or type(policy.get('max_speed_counts_s')) is not int or not 1<=policy['max_speed_counts_s']<=60
            or type(policy.get('max_sample_spread_counts')) is not int or not 0<=policy['max_sample_spread_counts']<=2
            or type(policy.get('timeout_s')) not in (int,float) or not 1<=policy['timeout_s']<=5):
        raise ValueError('Invalid disabled startup position recovery policy')
    overrides=policy.get('axis_overrides',{})
    if not isinstance(overrides,dict):raise ValueError('Invalid startup recovery axis overrides')
    for sid,values in overrides.items():
        if (not isinstance(sid,str) or not sid.isdecimal() or not 0<=int(sid)<=253
                or not isinstance(values,dict) or not set(values)<={'max_outside_counts','timeout_s'}):
            raise ValueError('Invalid startup recovery axis override')
        if ('max_outside_counts' in values and (type(values['max_outside_counts']) is not int
                or not 0<=values['max_outside_counts']<=200)):
            raise ValueError('Invalid per-axis startup recovery margin')
        if 'timeout_s' in values and (type(values['timeout_s']) not in (int,float) or not 1<=values['timeout_s']<=10):
            raise ValueError('Invalid per-axis startup recovery timeout')
    return policy


def home_check(state,snapshot,neutral=None,tolerance=0,telemetry_policy=None,position_recovery=None):
    sid=snapshot['id']
    position=state['position']
    near_home=neutral is not None and abs(position-neutral)<=tolerance
    recovering=(position_recovery is not None and
                position_recovery['feedback_min']<=position<=position_recovery['feedback_max'])
    if not 0<=position<=4095 or (not snapshot['configured_min']<=position<=snapshot['configured_max'] and not near_home and not recovering):
        raise RuntimeError(f'ID{sid}: home position outside device limits; position={position} '
                           f'limits=[{snapshot["configured_min"]},{snapshot["configured_max"]}] '
                           f'initial_torque={snapshot["torque_enable"]}')
    if state['status_register']:raise RuntimeError(f'ID{sid}: home servo alarm {state["status_register"]}')
    if not telemetry_policy and state['temperature_c']>=snapshot['configured_max_temperature']:
        raise RuntimeError(f'ID{sid}: home hardware temperature threshold reached')
    if not snapshot['configured_min_voltage_raw']<=state['voltage_raw']<=snapshot['configured_max_voltage_raw']:
        raise RuntimeError(f'ID{sid}: home voltage outside device limits')
    return evaluate(sid,state,telemetry_policy) if telemetry_policy else []


def homing_feedback_motor(motor,snapshot,position_recovery=None):
    """Neutral recovery has a device-wide corridor, independent of tracking stroke.

    Keep task temperature/load checks. This changes only the homing feedback
    position domain; the playback/following command and feedback limits stay intact.
    """
    tolerance=motor.get('initial_settle_tolerance_counts',20);neutral=motor['neutral']
    result={**motor,'safe_min':snapshot['configured_min'],'safe_max':snapshot['configured_max'],
            'feedback_min':max(0,min(snapshot['configured_min'],neutral-tolerance)),
            'feedback_max':min(4095,max(snapshot['configured_max'],neutral+tolerance))}
    if position_recovery is not None:
        result['feedback_min']=min(result['feedback_min'],position_recovery['feedback_min'])
        result['feedback_max']=max(result['feedback_max'],position_recovery['feedback_max'])
    return result


def confirm_home_recoveries(bus,snapshots,policy,guard,records,telemetry_policy=None):
    """Read-only confirmation of small passive offsets on disabled axes.

    This never changes a device/command limit. The temporary feedback corridor
    is bounded by observed startup positions, and may be used only for homing.
    """
    from vivis.runtime.playback import startup_feedback
    policy=recovery_policy(policy);plans={};confirmed={}
    if policy is None:return plans,confirmed
    deadline=time.monotonic()+10
    for snapshot in snapshots:
        sid=snapshot['id'];position=snapshot['position_raw']
        selected={**policy,**policy.get('axis_overrides',{}).get(str(sid),{})}
        low,high=snapshot['configured_min'],snapshot['configured_max']
        distance=max(low-position,position-high,0)
        if (snapshot['torque_enable']!=0 or not 0<=position<=4095
                or not 0<distance<=selected['max_outside_counts']):continue
        plan=dict(id=sid,initial_position=position,device_min=low,device_max=high,
                  feedback_min=max(0,low-selected['max_outside_counts']),
                  feedback_max=min(4095,high+selected['max_outside_counts']),
                  max_speed_counts_s=selected['max_speed_counts_s'],timeout_s=selected['timeout_s'])
        record={**plan,'confirmed':False,'samples':[]};records.append(record)
        positions=[]
        for _ in range(policy['confirmation_samples']):
            guard()
            if bus.read(sid,40,1)!=b'\x00':
                raise RuntimeError(f'ID{sid}: startup position recovery requires disabled torque')
            sample=startup_feedback(bus.feedback,sid,deadline,record.setdefault('communication_recoveries',[]),
                                    should_stop=lambda:guard() or False)
            sample,_=verify_sample(bus.feedback,sid,sample,
                lambda fresh:home_check(fresh,snapshot,telemetry_policy=telemetry_policy,position_recovery=plan),
                telemetry_policy,record.setdefault('sensor_confirmations',[]),should_stop=lambda:guard() or False)
            positions.append(sample['position']);record['samples'].append(dict(sample))
        if max(positions)-min(positions)>policy['max_sample_spread_counts']:
            raise RuntimeError(f'ID{sid}: inconsistent startup position rechecks {positions}')
        # Rechecks may jitter on either side of the first sample. After they
        # agree, retain only the observed interval, not the policy-wide margin.
        plan['feedback_min']=min(low,*positions)
        plan['feedback_max']=max(high,*positions)
        plan['start_position']=positions[-1]
        plan['prime_goal']=max(low,min(high,positions[-1]))
        record.update(confirmed=True,start_position=positions[-1],prime_goal=plan['prime_goal'],
                      feedback_min=plan['feedback_min'],feedback_max=plan['feedback_max'])
        confirmed[sid]=sample;plans[sid]=plan
    return plans,confirmed


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


def home_all(bus,axes,task_motors,guard,report,max_speed=60,restore_speeds=None,keep_enabled=(),telemetry_policy=None,position_recovery=None):
    """Home the configured inventory; restore speed RAM and temporary enables.

    Hardware acceleration/PID/deadband are preserved. Existing task protection
    is supplied by the caller for task axes. Other inventory axes use their live
    device thresholds, with no new per-axis software tuning.
    """
    from vivis.runtime.playback import check_feedback,startup_feedback
    recoveries=report.setdefault('communication_recoveries',[])
    guard();snapshot_deadline=time.monotonic()+10
    snapshots=[startup_feedback(bus.snapshot,a['id'],snapshot_deadline,recoveries,
                should_stop=lambda:guard() or False) for a in axes]
    validate_home_axes(axes,snapshots)
    plans,confirmed=confirm_home_recoveries(bus,snapshots,position_recovery,guard,
                                          report.setdefault('disabled_position_recoveries',[]),telemetry_policy)
    states={s['id']:s for s in snapshots};neutral={a['id']:a['neutral'] for a in axes}
    speed=int(max_speed)
    if speed!=max_speed or not 1<=speed<=32767:raise ValueError('Invalid integer home speed')
    if plans:speed=min(speed,min(plan['max_speed_counts_s'] for plan in plans.values()))
    start={s['id']:s['position_raw'] for s in snapshots}
    start.update({sid:plan['start_position'] for sid,plan in plans.items()})
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
    warning_recorder=WarningRecorder(report.setdefault('telemetry_warnings',[]))
    recovery_deadlines={};reentered=set();neutral_sent=set()
    def check(sid,state):
        tolerance=task_motors.get(sid,{}).get('initial_settle_tolerance_counts',20)
        def validate(sample):
            plan=plans.get(sid) if sid not in reentered else None
            if plan is not None:
                if states[sid]['configured_min']<=sample['position']<=states[sid]['configured_max']:
                    reentered.add(sid);plan=None
                elif (sid in neutral_sent and abs(sample['position']-neutral[sid])<=tolerance
                        and int.from_bytes(bus.read(sid,42,2),'little')==neutral[sid]):
                    # Preserve the established neutral encoder tolerance. It
                    # applies only after the actual neutral goal is verified,
                    # never to a clipped boundary goal or playback feedback.
                    reentered.add(sid);plan=None
                    report.setdefault('startup_neutral_residuals',[]).append(dict(
                        id=sid,position=sample['position'],target=neutral[sid],tolerance=tolerance))
                elif sid in recovery_deadlines and time.monotonic()>recovery_deadlines[sid]:
                    raise RuntimeError(f'ID{sid}: startup recovery did not reenter device limits or settle at neutral; '
                        f'position={sample["position"]} neutral={neutral[sid]} tolerance={tolerance}')
            warnings=home_check(sample,states[sid],neutral[sid],tolerance,telemetry_policy,plan)
            if sid in task_motors:warnings=check_feedback(sid,sample,homing_feedback_motor(task_motors[sid],states[sid],plan))
            return warnings
        state,warnings=verify_sample(bus.feedback,sid,state,validate,telemetry_policy,
            report.setdefault('sensor_confirmations',[]),should_stop=lambda:guard() or False)
        warning_recorder.observe(sid,state,warnings)
        return state
    def read(sid):
        guard()
        return check(sid,bus.feedback(sid))
    for s in snapshots:
        check(s['id'],confirmed.get(s['id'],dict(position=s['position_raw'],**{k:s[k] for k in ('load_raw','temperature_c','voltage_raw','status_register')})))
    deadline=time.monotonic()+duration+10
    restored=[];motion_complete=False;begin=None
    try:
        # Prime each disabled axis at its measured position before enabling it.
        # A goal update can itself enable some firmware. Check disabled torque
        # BEFORE sending the bounded target, then verify its goal/speed and
        # accept either automatic enable or the explicit enable path.
        for sid in sorted(disabled):
            initial=max(states[sid]['configured_min'],min(states[sid]['configured_max'],start[sid]))
            guard()
            if bus.read(sid,40,1)!=b'\x00':
                raise RuntimeError(f'ID{sid}: torque changed before startup target was sent')
            if sid in plans:
                # The common smooth curve can keep an axis at a clipped
                # boundary for seconds. Budget its whole planned motion before
                # the configured reentry grace; the global timeout still applies.
                recovery_deadlines[sid]=time.monotonic()+duration+plans[sid]['timeout_s']
            guard();bus.goals({sid:initial},{sid:speed})
            goal_readback=int.from_bytes(bus.read(sid,42,2),'little')
            speed_readback=int.from_bytes(bus.read(sid,46,2),'little')
            if goal_readback!=initial or speed_readback!=speed:
                raise RuntimeError(f'ID{sid}: bounded startup target/speed readback failed')
            torque=bus.read(sid,40,1)
            if torque not in (b'\x00',b'\x01'):
                raise RuntimeError(f'ID{sid}: unexpected torque mode after startup target')
            report.setdefault('startup_primes',[]).append(dict(id=sid,target=initial,speed=speed,
                goal_readback=goal_readback,speed_readback=speed_readback,torque_before=0,
                torque_after_goal=torque[0],enable_source='goal_update' if torque==b'\x01' else 'explicit_write'))
            if torque==b'\x00':
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
            neutral_sent.update(sid for sid,goal in goals.items() if goal==neutral[sid])
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
        report['startup_reentered_ids']=sorted(reentered)
