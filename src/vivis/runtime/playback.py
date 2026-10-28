"""Audio DAC clock drives motion; no catch-up burst, no automatic torque enabling."""
import json
import gc
import math
import signal
import threading
import time
import wave
from pathlib import Path
import numpy as np
from vivis.motion.mapping import live_preflight,limit_step
from vivis.hardware.feetech.protocol import Bus,ServoStatusError,CorruptResponse,ReadTimeout,ProtocolError
from vivis.runtime.telemetry import evaluate,WarningRecorder,CriticalTemperatureError,verify_sample

class SensorReadingError(RuntimeError):pass
class PositionEdgeReadingError(RuntimeError):pass

class AudioClock:
    def __init__(self,path,device,latency='low',gain=1.):
        if not (isinstance(latency,str) and latency in ('low','high') or
                type(latency) in (int,float) and math.isfinite(latency) and .01<=latency<=1):
            raise ValueError('Audio latency must be low/high or 0.01..1 seconds')
        if type(gain) not in (int,float) or not math.isfinite(gain) or not .01<=gain<=1:
            raise ValueError('Digital audio trim must be 0.01..1 to avoid clipping')
        import sounddevice as sd
        from scipy.signal import resample_poly
        self.sd=sd;self.first_dac=None;self.fault=None;self.cursor=0
        with wave.open(str(path)) as wav:
            if wav.getsampwidth()!=2:raise ValueError('Audio must be PCM16 WAV')
            original_rate=wav.getframerate()
            data=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').reshape(-1,wav.getnchannels()).astype(np.float32)/32768
        info=sd.query_devices(device,'output')
        self.rate=int(info['default_samplerate'])
        common=math.gcd(self.rate,original_rate)
        self.data=resample_poly(data,self.rate//common,original_rate//common,axis=0).astype(np.float32)*gain
        self.duration=len(data)/original_rate
        self.stream=sd.OutputStream(device=device,samplerate=self.rate,channels=self.data.shape[1],
            dtype='float32',blocksize=0,latency=latency,callback=self.callback)
    def callback(self,outdata,n,timing,status):
        if status:self.fault=str(status)
        outdata.fill(0)
        if self.first_dac is None:self.first_dac=float(timing.outputBufferDacTime)
        take=min(n,len(self.data)-self.cursor)
        if take>0:outdata[:take]=self.data[self.cursor:self.cursor+take];self.cursor+=take
        if take<n:raise self.sd.CallbackStop
    def start(self):self.stream.start()
    def time(self):return None if self.first_dac is None else self.stream.time-self.first_dac
    def close(self):self.stream.abort();self.stream.close()

def check_feedback(sid,state,motor):
    position_error=None
    if not 0<=state['position']<=4095:raise RuntimeError(f'{sid}: invalid single-turn encoder position')
    low=motor.get('feedback_min',motor['safe_min']);high=motor.get('feedback_max',motor['safe_max'])
    if not low<=state['position']<=high:
        message=f'{sid}: measured position {state["position"]} outside safe range'
        delta=max(low-state['position'],state['position']-high)
        tolerance=0 if 'feedback_min' in motor else motor.get('position_feedback_tolerance_counts',0)
        # An opt-in one-count edge sample may trigger hold-and-recheck, never
        # count as healthy feedback or enlarge the allowed command range.
        margin=motor.get('confirm_position_edge_counts',0)
        if delta<=tolerance:
            pass  # Explicit measured-position tolerance; targets remain bounded.
        elif (margin==1 and delta<=1
                and motor.get('device_min',motor['safe_min'])<=state['position']<=motor.get('device_max',motor['safe_max'])):
            position_error=PositionEdgeReadingError(message)
        else:raise RuntimeError(message)
    policy=motor.get('telemetry_policy')
    if not policy and (state['load_raw']&1023)>=motor['max_load_raw']:raise RuntimeError(f'{sid}: excessive load ({state["load_raw"]&1023})')
    if state['status_register']:raise ServoStatusError(sid,state['status_register'])
    if position_error is not None:raise position_error
    if not policy and state['temperature_c']>=motor['max_temperature_c']:raise SensorReadingError(f'{sid}: over temperature ({state["temperature_c"]} C >= {motor["max_temperature_c"]} C)')
    if motor.get('max_voltage_raw') is not None:
        if not motor['min_voltage_raw']<=state['voltage_raw']<=motor['max_voltage_raw']:
            raise SensorReadingError(f'{sid}: voltage feedback outside configured operating range')
    return evaluate(sid,state,policy) if policy else []

def verified_feedback(bus,sid,motor,confirm,records,should_stop=lambda:False,initial=None,on_warnings=None,on_received=None):
    """Hold command transmission while confirming sensor or opted-in edge samples.
    Never accept out-of-range positions as healthy, relax device/load limits,
    communication failures or non-voltage servo alarms.
    A transient needs TWO consecutive healthy reads within three fresh attempts.
    """
    record=None;clean=0
    for attempt in range(4):
        if should_stop():raise RuntimeError('Operator stop during feedback confirmation')
        if attempt:time.sleep(.01)
        state=None
        try:
            state=initial if attempt==0 and initial is not None else bus.feedback(sid)
            if on_received is not None:on_received(sid,state)
            state,warnings=verify_sample(bus.feedback,sid,state,lambda sample:check_feedback(sid,sample,motor),
                motor.get('telemetry_policy'),records,should_stop,on_received)
            if on_warnings is not None:on_warnings(sid,state,warnings)
        except (SensorReadingError,ServoStatusError,PositionEdgeReadingError) as exc:
            if not confirm or isinstance(exc,ServoStatusError) and exc.status!=1:raise
            if record is None:
                record={'id':sid,'monotonic':time.monotonic(),'samples':[],'recovered':False}
                records.append(record)
            record['samples'].append({'attempt':attempt,'state':state,'error':str(exc)})
            clean=0
        else:
            if record is None:return state
            record['samples'].append({'attempt':attempt,'state':state})
            clean+=1
            if clean==2:
                record['recovered']=True
                return state
    raise RuntimeError(f'{sid}: feedback anomaly did not clear with two consecutive healthy confirmations')


def startup_feedback(reader,sid,deadline,records,should_stop=lambda:False,wait=time.sleep,can_recover=lambda:True):
    """Hold startup targets while a damaged/missing sample is reacquired.

    Recovery needs two consecutive validated *whole* samples, within four fresh
    attempts and the caller's existing startup deadline. Servo alarms, invalid
    physical readings and the transport's error-rate gate are never retried here.
    """
    if should_stop():raise RuntimeError('Operator stop during startup feedback')
    try:return reader(sid)
    except (CorruptResponse,ReadTimeout) as first:
        if not can_recover():raise
        record=dict(id=sid,monotonic=time.monotonic(),error=str(first),samples=[],recovered=False)
        records.append(record)
        clean=0;last_error=first
        recovery_deadline=min(deadline,time.monotonic()+.5)
        for attempt in range(4):
            if should_stop():raise RuntimeError('Operator stop during startup feedback recovery')
            if time.monotonic()+.02>=recovery_deadline:break
            wait(.02)  # No position writes during this startup-only hold.
            if should_stop():raise RuntimeError('Operator stop during startup feedback recovery')
            try:state=reader(sid)
            except (CorruptResponse,ReadTimeout) as exc:
                if not can_recover():raise
                record['samples'].append(dict(attempt=attempt,error=str(exc)))
                clean=0;last_error=exc
            else:
                record['samples'].append(dict(attempt=attempt,state=state))
                clean=clean+1 if state is not None else 0
                if clean>=2 and time.monotonic()<=recovery_deadline:
                    record['recovered']=True
                    return state
        raise RuntimeError(f'ID{sid}: startup communication recovery failed; no two consecutive valid samples') from last_error

def play(profile,ids,trajectory,fps,audio_path,log_path,stop_file,max_seconds=None,gaze_override=None):
    if (not isinstance(ids,list) or not ids or len(set(ids))!=len(ids)
            or any(type(sid) is not int or not 0<=sid<=253 for sid in ids)):
        raise ValueError('Playback IDs must be nonempty, unique servo IDs')
    if type(fps) not in (int,float) or not math.isfinite(fps) or not 1<=fps<=120:
        raise ValueError('Playback FPS must be finite and within 1..120')
    live_preflight(profile,ids)
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds<=0):
        raise ValueError('Playback duration must be positive and finite')
    motors={m['id']:m for m in profile['motors']}
    command_ids=profile.get('diagnostic_command_ids',ids)
    if (not isinstance(command_ids,list) or not command_ids or
            any(type(sid) is not int or sid not in ids for sid in command_ids) or
            len(set(command_ids))!=len(command_ids)):
        raise ValueError('Diagnostic command IDs must be a nonempty unique subset of monitored IDs')
    command_ids=set(command_ids)
    mojia_gaze=False
    if gaze_override is not None and profile.get('schema_version') == 'mojia-playback-0.1':
        from vivis.motion.ownership import MojiaPlaybackGaze
        if not isinstance(gaze_override,MojiaPlaybackGaze):
            raise ValueError('Mojia tracking requires its own reviewed calibration')
        mojia_gaze=True
    if gaze_override is not None and not {a['id'] for a in gaze_override.axes} <= command_ids:
        raise ValueError('Tracked eye axes must be among commanded IDs')
    speed_limits=np.array([motors[sid]['max_speed_counts_s'] for sid in ids],dtype=float)
    direct_ids=profile.get('direct_position_ids',[])
    mouth_command_mode=profile.get('mouth_command_mode','limited')
    if (mouth_command_mode not in ('limited','lips-direct')
            or bool(direct_ids)!=(mouth_command_mode=='lips-direct')):
        raise ValueError('Mouth command mode and direct axes disagree')
    if (not isinstance(direct_ids,list) or len(set(direct_ids))!=len(direct_ids)
            or any(type(sid) is not int or sid not in command_ids for sid in direct_ids)
            or not set(direct_ids)<=set(profile.get('mouth_motion_ids',[]))
            or set(direct_ids)&set(profile.get('gaze_owner_ids',[]))):
        raise ValueError('Invalid direct mouth command ownership')
    if direct_ids and set(direct_ids)!=set(profile.get('round_lip_motion_ids',[])):
        raise ValueError('Direct lip commands must match the calibrated round-lip group')
    direct_columns=np.array([ids.index(sid) for sid in direct_ids],dtype=int)
    direct_low=np.array([motors[sid]['safe_min'] for sid in direct_ids],dtype=float)
    direct_high=np.array([motors[sid]['safe_max'] for sid in direct_ids],dtype=float)
    feedback_age_limit=profile.get('feedback_max_age_s')
    if feedback_age_limit is not None and not 0<feedback_age_limit<=1:
        raise ValueError('Feedback age limit must be positive and at most one second')
    polls_per_frame=profile.get('feedback_polls_per_frame',1)
    if type(polls_per_frame) is not int or not 1<=polls_per_frame<=4:
        raise ValueError('Feedback polls per frame must be 1..4')
    feedback_policy=profile.get('feedback_policy','stop_on_fault')
    if feedback_policy not in ('stop_on_fault','record_only'):
        raise ValueError('Unknown feedback policy')
    record_only=feedback_policy=='record_only'
    scheduled_feedback=profile.get('scheduled_feedback',False)
    if type(scheduled_feedback) is not bool:
        raise ValueError('Invalid scheduled feedback flag')
    reviewed_face=(profile.get('schema_version')=='mojia-playback-0.1'
                   and profile.get('developer_mode')=='face' and gaze_override is None)
    if scheduled_feedback and ((not mojia_gaze and not reviewed_face) or record_only or feedback_age_limit is None
            or profile.get('serial_backend')!='feetech_python' or profile.get('feedback_mode')!='short_registers'):
        raise ValueError('Scheduled feedback requires the reviewed Mojia SDK tracking path')
    control_hz=profile.get('control_hz',fps)
    if (type(control_hz) not in (int,float) or not math.isfinite(control_hz)
            or control_hz not in (30,60) and control_hz!=fps or not 1<=control_hz<=fps):
        raise ValueError('Invalid control frequency')
    if control_hz!=fps and not scheduled_feedback:
        raise ValueError('Separate control frequency requires scheduled Mojia playback')
    face_hz=profile.get('face_update_hz',fps)
    if type(face_hz) not in (int,float) or not math.isfinite(face_hz) or not 1<=face_hz<=fps:
        raise ValueError('Invalid face sampling frequency')
    if face_hz!=fps and not scheduled_feedback:
        raise ValueError('Separate face sampling requires scheduled Mojia playback')
    trajectory=np.asarray(trajectory,dtype=float)
    if (trajectory.ndim!=2 or not len(trajectory) or trajectory.shape[1]!=len(ids)
            or not np.isfinite(trajectory).all()):
        raise ValueError('Invalid prerecorded trajectory')
    for j,sid in enumerate(ids):
        if np.any(trajectory[:,j]<motors[sid]['safe_min']) or np.any(trajectory[:,j]>motors[sid]['safe_max']):
            raise ValueError('Prerecorded targets exceed command limits')
    last_good_feedback={}
    last_feedback_attempt={}
    speeds={sid:int(motors[sid]['max_speed_counts_s']) for sid in ids if sid in command_ids}
    stop=threading.Event();previous_handlers={}
    for sig in [signal.SIGINT,signal.SIGTERM]:
        previous_handlers[sig]=signal.signal(sig,lambda *_:stop.set())
    clock=None;bus=None;gc_suspended=False;events=[];result={'completed':False,'stop_policy':profile['stop_method'],
        'initial_approach_commands':0,'servo_ids':ids,'motion_gain':profile.get('motion_gain',1),
        'preflight_snapshots':[],'sensor_confirmations':[],'startup_communication_recoveries':[],
        'phase':'preflight',
        'command_ids':sorted(command_ids),'monitor_ids':list(ids),
        'playback_fps':fps,'control_hz':control_hz,'baud':profile['baud'],
        'face_update_hz':face_hz,'feedback_polls_per_frame':polls_per_frame,
        'feedback_policy':feedback_policy,'feedback_notifications':[]}
    result['acceleration_control']='servo_register_41'
    result.update(mouth_command_mode=mouth_command_mode,direct_position_ids=direct_ids)
    warning_recorder=WarningRecorder(result.setdefault('telemetry_warnings',[]))
    def received(sid,state):
        result['last_feedback_received']={'id':sid,**state}
    def should_stop():return stop.is_set() or Path(stop_file).exists()
    def feedback_checked(sid,initial=None,observed_at=None):
        if should_stop():raise RuntimeError('Operator stop during feedback')
        last_feedback_attempt[sid]=time.monotonic()
        if initial is not None:result['last_feedback_received']={'id':sid,**initial}
        if record_only:
            try:
                state=initial if initial is not None else bus.feedback(sid)
            except ProtocolError as exc:
                result['feedback_notifications'].append({'id':sid,'monotonic':time.monotonic(),
                    'error':str(exc),'valid_payload':False,
                    'sdk_response':getattr(bus,'last_sdk_response',None)})
                return None  # Never invent a position from damaged/missing data.
            try:check_feedback(sid,state,motors[sid])
            except CriticalTemperatureError:raise
            except RuntimeError as exc:
                result['feedback_notifications'].append({'id':sid,'monotonic':time.monotonic(),
                    'error':str(exc),'valid_payload':True,'state':state})
        else:
            state=verified_feedback(bus,sid,motors[sid],profile.get('confirm_sensor_transients') is True,
                                    result['sensor_confirmations'],should_stop,initial,
                                    on_warnings=warning_recorder.observe,on_received=received)
        last_good_feedback[sid]=time.monotonic()
        if not motors[sid]['safe_min']<=state['position']<=motors[sid]['safe_max']:
            result.setdefault('position_tolerance_samples',[]).append({
                'id':sid,'position':state['position'],'safe_min':motors[sid]['safe_min'],
                'safe_max':motors[sid]['safe_max'],
                'tolerance_counts':motors[sid].get('position_feedback_tolerance_counts',0)})
        result['last_feedback']={'id':sid,**state}
        result.setdefault('last_feedback_by_id',{})[str(sid)]=dict(state)
        if mojia_gaze:
            gaze_override.telemetry_warnings=[{k:r[k] for k in ('id','kind','value','threshold')}
                                             for r in warning_recorder.active.values()]
            gaze_override.observe(sid,state,time.monotonic() if observed_at is None else observed_at)
        return state

    def feedback_error(sid,exc):
        pending_sensor=any(r['id']==sid and not r['recovered'] for r in result['sensor_confirmations'])
        if feedback_age_limit is None or pending_sensor:raise exc
        events[-1]['feedback_error']={'id':sid,'error':str(exc)}
        if polls_per_frame>1 or scheduled_feedback:
            events[-1].setdefault('feedback_errors',[]).append(events[-1]['feedback_error'])

    def scheduled_sample(sid,state,observed_at):
        # An actual sensor/servo alarm still takes the existing hold-and-confirm
        # or stop path. Only communication loss is deferred between commands.
        checked=feedback_checked(sid,state,observed_at)
        if checked is not None:
            sample={'id':sid,**checked,'sample_started_monotonic':observed_at}
            events[-1]['feedback']=sample
            events[-1].setdefault('feedback_samples',[]).append(sample)

    def startup_feedback_checked(sid,deadline):
        if not scheduled_feedback:return feedback_checked(sid)
        return startup_feedback(feedback_checked,sid,deadline,result['startup_communication_recoveries'],
            should_stop,stop.wait,
            lambda:not any(r['id']==sid and not r['recovered'] for r in result['sensor_confirmations']))
    try:
        if should_stop():raise RuntimeError('Stop file exists')
        # Verify physical audio output has been configured before opening the bus.
        audio_latency=profile.get('audio_latency_s',profile.get('audio_latency_seconds','low'))
        audio_options={'gain':profile['audio_gain']} if 'audio_gain' in profile else {}
        clock=AudioClock(audio_path,profile['audio_device'],latency=audio_latency,**audio_options)
        result['audio_requested_latency']=audio_latency
        result['audio_requested_gain']=profile.get('audio_gain',1.)
        if abs(len(trajectory)/fps-clock.duration)>1/fps:raise ValueError('Clip/audio duration mismatch')
        run_duration=min(clock.duration,max_seconds) if max_seconds is not None else clock.duration
        result.update(requested_duration_seconds=run_duration,full_clip_requested=run_duration==clock.duration)
        if profile.get('serial_backend')=='feetech_python':
            from vivis.hardware.feetech.sdk import OfficialSDKBus
            trace_path=Path(log_path).with_name('serial_trace.json') if profile.get('capture_serial_trace') else None
            bus=OfficialSDKBus(profile['port'],profile['baud'],profile.get('feedback_mode','block'),profile.get('write_mode','position_speed'),profile.get('changed_goals_only',False),trace_path,profile.get('post_goal_quiet_ms',0),profile.get('sync_group_size',30),profile.get('sync_group_gap_ms',0),profile.get('sdk_transaction_timeout_ms'),servo_status_policy='record_only' if record_only else 'raise',snapshot_chunk_bytes=profile.get('snapshot_chunk_bytes'))
        else:
            bus=Bus(profile['port'],profile['baud'])
        result['serial_backend']=profile.get('serial_backend','local')
        if mojia_gaze and gaze_override.platform is not None:
            bus.coordinated_goal_ids=(33,34)
        result['sdk_transaction_timeout_ms']=profile.get('sdk_transaction_timeout_ms')
        positions=[];snapshot_deadline=time.monotonic()+10
        for sid in ids:
            if should_stop():raise RuntimeError('Stopped during preflight')
            state=(startup_feedback(bus.snapshot,sid,snapshot_deadline,result['startup_communication_recoveries'],
                                    should_stop,stop.wait) if scheduled_feedback else bus.snapshot(sid))
            motor=motors[sid]
            result['preflight_snapshots'].append(state)
            if state['model_raw']!=motor['model_raw_expected']:raise RuntimeError(f'{sid}: model mismatch')
            if state['mode']!=0 or state['torque_enable']!=1:raise RuntimeError(f'{sid}: requires existing position mode and enabled torque')
            if state['configured_min']>motor['safe_min'] or state['configured_max']<motor['safe_max']:raise RuntimeError(f'{sid}: software range exceeds device range')
            if motor.get('confirm_position_edge_counts'):
                if (state['configured_min'],state['configured_max'])!=(motor['device_min'],motor['device_max']):
                    raise RuntimeError(f'{sid}: device limits changed; edge confirmation requires review')
            if 'configured_max_temperature' in state and not motor.get('telemetry_policy') and motor['max_temperature_c']>state['configured_max_temperature']:
                raise RuntimeError(f'{sid}: software temperature limit exceeds current device protection threshold')
            feedback={'position':state['position_raw'],**{k:state[k] for k in ['temperature_c','load_raw','status_register']}}
            if 'voltage_raw' in state:feedback['voltage_raw']=state['voltage_raw']
            if record_only and not 0<=state['position_raw']<=4095:
                raise RuntimeError('Invalid startup encoder position')
            if record_only:
                tolerance=motor.get('position_feedback_tolerance_counts',0)
                if not motor['safe_min']-tolerance<=state['position_raw']<=motor['safe_max']+tolerance:
                    raise RuntimeError('Startup position is outside the bounded command envelope')
            feedback=feedback_checked(sid,feedback);positions.append(feedback['position'])
            if motor.get('telemetry_policy'):
                result.setdefault('hardware_temperature_limits',{})[str(sid)]=state.get('configured_max_temperature')
        last=np.clip(np.array(positions,dtype=float),
                     np.array([motors[sid]['safe_min'] for sid in ids]),
                     np.array([motors[sid]['safe_max'] for sid in ids]))
        if mojia_gaze:
            trajectory=gaze_override.prepare(trajectory,last,result['preflight_snapshots'],time.monotonic())
        last_sent=None;poll=0
        if len(command_ids)!=len(ids):
            # Diagnostic selection changes only commanded axes. Keep preflight,
            # runtime feedback and freshness guards for ALL monitored axes.
            trajectory=trajectory.copy()
            for column,sid in enumerate(ids):
                if sid not in command_ids:trajectory[:,column]=last[column]
        # Start from measured positions, and approach frame zero with the same rate limit.
        result['phase']='initial_approach'
        last_tick=time.monotonic();ramp_deadline=last_tick+15
        while np.max(abs(last-trajectory[0]))>.5:
            if should_stop():raise RuntimeError('Stopped during initial approach')
            now=time.monotonic()
            if now>ramp_deadline:raise RuntimeError('Initial approach timeout')
            dt=now-last_tick
            if dt<1/fps:stop.wait(1/fps-dt);continue
            last=limit_step(last,trajectory[0],speed_limits,dt);last_tick=now
            goals={sid:int(round(v)) for sid,v in zip(ids,last) if sid in command_ids}
            bus.goals(goals,speeds);last_sent=goals
            result['initial_approach_commands']+=1
            sid=ids[poll%len(ids)];poll+=1
            previous_recoveries=len(result['startup_communication_recoveries'])
            state=startup_feedback_checked(sid,ramp_deadline)
            if len(result['startup_communication_recoveries'])!=previous_recoveries:
                last_tick=time.monotonic()  # Recovery held transmission; resume with a fresh bounded step.
        # Verify that measured positions actually reached the first frame before audio.
        result['phase']='first_frame_settle'
        deadline=time.monotonic()+5
        while True:
            if should_stop():raise RuntimeError('Stopped before audio')
            settled=True
            for sid,target in zip(ids,trajectory[0]):
                state=startup_feedback_checked(sid,deadline)
                settled &= state is not None and abs(state['position']-target)<=motors[sid].get('initial_settle_tolerance_counts',20)
            if settled:break
            if time.monotonic()>deadline:raise RuntimeError('Servos did not settle at first frame')
            stop.wait(.05)
        if (mojia_gaze or scheduled_feedback) and gc.isenabled():
            # This bounded clip retains a large telemetry graph. Collect between
            # clips, never traverse that graph in the 60 Hz command deadline.
            gc.collect();gc.disable();gc_suspended=True
            result['cyclic_gc_suspended_during_audio']=True
        feedback_scheduler=None
        if scheduled_feedback:
            from vivis.runtime.feedback import BudgetedFeedback
            # Startup checks use the original timeout/retry. Playback reads are
            # individually bounded and retried by scheduling a fresh sample.
            bus.set_read_policy(3,retry=False)
            priority_ids=tuple(sid for sid in (gaze_override.camera_eye_id,32,*((33,34) if gaze_override.platform is not None else ()))
                               if sid in command_ids) if mojia_gaze else ()
            feedback_requests=int(8*60/control_hz)
            feedback_scheduler=BudgetedFeedback(bus,ids,priority_ids=priority_ids,max_feedback_age_s=feedback_age_limit,
                                                max_requests_per_step=feedback_requests)
            result['runtime_feedback_schedule']=dict(timeout_ms=3,budget_ms=1000/control_hz,
                budget_mode='remaining_control_period',command_guard_ms=1,max_requests_per_tick=feedback_requests,
                priority_ids=list(priority_ids),priority_period_s=.08,inline_retry=False,
                fair_attempt_scheduling=True,urgent_age_s=feedback_scheduler.urgent_age_s,
                fair_urgent_attempts=True,continue_other_axes_after_read_error=True)
        next_command_at=None
        result['phase']='playing'
        clock.start();last_tick=time.monotonic();previous_index=-1;skipped=0;started=time.monotonic()
        while True:
            if should_stop():raise RuntimeError('Operator stop')
            if clock.fault:raise RuntimeError('Audio callback fault: '+clock.fault)
            t=clock.time()
            if t is None:
                if time.monotonic()-started>2:raise RuntimeError('Audio clock did not start')
                stop.wait(.001);continue
            if t>=run_duration:break
            if t<0:stop.wait(min(.002,-t));continue
            now=time.monotonic()
            if scheduled_feedback and next_command_at is not None and now<next_command_at:
                stop.wait(min(.001,next_command_at-now));continue
            index=min(len(trajectory)-1,max(0,int((t+profile['actuator_latency_ms']/1000)*fps)))
            if index==previous_index and not scheduled_feedback:stop.wait(.001);continue
            dt=(1/control_hz if previous_index<0 else now-last_tick)
            if scheduled_feedback:
                if next_command_at is None:next_command_at=now
                next_command_at+=1/control_hz
                while next_command_at<=now:next_command_at+=1/control_hz
            # A blocked USB call must not cause an unbounded jump on its return.
            if dt>.1:
                result['watchdog_interval_s']=dt
                raise RuntimeError('Playback watchdog: stalled for >100 ms')
            if feedback_age_limit is not None and not record_only:
                stale=[sid for sid in ids if now-last_good_feedback[sid]>feedback_age_limit]
                if stale:
                    result['feedback_watchdog']={str(sid):dict(
                        valid_feedback_age_s=now-last_good_feedback[sid],
                        last_attempt_age_s=(now-feedback_scheduler.last_attempt[sid]
                            if feedback_scheduler is not None and sid in feedback_scheduler.last_attempt else None),
                        last_valid_state=result.get('last_feedback_by_id',{}).get(str(sid))) for sid in stale}
                    raise RuntimeError(f'Feedback watchdog: no recent valid feedback for IDs {stale}')
            face_index=index
            if face_hz!=fps:
                sample_time=max(0.,t+profile['actuator_latency_ms']/1000)
                face_index=min(len(trajectory)-1,int(math.floor(sample_time*face_hz)*fps/face_hz))
            target=trajectory[face_index]
            gaze_started=time.monotonic() if mojia_gaze else None
            if gaze_override is not None:
                if mojia_gaze:gaze_override.set_audio_time(t,run_duration)
                target=gaze_override.apply(target,last,max(dt,1e-6),now)
                # Validate dynamic targets independently of the vision worker;
                # feedback policy never widens command limits.
                if (np.shape(target)!=last.shape or not np.isfinite(target).all()
                        or any(not motors[sid]['safe_min']<=target[j]<=motors[sid]['safe_max']
                               for j,sid in enumerate(ids))):
                    raise ValueError('Live gaze target exceeds command limits')
            gaze_done=time.monotonic() if mojia_gaze else None
            last=limit_step(last,target,speed_limits,max(dt,1e-6));last_tick=now
            if direct_ids:
                nominal=np.asarray(target)[direct_columns]
                if not np.isfinite(nominal).all() or np.any(nominal<direct_low) or np.any(nominal>direct_high):
                    raise ValueError('Direct mouth targets exceed command limits')
                last[direct_columns]=nominal
            goals={sid:int(round(v)) for sid,v in zip(ids,last) if sid in command_ids}
            command_speeds=speeds
            overrides=gaze_override.speed_overrides if mojia_gaze else {}
            if overrides:
                if any(sid not in gaze_override.owned or sid not in command_ids
                       or type(speed) is not int or not 1<=speed<=speeds[sid]
                       for sid,speed in overrides.items()):
                    raise ValueError('Live gaze speed exceeds owned-axis speed limits')
                command_speeds={**speeds,**overrides}
            bus.goals(goals,command_speeds);last_sent=goals
            command_done=time.monotonic() if mojia_gaze else None
            if previous_index>=0:skipped+=max(0,index-previous_index-1)
            events.append({'audio_seconds':t,'frame':index,'sent_monotonic':now,'goals':goals})
            if overrides:events[-1]['speed_overrides']=dict(overrides)
            if face_hz!=fps:events[-1]['face_frame']=face_index
            if gaze_override is not None:
                events[-1]['person_gaze']=gaze_override.last_diagnostic
            if profile.get('changed_goals_only'):
                events[-1]['transmitted_ids']=list(bus.last_transmitted_ids)
            previous_index=index
            # Prioritize the stalest axis so a single lost reply does not wait
            # another complete rotation and unnecessarily hit its age limit.
            if feedback_scheduler is not None:
                # Use the actual slack after the goal packet/quiet interval.
                # A fixed7/14ms cap wasted usable time at lower baud, while
                # dense goal packets left too little time to recover bad reads.
                deadline=next_command_at-.001
                events[-1]['feedback_schedule']=feedback_scheduler.step(deadline,last_good_feedback,
                    scheduled_sample,feedback_error,should_stop)
            for _ in range(0 if feedback_scheduler is not None else polls_per_frame):
                clocks=last_feedback_attempt if record_only else last_good_feedback
                sid=min(ids,key=lambda axis:clocks[axis])
                # Keep the eye movement evidence fresh with the same number of
                # serial reads. Remaining axes retain oldest-first polling.
                if mojia_gaze and now-clocks[gaze_override.camera_eye_id]>=.1:sid=gaze_override.camera_eye_id
                try:
                    state=feedback_checked(sid)
                except (CorruptResponse,ReadTimeout) as exc:
                    # Discard invalid telemetry; never invent a healthy measurement.
                    # Independent per-axis freshness and the existing error-rate gate
                    # bound temporary loss, while every commanded pose remains limited.
                    feedback_error(sid,exc)
                else:
                    if state is not None:
                        events[-1]['feedback']={'id':sid,**state}
                        if polls_per_frame>1:events[-1].setdefault('feedback_samples',[]).append(events[-1]['feedback'])
            if mojia_gaze:
                events[-1]['stage_ms']=dict(gaze=1000*(gaze_done-gaze_started),
                    command=1000*(command_done-gaze_done),feedback=1000*(time.monotonic()-command_done))
        result.update(completed=True,skipped_source_frames=skipped,audio_seconds_at_stop=t)
    except Exception as exc:
        result['error']=str(exc)
        raise
    finally:
        # Stop clock and transmission. Do not suddenly neutralize or unload a gravity-loaded neck.
        if bus is not None and profile.get('changed_goals_only'):
            result['sparse_writes']={
                'enabled':True,'axes_sent':getattr(bus,'goal_axes_sent',0),
                'axes_suppressed':getattr(bus,'goal_axes_suppressed',0),
                'packets_sent':getattr(bus,'goal_packets_sent',0)}
        if bus is not None and isinstance(getattr(bus,'read_requests',None),int):
            result['serial_diagnostics']={'read_requests':bus.read_requests,'corrupt_responses':bus.corrupt_responses,'read_failures':bus.read_failures}
            if isinstance(getattr(bus,'servo_status_notifications',None),list):
                result['serial_diagnostics']['servo_status_notifications']=bus.servo_status_notifications
            if isinstance(getattr(bus,'last_sdk_response',None),dict):
                result['serial_diagnostics']['last_sdk_response']=bus.last_sdk_response
                result['serial_diagnostics']['goal_packets_sent']=bus.goal_packets_sent
        cleanup_errors=[]
        for resource in [clock,bus]:
            if resource is not None:
                try:resource.close()
                except Exception as exc:cleanup_errors.append(str(exc))
        if cleanup_errors:
            result['cleanup_errors']=cleanup_errors
            if result['completed']:
                result.update(completed=False,error='Playback resource cleanup failed: '+'; '.join(cleanup_errors))
        if gc_suspended:gc.enable()
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
        result['events']=events;result['sent_frames']=len(events)
        if mojia_gaze:
            preview_error=gaze_override.finish(result['completed'],result.get('error'))
            if preview_error:result['preview_error']=preview_error
        Path(log_path).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
