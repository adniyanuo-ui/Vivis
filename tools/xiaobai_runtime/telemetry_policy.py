"""Explicit software warning thresholds; never alter device protection registers."""
import time


class CriticalTemperatureError(RuntimeError):pass


def validate_policy(policy):
    if (not isinstance(policy,dict) or policy.get('schema_version')!='mojia-telemetry-policy-1.0'
            or policy.get('hardware_protection')!='preserve'):
        raise ValueError('Unsupported telemetry warning policy')
    load=policy.get('load_warning_raw');warn=policy.get('temperature_warning_c');stop=policy.get('temperature_stop_c')
    if (type(load) is not int or not 1<=load<=1023 or type(warn) is not int or type(stop) is not int
            or not 1<=warn<stop<=100):
        raise ValueError('Invalid telemetry warning thresholds')
    count=policy.get('temperature_stop_consecutive_samples',1)
    if type(count) is not int or not 1<=count<=8:
        raise ValueError('Invalid consecutive temperature sample count')


def apply_policy(profile,policy):
    validate_policy(policy)
    profile['telemetry_policy']=dict(policy)
    for motor in profile['motors']:
        motor.update(telemetry_policy=dict(policy),max_load_raw=policy['load_warning_raw'],
                     max_temperature_c=policy['temperature_stop_c'])


def evaluate(sid,state,policy):
    # Policy is validated before opening hardware. Strict > matches the user's
    # requested boundaries: exactly800,80,100 do not cross their thresholds.
    temperature=state['temperature_c']
    if temperature>policy['temperature_stop_c']:
        raise CriticalTemperatureError(f'{sid}: critical temperature {temperature} C > {policy["temperature_stop_c"]} C')
    warnings=[];load=state['load_raw']&1023
    if load>policy['load_warning_raw']:
        warnings.append(dict(id=sid,kind='load',value=load,threshold=policy['load_warning_raw']))
    if temperature>policy['temperature_warning_c']:
        warnings.append(dict(id=sid,kind='temperature',value=temperature,threshold=policy['temperature_warning_c']))
    return warnings


def verify_sample(reader,sid,initial,validate,policy,records,should_stop=lambda:False,on_received=None):
    """Hold command progression while checking consecutive hot whole samples.

    A normal sample resets the streak. Corrupt/missing replies and other faults
    propagate; they are never counted as hot or healthy temperature samples.
    The caller's validator must check nonthermal faults before temperature.
    """
    if should_stop():raise RuntimeError('Operator stop during temperature verification')
    try:return initial,validate(initial)
    except CriticalTemperatureError:
        required=(policy or {}).get('temperature_stop_consecutive_samples',1)
        if required==1:raise
    record=dict(id=sid,kind='temperature_confirmation',required_consecutive=required,
                consecutive_high_samples=1,recovered=False,monotonic=time.monotonic(),
                samples=[dict(state=dict(initial))])
    records.append(record)
    print(f'WARNING ID{sid} temperature={initial["temperature_c"]}C; holding goals for {required}-sample verification',flush=True)
    for _ in range(required-1):
        if should_stop():raise RuntimeError('Operator stop during temperature verification')
        time.sleep(.005)
        if should_stop():raise RuntimeError('Operator stop during temperature verification')
        try:
            state=reader(sid)
            record['samples'].append(dict(state=dict(state)))
            if on_received is not None:on_received(sid,state)
            warnings=validate(state)
        except CriticalTemperatureError:
            record['consecutive_high_samples']+=1
        except BaseException as exc:
            record['error']=str(exc)
            raise
        else:
            record.update(consecutive_high_samples=0,recovered=True)
            print(f'WARNING ID{sid} temperature recheck={state["temperature_c"]}C; streak reset, continuing',flush=True)
            return state,warnings
    record['error']=f'{required} consecutive samples above temperature stop threshold'
    raise CriticalTemperatureError(f'{sid}: critical temperature {state["temperature_c"]} C > '
        f'{policy["temperature_stop_c"]} C ({required} consecutive valid samples)')


class WarningRecorder:
    """One record per ongoing episode; print once, retain count and last sample."""
    def __init__(self,records):
        self.records=records;self.active={}

    def observe(self,sid,state,warnings):
        now=time.monotonic();present=set()
        for warning in warnings:
            key=(sid,warning['kind']);present.add(key)
            if key not in self.active:
                record=dict(**warning,first_value=warning['value'],peak_value=warning['value'],
                            first_state=dict(state),first_monotonic=now,last_monotonic=now,count=0,cleared=False)
                self.records.append(record);self.active[key]=record
                print(f'WARNING ID{sid} {warning["kind"]}={warning["value"]} > {warning["threshold"]}; continuing',flush=True)
            record=self.active[key]
            record.update(value=warning['value'],peak_value=max(record['peak_value'],warning['value']),
                          last_monotonic=now,count=record['count']+1,state=dict(state))
        for key in list(self.active):
            if key[0]==sid and key not in present:
                self.active.pop(key).update(cleared=True,cleared_monotonic=now)
