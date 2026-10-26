"""Restore explicitly configured tuning before homing, without enabling axes.

Both commissioned models use these addresses. Limits, phase, EEPROM lock,
torque enable and position targets are never written by this module.
"""
import time

from vivis.runtime.homing import write_ram_verified
from vivis.runtime.playback import startup_feedback


def startup_plan(preset, snapshots, home_speed):
    if preset.get('schema_version')!='mojia-startup-hardware-1.0':
        raise ValueError('Unsupported startup hardware schema')
    if type(home_speed) is not int or not 1<=home_speed<=32767:
        raise ValueError('Invalid startup home speed')
    states={s['id']:s for s in snapshots}
    axes=preset['axes'];excluded=preset['excluded_ids']
    selected=[a['id'] for a in axes]
    if (len(states)!=len(snapshots) or len(set(selected))!=len(selected)
            or len(set(excluded))!=len(excluded) or set(selected)&set(excluded)
            or set(selected)|set(excluded)!=set(states)):
        raise ValueError('Startup hardware inventory differs from home inventory')
    speeds={sid:int.from_bytes(bytes.fromhex(s['state_40_70_hex'])[6:8],'little')
            for sid,s in states.items()}
    plan=[]
    for a in axes:
        sid=a['id'];s=states[sid]
        if (a['model_raw'] not in (521,10760) or s['model_raw']!=a['model_raw']
                or s['mode']!=0 or s['status_register']
                or [s['configured_min'],s['configured_max']]!=a['device_limits']):
            raise RuntimeError(f'ID{sid}: startup model/mode/limits/status mismatch')
        config=bytes.fromhex(s['config_0_39_hex']);ram=bytes.fromhex(s['state_40_70_hex'])
        if len(config)!=40 or len(ram)!=31 or not config[18]&4:
            raise RuntimeError(f'ID{sid}: startup register layout/speed unit mismatch')
        acc=a['acceleration_raw'];speed=a['speed_raw']
        if type(acc) is not int or not 0<=acc<=254 or type(speed) is not int or not 1<=speed<=32767:
            raise ValueError(f'ID{sid}: invalid startup acceleration/speed')
        speeds[sid]=speed
        # Stage the small homing speed first, including already-enabled axes.
        # home_all restores the requested task speed only after neutral settles.
        fields=[(46,2,min(home_speed,speed)),(41,1,acc)]
        for key,start,count in [('pid_raw',21,3),('deadband_raw',26,2)]:
            if key not in a:continue
            values=a[key]
            if len(values)!=count or any(type(v) is not int or not 0<=v<=254 for v in values):
                raise ValueError(f'ID{sid}: invalid startup {key}')
            fields.extend((start+i,1,v) for i,v in enumerate(values))
        for address,width,value in fields:
            block,offset=(config,address) if address<40 else (ram,address-40)
            before=int.from_bytes(block[offset:offset+width],'little')
            if before!=value:
                plan.append(dict(id=sid,address=address,width=width,before=before,target=value))
    return speeds,plan


def restore_startup_settings(bus,preset,snapshots,home_speed,guard,report):
    speeds,plan=startup_plan(preset,snapshots,home_speed)
    report.update(completed=False,planned_writes=plan,writes=[],task_speeds=speeds)
    for row in plan:
        guard()
        report['writes'].append(write_ram_verified(bus,row['id'],row['address'],row['target'],row['width']))
    deadline=time.monotonic()+10
    after=[startup_feedback(bus.snapshot,s['id'],deadline,report.setdefault('communication_recoveries',[]),
                           should_stop=lambda:guard() or False) for s in snapshots]
    # Independent whole snapshots verify every requested field, as well as the
    # absence of torque/goal/limit changes. Never accept a partial restoration.
    remaining_speeds,remaining=startup_plan(preset,after,home_speed)
    if remaining or remaining_speeds!=speeds:
        raise RuntimeError('Startup hardware tuning readback differs from preset')
    before_by_id={s['id']:s for s in snapshots}
    allowed_config={}
    for row in plan:
        if row['address']<40:
            allowed_config.setdefault(row['id'],set()).update(range(row['address'],row['address']+row['width']))
    for s in after:
        old=before_by_id[s['id']]
        raw=bytes.fromhex(s['config_0_39_hex']);previous=bytes.fromhex(old['config_0_39_hex'])
        if (s['torque_enable']!=old['torque_enable'] or s['goal_position_raw']!=old['goal_position_raw']
                or any(x!=y and i not in allowed_config.get(s['id'],())
                       for i,(x,y) in enumerate(zip(raw,previous)))):
            raise RuntimeError(f'ID{s["id"]}: unexpected torque/goal/config change during startup restore')
    report.update(completed=True,snapshots_after=after)
    return speeds
