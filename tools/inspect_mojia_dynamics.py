#!/usr/bin/env python3
"""Read all Mojia motion parameters; never write motor RAM or EEPROM."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from xiaobai_runtime.mapping import read_json


def decode(state, factory=None, software_limit=None):
    config=bytes.fromhex(state['config_0_39_hex'])
    ram=bytes.fromhex(state['state_40_70_hex'])
    phase=config[18];speed=int.from_bytes(ram[6:8],'little')
    # Register 18 BIT2 selects 0.0146 or 0.732 RPM. Values below use nominal
    # 1 or 50 encoder counts/s; rounded RPM specifications are not exact conversions.
    speed_scale=1 if phase&4 else 50
    row=dict(id=state['id'],model_raw=state['model_raw'],firmware=f'{config[0]}.{config[1]}',
        phase_raw=phase,acceleration_raw_41=ram[1],
        acceleration_counts_s2_nominal=ram[1]*100 if ram[1] else None,
        acceleration_zero_special=ram[1]==0,
        goal_speed_raw_46=speed,speed_unit_counts_s_nominal=speed_scale,
        goal_speed_zero_mode=('factory_maximum' if phase&8 else 'stop') if (speed&32767)==0 else None,
        goal_speed_limit_counts_s_nominal=None if (speed&32767)==0 and phase&8 else (speed&32767)*speed_scale,
        software_limit_counts_s=software_limit,
        factory_max_speed_raw_84=None,factory_max_speed_counts_s_nominal=None,
        factory_acceleration_limit_raw_85=None,acceleration_multiplier_raw_86=None,
        torque_enable=state['torque_enable'],position_raw=state['position_raw'])
    if factory is not None:
        if len(factory)!=7:raise ValueError('Factory register block must cover 80..86')
        row.update(factory_max_speed_raw_84=factory[4],
            factory_max_speed_counts_s_nominal=factory[4]*50,
            factory_acceleration_limit_raw_85=factory[5],
            acceleration_multiplier_raw_86=factory[6])
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baud',type=int,default=1000000)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Choose a new output directory')
    profile=read_json(args.profile)
    motors={motor['id']:motor for motor in profile['motors']}
    from xiaobai_runtime.feetech_sdk import OfficialSDKBus
    bus=OfficialSDKBus(args.port,args.baud,transaction_timeout_ms=30)
    args.output.mkdir(parents=True)
    rows=[];raw=[]
    try:
        for sid in range(2,37):
            try:
                state=bus.snapshot(sid)
                expected=521 if sid<=30 or sid==36 else 10760
                if state['model_raw']!=expected:raise ValueError('Unexpected motor model')
                record=dict(state=state,factory_80_86_hex=None)
                factory=None
                try:
                    factory=bus.read(sid,80,7);record['factory_80_86_hex']=factory.hex()
                except Exception as exc:record['factory_read_error']=str(exc)
                row=decode(state,factory,motors.get(sid,{}).get('max_speed_counts_s'))
                if factory is None:row['error']=record['factory_read_error']
                raw.append(record);rows.append(row)
                print(json.dumps(row,ensure_ascii=False),flush=True)
            except Exception as exc:
                row=dict(id=sid,error=str(exc));rows.append(row);print(json.dumps(row),flush=True)
    finally:
        bus.close()
        report=dict(read_only=True,captured_at_utc=datetime.now(timezone.utc).isoformat(),
            port=args.port,baud=args.baud,
            interpretation='41 is requested acceleration, 46 is goal speed limit; factory 84/85 are distinct. Acceleration 0 has firmware-dependent multiplier semantics; 85 unit not inferred.',
            sources=['https://ftservo.github.io/ftservo-wiki/reference/parameter/memory-sts/',
                     'https://ftservo.github.io/ftservo-wiki/reference/parameter/memory-sms/'],
            motors=rows,raw=raw)
        (args.output/'motor_dynamics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        fields=list(dict.fromkeys(key for row in rows for key in row))
        with (args.output/'motor_dynamics.csv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    if any('error' in row for row in rows):raise SystemExit('Some parameters could not be read; see saved report')


if __name__=='__main__':main()
