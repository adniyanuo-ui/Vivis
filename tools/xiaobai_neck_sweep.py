#!/usr/bin/env python3
"""Read-only by default; verified, slow ID32-only poses independent of vision/audio."""
import argparse
import json
from pathlib import Path
import signal
import struct
import threading
import time

from xiaobai_runtime.feetech_sdk import OfficialSDKBus, COMM_SUCCESS
from xiaobai_runtime.mapping import ROOT, read_json


def sweep_plan(snapshots, offset=80):
    if type(offset) is not int or offset not in (40, 80, 160):
        raise ValueError('ID32 sweep supports only 40, 80 or 160 counts')
    if {s['id'] for s in snapshots} != {32, 33, 34} or len(snapshots) != 3:
        raise ValueError('Need all three neck snapshots')
    for state in snapshots:
        if (state['model_raw'] != 11272 or state['mode'] != 0 or state['torque_enable'] != 1
                or state['goal_position_raw'] != 2047 or abs(state['position_raw'] - 2047) > 10):
            raise ValueError('Neck must already be at home in enabled position mode')
    state = next(s for s in snapshots if s['id'] == 32)
    config = bytes.fromhex(state['config_0_39_hex'])
    if config[:3] != bytes([2, 53, 0]) or config[18] != 44 or config[30] != 1:
        raise ValueError('ID32 firmware/phase/resolution changed; review SMS units first')
    low, high = 2047-offset, 2047+offset
    if state['configured_min'] > low or state['configured_max'] < high:
        raise ValueError('Sweep would exceed current device limits')
    return dict(id=32, bus='RS485', neutral=2047, safe_min=low, safe_max=high,
                positions=[2047, high, 2047, low, 2047], step_counts=2,
                minimum_step_interval_s=.1, speed_raw=25, hold_seconds=3.)


def write_verified_goal(bus, position, plan, records):
    if (plan['id'] != 32 or type(position) is not int
            or not plan['safe_min'] <= position <= plan['safe_max']):
        raise ValueError('Only bounded ID32 sweep goals are permitted')
    result, status = bus.sdk.WritePosEx(32, position, plan['speed_raw'], bus.acceleration[32])
    records.append(dict(kind='write', monotonic=time.monotonic(), id=32,
                        target=position, communication_result=result, servo_status=status))
    if result != COMM_SUCCESS:
        raise RuntimeError('ID32 unicast write failed; holding last target')
    # A valid status alarm is recorded under the user's existing policy. A
    # missing/corrupt response or mismatching goal is never treated as success.
    if bus.read(32, 42, 6) != struct.pack('<HHH', position, 0, plan['speed_raw']):
        raise RuntimeError('ID32 goal register readback differs from the requested pose')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--play', action='store_true', help='Perform the ID32-only sweep after read-only checks')
    parser.add_argument('--offset-counts', type=int, choices=[40, 80, 160], default=80)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT/'outputs/xiaobai'/('neck32_sweep_'+str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    previous_handlers = {s: signal.signal(s, lambda *_: stop.set()) for s in (signal.SIGINT, signal.SIGTERM)}
    report = dict(completed=False, motion_requested=args.play, events=[],
                  stop_policy='Hold last verified target; never enable or unload neck torque',
                  visible_head_motion='requires onsite observation; encoder feedback is insufficient')
    profile = read_json(ROOT/'xiaobai/playback_profile_0001_observation.json')
    bus = None

    def guard():
        if stop.is_set() or (ROOT/'outputs/xiaobai/STOP').exists():
            raise RuntimeError('Operator stop; retaining the last neck target')

    def feedback():
        guard()
        state = bus.feedback(32)
        report['events'].append(dict(kind='feedback', monotonic=time.monotonic(), id=32, **state))
        return state

    try:
        guard()
        bus = OfficialSDKBus(profile['port'], profile['baud'], transaction_timeout_ms=8,
                             servo_status_policy='record_only', trace_path=output/'serial_trace.json')
        report['initial'] = [bus.snapshot(sid) for sid in (32, 33, 34)]
        plan = report['plan'] = sweep_plan(report['initial'], args.offset_counts)
        print(json.dumps(dict(ready=True, motion_requested=args.play, output=str(output), plan=plan), ensure_ascii=False), flush=True)
        if args.play:
            last = 2047
            for pose in plan['positions']:
                guard()
                print(f'ID32 target={pose}; then hold {plan["hold_seconds"]:g}s', flush=True)
                while last != pose:
                    stop.wait(plan['minimum_step_interval_s'])
                    guard()
                    last += max(-plan['step_counts'], min(plan['step_counts'], pose-last))
                    write_verified_goal(bus, last, plan, report['events'])
                    feedback()
                deadline = time.monotonic()+plan['hold_seconds']
                actual = None
                while time.monotonic() < deadline:
                    actual = feedback()
                    stop.wait(.1)
                if actual is None or abs(actual['position']-pose) > 10:
                    raise RuntimeError('ID32 encoder did not settle at this pose; no next pose commanded')
                report.setdefault('settled_poses', []).append(dict(target=pose, position=actual['position']))
            report['final'] = [bus.snapshot(sid) for sid in (32, 33, 34)]
            if any(s['goal_position_raw'] != 2047 or s['torque_enable'] != 1 for s in report['final']):
                raise RuntimeError('Final neck home/holding check failed')
        report['completed'] = True
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        if bus is not None:
            report['servo_status_notifications'] = bus.servo_status_notifications
            bus.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        (output/'sweep.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(completed=report['completed'], output=str(output)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
