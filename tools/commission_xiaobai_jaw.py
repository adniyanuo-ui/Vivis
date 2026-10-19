"""Scoped ID 31 RAM write/readback test; never changes mode, torque or EEPROM."""
import json
import signal
import struct
import time
from pathlib import Path
from xiaobai_runtime.mapping import ROOT, read_json
from xiaobai_runtime.playback import check_feedback
from xiaobai_runtime.protocol import Bus


def main():
    profile_path = ROOT / 'xiaobai/playback_profile.json'
    profile = read_json(profile_path)
    motor = next(m for m in profile['motors'] if m['id'] == 31)
    output = ROOT / 'outputs/xiaobai/jaw_commissioning.json'
    stop_file = ROOT / 'outputs/xiaobai/STOP'
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    def guard():
        if stopped or stop_file.exists():
            raise RuntimeError('Operator stop')
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    report = {'completed': False, 'id': 31, 'writes': [], 'feedback': [],
              'scope': 'RAM 42..47 only; tiny position steps; no EEPROM/torque/neck writes'}
    bus = None
    try:
        guard()
        if not profile['power_checked'] or not profile['hardware_stop_description']:
            raise RuntimeError('Missing commissioning prerequisites')
        bus = Bus(profile['port'], profile['baud'])
        before = bus.snapshot(31)
        report['before'] = before
        if before['model_raw'] != 11272 or before['mode'] != 0 or before['torque_enable'] != 1:
            raise RuntimeError('Unexpected model/mode/torque')
        if before['configured_min'] > motor['safe_min'] or before['configured_max'] < motor['safe_max']:
            raise RuntimeError('Device range differs')
        state = bus.feedback(31)
        check_feedback(31, state, motor)
        origin = state['position']
        target = origin - 12
        if not motor['safe_min'] <= target < origin <= motor['safe_max']:
            raise RuntimeError('Test endpoints exceed bounds')
        # Two-count increments also bound motion if speed units differ on this model.
        for destination in [target, origin]:
            start = origin if destination == target else target
            step = -2 if destination < start else 2
            for position in range(start + step, destination + step, step):
                guard()
                check_feedback(31, bus.feedback(31), motor)
                bus.goals({31: position}, {31: 60})
                report['writes'].append({'time': time.monotonic(), 'position': position, 'speed_field': 60})
                if bus.read(31, 42, 6) != struct.pack('<HHH', position, 0, 60):
                    raise RuntimeError('Goal/time/speed readback mismatch')
                time.sleep(.05)
            end = time.monotonic() + .8
            phase = []
            while time.monotonic() < end:
                guard()
                state = bus.feedback(31)
                check_feedback(31, state, motor)
                phase.append(state)
                report['feedback'].append({'target': destination, **state})
                time.sleep(.05)
            if abs(phase[-1]['position'] - destination) > 6:
                raise RuntimeError('Position failed to follow within six counts')
            if destination == target and origin - min(s['position'] for s in phase) < 5:
                raise RuntimeError('No measurable jaw movement')
        report['after'] = bus.snapshot(31)
        report['completed'] = True
        report['limitation'] = 'Verified this ID/firmware only; speed field roundtrip verified, physical speed units not independently measured'
        motor['register_layout_confirmed'] = True
        motor['register_write_verification'] = 'ID31: bounded 12-count motion and exact RAM42..47 roundtrip; outputs/xiaobai/jaw_commissioning.json'
        profile['audio_output_evidence'] += '; user confirmed sound normal'
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + '\n')
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        if bus is not None:
            bus.close()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'completed': True, 'id': 31, 'motion_commands': len(report['writes']),
                      'position_before': origin, 'position_after': report['after']['position_raw']}))


if __name__ == '__main__':
    main()
