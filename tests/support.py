"""Mock hardware/profile builders used only by isolated controller tests."""
import math

def proposal(shapes, baseline, jaw_policy):
    """Endpoint envelopes are proposals, never measured mechanical safety limits."""
    if type(baseline) is not int or not 0 <= baseline <= 4095:
        raise ValueError('Baseline must be an integer in 0..4095')
    if jaw_policy not in ('JawOpen', 'all_calibrated'):
        raise ValueError('Unknown jaw policy')
    motors = []
    for sid in sorted({sid for positions in shapes.values() for sid in positions}):
        points = [baseline] + [positions[sid] for positions in shapes.values() if sid in positions]
        motors.append(dict(id=sid, reported_model='STS3032' if sid<=30 or sid==36 else 'SM45BL', neutral=baseline,
                           endpoint_envelope_not_safe_limits=[min(points), max(points)],
                           safe_min=None, safe_max=None))
    return dict(mixing='bounded_weighted_deltas', jaw_owner=jaw_policy,
                paired_eye_look_is_already_in_lids=True,
                eye_angle_endpoints_rad={name:math.radians(30) for name in shapes if name.startswith('EyeLook')},
                neck_angle_position_maps=None, motion_gain=1., motors=motors)


def hardware_profile(profile, scan):
    """Intersect calibration proposals with device limits; use only the supplied fake scan."""
    rows=scan['servos']
    if len(rows)!=35 or {row['id'] for row in rows}!=set(range(2,37)):
        raise ValueError('Need a complete Mojia scan of IDs 2..36')
    states={row['id']:row for row in rows}
    for sid,state in states.items():
        expected=521 if sid<=30 or sid==36 else 10760
        if state.get('model_raw')!=expected or state.get('mode')!=0 or state.get('status_register')!=0:
            raise ValueError(f'ID{sid}: incomplete scan, model/mode mismatch or fault')
    command_ids=[motor['id'] for motor in profile['motors']]
    for sid in (32,33,34):
        profile['motors'].append(dict(id=sid,reported_model='SM45BL',neutral=2047,
            endpoint_envelope_not_safe_limits=[2047,2047],safe_min=None,safe_max=None))
    for motor in profile['motors']:
        sid=motor['id'];state=states[sid]
        low,high=motor['endpoint_envelope_not_safe_limits']
        if sid in (32,33,34):low,high=2027,2067
        low=max(low,state['configured_min']);high=min(high,state['configured_max'])
        if not low<=motor['neutral']<=high:
            raise ValueError(f'ID{sid}: zero-weight position outside device/calibration intersection')
        motor.update(safe_min=low,safe_max=high,neutral_confirmed=True,
            device_min=state['configured_min'],device_max=state['configured_max'],
            max_speed_counts_s=50 if sid in (32,33,34) else 300,
            max_temperature_c=min(60,state['configured_max_temperature']),
            device_max_temperature_c=state['configured_max_temperature'],max_load_raw=350,
            min_voltage_raw=state['configured_min_voltage_raw'],
            max_voltage_raw=state['configured_max_voltage_raw'],
            model_raw_expected=state['model_raw'],register_layout_confirmed=True,
            register_write_verification='Pending bounded RAM write/readback during explicit arm',
            position_feedback_tolerance_counts=5)
        if sid not in (32,33,34):
            motor.update(feedback_min=state['configured_min'],feedback_max=state['configured_max'])
        if sid==31:
            motor.update(initial_settle_tolerance_counts=32,
                         initial_settle_basis='Mojia startup observation: approximately 26 counts position error at a bounded goal; target/device/load guards remain active')
    profile.update(schema_version='mojia-playback-0.1',port=scan['port'],baud=1000000,
        mixing_confirmed=True,jaw_owner_confirmed=True,
        confirmation_source='User confirmed w=0 is 2047 and requested all Mojia calibration mappings',
        physical_audio_output_confirmed=True,audio_device='USB Audio Device',audio_latency_s=.2,
        actuator_latency_ms=0,power_checked=True,serial_backend='feetech_python',
        sdk_transaction_timeout_ms=20,feedback_mode='short_registers',write_mode='position_speed',
        feedback_policy='stop_on_fault',feedback_max_age_s=1.,feedback_polls_per_frame=1,
        changed_goals_only=True,capture_serial_trace=True,confirm_sensor_transients=True,
        diagnostic_command_ids=command_ids,
        stop_method='Stop sending; preserve last bounded goal and neck holding torque',
        hardware_stop_description='Onsite operator can cut servo power',
        held_neck_ids=[32,33,34])
    return profile
