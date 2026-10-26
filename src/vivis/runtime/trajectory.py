"""Validate prerecorded trajectories and initial states for the Mojia runtime."""
import hashlib
import numpy as np
from pathlib import Path
from vivis.motion.mapping import read_json

MOJIA_IDS = set(range(2,32)) | {35,36}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def startup_target(state,motor,neck=False,recover_disabled=False):
    """Explicit disabled-axis recovery may start outside limits; every goal stays bounded."""
    sid=motor['id']
    if state['model_raw']!=motor['model_raw_expected'] or state['mode']!=0 or state['torque_enable'] not in (0,1):
        raise ValueError(f'ID{sid}: unexpected model, mode or torque state')
    if (state['configured_min'],state['configured_max'])!=(motor['device_min'],motor['device_max']):
        raise ValueError(f'ID{sid}: device limits changed; regenerate profile')
    position=state['position_raw']
    if not 0<=position<=4095:raise ValueError(f'ID{sid}: invalid single-turn position')
    if recover_disabled:
        if sid not in (29,35,36) or state['torque_enable']!=0 or neck:
            raise ValueError(f'ID{sid}: disabled recovery is restricted to unpowered face axes 29/35/36')
        if not motor['device_min']<=2047<=motor['device_max'] or not motor['safe_min']<=2047<=motor['safe_max']:
            raise ValueError(f'ID{sid}: recovery target is outside reviewed limits')
        return 2047
    if not motor['device_min']<=position<=motor['device_max']:
        raise ValueError(f'ID{sid}: measured {position} outside device [{motor["device_min"]},{motor["device_max"]}]')
    if neck:return position
    target=max(motor['safe_min'],min(motor['safe_max'],position))
    if state['torque_enable']==1 and 'feedback_min' in motor:
        return target  # No priming write on enabled axes; check_feedback enforces device limits.
    tolerance=motor.get('position_feedback_tolerance_counts',0)
    if abs(target-position)>10 or state['torque_enable']==1 and abs(target-position)>tolerance:
        raise ValueError(f'ID{sid}: restart recovery exceeds bounded policy; measured {position}, allowed [{motor["safe_min"]},{motor["safe_max"]}]')
    return target


def feedback(state):
    return dict(position=state['position_raw'],**{k:state[k] for k in
                ['load_raw','temperature_c','voltage_raw','status_register']})


def load_targets(targets_path,profile,audio,calibration):
    if (sha256(targets_path)!=profile['targets_sha256'] or sha256(audio)!=profile['audio_sha256']
            or sha256(calibration)!=profile['calibration_sha256']):
        raise ValueError('Targets, audio or Mojia calibration differs from reviewed profile')
    payload=read_json(targets_path)
    if payload.get('schema_version')!='mojia-motor-targets-0.1' or payload.get('device_limits_applied') is not True:
        raise ValueError('Regenerate targets using a complete Mojia hardware scan')
    if payload['calibration_sha256']!=profile['calibration_sha256'] or payload['audio_sha256']!=profile['audio_sha256']:
        raise ValueError('Target provenance mismatch')
    ids=payload['motor_ids'];positions=np.asarray(payload['positions'],dtype=float)
    if (not ids or len(set(ids))!=len(ids) or any(type(sid) is not int or sid not in MOJIA_IDS for sid in ids)
            or positions.shape!=(payload['frames'],len(ids)) or not len(positions) or not np.isfinite(positions).all()):
        raise ValueError('Invalid Mojia target matrix')
    if set(ids)!=set(profile['diagnostic_command_ids']):raise ValueError('Command IDs differ from profile')
    fps=payload['fps']['num']/payload['fps']['den']
    if not np.isfinite(fps) or not 1<=fps<=120:raise ValueError('Invalid FPS')
    motors={m['id']:m for m in profile['motors']}
    for column,sid in enumerate(ids):
        if np.any(positions[:,column]<motors[sid]['safe_min']) or np.any(positions[:,column]>motors[sid]['safe_max']):
            raise ValueError(f'ID{sid}: target exceeds reviewed limits')
    return ids,positions,fps
