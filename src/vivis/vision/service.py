#!/usr/bin/env python3
"""Camera-only largest-face service, local preview and loopback target broadcast."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import threading
import time

import numpy as np

from vivis.motion.mapping import ROOT, read_json
from vivis.vision.policy import GazePolicy, validate_config, VERSION
from vivis.vision.detection import FaceVision, LatestCamera
from vivis.vision.preview import Preview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1', help='Preview HTTP bind address')
    parser.add_argument('--port', type=int, default=8765, help='Preview HTTP port')
    parser.add_argument('--seconds', type=float, help='Exit after this many seconds; default runs until Ctrl+C')
    parser.add_argument('--snapshot', action='store_true', help='Explicitly save the last annotated frame')
    parser.add_argument('--rotation', type=int, choices=[0, 90, 180, 270], help='Clockwise image rotation for visual calibration')
    parser.add_argument('--mirror-x', action=argparse.BooleanOptionalAction, default=None,
                        help='Flip the corrected image horizontally for visual calibration')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--managed-owner',type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('--seconds must be finite and positive')
    config = read_json(args.config)
    config_digest=hashlib.sha256(args.config.read_bytes()).hexdigest()
    validate_config(config)
    managed=None
    if args.managed_owner is not None:
        from vivis.vision.persistent import ModeControl,validate_mode
        managed=ModeControl(args.managed_owner,config)
        validate_mode(config,managed.base,managed.landmarker)
    camera_config, detector = config['camera'], config['detector']
    if args.rotation is not None:
        camera_config['rotation_deg'] = args.rotation
    if args.mirror_x is not None:
        camera_config['mirror_x'] = args.mirror_x
    identity=dict(config_sha256=config_digest,target_protocol_version=VERSION,target_features=['face_xy','eye_line_roll'],
        camera_device=camera_config['device'],camera_eye=config['control'].get('camera_eye'),
        camera_transform={k:camera_config[k] for k in ('rotation_deg','mirror_x')})
    social_config=config.get('social_smile',{})
    expression_config=config.get('face_expression',{})
    if expression_config.get('enabled') and expression_config.get('face_selection')!='largest':
        raise ValueError('Expression follow requires the existing largest-face selector')
    if social_config.get('enabled'):
        from vivis.interaction.smile import selected_face_allowed
        selected_face_allowed(social_config,0)
        identity['target_features'].append('social_smile')
    if expression_config.get('enabled'):
        identity['target_features'].append('face_expression')
    if (camera_config['rotation_deg'] not in (0, 90, 180, 270)
            or type(camera_config['mirror_x']) is not bool
            or any(type(camera_config[k]) is not int or not 64 <= camera_config[k] <= 1920
                   for k in ('width', 'height'))
            or not 1 <= camera_config['fps'] <= 60
            or not 1 <= detector['hz'] <= 60 or not 160 <= detector['max_width'] <= 1280
            or not 0 < detector['score_threshold'] <= 1
            or not 0 < detector['min_face_area_ratio'] <= 1):
        raise ValueError('Invalid camera/detector settings')
    output = args.output or ROOT / 'outputs/vision' / ('person_tracking_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if output.exists():
        raise ValueError('Use a new output directory')
    stop = threading.Event()
    previous_handlers = {s: signal.signal(s, lambda *_: stop.set()) for s in (signal.SIGINT, signal.SIGTERM)}
    processing, ages = [], []
    frames = with_face = 0
    landmark_inference_frames=0;mode_ack_pending=False
    jpeg = None
    begin = time.monotonic()
    def startup(stage,**details):
        state=dict(stage=stage,pid=os.getpid(),elapsed_s=time.monotonic()-begin,
                   camera_device=camera_config['device'],config_sha256=config_digest,**details)
        print(json.dumps(dict(event='vision_startup',**state),ensure_ascii=False),flush=True)
        status=output.parent/'vision_startup.json'
        try:
            status.parent.mkdir(parents=True,exist_ok=True)
            temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(state,ensure_ascii=False)+'\n')
            temporary.replace(status)
        except OSError:pass  # The flushed process log still contains the stage.
    summary = {'completed': False, 'actuators_opened': False, 'config': config,
               'preview_url': f'http://{args.host}:{args.port}/', 'output': str(output)}
    try:
        startup('vision_lock')
        with ExitStack() as resources:
            lock_path = ROOT / 'outputs/vision/service.lock'
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = resources.enter_context(lock_path.open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('A person-tracking service is already running')
            startup('face_detector_init')
            vision = FaceVision(config, ROOT)
            from vivis.interaction.cloud_control import PresencePublisher
            presence=PresencePublisher(json.loads((ROOT/'configs/perception/speech_presence.json').read_text()),ROOT)
            resources.callback(presence.close)
            from vivis.vision.person import PersonWorker,configuration as person_configuration
            startup('person_detector_init')
            people=PersonWorker(person_configuration(),ROOT)
            resources.callback(people.close)
            if people.error:print('人体检测不可用，仅关闭人体招呼入口：'+people.error,flush=True)
            familiar=None
            if config.get('familiar_faces',{}).get('enabled'):
                startup('identity_init')
                from vivis.vision.familiar import FamiliarFaces
                familiar=FamiliarFaces(config['familiar_faces'],ROOT)
                resources.callback(familiar.close)
            social=landmarker=None
            if managed is not None or social_config.get('enabled') or expression_config.get('enabled'):
                startup('face_landmarker_init')
                from vivis.vision.landmarks import FaceLandmarkerDetector
                wanted=managed.landmarker if managed is not None else expression_config if expression_config.get('enabled') else social_config
                if social_config.get('enabled') and expression_config.get('enabled'):
                    if any(expression_config[k]!=social_config[k] for k in ('model','sha256')):
                        raise ValueError('One camera service requires a single shared landmark model')
                landmarker=FaceLandmarkerDetector(wanted,ROOT)
                resources.callback(landmarker.close)
            if social_config.get('enabled'):
                from vivis.interaction.smile import SmileGate
                social=SmileGate(social_config['gate'],max_gap_s=config['transport']['max_age_s'])
            camera = LatestCamera(camera_config,progress=startup)
            resources.callback(camera.close)
            summary['camera_actual'] = camera.actual
            first_frame_started=time.monotonic()
            startup('first_frame_wait')
            preview = Preview(args.host, args.port)
            resources.callback(preview.close)
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sender.setblocking(False)
            resources.callback(sender.close)
            address = (config['transport']['host'], config['transport']['port'])
            policy = GazePolicy(config)
            output.mkdir(parents=True)
            records = resources.enter_context((output / 'detections.jsonl').open('w'))
            print(json.dumps({'preview_url': summary['preview_url'], 'camera_actual': camera.actual,
                              'actuators_opened': False, 'output': str(output)}, ensure_ascii=False), flush=True)
            last_sequence = -1
            tick = time.monotonic()
            printed = tick - 1
            while not stop.is_set() and (args.seconds is None or time.monotonic() - begin < args.seconds):
                change=managed.poll() if managed is not None else None
                if change is not None:
                    config,config_digest,request_id=change
                    social_config=config.get('social_smile',{});expression_config=config.get('face_expression',{})
                    vision.config=config;policy=GazePolicy(config)
                    identity.update(config_sha256=config_digest,target_features=['face_xy','eye_line_roll'],
                                    vision_request_id=request_id,vision_pid=os.getpid(),
                                    camera_eye=config['control'].get('camera_eye'))
                    social=None
                    if social_config.get('enabled'):
                        from vivis.interaction.smile import SmileGate,selected_face_allowed
                        social=SmileGate(social_config['gate'],max_gap_s=config['transport']['max_age_s'])
                        identity['target_features'].append('social_smile')
                    if expression_config.get('enabled'):identity['target_features'].append('face_expression')
                    mode_ack_pending=True
                    print(json.dumps(dict(event='vision_mode_applied',request_id=request_id,
                        landmark_inference_enabled=bool(social is not None or expression_config.get('enabled')),
                        camera_reopened=False,models_reloaded=False)),flush=True)
                sample = camera.get()
                now = time.monotonic()
                if sample is None or now - sample[1] > config['transport']['max_age_s']:
                    preview.update({**identity,'status': 'camera_stale', 'camera_error': camera.error,
                                    'actuators_opened': False})
                    if frames == 0 and now - first_frame_started > 5:
                        raise RuntimeError('Camera did not provide a fresh frame within five seconds')
                    stop.wait(.02)
                    continue
                if sample[0] == last_sequence:
                    stop.wait(.002)
                    continue
                cycle_begin = now
                people.submit(sample)
                packet, jpeg, metrics = vision.detect(sample)
                if managed is not None:
                    packet['vision_request_id']=managed.request_id
                    metrics.update(vision_pid=os.getpid(),landmarker_loaded=landmarker is not None,
                        landmark_inference_enabled=bool(social is not None or expression_config.get('enabled')))
                metrics['person_detection']=people.observation()
                body=metrics['person_detection'];body_stamp=body['observed_monotonic']
                body['fresh']=(body['available'] and type(body_stamp) in (int,float)
                               and 0<=time.monotonic()-body_stamp<=presence.c['max_age_s'])
                if frames==0:startup('first_frame_detected')
                if familiar is not None and config.get('familiar_faces',{}).get('enabled'):
                    recognized=familiar.step(sample,vision.selected_face_row,faces=metrics['faces'])
                    familiar.publish(recognized)
                    metrics['familiar_recognition_available']=familiar.available
                    metrics['familiar_identity']={k:recognized[k] for k in ('state','person_id','similarity','observed_monotonic')}
                    metrics['familiar_identity']['backend']=familiar.c['backend']
                    if 'observation' in recognized:metrics['familiar_identity']['observation']=recognized['observation']
                    if 'confirmed_monotonic' in recognized:metrics['familiar_identity']['confirmed_monotonic']=recognized['confirmed_monotonic']
                    if familiar.error:metrics['familiar_recognition_error']=familiar.error
                observation=None
                if landmarker is not None and (social is not None or expression_config.get('enabled')):
                    # Expression-follow uses the SAME selected largest-face ROI.
                    # Standby follows and responds to this SAME selected ROI.
                    # Legacy/P0 defaults retain the single-face requirement.
                    social_selected=social is not None and selected_face_allowed(social_config,metrics['faces'])
                    selected=expression_config.get('enabled') or social_selected
                    if selected and packet['target'] is not None:landmark_inference_frames+=1
                    observation=landmarker.observe(sample,packet['target'] if selected else None)
                    if expression_config.get('enabled'):packet['face_expression']=observation
                    if social is not None:
                        weights=observation['weights']
                        packet['social_smile']=social.step(weights.get('mouthSmileLeft',0.),weights.get('mouthSmileRight',0.),
                            observation['timestamp_monotonic'],observation['valid'] and social_selected)
                expression=None
                if observation is not None and observation['valid'] and metrics['faces']==1:
                    expression=dict(bilateral_smile=round(min(observation['weights'].get('mouthSmileLeft',0.),
                        observation['weights'].get('mouthSmileRight',0.)),1),observation_only=True)
                presence.publish(metrics['faces'],sample[1],expression=expression,person=metrics['person_detection'])
                now = time.monotonic()
                last_sequence = sample[0]
                age = now - sample[1]
                # Old inference results are shown as stale, never broadcast as fresh.
                fresh = age <= config['transport']['max_age_s']
                if fresh:
                    sender.sendto(json.dumps(packet, allow_nan=False).encode(), address)
                dt = min(.1, max(.001, now - tick))
                xy, weight, state = policy.step(packet, now, dt, policy.xy)
                tick = now
                frames += 1
                with_face += packet['target'] is not None
                processing.append(metrics['processing_ms'])
                ages.append(age * 1000)
                current = {**identity,'status': state if fresh else 'inference_stale', **metrics,
                           'config_sha256':config_digest,
                           'camera_device':camera_config['device'],
                           'frame_age_ms': age * 1000, 'target': packet['target'],
                           'simulated_xy': (xy * weight).tolist(), 'tracking_weight': weight,
                           'actuators_opened': False,
                           'mount': config['control']['camera_mount']}
                current['camera_eye']=config['control'].get('camera_eye')
                if social is not None:current['social_smile']=packet['social_smile']
                if expression_config.get('enabled'):current['face_expression']=packet['face_expression']
                if config.get('schema_version') in ('mojia-person-gaze-1.0','vivis-person-gaze-1.0'):
                    # The camera service has no actuator feedback; show image direction,
                    # not an integrator pretending the eye moved with its commands.
                    current['simulated_xy']=[packet['target']['x'],packet['target']['y'] if config['control'].get('vertical_enabled') else 0.] if fresh and packet['target'] else [0.,0.]
                    status_path=ROOT/'outputs/runtime/control_status.json'
                    try:
                        control=json.loads(status_path.read_text())
                        control['active']=control.get('active',False) and now-control['updated_monotonic']<1
                    except (OSError,ValueError,KeyError):
                        control={'active':False,'state':'not_running'}
                    current['robot_control']=control
                preview.update(current, jpeg)
                if fresh and mode_ack_pending:
                    managed.applied_frame(config_digest);mode_ack_pending=False
                records.write(json.dumps({**packet, 'metrics': metrics, 'frame_age_ms': age * 1000}) + '\n')
                if now - printed >= 1:
                    records.flush()
                    print(json.dumps(current, ensure_ascii=False), flush=True)
                    printed = now
                stop.wait(max(0, 1 / detector['hz'] - (time.monotonic() - cycle_begin)))
            if not frames:
                raise RuntimeError('No camera frames were processed')
            if args.snapshot and jpeg:
                (output / 'last_frame.jpg').write_bytes(jpeg)
            summary['completed'] = True
    except Exception as exc:
        summary['error'] = str(exc)
        startup('failed',error=str(exc))
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        elapsed = time.monotonic() - begin
        summary.update(frames=frames, frames_with_face=with_face, elapsed_s=elapsed,
                       managed=managed is not None,camera_open_count=int('camera_actual' in summary),
                       landmark_inference_frames=landmark_inference_frames,
                       mode_switches=managed.applied if managed is not None else 0,
                       detection_hz_measured=frames / elapsed,
                       processing_p50_ms=None if not processing else float(np.median(processing)),
                       host_frame_age_p95_ms=None if not ages else float(np.percentile(ages, 95)))
        if output.exists():
            (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
