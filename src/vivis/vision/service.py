#!/usr/bin/env python3
"""Camera-only largest-face service, local preview and loopback target broadcast."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import json
import math
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
    args = parser.parse_args(argv)
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('--seconds must be finite and positive')
    config = read_json(args.config)
    config_digest=hashlib.sha256(args.config.read_bytes()).hexdigest()
    validate_config(config)
    camera_config, detector = config['camera'], config['detector']
    if args.rotation is not None:
        camera_config['rotation_deg'] = args.rotation
    if args.mirror_x is not None:
        camera_config['mirror_x'] = args.mirror_x
    identity=dict(config_sha256=config_digest,target_protocol_version=VERSION,target_features=['face_xy','eye_line_roll'],
        camera_device=camera_config['device'],camera_eye=config['control'].get('camera_eye'),
        camera_transform={k:camera_config[k] for k in ('rotation_deg','mirror_x')})
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
    jpeg = None
    begin = time.monotonic()
    summary = {'completed': False, 'actuators_opened': False, 'config': config,
               'preview_url': f'http://{args.host}:{args.port}/', 'output': str(output)}
    try:
        with ExitStack() as resources:
            lock_path = ROOT / 'outputs/vision/service.lock'
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = resources.enter_context(lock_path.open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('A person-tracking service is already running')
            vision = FaceVision(config, ROOT)
            camera = LatestCamera(camera_config)
            resources.callback(camera.close)
            summary['camera_actual'] = camera.actual
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
                sample = camera.get()
                now = time.monotonic()
                if sample is None or now - sample[1] > config['transport']['max_age_s']:
                    preview.update({**identity,'status': 'camera_stale', 'camera_error': camera.error,
                                    'actuators_opened': False})
                    if frames == 0 and now - begin > 5:
                        raise RuntimeError('Camera did not provide a fresh frame within five seconds')
                    stop.wait(.02)
                    continue
                if sample[0] == last_sequence:
                    stop.wait(.002)
                    continue
                cycle_begin = now
                packet, jpeg, metrics = vision.detect(sample)
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
                if config.get('schema_version')=='mojia-person-gaze-1.0':
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
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        elapsed = time.monotonic() - begin
        summary.update(frames=frames, frames_with_face=with_face, elapsed_s=elapsed,
                       detection_hz_measured=frames / elapsed,
                       processing_p50_ms=None if not processing else float(np.median(processing)),
                       host_frame_age_p95_ms=None if not ages else float(np.percentile(ages, 95)))
        if output.exists():
            (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
