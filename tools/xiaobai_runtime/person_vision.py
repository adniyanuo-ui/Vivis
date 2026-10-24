"""USB capture and YuNet detector for the isolated visual service. No actuator I/O."""
import math
import os
import hashlib
import json
import threading
import time

from .person_gaze import VERSION


def select_largest(faces, width, height, min_area_ratio=.002, score_threshold=.8):
    """Rank visible (clipped) bounding-box area. Aim at landmark eye midpoint."""
    candidates = []
    for row in ([] if faces is None else faces):
        if len(row) < 15 or not all(math.isfinite(float(v)) for v in row):
            continue
        x, y, w, h = map(float, row[:4])
        score = float(row[14])
        if score < score_threshold or w <= 0 or h <= 0:
            continue
        x0, y0 = max(0., x), max(0., y)
        x1, y1 = min(float(width), x + w), min(float(height), y + h)
        area = max(0, x1 - x0) * max(0, y1 - y0)
        if area <= 0 or area / (width * height) < min_area_ratio:
            continue
        cx, cy = (float(row[4]) + float(row[6])) / 2, (float(row[5]) + float(row[7])) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        candidates.append({'bbox': [x0, y0, x1 - x0, y1 - y0],
                           'area_ratio': area / (width * height), 'score': score,
                           'point': [cx, cy], 'x': 2 * cx / width - 1, 'y': 2 * cy / height - 1})
    return (max(candidates, key=lambda face: face['area_ratio']) if candidates else None), candidates


class LatestCamera:
    def __init__(self, config):
        import cv2
        self.cv2 = cv2
        self.config = config
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.latest = None
        self.error = None
        self.sequence = 0
        device=config['device']
        if isinstance(device,str) and device.startswith('/dev/'):
            if not os.path.exists(device):
                raise RuntimeError('Camera device is missing: '+device+'; check USB connection and configured device path')
            if not os.access(device,os.R_OK|os.W_OK):
                raise RuntimeError('Camera device access denied: '+device+'; check video group/device ACL permissions')
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError('Cannot open camera: '+str(device)+
                '; check competing processes with sudo fuser -v '+os.path.realpath(str(device))+
                ' and verify V4L2 format support. System camera services may restart after reboot.')
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
        self.capture.set(cv2.CAP_PROP_FPS, config['fps'])
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        controls = {}
        for key, property_id in [('autofocus', cv2.CAP_PROP_AUTOFOCUS),
                                 ('backlight', cv2.CAP_PROP_BACKLIGHT),
                                 ('brightness', cv2.CAP_PROP_BRIGHTNESS),
                                 ('gain', cv2.CAP_PROP_GAIN)]:
            if key in config:
                value = config[key]
                if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
                    self.capture.release()
                    raise ValueError('Invalid camera control: ' + key)
                accepted = self.capture.set(property_id, float(value))
                controls[key] = {'requested': value, 'accepted': accepted,
                                 'reported': self.capture.get(property_id)}
        self.actual = {'width': self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                       'height': self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                       'fps_requested': config['fps'], 'fps_reported': self.capture.get(cv2.CAP_PROP_FPS),
                       'controls': controls}
        self.thread = threading.Thread(target=self._capture, daemon=True, name='latest-usb-frame')
        self.thread.start()

    def _capture(self):
        while not self.stop.is_set():
            # Timestamp the beginning, not completion, of a potentially blocked
            # read. This is a host acquisition bound, not a sensor exposure stamp.
            stamp = time.monotonic()
            ok, frame = self.capture.read()
            if not ok:
                with self.lock:
                    self.error = 'Camera frame read failed'
                self.stop.wait(.1)
                continue
            rotation = self.config['rotation_deg']
            if rotation:
                codes = {90: self.cv2.ROTATE_90_CLOCKWISE, 180: self.cv2.ROTATE_180,
                         270: self.cv2.ROTATE_90_COUNTERCLOCKWISE}
                frame = self.cv2.rotate(frame, codes[rotation])
            if self.config['mirror_x']:
                frame = self.cv2.flip(frame, 1)
            with self.lock:
                self.sequence += 1
                self.latest = (self.sequence, stamp, frame)
                self.error = None

    def get(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        # Avoid racing release() against a driver blocked in read(). The daemon
        # thread and process teardown close the descriptor if the camera hung.
        if not self.thread.is_alive():
            self.capture.release()


class FaceVision:
    def __init__(self, config, root):
        import cv2
        self.cv2 = cv2
        self.config = config
        cv2.setNumThreads(1)
        detector = config['detector']
        model_path = root / detector['model']
        manifest = json.loads((root / 'tools/model-manifests/yunet.json').read_text())
        if (not model_path.is_file()
                or hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest['sha256']):
            raise RuntimeError('YuNet model missing or changed; run tools/download_yunet.py')
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), '', (320, 320), detector['score_threshold'], .3, 5000)

    def detect(self, sample):
        sequence, stamp, frame = sample
        begin = time.monotonic()
        max_width = self.config['detector']['max_width']
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = self.cv2.resize(frame, (max_width, round(frame.shape[0] * scale)))
        frame = frame.copy()
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        selected, candidates = select_largest(
            faces, width, height, self.config['detector']['min_face_area_ratio'],
            self.config['detector']['score_threshold'])
        packet = {'schema_version': VERSION, 'sequence': sequence,
                  'timestamp_monotonic': stamp, 'target': selected,
                  'camera_transform': {k: self.config['camera'][k] for k in ('rotation_deg', 'mirror_x')}}
        for face in candidates:
            x, y, w, h = map(round, face['bbox'])
            color = (0, 220, 70) if face is selected else (100, 100, 100)
            self.cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if selected:
            self.cv2.drawMarker(frame, tuple(map(round, selected['point'])), (0, 220, 70),
                                self.cv2.MARKER_CROSS, 20, 2)
        self.cv2.drawMarker(frame, (width // 2, height // 2), (230, 180, 50), self.cv2.MARKER_CROSS, 16, 1)
        self.cv2.putText(frame, f'faces={len(candidates)} | largest visible area',
                         (10, 24), self.cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 70), 1)
        encoded, jpeg = self.cv2.imencode('.jpg', frame, [self.cv2.IMWRITE_JPEG_QUALITY, 80])
        if not encoded:
            raise RuntimeError('Cannot encode camera preview')
        return packet, jpeg.tobytes(), {'faces': len(candidates), 'size': [width, height],
                                       'processing_ms': (time.monotonic() - begin) * 1000}
