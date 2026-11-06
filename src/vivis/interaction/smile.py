"""Observable smile events, never an inference about a person's emotions."""
import math


class SmileGate:
    """One response per sustained smile; rearm after a low observation."""
    def __init__(self, config):
        self.config = dict(config)
        for key in ('enter', 'exit', 'sustain_s', 'filter_tau_s', 'attack_s', 'hold_s', 'release_s', 'cooldown_s'):
            if type(config[key]) not in (float, int) or not math.isfinite(config[key]) or config[key] < 0:
                raise ValueError('Invalid smile setting: ' + key)
        if not 0 <= config['exit'] < config['enter'] <= 1 or min(config[k] for k in ('filter_tau_s','attack_s','release_s')) <= 0:
            raise ValueError('Invalid smile hysteresis/timing')
        self.filtered = 0.
        self.last = None
        self.since = self.trigger = None
        self.armed = True
        self.last_trigger = -math.inf
        self.envelope = 0.

    def step(self, left, right, now, valid=True):
        if type(now) not in (int, float) or not math.isfinite(now) or self.last is not None and now < self.last:
            raise ValueError('Smile clock must be finite and monotonic')
        dt = 0. if self.last is None else min(1., now-self.last)
        self.last = now
        values = (left, right)
        if valid and any(type(v) not in (int, float) or not math.isfinite(v) or not 0<=v<=1 for v in values):
            raise ValueError('Invalid smile coefficients')
        c = self.config
        score = min(values) if valid else 0.
        self.filtered += (score-self.filtered)*(1-math.exp(-dt/c['filter_tau_s']))
        if not valid:
            self.since = self.trigger = None
            self.armed = True
        elif self.filtered <= c['exit']:
            self.armed = True
            self.since = None
            self.trigger = None
        elif self.filtered >= c['enter'] and self.armed:
            if self.since is None:
                self.since = now
            if now-self.since >= c['sustain_s'] and now-self.last_trigger >= c['cooldown_s']:
                self.trigger = self.last_trigger = now
                self.armed = False
                self.since = None
        else:
            self.since = None
        desired = 0.
        if valid and self.trigger is not None:
            age = now-self.trigger
            if age <= c['attack_s']:
                desired = age/c['attack_s']
            elif age <= c['attack_s']+c['hold_s']:
                desired = 1.
            else:
                desired = max(0.,1-(age-c['attack_s']-c['hold_s'])/c['release_s'])
        # Limit the release rate even when observations disappear abruptly.
        self.envelope = max(desired, self.envelope-dt/c['release_s']) if desired<self.envelope else desired
        return dict(schema_version='vivis-smile-event-1.0', timestamp_monotonic=now,
                    valid=bool(valid), strength=float(self.envelope), score=float(self.filtered))


class SmileDetector:
    """Landmarker runs only within the face selected by the existing tracker."""
    def __init__(self, config, root):
        import hashlib
        from pathlib import Path
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        path = Path(root)/config['model']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=config['sha256']:
            raise ValueError('Face Landmarker model hash mismatch')
        self.mp = mp
        self.model = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True, min_face_detection_confidence=.5,
            min_face_presence_confidence=.5, min_tracking_confidence=.5))
        self.gate = SmileGate(config['gate'])
        self.timestamp_ms = -1

    def detect(self, sample, target):
        import cv2
        _, stamp, image = sample
        left = right = 0.
        valid = False
        if target is not None:
            x,y,w,h = target['bbox']
            height,width=image.shape[:2]
            x0,y0=max(0,int(x-.15*w)),max(0,int(y-.15*h))
            x1,y1=min(width,math.ceil(x+1.15*w)),min(height,math.ceil(y+1.15*h))
            if x1>x0 and y1>y0:
                # YuNet's selected ROI prevents smile from a different face.
                roi=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)
                self.timestamp_ms=max(self.timestamp_ms+1,int(stamp*1000))
                result=self.model.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=roi),self.timestamp_ms)
                if len(result.face_blendshapes)==1:
                    weights={v.category_name:float(v.score) for v in result.face_blendshapes[0]}
                    left,right=weights['mouthSmileLeft'],weights['mouthSmileRight']
                    valid=True
        return self.gate.step(left,right,stamp,valid)

    def close(self):
        self.model.close()
