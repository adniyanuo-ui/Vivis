"""Observable smile events, never an inference about a person's emotions."""
import math


def selected_face_allowed(config,faces):
    """Allow the tracker's selected largest ROI, or require one face in frame."""
    selection=config.get('face_selection','single')
    if selection not in ('single','largest') or type(faces) is not int or faces<0:
        raise ValueError('Invalid smile face-selection policy/count')
    return faces>0 if selection=='largest' else faces==1


class SmileGate:
    """Legacy pulse or sustained response to an observable bilateral smile."""
    def __init__(self, config, max_gap_s=1.):
        self.config = dict(config)
        for key in ('enter', 'exit', 'sustain_s', 'filter_tau_s', 'attack_s', 'hold_s', 'release_s', 'cooldown_s'):
            if type(config[key]) not in (float, int) or not math.isfinite(config[key]) or config[key] < 0:
                raise ValueError('Invalid smile setting: ' + key)
        if not 0 <= config['exit'] < config['enter'] <= 1 or min(config[k] for k in ('filter_tau_s','attack_s','release_s')) <= 0:
            raise ValueError('Invalid smile hysteresis/timing')
        self.mode=config.get('response_mode','pulse')
        if self.mode not in ('pulse','sustained'):raise ValueError('Invalid smile response mode')
        if type(max_gap_s) not in (int,float) or not math.isfinite(max_gap_s) or not .02<=max_gap_s<=2:
            raise ValueError('Invalid smile observation freshness')
        self.max_gap=max_gap_s
        self.invalid_hold=config.get('invalid_hold_s',.25)
        if type(self.invalid_hold) not in (int,float) or not math.isfinite(self.invalid_hold) or not 0<=self.invalid_hold<=1:
            raise ValueError('Invalid smile observation grace')
        self.active=False;self.low_since=self.invalid_since=None
        self.filtered = 0.
        self.last = None
        self.since = self.trigger = None
        self.armed = True
        self.last_trigger = -math.inf
        self.envelope = 0.

    def step(self, left, right, now, valid=True):
        if type(now) not in (int, float) or not math.isfinite(now) or self.last is not None and now < self.last:
            raise ValueError('Smile clock must be finite and monotonic')
        elapsed=0. if self.last is None else now-self.last
        dt = min(1.,elapsed)
        self.last = now
        values = (left, right)
        if valid and any(type(v) not in (int, float) or not math.isfinite(v) or not 0<=v<=1 for v in values):
            raise ValueError('Invalid smile coefficients')
        c = self.config
        score = min(values) if valid else 0.
        self.filtered += (score-self.filtered)*(1-math.exp(-dt/c['filter_tau_s']))
        if self.mode=='sustained':
            if elapsed>self.max_gap:
                self.active=False;self.since=self.low_since=self.invalid_since=None
            return self._sustained(now,dt,valid,elapsed)
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

    def _sustained(self,now,dt,valid,elapsed):
        c=self.config
        if not valid:
            self.since=None
            if self.invalid_since is None:self.invalid_since=now
            if now-self.invalid_since>=self.invalid_hold:
                self.active=False;self.low_since=None
        else:
            self.invalid_since=None
            if self.active:
                if self.filtered<=c['exit']:
                    if self.low_since is None:self.low_since=now
                    if now-self.low_since>=c['hold_s']:self.active=False
                else:self.low_since=None
            elif self.filtered>=c['enter']:
                if self.since is None:self.since=now
                if now-self.since>=c['sustain_s'] and now-self.last_trigger>=c['cooldown_s']:
                    self.active=True;self.last_trigger=now;self.since=self.low_since=None
            else:self.since=None
        desired=1. if self.active else 0.
        self.envelope=(min(desired,self.envelope+dt/c['attack_s']) if desired>self.envelope
                       else max(desired,self.envelope-elapsed/c['release_s']))
        phase=('release_wait' if self.active and self.low_since is not None else 'rising' if self.active and self.envelope<1
               else 'holding' if self.active else 'releasing' if self.envelope>0 else 'neutral')
        return dict(schema_version='vivis-smile-event-1.0',timestamp_monotonic=now,
                    valid=bool(valid),strength=float(self.envelope),score=float(self.filtered),phase=phase)


class SmileDetector:
    """Landmarker runs only within the face selected by the existing tracker."""
    def __init__(self, config, root):
        from vivis.vision.landmarks import FaceLandmarkerDetector
        self.detector=FaceLandmarkerDetector(config,root)
        self.gate = SmileGate(config['gate'])

    def detect(self, sample, target):
        observation=self.detector.observe(sample,target)
        weights=observation['weights']
        return self.gate.step(weights.get('mouthSmileLeft',0.),weights.get('mouthSmileRight',0.),
                              observation['timestamp_monotonic'],observation['valid'])

    def close(self):
        self.detector.close()
