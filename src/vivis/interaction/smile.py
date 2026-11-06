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
