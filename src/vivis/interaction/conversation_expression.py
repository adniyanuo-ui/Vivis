"""Current-reply expression envelopes; no language model or device access."""
import json
import math
from pathlib import Path

from vivis.paths import ROOT
from vivis.interaction.expression_intent import channel_names


CONFIG = 'configs/interaction/conversation_expression.json'
PHASES = ('disabled', 'neutral', 'listening', 'thinking', 'speaking', 'relaxing')


def configuration(root=ROOT):
    c = json.loads((Path(root)/CONFIG).read_text())
    if (c.get('schema_version') != 'vivis-conversation-expression-1.0'
            or type(c.get('enabled')) is not bool):
        raise ValueError('Invalid conversation expression switch')
    for key, lo, hi in (('publish_interval_s', .02, .2), ('max_age_s', .1, 1.),
                        ('attack_s', .1, 2.), ('release_s', .1, 2.),
                        ('cancel_release_s', .05, .5),
                        ('late_intent_s', 0., 1.), ('attention_hold_s', 1., 15.),
                        ('resting_smile_intensity', 0., .5),
                        ('listening_intensity', 0., .6), ('thinking_intensity', 0., .6),
                        ('listening_smile_scale', 0., 1.)):
        if (type(c.get(key)) not in (int, float) or not math.isfinite(c[key])
                or not lo <= c[key] <= hi):
            raise ValueError('Invalid conversation expression '+key)
    if c['max_age_s'] < 2*c['publish_interval_s']:
        raise ValueError('Expression freshness must cover publication cadence')
    validate_weights(c.get('speaking_weights'))
    shared = c['shared_mouth']
    if type(shared.get('enabled')) is not bool:
        raise ValueError('Invalid shared mouth switch')
    for key, lo, hi in (('max_blend', 0., .2), ('articulation_threshold', .05, .4),
                        ('max_range_fraction', 0., .05)):
        if (type(shared.get(key)) not in (int, float) or not math.isfinite(shared[key])
                or not lo <= shared[key] <= hi):
            raise ValueError('Invalid shared-mouth limit '+key)
    return c


def resting_pose(c, expressions):
    """A mild version of the existing joy preset, never a new mechanical zero."""
    if not c['enabled']:
        return {}
    return {k: v*c['resting_smile_intensity'] for k, v in expressions['presets']['joy'].items()}


def validate_weights(weights):
    if (not isinstance(weights, dict) or not set(weights) <= set(channel_names())
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                   for v in weights.values())):
        raise ValueError('Invalid conversation expression weights')


class ConversationEnvelope:
    """Read a current producer lease; cancelled/stale poses release locally."""
    def __init__(self, path, c, default_weights=None):
        self.path, self.c = Path(path) if path is not None else None, c
        self.default_weights = dict(default_weights or {})
        validate_weights(self.default_weights)
        self.last_poll = -math.inf
        self.last = None
        self.owner = self.scope = None
        self.sequence = -1
        self.packet = None
        self.cancel_until = -math.inf
        self.weights = {}
        self.smile_scale = 1.
        self.diagnostic = {}

    def step(self, now, resting_allowed=True):
        if not self.c['enabled']:
            self.weights = {}
            self.smile_scale = 1.
            self.diagnostic = dict(valid=False,phase='disabled',control_target_only=True)
            return {}
        if self.path is not None and now-self.last_poll >= self.c['publish_interval_s']:
            self.last_poll = now
            try:
                if self.path.stat().st_size > 65536:
                    raise ValueError('Oversize conversation expression')
                packet = json.loads(self.path.read_text())
                validate_weights(packet.get('weights'))
                if (packet.get('schema_version') != 'vivis-conversation-pose-1.0'
                        or packet.get('phase') not in PHASES
                        or type(packet.get('enabled')) is not bool
                        or any(not isinstance(packet.get(k), str) or len(packet[k]) != 32
                               for k in ('owner', 'scope_id'))
                        or type(packet.get('sequence')) is not int or packet['sequence'] < 0
                        or type(packet.get('timestamp_monotonic')) not in (int, float)
                        or not math.isfinite(packet['timestamp_monotonic'])
                        or not 0 <= now-packet['timestamp_monotonic'] <= self.c['max_age_s']
                        or type(packet.get('smile_scale')) not in (int, float)
                        or not math.isfinite(packet['smile_scale']) or not 0 <= packet['smile_scale'] <= 1
                        or packet['phase']=='speaking' and (
                            not isinstance(packet.get('playback_id'),str) or not packet['playback_id']
                            or type(packet.get('played_seconds')) not in (int,float)
                            or not math.isfinite(packet['played_seconds']) or packet['played_seconds']<0)
                        or (packet['owner'] == self.owner and packet['sequence'] < self.sequence)):
                    raise ValueError('Stale/foreign conversation expression')
                self.packet = packet
            except (OSError, ValueError, TypeError, KeyError):
                self.packet = None
        packet = self.packet
        valid = bool(self.c['enabled'] and packet and packet['enabled']
                     and 0 <= now-packet['timestamp_monotonic'] <= self.c['max_age_s'])
        changed = bool(packet and (packet['owner'], packet['scope_id']) != (self.owner, self.scope))
        if changed:self.cancel_until=now+3*self.c['cancel_release_s']
        if packet:
            self.owner, self.scope, self.sequence = packet['owner'], packet['scope_id'], packet['sequence']
        dt = 1/60 if self.last is None else max(0., min(.1, now-self.last))
        self.last = now
        resting = not valid or packet['phase'] in ('neutral', 'relaxing') or packet.get('style') == 'default_smile'
        target = packet['weights'] if valid else self.default_weights
        if resting and not resting_allowed:
            target = {}
        cancelled = now<self.cancel_until or not valid or packet['phase'] in ('listening', 'disabled')
        for name in set(self.weights) | set(target):
            before, goal = self.weights.get(name, 0.), target.get(name, 0.)
            tau = self.c['attack_s'] if goal > before else (
                self.c['cancel_release_s'] if cancelled else self.c['release_s'])
            self.weights[name] = before+(goal-before)*(-math.expm1(-dt/tau))
        self.weights = {k: v for k, v in self.weights.items() if v > 1e-5}
        smile = packet['smile_scale'] if valid else 1.
        tau = self.c['cancel_release_s'] if cancelled else self.c['attack_s']
        self.smile_scale += (smile-self.smile_scale)*(-math.expm1(-dt/tau))
        self.diagnostic = dict(valid=valid, phase=packet['phase'] if valid else 'unavailable',
                               style=packet.get('style') if valid else 'default_smile' if self.default_weights else None,
                               playback_id=packet.get('playback_id') if valid else None,
                               scope_id=self.scope, weights=dict(self.weights),
                               smile_scale=self.smile_scale, control_target_only=True)
        return dict(self.weights)
