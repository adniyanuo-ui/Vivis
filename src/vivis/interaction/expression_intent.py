"""Robot-independent52-channel expression intents; no servo IDs or registers."""
import csv
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import time
import uuid

from vivis.paths import ROOT


CONFIG = 'configs/interaction/expression_intent.json'
NAMES = ('smile', 'neutral', 'attentive', 'auto', 'custom')


@lru_cache(maxsize=1)
def channel_names():
    with (ROOT/'configs/schemas/blendshape_channels_61.csv').open(encoding='utf-8-sig',newline='') as f:
        names=tuple(next(csv.reader(f))[2:54])
    if len(names)!=52 or len(set(names))!=52:raise ValueError('Need52canonical expression channels')
    return names


def configuration(root=ROOT):
    c = json.loads((Path(root) / CONFIG).read_text())
    if c.get('schema_version') != 'vivis-expression-intent-1.0' or type(c.get('enabled')) is not bool:
        raise ValueError('Invalid expression intent configuration')
    for key, low, high in (('default_duration_s', 1, 30), ('max_duration_s', 1, 60),
                           ('attack_s', .05, 3), ('release_s', .05, 3),
                           ('ack_timeout_s', .2, 5)):
        value = c.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid expression intent ' + key)
    if c['default_duration_s'] > c['max_duration_s']:
        raise ValueError('Expression duration exceeds maximum')
    return c


def arguments(value, c):
    if not isinstance(value, dict) or not {'expression'} <= value.keys() <= {'expression', 'intensity', 'duration_s', 'source', 'weights'}:
        raise ValueError('Expression needs expression and optional intensity/duration_s/source/weights')
    name = value['expression']
    if name not in (*NAMES,*channel_names()):
        raise ValueError('Choose a canonical52channel, custom weights, or smile/neutral/attentive/auto')
    supplied=value.get('weights',{})
    if (not isinstance(supplied,dict) or not set(supplied)<=set(channel_names())
            or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in supplied.values())):
        raise ValueError('Weights must name canonical52channels with finite values0..1')
    if name=='custom' and not supplied:raise ValueError('Custom expression needs nonempty weights')
    expected={name:1.} if name in channel_names() else supplied if name=='custom' else {}
    # Canonical on-disk packets include derived weights; do not accept a second,
    # conflicting definition for a named single channel or scalar smile intent.
    if 'weights' in value and supplied!=expected:raise ValueError('Weights do not match named expression')
    intensity = value.get('intensity', 1.)
    duration = value.get('duration_s', c['default_duration_s'])
    source = value.get('source', 'command')
    if type(intensity) not in (int, float) or not math.isfinite(intensity) or not 0 <= intensity <= 1:
        raise ValueError('Expression intensity must be0..1')
    if type(duration) not in (int, float) or not math.isfinite(duration) or not 1 <= duration <= c['max_duration_s']:
        raise ValueError('Expression duration outside configured range')
    if source not in ('command', 'social'):
        raise ValueError('Expression source must be command or social')
    return dict(expression=name, intensity=0. if name in ('neutral','attentive','auto') else float(intensity),
                duration_s=float(duration), source=source,weights=expected)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


class IntentWriter:
    def __init__(self, path, config):
        self.path = Path(path); self.c = config; self.active = None

    def clear(self):
        self.active = None
        self.path.unlink(missing_ok=True)
        self.path.with_suffix('.ack.json').unlink(missing_ok=True)

    def publish(self, value, revision, now=None):
        now = time.monotonic() if now is None else now
        a = arguments(value, self.c)
        if not self.c['enabled']:
            raise ValueError('Expression intents disabled')
        if (a['source'] == 'social' and self.active and self.active['source'] == 'command'
                and now < self.active['expires_monotonic'] and self.active['expression'] != 'auto'):
            return dict(status='suppressed_by_explicit_command', intent_id=self.active['id'])
        self.active = dict(a, schema_version='vivis-expression-intent-1.0', id=uuid.uuid4().hex,
                           revision=revision, created_monotonic=now, expires_monotonic=now+a['duration_s'])
        atomic_json(self.path, self.active)
        return dict(status='pending_expression_target', intent_id=self.active['id'])

    def acknowledgement(self, intent_id):
        try:
            value = json.loads(self.path.with_suffix('.ack.json').read_text())
            if value.get('intent_id') == intent_id and value.get('status') in ('target_accepted','no_calibrated_displacement','blocked_by_ownership'):
                return value
        except (OSError, ValueError):
            pass
        return None


class IntentEnvelope:
    """Blend an intent into existing social strength, leaving calibration unchanged."""
    def __init__(self, path, config):
        self.path = Path(path); self.c = config; self.active = None
        self.last_poll = -math.inf; self.last = None; self.blend = 0.; self.goal = 0.
        self.last_id = None
        self.channel_weights={};self.channel_blend=0.;self.channel_source='social'

    def step(self, base, now):
        if not self.c['enabled']:
            return base
        if now-self.last_poll >= .05:
            self.last_poll = now
            try:
                packet = json.loads(self.path.read_text())
                normalized=arguments({k:packet[k] for k in ('expression','intensity','duration_s','source','weights') if k in packet},self.c)
                packet.update(normalized)
                valid = (packet.get('schema_version') == 'vivis-expression-intent-1.0'
                         and isinstance(packet.get('id'), str) and len(packet['id']) == 32
                         and type(packet.get('revision')) is int
                         and type(packet.get('created_monotonic')) in (int,float)
                         and type(packet.get('expires_monotonic')) in (int,float)
                         and 0 <= now-packet['created_monotonic'] <= self.c['max_duration_s']
                         and packet['expires_monotonic'] == packet['created_monotonic']+packet['duration_s'])
                self.active = packet if valid else None
            except (OSError, ValueError, KeyError, TypeError):
                self.active = None
        active = self.active
        selected = active is not None and now < active['expires_monotonic'] and active['expression'] != 'auto'
        dt = 0. if self.last is None else max(0., min(.1, now-self.last)); self.last = now
        if selected:
            goal = active['intensity'] if active['expression']=='smile' else 0.
            goal_tau = self.c['attack_s'] if goal > self.goal else self.c['release_s']
            self.goal += (goal-self.goal)*(-math.expm1(-dt/goal_tau))
        target = 1. if selected else 0.
        tau = self.c['attack_s'] if target > self.blend else self.c['release_s']
        self.blend += (target-self.blend)*(-math.expm1(-dt/tau))
        channels=active['weights'] if selected else {}
        channel_target=1. if channels else 0.
        channel_tau=self.c['attack_s'] if channel_target>self.channel_blend else self.c['release_s']
        self.channel_blend+=(channel_target-self.channel_blend)*(-math.expm1(-dt/channel_tau))
        if channels:
            self.channel_source=active['source']
            for name in set(self.channel_weights)|set(channels):
                previous=self.channel_weights.get(name,0.);goal=channels.get(name,0.)*active['intensity']
                tau=self.c['attack_s'] if goal>previous else self.c['release_s']
                self.channel_weights[name]=previous+(goal-previous)*(-math.expm1(-dt/tau))
            self.channel_weights={k:v for k,v in self.channel_weights.items() if k in channels or v>1e-5}
        elif self.channel_blend<1e-5:self.channel_weights={}
        result = base*(1-self.blend)+self.goal*self.blend
        return min(1.,max(0.,result))

    def acknowledge(self,axes=None,blocked_axes=()):
        active=self.active
        if not active or active['id']==self.last_id:return
        requested=active['weights'];axes=axes or {}
        missing=sorted(name for name in requested if not axes.get(name))
        calibrated=set().union(*(axes.get(name,set()) for name in requested)) if requested else set()
        affected=sorted(calibrated-set(blocked_axes))
        status='no_calibrated_displacement' if requested and not calibrated else 'blocked_by_ownership' if calibrated and not affected else 'target_accepted'
        atomic_json(self.path.with_suffix('.ack.json'),dict(intent_id=active['id'],status=status,
            expression=active['expression'],source=active['source'],observed_monotonic=self.last,
            requested_channels=requested,unmapped_or_zero_channels=missing,mapped_actuator_ids=affected,
            ownership_excluded_ids=sorted(calibrated&set(blocked_axes)),
            note='Control-loop target acknowledgement; not encoder/skin-motion confirmation'))
        self.last_id=active['id']
