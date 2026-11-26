"""Bounded DAC-clock expression planning. Audio callback only records small spans."""
from collections import deque
import math
from pathlib import Path
import queue
import threading
import time
import uuid

from vivis.interaction.expression_intent import arguments, atomic_json


class ConversationAnimation:
    def __init__(self, path, c, expressions):
        self.path, self.c, self.expressions = Path(path), c, expressions
        self.owner, self.scope_id = uuid.uuid4().hex, uuid.uuid4().hex
        self.lock = threading.Lock()
        self.scope = None
        self.enabled = self.in_speech = False
        self.attention_since = None
        self.records = {}
        self.spans = deque(maxlen=256)
        self.paused = set()
        self.pending = None
        self.sequence = 0
        self.published = 0
        self.error = None
        self.events = queue.Queue(maxsize=32)
        self.stop = threading.Event()
        self.thread = None

    def context(self, scope, enabled, in_speech, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            changed = scope != self.scope or bool(enabled) != self.enabled
            if changed:
                self.scope, self.scope_id = scope, uuid.uuid4().hex
                self.records.clear()
                self.spans.clear()
                self.paused.clear()
                self.pending = None
                self.attention_since = now if in_speech else None
            if self.in_speech and not in_speech:
                self.attention_since = now
            if in_speech and not self.in_speech:
                self.attention_since = now
                self.spans.clear()
            self.enabled = bool(enabled and self.c['enabled'])
            self.in_speech = bool(in_speech)

    def register(self, rid, scope, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if not self.enabled or scope != self.scope:
                return False
            record=self.records.setdefault(rid, dict(scope=scope, created=now, first_dac=None))
            record['neutral_only']=False
            while len(self.records) > 32:
                old = next(iter(self.records))
                self.records.pop(old)
                self.paused.discard(old)
            return True

    def stage(self, value, scope, now=None):
        now = time.monotonic() if now is None else now
        normalized = arguments(value, self.expressions)
        if normalized['source'] != 'social':
            raise ValueError('Reply animation cannot execute explicit commands')
        with self.lock:
            if not self.enabled or scope != self.scope:
                raise ValueError('Reply expression belongs to an inactive conversation')
            if any(r['first_dac'] is not None and now-r['first_dac'] > self.c['late_intent_s']
                   for r in self.records.values()):
                return dict(status='skipped_late_expression', audio_blocked=False)
            identifier = uuid.uuid4().hex
            self.pending = dict(normalized, id=identifier, scope=scope, created=now, bound=None)
            return dict(status='scheduled_for_current_reply', intent_id=identifier,
                        audio_blocked=False, control_target_confirmed=False)

    def played(self, rid, start, end, dac):
        """Only called with real, noncancelled PCM from the shared VoiceBuffer."""
        with self.lock:
            record = self.records.get(rid)
            if not self.enabled or rid in self.paused:
                return
            if record is None:
                # Trusted DAC may resume a retracted interruption or play an isolated
                # fixed/ledger clip. Observe speech, never inherit a semantic intent.
                record=dict(scope=self.scope,created=dac,first_dac=None,neutral_only=True)
                self.records[rid]=record
                while len(self.records)>32:self.records.pop(next(iter(self.records)))
            if record['scope'] != self.scope:return
            if record['first_dac'] is None:
                record['first_dac'] = dac
            if self.pending and self.pending['bound'] is None and not record.get('neutral_only'):
                self.pending['bound'] = rid
            self.spans.append((rid, start, end, dac))

    def pause(self, ids):
        with self.lock:
            self.paused.update(set(ids)&self.records.keys())
            self.spans = deque((s for s in self.spans if s[0] not in self.paused), maxlen=256)

    def resume(self, ids):
        with self.lock:
            self.paused.difference_update(ids)

    def cancel(self, rid=None):
        with self.lock:
            if rid is None:
                self.records.clear()
                self.spans.clear()
                self.paused.clear()
                self.pending = None
            else:
                self.records.pop(rid, None)
                self.paused.discard(rid)
                self.spans = deque((s for s in self.spans if s[0] != rid), maxlen=256)
                if self.pending and self.pending['bound'] == rid:
                    self.pending = None

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            while len(self.spans) > 1 and self.spans[1][3] <= now:
                self.spans.popleft()
            span = self.spans[0] if self.spans else None
            if span and now >= span[3]:
                self.attention_since = None
            current = bool(span and span[3] <= now < span[3]+(span[2]-span[1])/48000
                           and span[0] in self.records and span[0] not in self.paused)
            phase, style, weights, smile, rid, elapsed = 'neutral', 'neutral', {}, 1., None, None
            intent_id=None
            if not self.enabled:
                phase = 'disabled'
            elif self.in_speech:
                phase, style = 'listening', 'listening'
                weights = {k: v*self.c['listening_intensity']
                           for k, v in self.expressions['presets']['listening'].items()}
                smile = self.c['listening_smile_scale']
            elif current:
                rid, start, end, dac = span
                self.attention_since = None
                elapsed = (start+min(end-start, (now-dac)*48000))/48000
                phase = 'speaking'
                weights = {} if self.records[rid].get('neutral_only') else dict(self.c['speaking_weights'])
                intent = self.pending
                if intent and intent['bound'] == rid:
                    intent_id=intent['id']
                    style = intent['expression']
                    chosen = (self.expressions['presets']['joy'] if style == 'smile' else
                              self.expressions['presets']['listening'] if style == 'attentive' else
                              intent['weights'])
                    intensity = intent['intensity']
                    if style == 'attentive':
                        intensity = self.c['listening_intensity']
                    # Initial emphasis relaxes over actual played samples, never text arrival.
                    strength = intensity*(.7+.3*math.exp(-elapsed/2.))
                    strength *= math.exp(-max(0., elapsed-intent['duration_s'])/self.c['release_s'])
                    weights = {k: v*strength for k, v in chosen.items()}
                    if style in ('concern', 'listening', 'attentive', 'neutral'):
                        smile = 0.
            elif (self.attention_since is not None
                  and 0 <= now-self.attention_since <= self.c['attention_hold_s']):
                phase, style = 'thinking', 'listening'
                weights = {k: v*self.c['thinking_intensity']
                           for k, v in self.expressions['presets']['listening'].items()}
            elif span:
                phase = 'relaxing'
            self.sequence += 1
            return dict(schema_version='vivis-conversation-pose-1.0', owner=self.owner,
                        scope_id=self.scope_id, sequence=self.sequence, timestamp_monotonic=now,
                        enabled=self.enabled, phase=phase, style=style, weights=weights,
                        smile_scale=smile, playback_id=rid, played_seconds=elapsed,
                        intent_id=intent_id)

    def start(self):
        if not self.c['enabled'] or self.thread is not None and self.thread.is_alive():
            return
        self.stop = threading.Event()
        def publish():
            previous = None
            while not self.stop.is_set():
                try:
                    packet = self.snapshot()
                    key = (packet['scope_id'], packet['phase'], packet['style'], packet['playback_id'])
                    if packet['enabled'] or key != previous:
                        atomic_json(self.path, packet)
                        self.published += 1
                        self.error = None
                    if key != previous:
                        try:
                            self.events.put_nowait({k: packet[k] for k in (
                                'scope_id', 'phase', 'style', 'playback_id', 'played_seconds', 'intent_id')})
                        except queue.Full:
                            pass
                        previous = key
                except (OSError, ValueError) as exc:
                    self.error = f'{type(exc).__name__}: {exc}'
                self.stop.wait(self.c['publish_interval_s'])
        self.thread = threading.Thread(target=publish, daemon=True, name='conversation-expression')
        self.thread.start()

    def close(self):
        self.context(None, False, False)
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=.5)
