"""Mirror speaker PCM to an array's analog AEC loopback on its DAC clock."""
from contextlib import ExitStack
import threading
import time

import numpy as np

from vivis.speech.audio_clock import AudioStreamClock
from vivis.speech.echo import ReferenceBuffer


MAX_CONSECUTIVE_UNDERFLOWS = 5


def reference_device(value):
    if value is not None and (not isinstance(value, str) or not value.strip() or value != value.strip()):
        raise ValueError('board_reference_device must be a device name or null')
    return value


class BoardReferenceOutput:
    """Two streams in one speaker lease. Only the main callback renders audio."""

    def __init__(self, sd, board, on_fault, *, device, samplerate, channels, dtype, blocksize, latency, callback,
                 on_warning=None):
        speaker_info = sd.query_devices(device, 'output')
        board_info = sd.query_devices(board, 'output')
        if speaker_info['index'] == board_info['index'] or board_info['max_output_channels'] < 2:
            raise ValueError('Board reference needs a separate stereo playback device')
        if channels not in (1, 2):
            raise ValueError('Board reference supports mono or stereo speaker audio')
        self.sd = sd
        self.callback = callback
        self.on_fault = on_fault
        self.on_warning = on_warning
        self.rate = samplerate
        self.lock = threading.Lock()
        self.buffers = [ReferenceBuffer(seconds=.8) for _ in range(channels)]
        self.first_dac = None
        self.end_dac = None
        self.last_reference = None
        self.started = None
        self.fault = None
        self.gap_streak = 0
        self.underflow_streak = 0
        self.stats = dict(device=board_info['name'], reference_callbacks=0,
                          reference_gap_callbacks=0, reference_frames=0,
                          output_underflow_callbacks=0, max_underflow_streak=0,
                          underflow_streak_limit=MAX_CONSECUTIVE_UNDERFLOWS)
        self.resources = ExitStack()
        try:
            # The speaker callback produces PCM before the board asks for the
            # same DAC time. The speaker's own DAC still drives lip sync.
            speaker_latency = max(.12, latency) if isinstance(latency, (int, float)) else .12
            if latency == 'high':
                speaker_latency = latency
            self.speaker = sd.OutputStream(device=device, samplerate=samplerate, channels=channels,
                dtype=dtype, blocksize=blocksize, latency=speaker_latency, callback=self.render_speaker)
            self.resources.callback(self.speaker.close)
            self.board = sd.OutputStream(device=board_info['index'], samplerate=48000, channels=2,
                dtype='float32', blocksize=960, latency='low', callback=self.render_reference)
            self.resources.callback(self.board.close)
            if self.speaker.latency - self.board.latency < .04:
                raise RuntimeError('Insufficient speaker DAC lookahead for DOV reference')
            self.speaker_clock = AudioStreamClock.calibrate(self.speaker)
            self.board_clock = AudioStreamClock.calibrate(self.board)
            self.stats.update(speaker_latency_s=self.speaker.latency, reference_latency_s=self.board.latency)
        except BaseException:
            self.resources.close()
            raise

    def fail(self, message):
        if self.fault is None:
            self.fault = 'Board AEC reference: ' + message
            self.stats['fault'] = self.fault
            self.on_fault(self.fault)

    def render_speaker(self, data, frames, timing, status):
        now = time.monotonic()
        if now - (self.last_reference or self.started or now) > .3:
            self.fail('DOV callback stopped')
        if self.fault:
            data.fill(0)
            raise self.sd.CallbackAbort
        ended = False
        try:
            self.callback(data, frames, timing, status)
        except self.sd.CallbackStop:
            ended = True  # The last buffer still plays and needs its reference.
        stamp = self.speaker_clock.convert(timing.outputBufferDacTime)
        np.clip(data, -1, 1, out=data)
        with self.lock:
            if self.first_dac is None:
                self.first_dac = stamp
            for channel, buffer in enumerate(self.buffers):
                buffer.append(stamp, self.rate, data[:, channel].copy(), 'speaker', now)
            if ended:
                self.end_dac = stamp + frames / self.rate
        if ended:
            raise self.sd.CallbackStop

    def render_reference(self, data, frames, timing, status):
        data.fill(0)
        self.last_reference = time.monotonic()
        # PortAudio reports a past output gap on a still-running stream. A
        # single flag is not a lost device: send the CURRENT DAC-time samples,
        # never replay old PCM or abort the microphone on this warning alone.
        underflow = bool(status and getattr(status, 'output_underflow', False))
        other_error = bool(status and any(getattr(status, field, False) for field in
                           ('input_underflow', 'input_overflow', 'output_overflow')))
        priming = bool(status and getattr(status, 'priming_output', False))
        if underflow and not other_error:
            self.underflow_streak += 1
            self.stats['output_underflow_callbacks'] += 1
            self.stats['max_underflow_streak'] = max(self.stats['max_underflow_streak'], self.underflow_streak)
            if self.underflow_streak == 1 and self.on_warning:
                self.on_warning('DOV output underflow; continuing with current DAC-time PCM')
            if self.underflow_streak >= MAX_CONSECUTIVE_UNDERFLOWS:
                self.fail(f'DOV output underflow for {MAX_CONSECUTIVE_UNDERFLOWS} consecutive callbacks')
        elif status and (other_error or not priming):
            self.fail(str(status))
        else:
            if self.underflow_streak and self.on_warning:
                self.on_warning('DOV output recovered after '+str(self.underflow_streak)+' underflow callback(s)')
            self.underflow_streak = 0
        if self.fault:
            raise self.sd.CallbackAbort
        try:
            stamp = self.board_clock.convert(timing.outputBufferDacTime)
            with self.lock:
                coverage = 1.
                for channel, buffer in enumerate(self.buffers):
                    pcm, covered = buffer.sample(stamp, frames, 48000)
                    data[:, channel] = pcm
                    coverage = min(coverage, covered)
                in_playback = (self.first_dac is not None and stamp >= self.first_dac
                               and (self.end_dac is None or stamp + frames / 48000 <= self.end_dac))
            if len(self.buffers) == 1:
                data[:, 1] = data[:, 0]
            self.stats['reference_callbacks'] += 1
            self.stats['reference_frames'] += round(coverage * frames)
            if in_playback and coverage < .95:
                self.stats['reference_gap_callbacks'] += 1
                self.gap_streak += 1
                if self.gap_streak >= 3:
                    raise RuntimeError('DOV outran speaker PCM for three callbacks')
            else:
                self.gap_streak = 0
        except Exception as exc:
            data.fill(0)
            self.fail(str(exc))
            raise self.sd.CallbackAbort from None

    @property
    def active(self):
        return self.speaker.active

    @property
    def time(self):
        return self.speaker.time

    def start(self):
        self.started = time.monotonic()
        self.speaker.start()
        try:
            self.board.start()
        except BaseException:
            self.speaker.abort()
            raise

    def stop(self):
        try:
            self.speaker.stop()
        finally:
            self.board.stop()

    def abort(self):
        try:
            self.speaker.abort()
        finally:
            self.board.abort()

    def close(self):
        self.resources.close()
