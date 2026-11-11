"""The board receives the speaker's samples and shares its output lease."""
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vivis.speech.audio_clock import AudioStreamClock
from vivis.speech.board_reference import BoardReferenceOutput


class CallbackStop(Exception):
    pass


class CallbackAbort(Exception):
    pass


class Status:
    def __init__(self, **flags):self.__dict__.update(flags)
    def __bool__(self):return any(self.__dict__.values())
    def __str__(self):return ', '.join(k for k,v in self.__dict__.items() if v)


class BoardReferenceTests(unittest.TestCase):
    def open_pair(self, callback):
        streams = []

        def output_stream(**options):
            latency = options['latency'] if options['latency'] != 'low' else .02
            stream = Mock(latency=latency, callback=options['callback'])
            streams.append(stream)
            return stream

        sd = SimpleNamespace(OutputStream=output_stream, CallbackStop=CallbackStop, CallbackAbort=CallbackAbort,
            query_devices=lambda name, _: dict(index=int(name == 'board'), name=name, max_output_channels=2))
        self.fault = Mock()
        with patch.object(AudioStreamClock, 'calibrate', side_effect=[AudioStreamClock(5., 0., 0.),
                                                                     AudioStreamClock(9., 0., 0.)]):
            pair = BoardReferenceOutput(sd, 'board', self.fault, device='speaker', samplerate=48000,
                channels=2, dtype='float32', blocksize=960, latency='low', callback=callback)
        self.addCleanup(pair.close)
        return pair, *streams

    def test_stereo_final_buffer_is_mirrored_on_its_dac_time(self):
        samples = np.column_stack((np.linspace(0, .5, 960), np.linspace(-.4, 0, 960))).astype('float32')

        def render(data, *_):
            data[:] = samples
            raise CallbackStop

        callback = Mock(side_effect=render)
        pair, speaker, board = self.open_pair(callback)
        start = time.monotonic() + .12
        main = np.zeros_like(samples)
        mirror = np.zeros_like(samples)
        with self.assertRaises(CallbackStop):
            speaker.callback(main, 960, SimpleNamespace(outputBufferDacTime=start - 5.), None)
        board.callback(mirror, 960, SimpleNamespace(outputBufferDacTime=start - 9.), None)
        np.testing.assert_allclose(mirror, samples, atol=1e-6)
        callback.assert_called_once()
        self.fault.assert_not_called()
        self.assertEqual(pair.stats['reference_frames'], 960)

    def test_reference_gap_stops_the_speaker(self):
        pair, speaker, board = self.open_pair(lambda data, *_: data.fill(.25))
        pcm = np.zeros((960, 2), dtype='float32')
        start = time.monotonic() + .12
        speaker.callback(pcm, 960, SimpleNamespace(outputBufferDacTime=start - 5.), None)
        for frame in (1, 2):
            board.callback(pcm, 960, SimpleNamespace(outputBufferDacTime=start - 9. + frame / 50), None)
        with self.assertRaises(CallbackAbort):
            board.callback(pcm, 960, SimpleNamespace(outputBufferDacTime=start - 9. + 3 / 50), None)
        self.fault.assert_called_once()
        with self.assertRaises(CallbackAbort):
            speaker.callback(pcm, 960, SimpleNamespace(outputBufferDacTime=start - 5. + .08), None)
        self.assertFalse(pcm.any())

    def test_board_start_failure_releases_both_streams(self):
        pair, speaker, board = self.open_pair(Mock())
        board.start.side_effect = RuntimeError('device unavailable')
        with self.assertRaisesRegex(RuntimeError, 'device unavailable'):
            pair.start()
        speaker.abort.assert_called_once()
        pair.close()
        speaker.close.assert_called_once()
        board.close.assert_called_once()

    def test_isolated_underflow_keeps_exact_current_pcm_and_recovers(self):
        pair, speaker, board = self.open_pair(lambda data, *_: data.fill(.25))
        pair.on_warning = Mock()
        pcm = np.zeros((960, 2), dtype='float32');start = time.monotonic()+.12
        for frame,status in enumerate((Status(output_underflow=True),None,Status(output_underflow=True),None)):
            stamp=start+frame/50
            speaker.callback(pcm,960,SimpleNamespace(outputBufferDacTime=stamp-5.),None)
            board.callback(pcm,960,SimpleNamespace(outputBufferDacTime=stamp-9.),status)
            np.testing.assert_allclose(pcm,.25,atol=1e-6)
        self.fault.assert_not_called()
        self.assertEqual(pair.stats['output_underflow_callbacks'],2)
        self.assertEqual(pair.stats['max_underflow_streak'],1)
        self.assertEqual(pair.underflow_streak,0)
        self.assertEqual(pair.on_warning.call_count,4)

    def test_five_consecutive_underflows_still_abort_and_mute_speaker(self):
        pair, speaker, board = self.open_pair(lambda data, *_: data.fill(.25))
        pcm=np.zeros((960,2),dtype='float32');start=time.monotonic()+.12
        for frame in range(5):
            stamp=start+frame/50
            speaker.callback(pcm,960,SimpleNamespace(outputBufferDacTime=stamp-5.),None)
            if frame<4:board.callback(pcm,960,SimpleNamespace(outputBufferDacTime=stamp-9.),Status(output_underflow=True))
            else:
                with self.assertRaises(CallbackAbort):
                    board.callback(pcm,960,SimpleNamespace(outputBufferDacTime=stamp-9.),Status(output_underflow=True))
        self.fault.assert_called_once()
        with self.assertRaises(CallbackAbort):
            speaker.callback(pcm,960,SimpleNamespace(outputBufferDacTime=start-5.+.1),None)
        self.assertFalse(pcm.any())

    def test_priming_is_normal_but_other_status_errors_remain_fatal(self):
        pair, _, board = self.open_pair(Mock())
        pcm=np.zeros((960,2),dtype='float32');timing=SimpleNamespace(outputBufferDacTime=time.monotonic()-9.)
        board.callback(pcm,960,timing,Status(priming_output=True))
        self.fault.assert_not_called()
        with self.assertRaises(CallbackAbort):board.callback(pcm,960,timing,Status(output_underflow=True,input_overflow=True))
        self.fault.assert_called_once()

    def test_stopped_reference_callback_still_aborts_speaker(self):
        pair, speaker, _ = self.open_pair(Mock())
        pair.last_reference=time.monotonic()-.4
        pcm=np.ones((960,2),dtype='float32')
        with self.assertRaises(CallbackAbort):
            speaker.callback(pcm,960,SimpleNamespace(outputBufferDacTime=time.monotonic()-5.),None)
        self.assertFalse(pcm.any());self.fault.assert_called_once()
