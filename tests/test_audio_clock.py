"""Reference time must follow audio clocks, not Python scheduling or rendering cost."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.speech.audio_clock import AudioStreamClock
from vivis.speech.echo import ReferencePublisher


class AudioClockTests(unittest.TestCase):
    def test_calibration_uses_shortest_query_and_maps_callback_timestamps(self):
        clock=iter([100.,100.01,100.1,100.100002,100.2,100.205])
        values=iter([20.,20.1,20.2])
        class Stream:
            @property
            def time(self):return next(values)
        c=AudioStreamClock.calibrate(Stream(),clock=lambda:next(clock),samples=3)
        self.assertAlmostEqual(c.offset_s,80.000001)
        self.assertAlmostEqual(c.calibration_roundtrip_s,.000002)
        self.assertAlmostEqual(c.convert(20.3),100.300001)

    def test_callback_scheduling_delay_cannot_change_the_reference_clock(self):
        c=AudioStreamClock(80.,.000002,20.)
        with patch('vivis.speech.audio_clock.time.monotonic',side_effect=AssertionError('callback time queried')):
            self.assertAlmostEqual(c.convert(20.04),100.04)
            self.assertAlmostEqual(c.convert(20.04),100.04)
        with self.assertRaises(RuntimeError):c.convert(float('nan'))

    def test_invalid_stream_clock_fails_without_fabricating_timing(self):
        for value in (0.,float('nan'),float('inf')):
            with self.assertRaises(RuntimeError):
                AudioStreamClock.calibrate(SimpleNamespace(time=value),clock=lambda:100.,samples=1)

    def test_explicit_dac_stamp_is_not_recomputed_after_rendering(self):
        sock=Mock()
        with patch('vivis.speech.echo.socket.socket',return_value=sock):
            publisher=ReferencePublisher(dict(host='127.0.0.1',port=14788,session_id='clock-test'),48000)
        pcm=np.zeros((960,2),dtype=np.float32)
        timing=SimpleNamespace(outputBufferDacTime=20.04,currentTime=20.)
        with patch('vivis.speech.echo.time.monotonic',side_effect=AssertionError('post-render clock used')):
            publisher.send(pcm,timing,timestamp=100.04)
        raw=sock.sendto.call_args.args[0];header,body=raw.split(b'\n',1)
        self.assertEqual(json.loads(header)['timestamp'],100.04)
        self.assertEqual(len(body),960*4)
        # Other producers retain their previous API until separately migrated.
        with patch('vivis.speech.echo.time.monotonic',return_value=100.):publisher.send(pcm,timing)
        header=sock.sendto.call_args.args[0].split(b'\n',1)[0]
        self.assertAlmostEqual(json.loads(header)['timestamp'],100.04)


if __name__=='__main__':unittest.main()
