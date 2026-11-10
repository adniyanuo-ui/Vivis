"""Prepared audio uses calibrated DAC time, not Python callback scheduling time."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock,patch
import wave

import numpy as np

from vivis.runtime.playback import AudioClock


class ProgramReferenceTests(unittest.TestCase):
    def test_callback_jitter_and_pa_current_time_do_not_warp_reference_pcm(self):
        clock=AudioClock.__new__(AudioClock);clock.data=np.full((10000,1),.2,np.float32)
        clock.cursor=0;clock.first_dac=None;clock.fault=None
        publisher=Mock();clock.reference=publisher
        clock.reference_clock=SimpleNamespace(convert=lambda value:1000.+value)
        clock.callback(np.empty((960,1),np.float32),960,SimpleNamespace(outputBufferDacTime=10.,currentTime=9.9),None)
        clock.callback(np.empty((960,1),np.float32),960,SimpleNamespace(outputBufferDacTime=10.02,currentTime=10.01),None)
        stamps=[c.kwargs['timestamp'] for c in publisher.send.call_args_list]
        self.assertAlmostEqual(stamps[1]-stamps[0],.02)

    def test_reference_calibration_failure_closes_open_stream_and_publisher(self):
        sd=Mock();sd.query_devices.return_value={'default_samplerate':48000};publisher=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            wav=Path(tmp)/'test.wav'
            with wave.open(str(wav),'wb') as w:
                w.setparams((1,2,48000,0,'NONE','not compressed'));w.writeframes(b'\0\0'*4800)
            with patch.dict(sys.modules,sounddevice=sd),patch('vivis.speech.echo.ReferencePublisher',return_value=publisher), \
                 patch('vivis.speech.audio_clock.AudioStreamClock.calibrate',side_effect=RuntimeError('invalid clock')), \
                 self.assertRaisesRegex(RuntimeError,'invalid clock'):
                AudioClock(wav,'fake',reference={'host':'127.0.0.1','port':14788,'session_id':'test'})
        sd.OutputStream.return_value.close.assert_called_once();publisher.close.assert_called_once()
