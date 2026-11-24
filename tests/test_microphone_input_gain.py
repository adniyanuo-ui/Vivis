"""Microphone calibration stays downstream of AEC and outside the output reference."""
import json
import time
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.paths import ROOT
from vivis.speech.realtime_audio import AudioIO,input_gain_parameters,input_latency


class InputGainTests(unittest.TestCase):
    def test_input_jitter_buffer_is_bounded_and_does_not_change_frame_size(self):
        self.assertEqual(input_latency({}),'low');self.assertEqual(input_latency(dict(input_latency_ms=80)),.08)
        for bad in (0,True,'80',201,float('nan')):
            with self.assertRaises(ValueError):input_latency(dict(input_latency_ms=bad))
    def frame(self,gain_db=None,amplitude=.04):
        speech=json.loads((ROOT/'configs/perception/speech.json').read_text())
        if gain_db is None:speech['capture'].pop('input_gain_db',None)
        else:speech['capture']['input_gain_db']=gain_db
        player=Mock();io=AudioIO(speech,'unused',player,aec=False)
        io.stats['processed_frames']=49
        near=(np.sin(np.arange(960)*.1)*amplitude).astype(np.float32)
        far=np.full(960,.025,np.float32)
        front=Mock();front.process48.side_effect=lambda n,f:(n[::3],f[::3],n[::3]*.5)
        io.capture.put_nowait((time.monotonic(),near.copy()))
        original=io.uplink.put_nowait
        def put(item):original(item);io.stop.set()
        io.uplink.put_nowait=put
        udp=Mock();udp.recvfrom.side_effect=BlockingIOError
        with patch('vivis.speech.realtime_audio.ReferenceBuffer') as reference:
            reference.return_value.sample.return_value=(far.copy(),1.)
            io.process(udp,front)
        self.assertIsNone(io.fault)
        item=io.uplink.get_nowait();meter=io.meters.get_nowait()
        return io,item,meter,near,far,front,player

    def test_default_is_exactly_zero_db_and_nine_db_is_a_bounded_factor(self):
        self.assertEqual(input_gain_parameters({}),(0.,1.))
        self.assertAlmostEqual(input_gain_parameters({'input_gain_db':9})[1],2.818382931)
        for bad in (True,'9',None,-1,13,float('nan'),float('inf')):
            with self.assertRaises(ValueError):input_gain_parameters({'input_gain_db':bad})

    def test_zero_db_preserves_existing_pcm_without_new_processing(self):
        io,(_,pcm,raw,ref),meter,near,far,front,player=self.frame()
        encode=lambda x:np.clip(x*32768,-32768,32767).astype('<i2').tobytes()
        self.assertEqual(pcm,encode(near[::3]*.5));self.assertEqual(raw,encode(near[::3]))
        self.assertEqual(ref,encode(far[::3]));self.assertEqual(len(pcm),640)
        self.assertEqual(io.stats['input_gain_db'],0);player.assert_not_called()

    def test_nine_db_boosts_only_processed_microphone_after_frontend(self):
        io,(_,pcm,raw,ref),meter,near,far,front,player=self.frame(9)
        encode=lambda x:np.clip(x*32768,-32768,32767).astype('<i2').tobytes()
        self.assertEqual(pcm,encode(near[::3]*.5*10**(9/20)))
        self.assertEqual(raw,encode(near[::3]));self.assertEqual(ref,encode(far[::3]))
        np.testing.assert_array_equal(front.process48.call_args.args[0],near)
        np.testing.assert_array_equal(front.process48.call_args.args[1],far)
        self.assertEqual(len(pcm),640);self.assertEqual(io.stats['uplink_clipped_samples'],0)
        self.assertAlmostEqual(meter['uplink_rms']/meter['pre_gain_rms'],10**(9/20),places=5)
        self.assertEqual(meter['input_gain_db'],9);self.assertEqual(meter['reference_coverage'],1.)
        player.assert_not_called()

    def test_loud_input_is_saturated_at_pcm_boundary_and_counted_without_wraparound(self):
        io,(_,pcm,raw,ref),meter,near,far,_,_=self.frame(9,amplitude=1.)
        actual=np.frombuffer(pcm,dtype='<i2');unclipped=near[::3]*.5*10**(9/20)
        self.assertTrue(np.all(actual[unclipped>1]==32767))
        self.assertTrue(np.all(actual[unclipped<-1]==-32768))
        self.assertGreater(io.stats['uplink_clipped_samples'],0)
        self.assertEqual(meter['uplink_clipped_samples'],io.stats['uplink_clipped_samples'])
        self.assertGreater(meter['uplink_peak_abs'],1.)

    def test_capture_gap_retains_exact_diagnostics_without_sending_a_broken_block(self):
        speech=json.loads((ROOT/'configs/perception/speech.json').read_text())
        for interval,broken in ((.02,False),(.034,False),(.036,True)):
            io=AudioIO(speech,'unused',Mock(),aec=False)
            io.stop.is_set=Mock(side_effect=[False,False,True])
            near=np.zeros(960,np.float32)
            for stamp in (100.,100.+interval):io.capture.put_nowait((stamp,near))
            front=Mock();front.process48.side_effect=lambda n,f:(n[::3],f[::3],n[::3])
            udp=Mock();udp.recvfrom.side_effect=BlockingIOError
            with patch('vivis.speech.realtime_audio.time.monotonic',return_value=100.1):io.process(udp,front)
            self.assertEqual(io.uplink.qsize(),1 if broken else 2)
            if broken:
                self.assertIn('interval=36.00ms',io.fault)
                self.assertAlmostEqual(io.stats['capture_clock_fault']['interval_ms'],36.)
            else:self.assertIsNone(io.fault);self.assertIsNone(io.stats['capture_clock_fault'])


if __name__=='__main__':unittest.main()
