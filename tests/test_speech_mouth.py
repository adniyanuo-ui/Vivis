"""Speech-mouth timing, cancellation and composition without native/GPU/devices."""
from collections import deque
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.interaction.standby import plan
from vivis.speech.mouth import SpeechMouth,MouthReceiver
from vivis.speech.realtime_audio import VoiceBuffer


class SpeechMouthTests(unittest.TestCase):
    def waiting_mouth(self):
        m=SpeechMouth.__new__(SpeechMouth);m.lock=threading.Lock();m.error=None;m.warning=None
        m.prebuffer_timeouts=0;m.gated=0;m.responses={'a':dict(frames=deque([(1000,[.2]*52)]),done=False,created=0.)}
        return m

    def test_long_reply_with_four_second_cloud_gap_does_not_disable_mouth(self):
        m=self.waiting_mouth()
        with patch('vivis.speech.mouth.time.monotonic',return_value=20.):self.assertFalse(m.ready('a',3000))
        self.assertIsNone(m.error)
        m.responses['a']['frames'].append((1600,[.4]*52))
        with patch('vivis.speech.mouth.time.monotonic',return_value=24.3):self.assertTrue(m.ready('a',3000))
        self.assertEqual(m.prebuffer_timeouts,0);self.assertIsNone(m.warning)

    def test_new_animation_progress_resets_stall_timer(self):
        m=self.waiting_mouth()
        for now,sample in ((20.,1000),(27.,1100),(34.,1200)):
            m.responses['a']['frames'].append((sample,[.2]*52))
            with patch('vivis.speech.mouth.time.monotonic',return_value=now):self.assertFalse(m.ready('a',12000))
        m.responses['a']['frames'].append((5000,[.5]*52))
        with patch('vivis.speech.mouth.time.monotonic',return_value=40.):self.assertTrue(m.ready('a',12000))
        self.assertIsNone(m.error);self.assertEqual(m.prebuffer_timeouts,0)

    def test_actual_stall_affects_one_reply_and_next_reply_can_animate(self):
        m=self.waiting_mouth()
        with patch('vivis.speech.mouth.time.monotonic',return_value=20.):self.assertFalse(m.ready('a',3000))
        with patch('vivis.speech.mouth.time.monotonic',return_value=28.1):self.assertTrue(m.ready('a',3000))
        self.assertTrue(m.responses['a']['prebuffer_error']);self.assertEqual(m.prebuffer_timeouts,1)
        self.assertIsNone(m.error)
        m.responses['b']=dict(frames=deque([(0,[0.]*52),(16000,[1.]*52)]),done=True,created=29.)
        with patch('vivis.speech.mouth.time.monotonic',return_value=29.):self.assertTrue(m.ready('b',3000))
        m.heads=deque(maxlen=128);m.played('b',0,960,100.)
        self.assertGreater(m.weights(100.01)[0],0.)

    def test_dac_future_spans_do_not_replace_current_playhead_or_advance_during_pause(self):
        m=SpeechMouth.__new__(SpeechMouth);m.lock=threading.Lock();m.heads=deque(maxlen=128);m.error=None
        m.responses={'a':dict(frames=deque([(0,[0.]*52),(16000,[1.]*52)]))}
        m.played('a',0,960,100.)
        m.played('a',960,1920,100.02)
        self.assertIsNone(m.weights(99.99))
        self.assertAlmostEqual(m.weights(100.01)[0],.01)
        self.assertAlmostEqual(m.weights(100.03)[0],.03)
        self.assertIsNone(m.weights(100.2))
        m.deactivate();self.assertIsNone(m.weights(100.03))

    def test_playback_prebuffer_has_no_sample_loss_and_cancel_cannot_revive_mouth(self):
        class Mouth:
            ready_now=False
            def __init__(self):self.head=[];self.inputs=[]
            def add(self,rid,pcm):self.inputs.append((rid,pcm))
            def finish(self,rid):pass
            def ready(self,*_):return self.ready_now
            def played(self,*args):self.head.append(args)
            def deactivate(self):self.head.clear()
        m=Mouth();p=VoiceBuffer();p.mouth=m;pcm=(np.ones(2400)*5000).astype('<i2').tobytes()
        p.add('a',pcm);p.finish('a');before=p.count
        self.assertFalse(np.any(p.render(960,100.)));self.assertEqual(p.count,before)
        m.ready_now=True;self.assertTrue(np.any(p.render(960,100.02)))
        self.assertEqual(m.head[-1][:3],('a',0,960))
        p.pause('a');p.render(1920,100.04);held=p.count
        p.render(960,100.08);self.assertEqual(p.count,held)
        p.resume(['a']);p.render(960,100.1)
        self.assertEqual(m.head[-1][1],2880)
        p.cancel('a');self.assertEqual(m.head,[])
        p.render(960,100.12);self.assertEqual(m.head,[])
        self.assertFalse(p.add('a',pcm))

    def test_receiver_rejects_stale_invalid_foreign_channels_and_model_failure(self):
        order=['c'+str(i) for i in range(52)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mouth.json';r=MouthReceiver(path,order)
            good=dict(schema_version='vivis-speech-mouth-1.0',timestamp_monotonic=100.,order=order,weights=[.5]*52)
            path.write_text(json.dumps(good));self.assertEqual(r.sample(100.1),[.5]*52)
            for update in (dict(timestamp_monotonic=99.),dict(timestamp_monotonic=101.),
                           dict(order=order[::-1]),dict(weights=[2.]*52),dict(weights=None),dict(error='model failed')):
                path.write_text(json.dumps({**good,**update}));self.assertIsNone(r.sample(100.1))

    def test_idle_uses_only_speech_mouth_channels_and_retains_blink_and_gaze_ownership(self):
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        settings=json.loads((ROOT/'configs/interaction/interaction.json').read_text())
        _,ids,_,mapper,source,_=plan(robot,settings,1.)
        class Mouth:
            active=True
            def sample(self,now):return [1.]*52 if self.active else None
        source.mouth=Mouth();source.sample(0.,100.)
        speech=np.array([float(n.startswith(('Mouth','Jaw','Tongue'))) for n in mapper.order[:52]])
        np.testing.assert_array_equal(mapper.last_fused,speech)
        self.assertTrue(mapper.speech_owned_ids)
        self.assertFalse(set(mapper.speech_owned_ids)&{7,8,22,23,32,33,34})
        self.assertFalse({9,24}&set(ids))
        source.mouth.active=False;source.sample(.01,100.01)
        np.testing.assert_array_equal(mapper.last_fused,np.zeros(52));self.assertEqual(mapper.speech_owned_ids,[])


if __name__=='__main__':unittest.main()
