"""Reply bursts, provisional interruptions and animation retention regressions."""
import asyncio
import base64
from collections import deque
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.paths import ROOT
from vivis.speech.mouth import SpeechMouth
from vivis.speech.realtime import VoiceSession,receive_event,configuration,connect_session
from vivis.speech.realtime_audio import VoiceBuffer


def packet(rid,seconds):
    return dict(type='response.audio.delta',response_id=rid,
        delta=base64.b64encode(np.full(round(seconds*24000),1000,dtype='<i2').tobytes()).decode())


class VoiceResilienceTests(unittest.TestCase):
    def test_validated_profile_sets_server_sensitivity_hold_and_aligned_bounded_horizon(self):
        from test_realtime_voice import arguments
        config,_,_,_,session=configuration(arguments())
        self.assertEqual(session['turn_detection'],dict(type='smart_turn_v2',threshold=.65))
        self.assertEqual(config['interruption_hold_ms'],180)
        self.assertEqual(config['max_playback_buffer_s'],120)
        class WS:
            async def send(self,_):pass
            async def recv(self):
                return json.dumps(dict(type='session.updated',session={**session,'turn_detection':dict(type='smart_turn_v2',threshold=.5)}))
        with self.assertRaisesRegex(RuntimeError,'confirm'):
            asyncio.run(connect_session(WS(),session,Mock()))

    def test_short_invalid_turn_never_ducks_audio_but_confirmed_stop_does_not_wait(self):
        player=VoiceBuffer();state=VoiceSession(player,Mock(),interruption_hold_ms=180)
        state.event(dict(type='response.created',response=dict(id='old')));state.event(packet('old',1))
        with patch('vivis.speech.realtime.time.monotonic',return_value=10):
            state.event(dict(type='input_audio_buffer.speech_started',item_id='short'))
        with patch('vivis.speech.realtime.time.monotonic',return_value=10.1):state.poll_playback()
        self.assertFalse(player.paused);self.assertTrue(np.any(player.render(960,10.1)))
        state.event(dict(type='input_audio_buffer.speech_stopped',item_id='short',reason='turn_invalid'))
        self.assertEqual(state.provisional_pauses,0);self.assertFalse(player.paused)
        with patch('vivis.speech.realtime.time.monotonic',return_value=11):
            state.event(dict(type='input_audio_buffer.speech_started',item_id='real'))
            state.event(dict(type='input_audio_buffer.committed',item_id='real'))
        self.assertIn('old',state.blocked);self.assertIsNone(state.pending_interruption)

    def test_sustained_candidate_pauses_after_hold_then_invalid_resumes_same_samples(self):
        player=VoiceBuffer();state=VoiceSession(player,Mock(),interruption_hold_ms=180)
        state.event(dict(type='response.created',response=dict(id='old')));state.event(packet('old',1))
        with patch('vivis.speech.realtime.time.monotonic',return_value=10):
            state.event(dict(type='input_audio_buffer.speech_started',item_id='candidate'))
        with patch('vivis.speech.realtime.time.monotonic',return_value=10.181):state.poll_playback()
        self.assertIn('old',player.paused);retained=player.count
        self.assertFalse(np.any(player.render(960,10.2)))  # Still-unheard audio does not begin on a pause.
        self.assertEqual(retained,player.count)
        state.event(dict(type='input_audio_buffer.speech_stopped',item_id='candidate',reason='turn_invalid'))
        self.assertTrue(np.any(player.render(960,10.3)));self.assertEqual(state.invalid_turn_resumes,1)

    def test_logged_33_second_burst_fits_without_audio_loss_or_global_error(self):
        player=VoiceBuffer(max_seconds=120);state=VoiceSession(player,Mock())
        state.event(dict(type='response.created',response=dict(id='burst')))
        # Actual incident had >=33.6seconds arriving in~3.6seconds; zero drain is stricter.
        data=packet('burst',.4)
        for _ in range(84):self.assertIsNone(state.event(data))
        state.event(dict(type='response.done',response=dict(id='burst',status='completed')))
        self.assertEqual(player.count,round(33.6*48000));self.assertEqual(state.playback_overflows,0)
        while player.count:player.render(4800,100)
        self.assertEqual(player.consumed['burst'],round(33.6*48000))

    def test_true_overflow_cancels_only_bad_reply_and_next_reply_can_play(self):
        player=VoiceBuffer(max_seconds=1);state=VoiceSession(player,Mock());sent=[]
        async def send(value):sent.append(json.loads(value))
        ws=SimpleNamespace(send=send)
        state.event(dict(type='response.created',response=dict(id='large')))
        state.event(packet('large',1))
        warning=asyncio.run(receive_event(ws,state,packet('large',.1)))
        self.assertEqual(warning['response_id'],'large');self.assertEqual(player.count,0)
        self.assertEqual(sent,[dict(type='response.cancel')]);self.assertEqual(state.playback_overflows,1)
        self.assertIsNone(state.event(packet('large',.1)))  # Late packets cannot revive cancelled audio.
        state.event(dict(type='response.created',response=dict(id='next')));state.event(packet('next',.1))
        self.assertTrue(np.any(player.render(960,100)));self.assertNotIn('next',state.blocked)

    def test_rejected_overflow_packet_never_reaches_animation_worker(self):
        player=VoiceBuffer(max_seconds=1);player.mouth=Mock()
        player.add('reply',np.zeros(24000,dtype='<i2').tobytes())
        with self.assertRaisesRegex(RuntimeError,'bounded buffer'):
            player.add('reply',np.zeros(2400,dtype='<i2').tobytes())
        self.assertEqual(player.mouth.add.call_count,1)

    def test_animation_horizon_retains_the_start_of_fast_35_second_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            process=Mock();process.poll.return_value=0
            with patch('vivis.speech.mouth.subprocess.Popen',return_value=process), \
                 patch('vivis.speech.mouth.threading.Thread'):
                mouth=SpeechMouth('not-executed','model',Path(tmp)/'mouth.json',['c']*52,buffer_seconds=120)
            mouth.add('long',np.zeros(240,dtype='<i2').tobytes())
            frames=mouth.responses['long']['frames']
            self.assertGreaterEqual(frames.maxlen,7200)
            for i in range(2100):frames.append((i*266,[float(i%2)]*52))
            mouth.played('long',0,960,10)
            self.assertEqual(frames[0][0],0);self.assertIsNotNone(mouth.weights(10.01))
            mouth.close()


if __name__=='__main__':unittest.main()
