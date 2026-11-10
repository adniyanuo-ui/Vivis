"""Short invalid turns must not destroy completed-but-unplayed voice audio."""
import base64
from unittest.mock import Mock
import unittest

import numpy as np

from vivis.speech.realtime import VoiceSession
from vivis.speech.realtime_audio import VoiceBuffer


class InterruptionTests(unittest.TestCase):
    def setUp(self):
        self.player=VoiceBuffer();self.log=Mock();self.state=VoiceSession(self.player,self.log)

    def response(self,rid='old',seconds=7.28,complete=True):
        self.state.event(dict(type='response.created',response=dict(id=rid)))
        pcm=np.full(round(seconds*24000),2000,dtype='<i2').tobytes()
        self.state.event(dict(type='response.audio.delta',response_id=rid,delta=base64.b64encode(pcm).decode()))
        if complete:self.state.event(dict(type='response.done',response=dict(id=rid,status='completed')))

    def start(self,item='short'):
        self.state.event(dict(type='input_audio_buffer.speech_started',item_id=item))

    def stop(self,item='short',invalid=False):
        self.state.event(dict(type='input_audio_buffer.speech_stopped',item_id=item,
                              **({'reason':'turn_invalid'} if invalid else {})))

    def fade_and_hold(self):
        self.player.render(1920,101.5)
        remaining=self.player.count
        self.assertFalse(np.any(self.player.render(4800,101.54)))
        self.assertEqual(remaining,self.player.count)
        return remaining

    def test_logged_hotpot_sequence_retains_5_point_74_seconds_after_invalid_ack(self):
        self.response();self.player.render(72000,100.)
        self.start();remaining=self.fade_and_hold()
        self.assertEqual(remaining,275520)  #7.28s received,1.5s played,40ms fade.
        self.assertNotIn('old',self.state.blocked)
        self.stop(invalid=True)
        self.assertEqual(self.state.invalid_turn_resumes,1)
        self.assertTrue(np.any(self.player.render(remaining,104.8)))
        self.assertEqual(self.player.count,0)
        self.assertEqual(self.state.cancel_operations,0)

    def test_real_interruption_pauses_immediately_and_discards_only_after_confirmation(self):
        self.response();self.player.render(960,100.);self.start();self.fade_and_hold()
        self.stop()
        self.assertIn('old',self.state.blocked)
        self.assertFalse(np.any(self.player.render(960,105.)))
        self.assertEqual(self.player.count,0)  #No second fade after already-paused audio.

    def test_ack_before_first_audio_does_not_start_unheard_speech(self):
        self.state.event(dict(type='response.created',response=dict(id='old')))
        self.start()
        self.player.add('old',np.full(2400,1000,dtype='<i2').tobytes())
        self.assertFalse(np.any(self.player.render(960,100.)))
        self.assertEqual(self.player.count,4800)
        self.stop(invalid=True)
        self.assertTrue(np.any(self.player.render(4800,101.)))

    def test_keyboard_stop_during_pause_cannot_be_undone_by_late_invalid(self):
        self.response();self.player.render(960,100.);self.start();self.fade_and_hold()
        self.state.interrupt(source='keyboard');self.stop(invalid=True)
        self.assertFalse(np.any(self.player.render(960,106.)))
        self.assertIn('old',self.state.blocked)
        self.assertEqual(self.state.invalid_turn_resumes,0)

    def test_new_answer_supersedes_old_paused_audio_without_needing_stop_event(self):
        self.response();self.player.render(960,100.);self.start();self.fade_and_hold()
        self.response('new',.1)
        self.stop(invalid=True)
        self.assertIn('old',self.state.blocked);self.assertNotIn('new',self.state.blocked)
        self.assertEqual({sid for sid,_ in self.player.blocks},{'new'})
        self.assertTrue(np.any(self.player.render(4800,106.)))
        self.assertEqual(self.state.invalid_turn_resumes,0)

    def test_stale_invalid_event_does_not_resume_a_newer_pending_turn(self):
        self.response();self.player.render(960,100.);self.start('one');self.fade_and_hold()
        self.start('two');self.stop('one',invalid=True)
        self.assertFalse(np.any(self.player.render(960,105.)))
        self.assertIn('old',self.player.paused)
        self.stop('two',invalid=True)
        self.assertTrue(np.any(self.player.render(960,106.)))

    def test_server_cancelled_response_stays_cancelled_after_late_invalid(self):
        self.response(complete=False);self.player.render(960,100.);self.start();self.fade_and_hold()
        self.state.event(dict(type='response.done',response=dict(id='old',status='cancelled')))
        self.stop(invalid=True)
        self.assertFalse(np.any(self.player.render(960,105.)))
        self.assertIn('old',self.state.blocked)

    def test_input_commit_without_stopped_event_confirms_pending_interruption(self):
        self.response();self.player.render(960,100.);self.start();self.fade_and_hold()
        self.state.event(dict(type='input_audio_buffer.committed',item_id='short'))
        self.assertIn('old',self.state.blocked)
        self.assertEqual(self.player.count,0)

    def test_cancel_partway_through_pause_fade_uses_only_the_remaining_fade(self):
        self.response();self.player.render(960,100.);self.start()
        self.player.render(960,100.02)
        self.state.interrupt('old',source='server_cancelled')
        self.assertEqual(self.player.count,960)
        self.assertTrue(np.any(self.player.render(960,100.04)))
        self.assertFalse(np.any(self.player.render(960,100.06)))

    def test_paused_audio_and_new_packets_still_share_the_original_buffer_bound(self):
        player=VoiceBuffer(max_seconds=1.)
        player.pause('old')
        self.assertTrue(player.add('old',np.full(24000,1000,dtype='<i2').tobytes()))
        self.assertFalse(np.any(player.render(960,100.)))
        with self.assertRaisesRegex(RuntimeError,'bounded buffer'):
            player.add('old',np.full(480,1000,dtype='<i2').tobytes())
        self.assertEqual(player.count,48000)


if __name__=='__main__':unittest.main()
