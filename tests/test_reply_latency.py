"""Debounce observed face loss and measure first audible reply without replays."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vivis.paths import ROOT
from vivis.interaction.cloud_control import SpeechPresence,PresencePublisher
from vivis.speech.realtime import VoiceSession
from vivis.speech.realtime_audio import VoiceBuffer
from vivis.speech.cloud_managed import CloudManagedVoice
from types import SimpleNamespace


class ReplyTimingTests(unittest.TestCase):
    def test_brief_face_gap_keeps_voice_permission_but_departure_still_mutes(self):
        c=json.loads((ROOT/'configs/perception/speech_presence.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            presence=SpeechPresence(c,tmp);publisher=PresencePublisher(c,tmp)
            publisher.publish(1,10,10);self.assertTrue(presence.allowed(10))
            publisher.publish(0,10.2,10.2)
            self.assertTrue(presence.allowed(11.8))
            publisher.publish(1,11.9,11.9);self.assertTrue(presence.allowed(11.9))
            publisher.publish(0,12.2,12.2)
            self.assertTrue(presence.allowed(26.8));self.assertFalse(presence.allowed(27.))

    def test_departure_cleanup_cannot_cancel_voice_before_fifteen_second_grace(self):
        c=json.loads((ROOT/'configs/perception/speech_presence.json').read_text())
        reception=json.loads((ROOT/'configs/interaction/reception.json').read_text())
        self.assertEqual(c['face_loss_grace_s'],15.)
        self.assertEqual(reception['return_to_standby_after_s'],60)
        with tempfile.TemporaryDirectory() as tmp:
            presence=SpeechPresence(c,tmp);publisher=PresencePublisher(c,tmp)
            publisher.publish(1,10,10);presence.allowed(10)
            publisher.publish(0,10.2,10.2)
            service=CloudManagedVoice.__new__(CloudManagedVoice)
            service.presence=presence;service.reception=SimpleNamespace(c=reception)
            service.visit_seen=True;service.absence_since=10.2
            self.assertFalse(service.departure_requires_reset(14.))
            self.assertFalse(service.departure_requires_reset(24.9))
            self.assertTrue(service.departure_requires_reset(25.1))
            presence.wake(25.1)
            self.assertFalse(service.departure_requires_reset(30.))

    def session(self):
        records=[]
        class Log:
            def write(self,kind,**value):records.append(dict(kind=kind,**value))
        player=VoiceBuffer();return VoiceSession(player,Log()),player,records

    def test_latency_uses_first_dac_for_one_user_turn_and_ignores_later_continuations(self):
        session,player,records=self.session()
        with patch('vivis.speech.realtime.time.monotonic',return_value=100.):
            session.event(dict(type='input_audio_buffer.speech_stopped',item_id='user-a'))
        with patch('vivis.speech.realtime.time.monotonic',return_value=100.1):
            session.event(dict(type='response.created',response=dict(id='reply-a')))
        session.responses['reply-a']['first_audio_received']=100.2
        player.notices.put(dict(kind='playback_started',response_id='reply-a',dac_monotonic=100.5))
        session.poll_playback()
        first=[r for r in records if r['kind']=='reply_latency'];self.assertEqual(len(first),1)
        self.assertAlmostEqual(first[0]['confirmed_end_to_first_dac_s'],.5)
        session.event(dict(type='response.created',response=dict(id='reply-b')))
        player.notices.put(dict(kind='playback_started',response_id='reply-b',dac_monotonic=104.))
        session.poll_playback();self.assertEqual(sum(r['kind']=='reply_latency' for r in records),1)

    def test_proactive_greeting_is_not_mislabeled_as_long_user_response_delay(self):
        session,player,records=self.session()
        session.event(dict(type='input_audio_buffer.speech_stopped',item_id='old-user'))
        session.mark_proactive();session.event(dict(type='response.created',response=dict(id='welcome')))
        player.notices.put(dict(kind='playback_started',response_id='welcome',dac_monotonic=100.))
        session.poll_playback();self.assertFalse(any(r['kind']=='reply_latency' for r in records))
