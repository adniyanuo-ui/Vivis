"""Trusted TTS is checked by DAC consumption, never by model reply wording."""
import asyncio
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.speech.message_voice import MessageSpeaker
from vivis.speech.realtime_audio import VoiceBuffer
from vivis.speech.realtime import VoiceSession
from vivis.interaction.message_lane import StaleMessageTurn


class MessageVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def wait_for(self,predicate):
        end=time.monotonic()+2
        while not predicate():
            if time.monotonic()>end:raise AssertionError('voice test timeout')
            await asyncio.sleep(.005)

    def speaker(self,seconds=.05):
        player=VoiceBuffer(max_seconds=120);state=VoiceSession(player,Mock());spoken=[]
        engine=SimpleNamespace(generate=Mock(return_value=SimpleNamespace(samples=np.ones(round(24000*seconds))*.01,sample_rate=24000)))
        speaker=MessageSpeaker(engine,player,state,lambda _:True,spoken.append,lambda *a,**k:None)
        return speaker,player,state,spoken

    async def test_full_audio_consumption_and_dac_time_are_required(self):
        speaker,player,state,spoken=self.speaker()
        task=asyncio.create_task(speaker.say('测试留言。',None))
        await self.wait_for(lambda:player.count>0)
        self.assertFalse(task.done())
        player.render(player.count,time.monotonic()+.05)
        await asyncio.sleep(.02);self.assertFalse(task.done())
        self.assertTrue(await asyncio.wait_for(task,.5));self.assertEqual(spoken,['测试留言。'])

    async def test_interrupted_audio_cannot_report_complete(self):
        speaker,player,state,_=self.speaker(1)
        task=asyncio.create_task(speaker.say('尚未播完的留言。',None))
        await self.wait_for(lambda:player.count>0)
        state.interrupt(source='onsite-stop')
        with self.assertRaises(StaleMessageTurn):await asyncio.wait_for(task,.5)

    async def test_readonly_list_longer_than_short_receipt_limit_uses_same_audio_route(self):
        speaker,player,state,_=self.speaker(16)
        task=asyncio.create_task(speaker.say('这是一段超过十五秒的查询列表。',None))
        await self.wait_for(lambda:player.count>=16*48000)
        player.render(player.count,time.monotonic()-20)
        self.assertTrue(await asyncio.wait_for(task,.5))

    async def test_stalled_output_ends_only_current_speech(self):
        from vivis.speech.runtime_policy import policy,PlaybackStalled
        speaker,player,state,_=self.speaker(1)
        with patch('vivis.speech.message_voice.policy',return_value={**policy(),'playback_stall_s':.06}):
            with self.assertRaises(PlaybackStalled):await asyncio.wait_for(speaker.say('停滞测试',None),1)
        self.assertTrue(state.blocked)
        self.assertTrue(player.add('next-response',b'\0'*4800))

    async def test_progressing_speech_can_exceed_stall_interval(self):
        from vivis.speech.runtime_policy import policy
        speaker,player,state,_=self.speaker(.2)
        with patch('vivis.speech.message_voice.policy',return_value={**policy(),'playback_stall_s':.08}):
            task=asyncio.create_task(speaker.say('持续有进展',None))
            await self.wait_for(lambda:player.count>0)
            started=time.monotonic()
            while not task.done():
                player.render(960,time.monotonic()-.02)
                await asyncio.sleep(.02)
            self.assertTrue(await task);self.assertGreater(time.monotonic()-started,.08)

    async def test_tts_generation_timeout_does_not_mark_playback_complete(self):
        from vivis.speech.runtime_policy import policy
        speaker,player,state,_=self.speaker()
        def slow(*_):time.sleep(.1);return None
        speaker.engine.generate=slow
        with patch('vivis.speech.message_voice.policy',return_value={**policy(),'message_tts_operation_s':.02}):
            with self.assertRaises(TimeoutError):await speaker.say('生成超时',None)
        self.assertTrue(state.blocked);self.assertEqual(player.count,0)

    async def test_long_tts_with_mouth_cost_does_not_starve_microphone_sender(self):
        speaker,player,state,_=self.speaker(16)
        original=player.add
        def add_with_mouth_cost(rid,pcm):
            time.sleep(.004)  # Bounded per-packet A2F/filter work, no real model.
            return original(rid,pcm)
        player.add=add_with_mouth_cost
        gaps=[];last=time.monotonic();finished=False
        async def capture_sender():
            nonlocal last
            while not finished:
                now=time.monotonic();gaps.append(now-last);last=now
                if player.count:player.render(player.count,time.monotonic()-20)
                await asyncio.sleep(.01)
        sender=asyncio.create_task(capture_sender())
        try:await asyncio.wait_for(speaker.say('带口型处理开销的长查询。',None),3.)
        finally:
            finished=True;await sender
        # The production microphone sender rejects frames older than500ms.
        # Preserve that guard; no TTS batch should monopolize the loop near it.
        self.assertLess(max(gaps),.2,f'event loop blocked for {max(gaps):.3f}s')

    async def test_cancelled_synthesis_does_not_queue_another_background_http_request(self):
        import threading
        from vivis.speech.message_voice import BoundedTTS
        entered=threading.Event();release=threading.Event();calls=[]
        def generate(text):
            calls.append(text);entered.set();release.wait(1);return 'audio'
        tts=BoundedTTS(SimpleNamespace(generate=generate))
        task=asyncio.create_task(tts.generate('first'))
        try:
            await self.wait_for(entered.is_set);task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
            with self.assertRaisesRegex(RuntimeError,'still finishing'):await tts.generate('second')
            self.assertEqual(calls,['first'])
        finally:release.set()
        await self.wait_for(lambda:tts.flight.done())
        self.assertEqual(await tts.generate('third'),'audio')
