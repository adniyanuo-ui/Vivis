"""Trusted TTS is checked by DAC consumption, never by model reply wording."""
import asyncio
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock

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

