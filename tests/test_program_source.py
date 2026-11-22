"""Final-source gating: a mistaken cloud stop cannot execute before its negation arrives."""
import json
import unittest
from unittest.mock import patch

from vivis.interaction.program_source import veto
import test_program_followup as fixtures


class ProgramSourceTextTests(unittest.TestCase):
    def test_clear_negations_and_quotes_veto_without_proposing_actions(self):
        for text in ('不要停止。','请不要停下来，继续唱。','小帅，别结束。','不用退出。','歌词里写着停止。',
                '不要停止，而是继续播放。','不要停止，还是继续唱吧。'):
            self.assertIsNotNone(veto(text,'stop'),text)
        for text in ('停止。','不要继续唱了，停止。','不要停止，不对，停下来。','换一首歌曲。'):
            self.assertIsNone(veto(text,'stop'),text)
        self.assertIsNotNone(veto('不要播放歌曲。','play'))
        self.assertIsNotNone(veto('请别休息。','standby'))


class ProgramSourceTests(unittest.IsolatedAsyncioTestCase):
    setUp=fixtures.ProgramFollowupTests.setUp
    turn=fixtures.ProgramFollowupTests.turn
    response=fixtures.ProgramFollowupTests.response
    async def test_stop_before_final_waits_then_negative_produces_no_action_or_audio(self):
        s=self.service;self.turn('negative','wrong-stop')
        await s.tool(self.ws,dict(name='robot_program',call_id='bad',response_id='wrong-stop',arguments='{"action":"stop"}'))
        self.assertFalse(self.events);self.assertFalse(s.control_pending);self.assertIsNone(s.farewell)
        self.assertIn('bad',s.program_waiters)
        s.question_onset['final_text']='不要停止。'
        await s.resume_program_calls(self.ws)
        self.assertFalse(self.events);self.assertIsNone(s.farewell);self.assertFalse(s.program_waiters)
        result=json.loads(json.loads(self.ws.send.await_args.args[0])['item']['output'])
        self.assertFalse(result['accepted']);self.assertEqual(result['status'],'negated_control')

    async def test_positive_final_releases_original_stop_exactly_once(self):
        s=self.service;self.turn('positive','stop')
        event=dict(name='robot_program',call_id='good',response_id='stop',arguments='{"action":"stop"}')
        await s.tool(self.ws,event);self.assertFalse(self.events)
        s.question_onset['final_text']='好了，停止吧。'
        await s.resume_program_calls(self.ws);await s.resume_program_calls(self.ws);await s.tool(self.ws,event)
        self.assertEqual(len(self.events),1);self.assertEqual(self.events[0]['arguments']['action'],'stop')

    async def test_new_turn_discards_unsubmitted_old_stop(self):
        s=self.service;self.turn('old','stop')
        await s.tool(self.ws,dict(name='robot_program',call_id='old-call',response_id='stop',arguments='{"action":"stop"}'))
        self.turn('new','latest');s.question_onset['final_text']='继续唱。'
        await s.resume_program_calls(self.ws)
        self.assertFalse(self.events);self.assertFalse(s.program_waiters)

    async def test_missing_final_expires_without_executing_and_releases_speech_gate(self):
        s=self.service;self.turn('old','stop')
        await s.tool(self.ws,dict(name='robot_program',call_id='expired',response_id='stop',arguments='{"action":"stop"}'))
        deadline=s.program_waiters['expired']['deadline']
        with patch('vivis.speech.cloud_managed.time.monotonic',return_value=deadline+1):
            await s.resume_program_calls(self.ws)
        self.assertFalse(self.events);self.assertFalse(s.program_waiters);self.assertIsNone(s.farewell)

    async def test_only_one_program_proposal_waits_for_a_single_source(self):
        s=self.service;self.turn('one','reply')
        await s.tool(self.ws,dict(name='robot_program',call_id='first',response_id='reply',arguments='{"action":"stop"}'))
        await s.tool(self.ws,dict(name='robot_program',call_id='second',response_id='reply',arguments='{"action":"standby"}'))
        self.assertEqual(list(s.program_waiters),['first']);self.assertFalse(self.events)
