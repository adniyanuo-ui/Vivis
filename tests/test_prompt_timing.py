"""Offline prompt segmentation and bounded fault notification, without audio I/O."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from vivis.speech.prompts import synthesize_prompt
from vivis.interaction.dialogue import PromptPlayer


class PromptTimingTests(unittest.TestCase):
    def test_clause_and_sentence_pauses_are_real_silence_between_speech_segments(self):
        engine=Mock();engine.generate.return_value=SimpleNamespace(samples=np.ones(500)*.2,sample_rate=1000)
        timing=dict(clause_pause_ms=180,sentence_pause_ms=420,edge_fade_ms=5)
        samples,rate,report=synthesize_prompt(engine,'你好，欢迎。请选一个！',58,1.,timing)
        self.assertEqual([c.args[0] for c in engine.generate.call_args_list],['你好，','欢迎。','请选一个！'])
        self.assertEqual(rate,1000);self.assertEqual(len(samples),2100)
        self.assertEqual([p['samples'] for p in report['pauses']],[180,420])
        for pause in report['pauses']:
            np.testing.assert_array_equal(samples[pause['start_sample']:pause['start_sample']+pause['samples']],0.)
        self.assertEqual(samples[0],0.);self.assertEqual(samples[-1],0.)

    def test_invalid_timing_empty_or_silent_synthesis_fails_preparation(self):
        engine=Mock();engine.generate.return_value=SimpleNamespace(samples=np.zeros(20),sample_rate=1000)
        timing=dict(clause_pause_ms=180,sentence_pause_ms=420,edge_fade_ms=5)
        with self.assertRaisesRegex(ValueError,'Silent'):synthesize_prompt(engine,'你好。',58,1.,timing)
        with self.assertRaisesRegex(ValueError,'Empty'):synthesize_prompt(engine,'',58,1.,timing)
        with self.assertRaisesRegex(ValueError,'timing'):
            synthesize_prompt(engine,'你好。',58,1.,{**timing,'sentence_pause_ms':float('nan')})

    def test_fault_notification_wait_is_cancellable_and_has_a_deadline(self):
        player=PromptPlayer.__new__(PromptPlayer);player.clock=object();player.stop=Mock()
        player.finish(lambda:True);player.stop.assert_called_once()
        player.poll=Mock();player.stop.reset_mock()
        with patch('vivis.interaction.dialogue.time.monotonic',side_effect=[0.,20.]):
            with self.assertRaisesRegex(RuntimeError,'timed out'):player.finish(timeout_s=12.)
        player.poll.assert_called_once()
