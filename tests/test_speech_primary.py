"""The streaming primary remains available when optional review stalls/fails."""
import copy
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np

from vivis.paths import ROOT
from vivis.interaction.dialogue import EventGate
from vivis.interaction.intents import IntentRouter
from vivis.runtime.best_effort import BestEffortJob
from vivis.speech.listen import Decoder,recognizer


class PrimarySpeechTests(unittest.TestCase):
    def fixture(self, finalizer):
        cfg=json.loads((ROOT/'configs/perception/speech.json').read_text())
        cfg['control_keywords']={'enabled':False}
        engine=Mock();engine.is_ready.return_value=False;engine.get_result.return_value='停止'
        events=[];sdk=SimpleNamespace(OfflineRecognizer=SimpleNamespace(from_sense_voice=lambda **_:finalizer))
        with patch('vivis.speech.listen.recognizer',return_value=engine),patch('vivis.speech.segment.Segmenter'), \
             patch.dict('sys.modules',sherpa_onnx=sdk):
            decoder=Decoder(cfg,events.append)
        decoder.segmenter=Mock()
        decoder.segmenter.feed.return_value=dict(audio=np.zeros(320,np.float32),start=1.,end=1.5,speech_end=1.2,metrics={})
        return decoder,events

    def test_stalled_review_cannot_delay_primary_stop_or_shutdown_and_has_no_queue(self):
        entered=threading.Event();release=threading.Event();finalizer=Mock()
        def decode(_):entered.set();release.wait(2.)
        finalizer.decode_stream.side_effect=decode;finalizer.create_stream.return_value.result.text='播放白骨夫人'
        decoder,events=self.fixture(finalizer)
        try:
            decoder.feed(np.zeros(320,np.float32),1.);self.assertTrue(entered.wait(1.))
            start=time.monotonic()
            for _ in range(4):decoder.feed(np.zeros(320,np.float32),1.)
            decoder.close()
            self.assertLess(time.monotonic()-start,.2)
            self.assertEqual(len([e for e in events if e['kind']=='asr_final']),5)
            self.assertEqual(decoder.review_counts['skipped_busy'],4)
            self.assertEqual(finalizer.decode_stream.call_count,1)
            self.assertFalse(any(e['kind'] in ('asr_error','asr_overload') for e in events))
        finally:release.set();decoder.review.wait(2.)

    def test_late_review_cannot_become_another_committed_command(self):
        finalizer=Mock();finalizer.create_stream.return_value.result.text='播放白骨夫人'
        decoder,events=self.fixture(finalizer)
        try:
            decoder.feed(np.zeros(320,np.float32),1.);decoder.review.wait(2.)
            gate=EventGate(2.);accepted=[e for e in events if gate.accept(e,1.6)]
            self.assertEqual([e['text'] for e in accepted],['停止'])
            self.assertTrue(any(e['kind']=='asr_refinement' and e['text']=='播放白骨夫人' for e in events))
            self.assertLess(next(i for i,e in enumerate(events) if e['kind']=='asr_final'),
                            next(i for i,e in enumerate(events) if e['kind']=='asr_refinement'))
        finally:decoder.close()

    def test_review_failure_is_diagnostic_and_next_primary_still_arrives(self):
        finalizer=Mock();finalizer.decode_stream.side_effect=RuntimeError('optional review failed')
        decoder,events=self.fixture(finalizer)
        try:
            for _ in range(2):decoder.feed(np.zeros(320,np.float32),1.);decoder.review.wait(2.)
            self.assertEqual(len([e for e in events if e['kind']=='asr_final']),2)
            self.assertEqual(len([e for e in events if e['kind']=='asr_refinement_error']),2)
            self.assertFalse(any(e['kind']=='asr_error' for e in events))
        finally:decoder.close()

    def test_paraformer_uses_its_native_streaming_api_without_transducer_joiner(self):
        cfg=json.loads((ROOT/'configs/perception/speech.json').read_text());sdk=Mock()
        with patch.dict('sys.modules',sherpa_onnx=sdk):recognizer(cfg)
        sdk.OnlineRecognizer.from_paraformer.assert_called_once()
        self.assertNotIn('joiner',sdk.OnlineRecognizer.from_paraformer.call_args.kwargs)
        sdk.OnlineRecognizer.from_transducer.assert_not_called()
        cfg=copy.deepcopy(cfg);cfg['asr']['hotwords_score']=1.5
        with patch.dict('sys.modules',sherpa_onnx=sdk),self.assertRaisesRegex(ValueError,'hotword'):recognizer(cfg)

    def test_observed_natural_request_is_supported_and_negation_is_not_a_request(self):
        router=IntentRouter(json.loads((ROOT/'configs/interaction/programs.json').read_text()))
        self.assertEqual(router.parse('来一段白骨夫人吧',state='interaction'),dict(kind='select',program='baigufuren'))
        for text in ('不要来一段白骨夫人吧','他说来一段白骨夫人吧','来一段白骨夫人或者魏总剧本吧'):
            self.assertNotEqual(router.parse(text,state='interaction')['kind'],'select')

    def test_supervisor_ignores_review_flood_and_legacy_overload_but_keeps_stop(self):
        from vivis.interaction.dialogue import main
        actual_popen=subprocess.Popen
        cfg=json.loads((ROOT/'configs/interaction/p1.json').read_text())
        speech=json.loads((ROOT/cfg['speech']).read_text())
        cfg['programs']=str(ROOT/cfg['programs']);cfg['semantic']=str(ROOT/cfg['semantic']);cfg['conversation']=str(ROOT/cfg['conversation'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'p1.json';config.write_text(json.dumps(cfg))
            script=root/'asr.py';script.write_text('''import json,time
def emit(**event):print(json.dumps(event,ensure_ascii=False),flush=True)
emit(kind='ready')
emit(kind='asr_final',utterance_id=1,text='小帅',captured_end_monotonic=time.monotonic())
for i in range(200):emit(kind='asr_refinement',utterance_id=1,text='播放白骨夫人')
emit(kind='asr_overload',utterance_id=2)
emit(kind='asr_final',utterance_id=3,text='待机',captured_end_monotonic=time.monotonic())
time.sleep(3)
''')
            def launch(command,**kwargs):return actual_popen([sys.executable,str(script)],**kwargs)
            with patch('vivis.interaction.dialogue.ROOT',root), \
                 patch('vivis.interaction.dialogue.dependencies',return_value=(speech,[])), \
                 patch('vivis.interaction.dialogue.subprocess.Popen',side_effect=launch),redirect_stdout(io.StringIO()):
                code=main(['--config',str(config),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),
                           '--listen-only','--seconds','.5','--output',str(root/'run')])
            self.assertEqual(code,0)
            summary=json.loads((root/'run/summary.json').read_text())
            self.assertEqual(summary['desired_mode'],'standby');self.assertIsNone(summary['error'])
            self.assertEqual(summary['program_sessions'],0)
            self.assertIn('asr_overload',(root/'run/asr_events.jsonl').read_text())


if __name__=='__main__':unittest.main()
