"""Device-loss reporting and optional TTS isolation; no real audio or models."""
from contextlib import redirect_stdout
import io
import json
import hashlib
import wave
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np

from vivis.paths import ROOT
from vivis.speech.prompts import prepare
from vivis.speech.qwen import QwenEngine


class SpeechModelSwitchTests(unittest.TestCase):
    def test_capture_watchdog_is_reported_as_structured_error_before_exception(self):
        from vivis.speech.listen import main
        with patch('vivis.speech.listen.live',side_effect=RuntimeError('Microphone watchdog: no fresh capture frames')), \
                redirect_stdout(io.StringIO()) as out:
            with self.assertRaisesRegex(RuntimeError,'Microphone watchdog'):
                main(['--config',str(ROOT/'configs/perception/speech.json')])
        event=json.loads(out.getvalue());self.assertEqual(event['kind'],'asr_error')
        self.assertEqual(event['source'],'microphone');self.assertEqual(event['device'],'DOV:')

    def test_qwen_preparation_delegates_to_isolated_python_without_importing_heavy_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            config=dict(tts=dict(engine='qwen3',python='outputs/tts_qwen/venv/bin/python',directory='unused'),
                        prompts=temporary)
            with patch('vivis.speech.prompts.subprocess.run') as run, \
                    patch('vivis.speech.qwen.QwenEngine',side_effect=AssertionError('Loaded Qwen in ASR environment')):
                prepare(config)
            command=run.call_args.args[0]
            self.assertEqual(command[0],str(ROOT/'outputs/tts_qwen/venv/bin/python'))
            job=Path(command[-1]);self.assertEqual(json.loads(job.read_text()),config)
            self.assertEqual(run.call_args.kwargs['env']['PYTHONNOUSERSITE'],'1')

    def test_qwen_named_chinese_voice_uses_native_rate_and_local_cpu_model(self):
        model=Mock();model.get_supported_speakers.return_value=['dylan','uncle_fu']
        model.generate_custom_voice.return_value=([np.ones(2400,dtype=np.float32)*.1],24000)
        factory=Mock();factory.from_pretrained.return_value=model
        torch=SimpleNamespace(float32='float32',set_num_threads=Mock())
        config=dict(tts=dict(directory='outputs/tts_qwen/model',speaker_name='Dylan',language='Chinese',threads=6))
        with patch.dict('sys.modules',{'torch':torch,'qwen_tts':SimpleNamespace(Qwen3TTSModel=factory)}):
            engine=QwenEngine(config);wave=engine.generate('你好。')
        self.assertEqual(factory.from_pretrained.call_args.kwargs['device_map'],'cpu')
        self.assertTrue(factory.from_pretrained.call_args.kwargs['local_files_only'])
        self.assertEqual(wave.sample_rate,24000)
        self.assertEqual(model.generate_custom_voice.call_args.kwargs['speaker'],'dylan')
        self.assertEqual(model.generate_custom_voice.call_args.kwargs['language'],'Chinese')
        with self.assertRaisesRegex(ValueError,'native speed'):engine.generate('你好。',speed=1.5)
        with self.assertRaisesRegex(ValueError,'Unknown Qwen voice'):engine.set_voice('NotAVoice')

    def test_capability_variant_is_supported_without_using_wrong_wake_words(self):
        from vivis.interaction.intents import IntentRouter
        catalog=json.loads((ROOT/'configs/interaction/programs.json').read_text());r=IntentRouter(catalog)
        self.assertEqual(r.parse('你会点啥',state='interaction'),dict(kind='prompt',name='menu'))
        for wrong in ('你看','表帅'):
            self.assertEqual(r.parse(wrong,state='standby')['kind'],'ignore')

    def test_standard_chinese_selects_mandarin_then_restores_the_provider_dialect_map(self):
        model=Mock();model.get_supported_speakers.return_value=['dylan']
        original={'dylan':'beijing_dialect'}
        talker=SimpleNamespace(spk_is_dialect=original)
        model.model.config.talker_config=talker
        def generate(**kwargs):
            self.assertIs(talker.spk_is_dialect['dylan'],False)
            self.assertEqual(kwargs['language'],'Chinese')
            return [np.ones(100,dtype=np.float32)],24000
        model.generate_custom_voice.side_effect=generate
        factory=Mock();factory.from_pretrained.return_value=model
        torch=SimpleNamespace(float32='float32',set_num_threads=Mock())
        c=dict(tts=dict(directory='unused',speaker_name='Dylan',force_standard_chinese=True))
        with patch.dict('sys.modules',{'torch':torch,'qwen_tts':SimpleNamespace(Qwen3TTSModel=factory)}):
            engine=QwenEngine(c);engine.generate('你好。')
        self.assertIs(talker.spk_is_dialect,original)
        model.generate_custom_voice.side_effect=RuntimeError('generation failed')
        with self.assertRaises(RuntimeError):engine.generate('你好。')
        self.assertIs(talker.spk_is_dialect,original)

    def test_runtime_cache_requires_every_spoken_prompt_not_optional_asr_diagnostics(self):
        from vivis.speech.prompts import cache_matches,PROCESSING_VERSION,RUNTIME_PROMPTS,TEXTS
        c=json.loads((ROOT/'configs/perception/speech.json').read_text())
        metadata=dict(processing_version=PROCESSING_VERSION,tts=c['tts'],timing=c.get('prompt_timing'),
                      level={'config':c['prompt_level']},texts={name:c.get('prompt_text_overrides',{}).get(name,TEXTS[name]) for name in RUNTIME_PROMPTS})
        self.assertTrue(cache_matches(c,metadata))
        del metadata['texts']['fault'];self.assertFalse(cache_matches(c,metadata))

    def test_gpu_prepared_voice_cache_is_portable_but_cannot_change_mandarin_selection(self):
        from vivis.speech.prompts import cache_matches,PROCESSING_VERSION,RUNTIME_PROMPTS,TEXTS
        t=dict(engine='qwen3',speaker_name='Dylan',force_standard_chinese=True,device='cpu',threads=6)
        c=dict(tts=t,prompt_level={})
        metadata=dict(processing_version=PROCESSING_VERSION,tts={**t,'device':'cuda:0','threads':2},timing=None,
                      level={'config':{}},texts={n:TEXTS[n] for n in RUNTIME_PROMPTS})
        self.assertTrue(cache_matches(c,metadata))
        metadata['tts']['force_standard_chinese']=False;self.assertFalse(cache_matches(c,metadata))

    def test_cached_wav_tampering_is_detected_without_opening_models_or_devices(self):
        from vivis.speech.prompts import prompt_cache_ready,PROCESSING_VERSION,RUNTIME_PROMPTS,TEXTS
        with tempfile.TemporaryDirectory() as temp:
            c=dict(tts=dict(engine='qwen3',speaker_name='Dylan'),prompts=temp,prompt_level={})
            metadata=dict(processing_version=PROCESSING_VERSION,tts=c['tts'],timing=None,level={'config':{}},
                texts={n:TEXTS[n] for n in RUNTIME_PROMPTS},prompt_sha256={})
            for n in RUNTIME_PROMPTS:
                p=Path(temp)/(n+'.wav')
                with wave.open(str(p),'wb') as out:
                    out.setnchannels(1);out.setsampwidth(2);out.setframerate(24000);out.writeframes(b'\0\0'*100)
                metadata['prompt_sha256'][n]=hashlib.sha256(p.read_bytes()).hexdigest()
            (Path(temp)/'metadata.json').write_text(json.dumps(metadata))
            self.assertTrue(prompt_cache_ready(c))
            (Path(temp)/'welcome.wav').write_bytes(b'corrupt')
            self.assertFalse(prompt_cache_ready(c))


if __name__=='__main__':unittest.main()
