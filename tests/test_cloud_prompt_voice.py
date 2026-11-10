"""Explicit cloud cache preparation, credentials and atomic activation; no API calls."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import wave

import numpy as np

from vivis.paths import ROOT
from vivis.speech.cloud_tts import CloudTTS,main,decode_wav
from scipy.io import wavfile
from vivis.speech.prompts import RUNTIME_PROMPTS,prompt_cache_ready


class CloudPromptVoiceTests(unittest.TestCase):
    def test_text_update_prepares_only_changed_prompts_and_retains_other_cached_wavs(self):
        from vivis.speech.prompts import prepare
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            config=dict(tts=self.settings(),prompts=tmp)
            engine=SimpleNamespace(num_speakers=1,execution_info={'request_ids':[]},
                generate=lambda *a,**k:SimpleNamespace(samples=np.ones(2400)*.2,sample_rate=24000))
            with patch('vivis.speech.cloud_tts.CloudTTS',return_value=engine):prepare(config)
            config['prompt_text_overrides']={'menu':'我会唱歌、演剧本、学表情。','identity':'我是小帅，会唱歌、演剧本、学表情。'}
            with patch('vivis.speech.cloud_tts.CloudTTS',return_value=engine), \
                 patch.object(engine,'generate',wraps=engine.generate) as generate:prepare(config)
            self.assertEqual(generate.call_count,2);self.assertTrue(prompt_cache_ready(config))
    def test_pcm32_float_and_stereo_cloud_wav_are_converted_without_changing_rate(self):
        for dtype in (np.int16,np.int32,np.float32,np.float64,np.uint8):
            for channels in (1,2):
                values=np.sin(np.arange(2400)*.1)*.2
                if np.dtype(dtype).kind=='i':values=(values*(2**(np.dtype(dtype).itemsize*8-1))).astype(dtype)
                elif dtype==np.uint8:values=(values*128+128).astype(dtype)
                else:values=values.astype(dtype)
                if channels==2:values=np.repeat(values[:,None],2,axis=1)
                data=io.BytesIO();wavfile.write(data,24000,values)
                samples,rate,source=decode_wav(data.getvalue())
                self.assertEqual(samples.shape,(2400,));self.assertEqual(rate,24000)
                self.assertLessEqual(np.max(np.abs(samples)),.21);self.assertEqual(source['channels'],channels)
        for data in (b'invalid',):
            with self.assertRaises(ValueError):decode_wav(data)

    def test_pcm24_and_unknown_streaming_header_use_actual_complete_samples(self):
        values=(np.sin(np.arange(2400)*.1)*(2**21)).astype('<i4')
        raw=io.BytesIO()
        with wave.open(raw,'wb') as w:
            w.setparams((1,3,24000,0,'NONE','not compressed'))
            w.writeframes(values.view(np.uint8).reshape(-1,4)[:,:3].tobytes())
        data=raw.getvalue();samples,_,_=decode_wav(data)
        np.testing.assert_allclose(samples,values/(2**23))
        stream=bytearray(data);stream[4:8]=(0xFFFFFFFF).to_bytes(4,'little')
        samples,_,source=decode_wav(bytes(stream))
        self.assertEqual(len(samples),2400);self.assertIn('header_warnings',source)
        with self.assertRaises(ValueError):decode_wav(data[:-30])

    def test_exact_reported_99885_byte_stream_header_does_not_require_two_gigabytes(self):
        raw=io.BytesIO()
        with wave.open(raw,'wb') as w:
            w.setparams((1,2,24000,0,'NONE','not compressed'))
            w.writeframes((np.sin(np.arange(49920)*.1)*5000).astype('<i2').tobytes())
        data=bytearray(raw.getvalue());data[4:8]=(2147483591-8).to_bytes(4,'little');data+=b'\0'
        samples,rate,source=decode_wav(bytes(data))
        self.assertEqual(len(data),99885);self.assertEqual(len(samples),49920)
        self.assertEqual(rate,24000);self.assertTrue(source['header_warnings'])

    def test_decode_failure_keeps_original_wav_and_retry_does_not_resynthesize(self):
        raw=io.BytesIO()
        with wave.open(raw,'wb') as w:
            w.setparams((1,2,24000,0,'NONE','not compressed'))
            w.writeframes(np.full(2400,5000,dtype='<i2').tobytes())
        result=dict(request_id='test-request',output=dict(audio=dict(url='https://bucket.aliyuncs.com/a.wav?Signature=private')))
        with tempfile.TemporaryDirectory() as tmp:
            engine=CloudTTS(self.settings(),secret='unit-secret',cache_directory=tmp)
            with patch.object(engine,'request',side_effect=[json.dumps(result).encode(),raw.getvalue()]), \
                 patch('vivis.speech.cloud_tts.decode_wav',side_effect=ValueError('simulated format error')), \
                 self.assertRaisesRegex(ValueError,'format error'):
                engine.generate('在呢，你说！')
            self.assertEqual(len(list(Path(tmp).glob('*.wav'))),1)
            with patch.object(engine,'request',side_effect=AssertionError('Repeated paid synthesis')):
                self.assertEqual(engine.generate('在呢，你说！').sample_rate,24000)
            for p in Path(tmp).glob('*.json'):
                self.assertNotIn('unit-secret',p.read_text());self.assertNotIn('Signature',p.read_text())

    def settings(self):
        return json.loads((ROOT/'configs/perception/prompt_voice_ethan.json').read_text())['tts']

    def test_provider_uses_ethan_and_download_does_not_receive_api_key(self):
        raw=io.BytesIO()
        with wave.open(raw,'wb') as w:
            w.setparams((1,2,24000,0,'NONE','not compressed'))
            w.writeframes((np.sin(np.arange(2400)*.1)*5000).astype('<i2').tobytes())
        result=dict(request_id='test-request',output=dict(audio=dict(url='http://bucket.oss-cn-beijing.aliyuncs.com/test.wav?Signature=private')))
        class Response(io.BytesIO):pass
        with patch('urllib.request.urlopen',side_effect=[Response(json.dumps(result).encode()),Response(raw.getvalue())]) as request:
            engine=CloudTTS(self.settings(),secret='unit-secret',cache_directory=False);audio=engine.generate('在呢，你说！')
        first,second=[c.args[0] for c in request.call_args_list]
        self.assertEqual(json.loads(first.data)['input']['voice'],'Ethan')
        self.assertEqual(first.get_header('Authorization'),'Bearer unit-secret')
        self.assertIsNone(second.get_header('Authorization'));self.assertTrue(second.full_url.startswith('https://'))
        self.assertEqual(audio.sample_rate,24000);self.assertEqual(len(audio.samples),2400)
        self.assertNotIn('unit-secret',str(engine.execution_info));self.assertNotIn('Signature',str(engine.execution_info))

    def test_provider_rejects_unexpected_host_silence_and_redacts_errors(self):
        engine=CloudTTS(self.settings(),secret='unit-secret',cache_directory=False)
        with patch.object(engine,'request',return_value=json.dumps(dict(output=dict(audio=dict(url='https://evil.example/audio.wav')))).encode()):
            with self.assertRaisesRegex(ValueError,'host'):engine.generate('你好')
        exc=urllib.error.HTTPError(self.settings()['endpoint'],401,'Rejected',{},
            io.BytesIO(b'{"code":"InvalidApiKey","message":"unit-secret rejected"}'))
        with patch('urllib.request.urlopen',side_effect=exc),self.assertRaises(RuntimeError) as caught:
            engine.generate('你好')
        self.assertNotIn('unit-secret',str(caught.exception));self.assertIn('[redacted]',str(caught.exception))

    def test_all_seven_verified_before_activation_then_runtime_cache_needs_no_key(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);config=root/'speech.json';preset=root/'ethan.json'
            original=dict(tts=dict(engine='old'),prompts='unused');config.write_text(json.dumps(original))
            overlay=dict(tts=self.settings(),prompts=str(root/'cache'));preset.write_text(json.dumps(overlay))
            class Engine:
                num_speakers=1
                def __init__(self,*_,**__):self.calls=0
                def generate(self,*_,**__):
                    self.calls+=1
                    if json.loads(config.read_text())!=original:raise AssertionError('Activated before all prompts were prepared')
                    return SimpleNamespace(samples=np.ones(2400)*.2,sample_rate=24000)
            engine=Engine()
            with patch('vivis.speech.cloud_tts.CloudTTS',return_value=engine):
                self.assertEqual(main(['--config',str(config),'--preset',str(preset),'--activate']),0)
            active=json.loads(config.read_text());self.assertTrue(prompt_cache_ready(active))
            self.assertEqual(engine.calls,len(RUNTIME_PROMPTS));self.assertEqual(active['tts']['voice'],'Ethan')
            with patch('vivis.speech.cloud_tts.CloudTTS',side_effect=AssertionError('Unexpected cloud synthesis')):
                self.assertEqual(main(['--config',str(config),'--preset',str(preset),'--activate']),0)
            (root/'cache/welcome.wav').write_bytes(b'corrupt')
            self.assertFalse(prompt_cache_ready(active))

    def test_preparation_failure_preserves_active_dylan_config(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);config=root/'speech.json';preset=root/'ethan.json'
            before=json.dumps(dict(tts=dict(engine='qwen3',speaker_name='Dylan'),prompts='unused'))
            config.write_text(before);preset.write_text(json.dumps(dict(tts=self.settings(),prompts=str(root/'cache'))))
            with patch('vivis.speech.cloud_tts.CloudTTS',side_effect=RuntimeError('simulated permission failure')), \
                 self.assertRaisesRegex(RuntimeError,'permission'):
                main(['--config',str(config),'--preset',str(preset),'--activate'])
            self.assertEqual(config.read_text(),before)


if __name__=='__main__':unittest.main()
