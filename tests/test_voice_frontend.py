"""Front-end contracts, local diagnostic timing and optional real AEC3 regression."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock,patch
import wave

import numpy as np
from scipy.signal import firwin,lfilter

from vivis.paths import ROOT
from vivis.speech.voice_frontend import Aec3,VoiceFrontEnd,validate_frontend
from vivis.speech.voice_check import main,source_audio,STAGES
from vivis.speech.voice_analysis import stage_ranges,correlation_peak,replay,analyze,attenuation


def speech():return json.loads((ROOT/'configs/perception/speech.json').read_text())


class FrontendTests(unittest.TestCase):
    def test_backend_and_delay_options_are_bounded(self):
        self.assertEqual(validate_frontend({}),'speex')
        for config in ({'backend':'unknown'},{'aec3':{'ns_level':4}},
                       {'aec3':{'ns_level':True}},{'aec3':{'stream_delay_ms':-1}},
                       {'aec3':{'stream_delay_ms':float('nan')}}):
            with self.assertRaises(ValueError):validate_frontend(config)

    def test_capture_filter_is_continuous_and_aec_off_does_not_gate_near_speech(self):
        processor=VoiceFrontEnd(speech(),enabled=False)
        rng=np.random.default_rng(12);near=rng.normal(0,.1,960*4).astype(np.float32)
        far=np.ones_like(near)*.5;parts=[]
        for i in range(0,len(near),960):
            raw,reference,clean=processor.process48(near[i:i+960],far[i:i+960])
            np.testing.assert_array_equal(raw,clean);parts.append(clean)
        expected=lfilter(firwin(63,1/3-.015),1,near)[::3].astype(np.float32)
        np.testing.assert_array_equal(np.concatenate(parts),expected)
        with self.assertRaises(ValueError):processor.process16(np.zeros(319),np.zeros(319))
        with self.assertRaises(ValueError):processor.process16(np.full(320,np.nan),np.zeros(320))
        processor.close()

    def test_aec3_keeps_one_state_and_disables_agc_without_muting_during_far_audio(self):
        module=ModuleType('pywebrtc_audio');engine=Mock()
        engine.process.side_effect=lambda near,far:near.copy();module.AudioProcessor=Mock(return_value=engine)
        with patch.dict(sys.modules,{'pywebrtc_audio':module}):
            proc=VoiceFrontEnd(speech(),{'backend':'aec3','aec3':{'ns_level':1,'stream_delay_ms':0}})
            for amplitude in (.0,.4,.0):
                near=np.linspace(-.2,.2,320).astype(np.float32);far=np.full(320,amplitude,dtype=np.float32)
                np.testing.assert_array_equal(proc.process16(near,far),near)
            self.assertEqual(module.AudioProcessor.call_count,1)
            self.assertEqual(engine.process.call_count,3)
            self.assertFalse(module.AudioProcessor.call_args.kwargs['auto_gain_control'])
            self.assertTrue(module.AudioProcessor.call_args.kwargs['echo_cancellation'])
            proc.close()

    def test_missing_aec3_has_install_message_not_silent_speex_fallback(self):
        with patch.dict(sys.modules,{'pywebrtc_audio':None}):
            with self.assertRaisesRegex(RuntimeError,'setup_realtime_voice'):Aec3({})

    def test_check_mode_never_opens_devices_or_requires_api_key(self):
        with patch('vivis.speech.voice_check.AudioIO') as audio,redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['--check']),0)
        audio.assert_not_called();self.assertFalse(json.loads(out.getvalue())['hardware_opened'])
        self.assertEqual(sum(duration for _,duration in STAGES),67)

    def test_local_gain_trial_is_finite_and_does_not_change_canonical_config(self):
        path=ROOT/'configs/perception/realtime_voice.json';before=path.read_bytes()
        with patch('vivis.speech.voice_check.AudioIO') as audio,redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['--check','--output-gain','0.5']),0)
        self.assertEqual(json.loads(out.getvalue())['output_gain'],.5)
        self.assertEqual(path.read_bytes(),before);audio.assert_not_called()
        with redirect_stdout(io.StringIO()),patch('sys.stderr',io.StringIO()):
            with self.assertRaises(SystemExit):main(['--check','--output-gain','nan'])

    def test_acoustic_probe_uses_bounded_exact_duration_existing_pcm(self):
        pcm=source_audio(1.25)
        self.assertEqual(len(pcm),int(1.25*24000)*2)
        self.assertGreater(np.max(np.abs(np.frombuffer(pcm,'<i2').astype(float))),0)

    def test_stage_scoring_uses_adc_clock_and_excludes_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'frames.jsonl').write_text('\n'.join(json.dumps(dict(sample=i*320,adc_monotonic=100+i*.02)) for i in range(300)))
            ranges=stage_ranges(folder,{'stages':[dict(name='far_only',start_monotonic=101.,end_monotonic=105.)]},96000)
        self.assertEqual(ranges[0]['start_sample'],32000)
        self.assertEqual(ranges[0]['end_sample'],64000)

    def test_correlation_reports_delay_without_claiming_physical_path(self):
        rng=np.random.default_rng(3);far=rng.normal(0,.1,16000).astype(np.float32)
        near=np.r_[np.zeros(1600),far[:-1600]]
        self.assertEqual(correlation_peak(near,far)['delay_ms'],100.)
        self.assertIsNone(correlation_peak(np.zeros(16000),far))
        self.assertIsNone(attenuation(np.zeros(320),np.zeros(320)))

    def test_report_keeps_user_stage_correction_and_preserves_original_pcm(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'recording';folder.mkdir()
            (folder/'summary.json').write_text(json.dumps(dict(stages=[dict(name='double_talk',
                start_monotonic=100.,duration_s=3,end_monotonic=103.)],error=None)))
            (folder/'annotations.json').write_text(json.dumps(dict(stage_overrides={
                '0':dict(actual_condition='far_only',user_speech_present=False)})))
            (folder/'frames.jsonl').write_text('\n'.join(json.dumps(dict(sample=i*320,adc_monotonic=100+i*.02)) for i in range(150)))
            for name in ('microphone','reference','uplink'):
                with wave.open(str(folder/(name+'_16k.wav')),'wb') as w:
                    w.setparams((1,2,16000,0,'NONE','not compressed'))
                    w.writeframes(np.full(48000,1000 if name!='uplink' else 500,dtype='<i2').tobytes())
            before={p.name:p.read_bytes() for p in folder.glob('*.wav')}
            report=analyze(folder,Path(tmp)/'analysis')
            stage=report['stages'][0]
            self.assertFalse(stage['operator_annotation']['user_speech_present'])
            self.assertEqual(stage['operator_annotation']['actual_condition'],'far_only')
            self.assertAlmostEqual(stage['backends']['uplink']['attenuation_db'],6.0206,places=3)
            for name,data in before.items():self.assertEqual((folder/name).read_bytes(),data)
            self.assertNotIn('passed',report)  # Numerical reports must not invent acoustic acceptance.

    @unittest.skipUnless(importlib.util.find_spec('pywebrtc_audio'),'optional audio environment only')
    def test_real_aec3_preserves_near_speech_and_attenuates_delayed_echo(self):
        rng=np.random.default_rng(23);length=16000*12
        coeff=firwin(31,[.04,.6],pass_zero=False)
        far=lfilter(coeff,1,rng.normal(0,.15,length)).astype(np.float32)
        # Harmonic, syllabic near speech surrogate: broadband noise is an NS target,
        # so preserving random noise would test the wrong contract.
        t=np.arange(length)/16000;phase=2*np.pi*(150*t+1.5*np.sin(2*np.pi*2*t))
        dry=(.16*(.5+.5*np.sin(2*np.pi*3*t))**2
             *(np.sin(phase)+.5*np.sin(phase*2)+.2*np.sin(phase*3))).astype(np.float32)
        echo=np.r_[np.zeros(1600),far[:-1600]]*.4
        near=echo.copy();near[8*16000:]+=dry[8*16000:]
        clean,timing=replay(near.astype(np.float32),far,speech(),{'backend':'aec3'})
        # Synthetic far-only and double-talk ground truth, unlike unlabeled onsite recording.
        use=slice(4*16000,7*16000)
        reduction=10*np.log10(np.mean(near[use]**2)/max(np.mean(clean[use]**2),1e-12))
        self.assertGreater(reduction,20.)
        target=dry[9*16000:];result=clean[9*16000:]
        peak=correlation_peak(result,target,max_delay_ms=30)
        self.assertGreater(abs(peak['normalized_correlation']),.6)
        self.assertGreater(np.sqrt(np.mean(result**2)),.25*np.sqrt(np.mean(target**2)))


if __name__=='__main__':unittest.main()
