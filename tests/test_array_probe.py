"""Contracts for a diagnostic that must not change the production voice path."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from vivis.speech.array_probe import make_plan,waveforms,render,main,RATE
from vivis.speech.array_analysis import channel_identity,load,analyze


class ArrayProbeTests(unittest.TestCase):
    def test_aba_reference_changes_only_the_board_feed(self):
        stages=make_plan('far-only',10);external,board=waveforms(stages,.5)
        parts=[s for s in stages if s['name']!='quiet']
        a,b,c=[slice(round(s['start_s']*RATE),round((s['start_s']+10)*RATE)) for s in parts]
        np.testing.assert_array_equal(external[a],external[b])
        np.testing.assert_array_equal(external[a],external[c])
        np.testing.assert_array_equal(board[b],external[b])
        self.assertFalse(np.any(board[a]));self.assertFalse(np.any(board[c]))

    def test_near_only_does_not_play_either_output(self):
        stages=make_plan('speech',10);external,board=waveforms(stages,.5)
        stage=next(s for s in stages if s['name']=='near')
        use=slice(round(stage['start_s']*RATE),round((stage['start_s']+10)*RATE))
        self.assertFalse(np.any(external[use]));self.assertFalse(np.any(board[use]))

    def test_render_is_clock_based_and_zero_outside_the_source(self):
        signal=np.arange(100,dtype=np.float32)
        np.testing.assert_allclose(render(signal,10+2.5/RATE,10,4),[2.5,3.5,4.5,5.5],atol=1e-8)
        self.assertFalse(np.any(render(signal,9,10,4)))
        self.assertFalse(np.any(render(signal,11,10,4)))

    def test_check_opens_no_hardware_and_reference_needs_physical_confirmation(self):
        with patch('vivis.speech.array_probe.run') as run,redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--check']),0)
        self.assertFalse(json.loads(output.getvalue())['hardware_opened']);run.assert_not_called()
        with redirect_stdout(io.StringIO()),patch('sys.stderr',io.StringIO()):
            with self.assertRaises(SystemExit):main([])
            with self.assertRaises(SystemExit):main(['--check','--output-gain','nan'])

    def test_shifted_duplicate_does_not_imply_a_raw_channel(self):
        a=np.random.default_rng(8).normal(0,.1,4800).astype(np.float32)
        result=channel_identity(np.c_[a,np.r_[0,a[:-1]]])
        self.assertEqual(result['channel1_delay_samples'],1)
        self.assertEqual(result['equal_fraction'],1.)

    def test_recorded_clock_indices_are_checked_before_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            with wave.open(str(path/'capture_48k.wav'),'wb') as wav:
                wav.setparams((2,2,RATE,0,'NONE','not compressed'));wav.writeframes(np.zeros((1920,2),dtype='<i2').tobytes())
            frames=[dict(sample=0,frames=960,time_monotonic=10.),dict(sample=960,frames=960,time_monotonic=10.02)]
            index=path/'capture_frames.jsonl';index.write_text('\n'.join(map(json.dumps,frames)))
            data,times,timing=load(path,'capture')
            self.assertEqual(data.shape,(1920,2));self.assertAlmostEqual(times[960],10.02)
            frames[1]['sample']=959;index.write_text('\n'.join(map(json.dumps,frames)))
            with self.assertRaisesRegex(ValueError,'Noncontiguous'):load(path,'capture')

    def test_analysis_exports_complete_speech_and_preserves_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'recording';folder.mkdir();size=3*RATE
            signal=(np.sin(np.arange(size)*.04)*3000).astype('<i2')
            for key in ('capture','external','board'):
                data=np.c_[signal,np.r_[0,signal[:-1]]] if key=='capture' else np.zeros((size,2),dtype='<i2')
                with wave.open(str(folder/(key+'_48k.wav')),'wb') as wav:
                    wav.setparams((2,2,RATE,0,'NONE','not compressed'));wav.writeframes(data.astype('<i2').tobytes())
                (folder/(key+'_frames.jsonl')).write_text('\n'.join(json.dumps(dict(sample=i,frames=960,time_monotonic=10+i/RATE)) for i in range(0,size,960)))
            (folder/'summary.json').write_text(json.dumps(dict(start_monotonic=10,error=None,stages=[dict(name='near',start_s=0,duration_s=3,board_reference=False)])))
            before=(folder/'capture_48k.wav').read_bytes()
            out=Path(tmp)/'analysis';report=analyze(folder,out)
            with wave.open(str(out/'near_channel0_16k.wav')) as wav:self.assertEqual(wav.getnframes(),48000)
            self.assertEqual((folder/'capture_48k.wav').read_bytes(),before)
            self.assertEqual(report['digital_channel_identity']['equal_fraction'],1.)
            self.assertNotIn('passed',report)


if __name__=='__main__':unittest.main()
