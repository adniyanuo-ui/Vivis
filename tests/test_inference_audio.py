"""Separate, verified model/playback audio without old animation or device I/O."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.interaction.singing import inference_source,plan,main


class InferenceAudioTests(unittest.TestCase):
    def test_both_heads_use_the_original_vocals_but_play_the_same_mixed_song(self):
        settings=json.loads((ROOT/'configs/interaction/p0.json').read_text())
        for name in ('mojia','xiaoshuai'):
            robot=Robot(ROOT/f'configs/robots/{name}/robot.json')
            p=plan(robot,'0001','eyes-neck',settings)
            self.assertEqual(p['audio'],ROOT/'assets/works/0001/audio.wav')
            self.assertEqual(p['inference_audio'],ROOT/'assets/works/0001/inference_vocals.wav')
            self.assertNotEqual(p['inference_audio'],p['audio'])
            with wave.open(str(p['audio'])) as a,wave.open(str(p['inference_audio'])) as b:
                self.assertEqual(a.getnframes(),b.getnframes());self.assertEqual(a.getframerate(),b.getframerate())

    def test_changed_audio_origin_pair_or_duration_cannot_pass_alignment_verification(self):
        original=Robot(ROOT/'configs/robots/xiaoshuai/robot.json').work('0001')
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);path=folder/'vocals.wav';path.write_bytes(original['inference_audio'].read_bytes())
            alignment=folder/'alignment.json';record=json.loads(original['inference_alignment'].read_text())
            work={**original,'inference_audio':path,'inference_alignment':alignment}
            def save(value):
                alignment.write_text(json.dumps(value));work['inference_alignment_sha256']=hashlib.sha256(alignment.read_bytes()).hexdigest()
            save(record);self.assertEqual(inference_source(work),path)
            for change in ({'time_origin':'shifted'},{'playback_sha256':'0'*64},{'sample_count':1}):
                save({**record,**change})
                with self.assertRaisesRegex(ValueError,'alignment'):inference_source(work)
            save(record)
            raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'hash'):inference_source(work)
            # Updating only the file hash is insufficient: the alignment record
            # must refer to the exact audio pair that was checked.
            work['inference_audio_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'alignment'):inference_source(work)
            with wave.open(str(path),'wb') as f:
                f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(bytes(16000*2))
            work['inference_audio_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            save({**record,'inference_sha256':work['inference_audio_sha256'],'sample_count':16000})
            with self.assertRaisesRegex(ValueError,'alignment'):inference_source(work)

    def test_works_without_a_separate_track_keep_their_original_audio(self):
        work=Robot().work('wz2');self.assertNotIn('inference_audio',work)
        self.assertEqual(inference_source(work),work['audio'])

    def test_cli_reports_selected_track_and_retains_explicit_mix_comparison_without_devices(self):
        robot=ROOT/'configs/robots/xiaoshuai/robot.json'
        with patch('vivis.interaction.stream.subprocess.Popen',side_effect=AssertionError('process opened')), \
             patch('vivis.hardware.feetech.sdk.OfficialSDKBus',side_effect=AssertionError('serial opened')):
            for override in (False,True):
                args=['--robot',str(robot),'--smile','off','--check']
                if override:args+=['--inference-audio',str(ROOT/'assets/works/0001/audio.wav')]
                with redirect_stdout(io.StringIO()) as out:main(args)
                report=json.loads(out.getvalue())
                self.assertEqual(report['inference_audio_override'],override)
                self.assertEqual(report['inference_audio'],str(ROOT/'assets/works/0001'/('audio.wav' if override else 'inference_vocals.wav')))
                self.assertEqual(report['audio'],str(ROOT/'assets/works/0001/audio.wav'))
