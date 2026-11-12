"""Offline song timing/semantic and catalogue publication contracts."""
import json
import csv
import wave
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from vivis.paths import ROOT

sys.path.insert(0,str(ROOT/'tools'))
from prepare_song_library import source_clip,register_programs
from separate_song_vocals import stft,istft,CHUNK


class SongPreparationTests(unittest.TestCase):
    def test_fft_roundtrip_preserves_phase_and_stereo_order(self):
        times=np.arange(CHUNK)/44100
        samples=np.array([np.sin(2*np.pi*440*times),.3*np.cos(2*np.pi*900*times)],np.float32)
        result=istft(stft(samples))
        self.assertEqual(result.shape,samples.shape)
        np.testing.assert_allclose(result[:,4000:-4000],samples[:,4000:-4000],atol=1e-5)

    def test_timestamp_resampling_uses_named_channels_and_zero_gaze(self):
        names=['shape'+str(i) for i in range(52)]
        order=[n[0].upper()+n[1:] for n in reversed(names)]+['gaze'+str(i) for i in range(9)]
        frames=[dict(frame_seq=i,audio_sample_index=round(i*16000/60),sample_rate_hz=16000,
                     weights=[i*.1]+[0.]*51) for i in range(3)]
        clip=source_clip(dict(names=dict(skin=names,tongue=[])),frames,order,.05)
        self.assertEqual(len(clip['blendShapes']),3)
        self.assertAlmostEqual(clip['blendShapes'][2][51],.2,places=3)
        self.assertTrue(np.all(np.asarray(clip['blendShapes'])[:,52:]==0))
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            source_clip(dict(names=dict(skin=names,tongue=[])),frames,order,1.)

    def test_pending_audio_never_enters_catalog_and_retry_does_not_duplicate(self):
        from types import SimpleNamespace
        rows=[dict(work_id='ready_song',status='ready',title='歌曲',artist='歌手',aliases='别名'),
              dict(work_id='pending_song',status='awaiting_audio')]
        robot=SimpleNamespace(work=lambda name: {} if name=='ready_song' else self.fail(name))
        with tempfile.TemporaryDirectory() as tmp,patch('vivis.runtime.assets.verify_assets'):
            p=Path(tmp)/'programs.json';p.write_text(json.dumps(dict(programs=[])))
            register_programs(rows,robot,p);register_programs(rows,robot,p)
            programs=json.loads(p.read_text())['programs']
            self.assertEqual(len(programs),1);self.assertEqual(programs[0]['work'],'ready_song')
            self.assertFalse(programs[0]['realtime_response'])
            self.assertEqual(programs[0]['kind'],'prerecorded')

    def test_ready_library_has_full_audio_and_specific_voice_aliases(self):
        from vivis.runtime.robot import Robot
        from vivis.interaction.intents import IntentRouter
        catalog=json.loads((ROOT/'configs/interaction/programs.json').read_text())
        router=IntentRouter(catalog);robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        with (ROOT/'assets/program_requests/songs_20261112.csv').open() as f:rows=list(csv.DictReader(f))
        ready=[row for row in rows if row['status']=='ready']
        self.assertGreaterEqual(len(ready),14)
        for row in ready:
            with self.subTest(work=row['work_id']):
                self.assertEqual(router.parse('播放'+row['title'],state='interaction'),
                                 dict(kind='select',program=row['work_id']))
                work=robot.work(row['work_id'])
                with wave.open(str(work['audio'])) as f:duration=f.getnframes()/f.getframerate()
                self.assertAlmostEqual(duration,float(row['duration_s']),delta=1/16000)
                self.assertLessEqual(abs(work['frames']/60-duration),1/60)
        self.assertNotIn('fenghuang_shan_he_tu',{p['id'] for p in catalog['programs']})
        self.assertEqual(router.parse('播放想要什么给你',state='interaction')['program'],'li_jian_gei_ni')
        self.assertEqual(router.parse('播放You Are Beautiful',state='interaction')['program'],'james_blunt_youre_beautiful')


if __name__=='__main__':unittest.main()
