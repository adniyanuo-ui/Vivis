"""Only committed files count as collected; busy logging retries without duplication."""
import json
from pathlib import Path
import tempfile
import unittest

from vivis.interaction.sample_progress import SampleProgress


class SampleProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        c=self.root/'configs/perception';c.mkdir(parents=True)
        (c/'familiar_faces.json').write_text(json.dumps(dict(people_directory='private/people')))
        (c/'speaker_observation.json').write_text(json.dumps(dict(directory='private/speakers')))
        self.person=self.root/'private/people/alpha';self.person.mkdir(parents=True)
        (self.person/'profile.json').write_text(json.dumps(dict(preferred_address='何工')))
        self.voice=self.root/'private/speakers';self.voice.mkdir()
        self.events=[];self.s=SampleProgress(self.root,self.events.append);self.addCleanup(self.s.close)

    def photo(self,ids):
        rows=[]
        for key in ids:
            (self.person/(key+'.jpg')).write_bytes(b'image')
            rows.append(dict(hash=key,path=key+'.jpg'))
        (self.person/'photo_pool.json').write_text(json.dumps(dict(rows=rows,active=ids[:16])))

    def test_existing_samples_are_reported_as_baseline_not_new_collection(self):
        self.photo(['old']);self.s.scan()
        e=next(e for e in self.events if e['sample_type']=='photo')
        self.assertTrue(e['initial']);self.assertEqual(e['added'],0);self.assertEqual(e['total'],1)
        self.assertIn('照片样本库',e['message'])

    def test_new_photo_batch_reports_success_once_after_manifest_and_files_exist(self):
        self.s.scan();self.events.clear();self.photo(['a','b']);self.s.scan();self.s.scan()
        self.assertEqual(len(self.events),1);e=self.events[0]
        self.assertEqual(e['added'],2);self.assertTrue(e['committed']);self.assertIn('照片采集成功',e['message'])

    def test_voice_commit_has_separate_clear_success_log(self):
        self.s.scan();self.events.clear();p=self.voice/'alpha';p.mkdir();(p/'one.wav').write_bytes(b'pcm')
        (self.voice/'gallery.json').write_text(json.dumps(dict(people=dict(alpha=[dict(turn_id='one')]))))
        self.s.scan();self.assertEqual(len(self.events),1)
        self.assertEqual(self.events[0]['sample_type'],'voice');self.assertIn('声纹参考采集成功',self.events[0]['message'])

    def test_uncommitted_or_missing_file_is_not_reported_as_collection(self):
        self.s.scan();self.events.clear();self.photo(['missing']);(self.person/'missing.jpg').unlink();self.s.scan()
        self.assertEqual(self.events,[])

    def test_full_control_queue_retries_the_same_committed_change(self):
        self.s.scan();self.events.clear();self.photo(['a'])
        self.s.emit=lambda e:False;self.s.scan();self.assertEqual(self.events,[])
        self.s.emit=self.events.append;self.s.scan();self.s.scan()
        self.assertEqual(len(self.events),1);self.assertEqual(self.events[0]['added'],1)

    def test_missing_optional_configs_do_not_raise_into_dialogue(self):
        (self.root/'configs/perception/familiar_faces.json').unlink();self.s.scan()
        self.assertEqual(self.s.error,'FileNotFoundError')


if __name__=='__main__':unittest.main()
