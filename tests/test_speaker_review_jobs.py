import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch

from vivis.interaction.review_jobs import VoiceImportJobs


class VoiceImportJobsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name)
        python=root/'outputs/speech/venv/bin/python';python.parent.mkdir(parents=True);python.write_bytes(b'python')
        self.jobs=VoiceImportJobs(root);self.addCleanup(self.jobs.close)

    def test_queue_is_bounded_and_result_is_not_success_until_worker_commits(self):
        entered=threading.Event();release=threading.Event();process=Mock()
        def communicate(*a,**kw):entered.set();release.wait(2);return json.dumps(dict(ok=True,message='已保存候选')),''
        process.communicate.side_effect=communicate;process.poll.return_value=0
        with patch('vivis.interaction.review_jobs.subprocess.Popen',return_value=process):
            try:
                result=self.jobs.start(dict(person='heyude'));self.assertTrue(entered.wait(1))
                self.assertEqual(self.jobs.get(result['job_id'])['state'],'running')
                with self.assertRaises(ValueError):self.jobs.start({})
            finally:release.set();self.jobs.job.wait(2)
        self.assertEqual(self.jobs.get(result['job_id'])['state'],'complete')

    def test_failed_worker_is_honest_and_not_a_saved_candidate(self):
        process=Mock();process.communicate.return_value=(json.dumps(dict(ok=False,error='录音混说')),'');process.poll.return_value=1
        with patch('vivis.interaction.review_jobs.subprocess.Popen',return_value=process):
            result=self.jobs.start({});self.jobs.job.wait(2)
        self.assertEqual(self.jobs.get(result['job_id'])['state'],'failed')
        self.assertFalse(self.jobs.get(result['job_id'])['ok'])


if __name__=='__main__':unittest.main()
