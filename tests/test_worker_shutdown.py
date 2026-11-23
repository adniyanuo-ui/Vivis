"""Kernel/device I/O may delay process exit even after SIGKILL. No hardware."""
import subprocess
import unittest
from unittest.mock import Mock,patch

from vivis.interaction.dialogue import Worker


class WorkerShutdownTests(unittest.TestCase):
    def worker(self):
        worker=Worker.__new__(Worker);worker.kill_requested_at=None
        worker.process=Mock(pid=12345);worker.thread=Mock();worker.log=Mock()
        worker.kill_owned_group=Mock();return worker

    def test_unreaped_sigkill_is_a_result_not_an_uncaught_exception(self):
        worker=self.worker();worker.process.wait.side_effect=subprocess.TimeoutExpired('worker',2)
        self.assertFalse(worker.force_stop());worker.kill_owned_group.assert_called_once()

    def test_final_cleanup_does_not_restart_kill_wait_budget(self):
        worker=self.worker();worker.process.wait.side_effect=subprocess.TimeoutExpired('worker',2)
        with patch('vivis.interaction.dialogue.time.monotonic',side_effect=[10.,10.,12.1]):
            self.assertFalse(worker.force_stop());self.assertFalse(worker.force_stop())
        worker.kill_owned_group.assert_called_once()
        self.assertEqual(worker.process.wait.call_args_list[-1].kwargs['timeout'],0.)

    def test_eventual_exit_can_be_reaped_after_initial_timeout(self):
        worker=self.worker();worker.process.wait.side_effect=[subprocess.TimeoutExpired('worker',2),-9]
        self.assertFalse(worker.force_stop());self.assertTrue(worker.force_stop())
        worker.kill_owned_group.assert_called_once()

    def test_pending_reader_is_not_closed_from_another_thread(self):
        worker=self.worker();worker.thread.is_alive.return_value=True
        self.assertFalse(worker.finish());worker.process.stdout.close.assert_not_called();worker.log.close.assert_not_called()

    def test_completed_reader_closes_its_pipes_normally(self):
        worker=self.worker();worker.thread.is_alive.return_value=False
        self.assertTrue(worker.finish());worker.process.stdout.close.assert_called_once();worker.log.close.assert_called_once()


if __name__=='__main__':unittest.main()
