import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch
from contextlib import redirect_stdout

from vivis.paths import ROOT
from vivis.interaction.failure_report import describe_failure,failure_lines


class FailureReportTests(unittest.TestCase):
    def test_watchdog_reports_real_interval_and_recovery_without_inventing_hardware_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'worker';folder.mkdir()
            (folder/'playback_log.json').write_text(json.dumps(dict(
                error='Playback watchdog: stalled for >100 ms',watchdog_interval_s=.11672675,
                feedback_rescue_events=[dict(critical_ids=[4,10],feedback_ages_s={'4':.991264,'10':.959302})],
                serial_diagnostics=dict(read_failures=[dict(id=4,error='Official SDK ID 4: read timeout')]))))
            report=describe_failure(folder,1,'interaction');text='\n'.join(failure_lines(report))
            self.assertEqual(report['code'],'CONTROL_WATCHDOG')
            for expected in ['交互态','116.7毫秒','ID4=991.3毫秒','ID10=959.3毫秒','主动暂停','read timeout']:
                self.assertIn(expected,text)
            self.assertNotIn('硬件损坏',text)

    def test_traceback_tail_is_not_replaced_by_cleanup_or_chat_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'worker'
            folder.with_suffix('.log').write_text('noise\n'*10000+
                'Traceback (most recent call last):\n  raise RuntimeError("old")\n'
                'vivis.hardware.feetech.protocol.ServoStatusError: Servo 20: status 0x01\n'
                '清理完成\n用户：私密聊天\n')
            report=describe_failure(folder,1,'song')
            self.assertEqual(report['code'],'SERVO_STATUS');self.assertIn('Servo 20: status 0x01',report['raw_error'])
            self.assertNotIn('私密聊天',str(report))

    def test_missing_corrupt_or_oversized_artifacts_do_not_mask_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'worker';folder.mkdir()
            (folder/'summary.json').write_text('{broken')
            with (folder/'playback_log.json').open('wb') as f:f.truncate(16*1024*1024+1)
            self.assertEqual(describe_failure(folder,1)['code'],'WORKER_EXIT')
            report=describe_failure(folder,-signal.SIGKILL)
            self.assertEqual(report['code'],'WORKER_SIGNAL');self.assertIn('SIGKILL',report['message'])
            self.assertNotIn('内存不足',report['message'])

    def test_generic_exception_and_feedback_axes_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'worker';folder.mkdir()
            folder.with_suffix('.log').write_text('Exception: custom model failed\n')
            self.assertEqual(describe_failure(folder,1)['raw_error'],'Exception: custom model failed')
            folder.with_suffix('.log').unlink()
            (folder/'playback_log.json').write_text(json.dumps(dict(error='Feedback watchdog: no recent valid feedback for IDs [21]',
                feedback_watchdog={'21':{'valid_feedback_age_s':1.05}})))
            report=describe_failure(folder,1)
            self.assertEqual(report['code'],'FEEDBACK_TIMEOUT');self.assertIn('ID21有效反馈已间隔1050.0毫秒',' '.join(report['evidence']))

    def test_motion_failure_does_not_present_old_serial_errors_as_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'worker';folder.mkdir()
            (folder/'playback_log.json').write_text(json.dumps(dict(
                error='Standby pursuit target exceeds command limits: ID33 target=2348 limits=[1500,2347]',
                serial_diagnostics=dict(read_failures=[dict(error='Official SDK ID 10: corrupt response')]))))
            report=describe_failure(folder,1,'interaction')
            self.assertEqual(report['code'],'MOTION_TARGET_LIMIT')
            self.assertIn('ID33 target=2348',report['raw_error']);self.assertNotIn('corrupt response',str(report))

    def test_real_child_failure_reaches_console_status_event_and_sidecar(self):
        from vivis.interaction.dialogue import main
        finished=threading.Event()
        class WaitingInput:
            def __iter__(self):finished.wait(5);return iter(())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings=json.loads((ROOT/'configs/interaction/p1.json').read_text())
            for key in ('programs','semantic','conversation'):settings[key]=str(ROOT/settings[key])
            config=root/'config.json';config.write_text(json.dumps(settings))
            speech=json.loads((ROOT/'configs/perception/speech.json').read_text());console=io.StringIO()
            def command(*_):return [sys.executable,'-c','raise RuntimeError("ID33: device limits changed; actual=[1500,2347] expected=[1600,2300]")']
            try:
                with patch('vivis.interaction.dialogue.ROOT',root),patch('vivis.interaction.dialogue.dependencies',return_value=(speech,[])), \
                     patch('vivis.interaction.dialogue.check_competitors'),patch('vivis.interaction.dialogue.check_serial_access'), \
                     patch('vivis.interaction.dialogue.preview_state',return_value=None),patch('vivis.interaction.dialogue.PromptPlayer',return_value=Mock()), \
                     patch('vivis.interaction.dialogue.program_command',side_effect=command),patch('sys.stdin',WaitingInput()),redirect_stdout(console):
                    code=main(['--config',str(config),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),
                        '--text-input','--semantic','off','--cloud-chat','off','--realtime-voice','off','--seconds','2','--output',str(root/'run')])
            finally:finished.set()
            self.assertEqual(code,1);self.assertIn('节目故障[HOME_CHECK]',console.getvalue());self.assertIn('ID33',console.getvalue())
            status=json.loads((root/'outputs/runtime/program_status.json').read_text())
            self.assertIn('actual=[1500,2347]',status['error'])
            events=[json.loads(line) for line in (root/'run/events.jsonl').read_text().splitlines()]
            failure=next(e for e in events if e['kind']=='worker_failure');self.assertEqual(failure['code'],'HOME_CHECK')
            reports=list((root/'run').glob('*.failure.json'));self.assertEqual(len(reports),1)
            self.assertEqual(json.loads(reports[0].read_text())['code'],'HOME_CHECK')


if __name__=='__main__':unittest.main()
