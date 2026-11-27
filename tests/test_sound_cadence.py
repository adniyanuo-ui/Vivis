"""Receipt times are not periodic UI refreshes or board generation timestamps."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.check_sound_direction_cadence import CadenceEvidence,main
from vivis.speech.sound_direction import configuration


class CadenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.c=configuration();self.now=0.
        self.folder=Path(self.tmp.name)/'capture'
        self.e=CadenceEvidence(self.folder,self.c,clock=lambda:self.now);self.e.begin=0.

    def feed(self,now,data):self.now=now;self.e.raw_bytes(data)

    def finish(self):
        report={};self.e.finish(report);return report['cadence']

    def test_real_receipt_gaps_and_same_sector_updates_are_counted_without_ui_refreshes(self):
        for now in (10.1,10.35,10.6,13.5):self.feed(now,bytes.fromhex('55 AA 05 FA'))
        self.feed(14.,b'noise')
        with patch('builtins.print'):
            for now in (14.2,14.45,14.7):
                self.now=now;self.e.emit_phase(None)  # Displaying is not receiving.
        self.now=30.
        report=self.finish();p=report['phases']['fixed_voice']
        self.assertEqual(p['valid_frames'],4);self.assertEqual(p['same_sector_frames'],3)
        self.assertAlmostEqual(p['gap_max_s'],2.9);self.assertEqual(p['gaps_above_max_age'],1)
        self.assertAlmostEqual(p['tail_without_coordinate_s'],16.5)
        records=[json.loads(l) for l in (self.folder/'receipts.jsonl').read_text().splitlines()]
        self.assertEqual(len(records),5);self.assertEqual(records[-1]['sectors'],[])

    def test_split_frames_and_batched_frames_keep_host_resolution_honest(self):
        self.feed(10.1,b'\x55');self.feed(10.2,bytes.fromhex('AA 00 FF 55 AA 01 FE'))
        report=self.finish();p=report['phases']['fixed_voice']
        self.assertEqual(p['valid_frames'],2);self.assertEqual(p['valid_frame_batches'],1)
        self.assertIsNone(p['gap_max_s']);self.assertEqual(p['sector_changes'],1)
        self.assertIn('not_board_emission',report['timing_basis'])

    def test_timing_quota_does_not_stop_frame_counting(self):
        self.e.settings['max_event_bytes']=1
        self.feed(10.1,bytes.fromhex('55 AA 05 FA'))
        report=self.finish()
        self.assertTrue(report['receipts_truncated']);self.assertEqual(report['phases']['fixed_voice']['valid_frames'],1)

    def test_offline_check_never_enumerates_or_opens_devices(self):
        self.finish()
        with patch('serial.tools.list_ports.comports',side_effect=AssertionError('enumerated')), \
                patch('tools.check_sound_direction_cadence.open_receiver',side_effect=AssertionError('opened')), \
                patch('tools.check_sound_direction_cadence.subprocess.run',side_effect=AssertionError('systemctl')):
            self.assertEqual(main(['--check']),0)
