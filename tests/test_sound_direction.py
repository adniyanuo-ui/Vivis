"""Vendor frame replay, UART isolation and stale data without physical devices."""
from copy import deepcopy
import json
import os
from pathlib import Path
import pty
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vivis.hardware.haokai import DirectionFrame, DirectionParser, DirectionState, open_receiver, select_port
from vivis.speech.sound_direction import CONFIG, Evidence, configuration, main, monitor_receiver


def frame(sector):
    return bytes((0x55, 0xaa, sector, 0xff-sector))


class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.c = configuration()

    def test_vendor_example_and_all_twelve_sectors_at_every_split(self):
        stream = bytes.fromhex('55 AA 05 FA 00 E2 7E') + b''.join(frame(i) for i in range(12))
        for cut in range(len(stream)+1):
            parser = DirectionParser()
            result = parser.feed(stream[:cut])+parser.feed(stream[cut:])
            self.assertEqual([r.sector for r in result], [5]+list(range(12)))
            self.assertLessEqual(len(parser.pending), 3)

    def test_bad_checksum_out_of_range_header_noise_and_resynchronisation(self):
        parser = DirectionParser()
        data = b'\x55' + bytes.fromhex('55 AA 05 00 55 AA 0C F3') + frame(0)+frame(11)
        result = parser.feed(data)
        self.assertEqual([r.sector for r in result], [0, 11])
        self.assertEqual(parser.invalid, 2)
        self.assertGreater(parser.discarded, 0)

    def test_partial_header_and_long_noise_do_not_accumulate(self):
        parser = DirectionParser()
        self.assertEqual(parser.feed(b'x'*100000+b'\x55'), [])
        self.assertEqual(parser.pending, b'\x55')
        self.assertEqual(parser.feed(b'\xaa\x00'), [])
        self.assertEqual(parser.feed(b'\xff')[0].sector, 0)
        self.assertFalse(parser.pending)

    def test_fragmented_invalid_frame_cannot_publish_a_direction(self):
        parser = DirectionParser()
        for b in bytes.fromhex('55 AA 02 00'):
            self.assertEqual(parser.feed(bytes([b])), [])
        self.assertEqual(parser.frames, 0)

    def test_zero_is_a_real_sector_but_not_robot_forward_or_speech(self):
        state = DirectionState(self.c['coordinates'], .75)
        self.assertEqual(state.snapshot(10.)['status'], 'waiting')
        state.observe(DirectionFrame(0, frame(0)), 10.)
        packet = state.snapshot(10.1)
        self.assertEqual(packet['sector'], 0)
        self.assertEqual(packet['board_nominal_azimuth_deg'], 180)
        for key in ('speech_active', 'confidence', 'robot_azimuth_deg', 'elevation_deg', 'distance_m'):
            self.assertIsNone(packet[key])
        self.assertFalse(packet['motion_authorized'])

    def test_expired_and_future_host_clock_clear_current_direction(self):
        state = DirectionState(self.c['coordinates'], .75)
        state.observe(DirectionFrame(5, frame(5)), 10.)
        self.assertEqual(state.snapshot(10.75)['board_nominal_azimuth_deg'], 330)
        for now in (10.751, 9.):
            packet = state.snapshot(now)
            self.assertEqual(packet['status'], 'stale')
            self.assertIsNone(packet['sector'])
            self.assertIsNone(packet['raw_frame_hex'])

    def test_bad_or_fabricated_observation_rejected(self):
        state = DirectionState(self.c['coordinates'], .75)
        for observation, now in ((DirectionFrame(12, b''), 10.),
                                 (DirectionFrame(1, frame(2)), 10.),
                                 (DirectionFrame(1, frame(1)), float('nan'))):
            with self.assertRaises(ValueError):
                state.observe(observation, now)

    def ports(self):
        return [SimpleNamespace(device='/dev/ttyCH343USB0', vid=0x1a86, pid=0x55d3),
                SimpleNamespace(device='/dev/ttyUSB0', vid=0x1a86, pid=0x7523)]

    def test_selects_unique_ch340_not_head_and_never_guesses_unknown_adapter(self):
        self.assertEqual(select_port(self.ports(), self.c['serial'], '/dev/ttyCH343USB0'),
                         '/dev/ttyUSB0')
        for requested in ('/dev/ttyCH343USB0', '/dev/ttyTHS1', '/dev/missing'):
            with self.assertRaises(ValueError):
                select_port(self.ports(), self.c['serial'], '/dev/ttyCH343USB0', requested)
        with self.assertRaises(ValueError):
            select_port(self.ports()[:1], self.c['serial'], '/dev/ttyCH343USB0')

    def test_multiple_adapters_need_explicit_selection(self):
        ports = self.ports()+[SimpleNamespace(device='/dev/ttyUSB1', vid=0x1a86, pid=0x7523)]
        with self.assertRaises(ValueError):
            select_port(ports, self.c['serial'], '/dev/ttyCH343USB0')
        self.assertEqual(select_port(ports, self.c['serial'], '/dev/ttyCH343USB0', '/dev/ttyUSB1'),
                         '/dev/ttyUSB1')

    def test_listing_does_not_claim_builtin_uart_is_a_usb_adapter(self):
        local = SimpleNamespace(device='/dev/ttyS0', vid=None, pid=None)
        with patch('serial.tools.list_ports.comports', return_value=[local]), \
                patch('serial.Serial', side_effect=AssertionError('hardware opened')), \
                patch('builtins.print') as output:
            self.assertEqual(main(['--list']), 0)
        self.assertIn('未枚举到USB串口', output.call_args.args[0])

    def test_symlink_cannot_bypass_protected_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            alias = Path(tmp)/'array'
            alias.symlink_to('/dev/ttyCH343USB0')
            with self.assertRaises(ValueError):
                select_port(self.ports(), self.c['serial'], '/dev/ttyCH343USB0', str(alias))

    def test_host_setup_is_exclusive_without_transmit_or_flow_control(self):
        receiver = Mock()
        with patch('serial.Serial', return_value=receiver) as factory:
            self.assertIs(open_receiver('/dev/ttyUSB0', self.c['serial']), receiver)
        self.assertEqual(factory.call_args.kwargs['baudrate'], 1000000)
        self.assertTrue(factory.call_args.kwargs['exclusive'])
        self.assertFalse(factory.call_args.kwargs['rtscts'])
        self.assertFalse(receiver.dtr)
        self.assertFalse(receiver.rts)
        receiver.open.assert_called_once()
        receiver.write.assert_not_called()

    def test_open_failure_closes_receiver(self):
        receiver = Mock()
        receiver.open.side_effect = OSError('device removed')
        with patch('serial.Serial', return_value=receiver), self.assertRaises(OSError):
            open_receiver('/dev/ttyUSB0', self.c['serial'])
        receiver.close.assert_called_once()

    def test_configuration_and_offline_check_open_no_device(self):
        with patch('serial.Serial', side_effect=AssertionError('hardware opened')), \
                patch('serial.tools.list_ports.comports', side_effect=AssertionError('hardware enumerated')):
            self.assertEqual(main(['--check']), 0)
        with tempfile.TemporaryDirectory() as tmp:
            for mutate in (lambda c: c.update(motion_enabled=True),
                           lambda c: c['serial'].update(baud_rate=115200),
                           lambda c: c['monitor'].update(max_age_s=float('nan')),
                           lambda c: c['coordinates'].update(sector_zero_nominal_deg=0)):
                c = deepcopy(self.c)
                mutate(c)
                path = Path(tmp)/'bad.json'
                path.write_text(json.dumps(c))
                with self.assertRaises(ValueError):
                    configuration(path)

    def test_log_quotas_do_not_stop_parsing_and_existing_evidence_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'evidence'
            e = Evidence(path, dict(max_raw_bytes=4, max_event_bytes=3))
            e.raw_bytes(frame(5)+frame(6))
            e.raw_bytes(frame(7))
            e.event(dict(sector=5))
            e.finish({})
            self.assertEqual((path/'raw.bin').read_bytes(), frame(5))
            summary = json.loads((path/'summary.json').read_text())
            self.assertTrue(summary['raw_truncated'])
            self.assertTrue(summary['events_truncated'])
            with self.assertRaises(FileExistsError):
                Evidence(path, self.c['monitor'])

    def test_read_error_closes_device_and_preserves_explicit_failure(self):
        receiver = Mock(in_waiting=0)
        receiver.read.side_effect = OSError('USB removed')
        with tempfile.TemporaryDirectory() as tmp:
            e = Evidence(Path(tmp)/'run', self.c['monitor'])
            result = monitor_receiver(receiver, self.c, .2, e, emit=lambda _: None)
            self.assertEqual(result['outcome'], 'device_error')
            self.assertIn('USB removed', result['error'])
            self.assertEqual(result['statistics']['valid_frames'], 0)
        receiver.close.assert_called_once()
        receiver.write.assert_not_called()

    def test_usb_close_failure_is_recorded_without_losing_capture_summary(self):
        receiver = Mock(in_waiting=0)
        receiver.read.side_effect = KeyboardInterrupt
        receiver.close.side_effect = OSError('USB close failed')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run'
            result = monitor_receiver(receiver, self.c, .2, Evidence(path, self.c['monitor']),
                                      emit=lambda _: None)
            self.assertEqual(result['outcome'], 'device_error')
            self.assertEqual(result['failure_stage'], 'serial_close')
            self.assertIn('USB close failed', result['error'])
            self.assertTrue((path/'summary.json').is_file())

    def test_real_pty_stream_reads_documented_frames_sends_no_protocol_bytes(self):
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        receiver = open_receiver(os.ttyname(slave), self.c['serial'])
        def transmit():
            time.sleep(.03)
            for data in (b'\x55', b'\xaa\x05', b'\xfa\x00\xe2\x7e', frame(0), frame(11)):
                os.write(master, data)
                time.sleep(.01)
        writer = threading.Thread(target=transmit)
        writer.start()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run'
            e = Evidence(path, self.c['monitor'])
            result = monitor_receiver(receiver, self.c, .3, e, emit=lambda _: None)
            self.assertEqual(result['statistics']['valid_frames'], 3)
            self.assertEqual(result['serial_protocol_bytes_sent'], 0)
            self.assertFalse(receiver.is_open)
            events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(v['frame_valid'] for v in events))
            self.assertEqual(result['last_observation']['sector'], 11)
        writer.join(timeout=1.)
        self.assertFalse(writer.is_alive())
        os.set_blocking(master, False)
        with self.assertRaises(BlockingIOError):
            os.read(master, 1)


if __name__ == '__main__':
    unittest.main()
