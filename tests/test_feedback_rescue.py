"""Near-deadline read-only slots recover feedback without extending protection."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.runtime.feedback import BudgetedFeedback
from vivis.hardware.feetech.protocol import CorruptResponse,ReadTimeout,ServoStatusError


class FeedbackRescueTests(unittest.TestCase):
    def fixture(self):
        now=[.94];bus=Mock();samples=[];errors=[];clocks={21:0.,22:.01}
        fields={56:b'\xff\x07',60:b'\x20\x00',62:b'\x3d\x1c',65:b'\x00'}
        def read(sid,address,count):now[0]+=.0006;return fields[address]
        bus.read.side_effect=read
        reader=BudgetedFeedback(bus,[21,22],priority_ids=())
        def sample(sid,state,started):clocks[sid]=started;samples.append((sid,state))
        return now,bus,reader,clocks,sample,samples,errors,SimpleNamespace(monotonic=lambda:now[0])

    def test_hold_reserve_uses_existing_period_timeout_backoff_and_does_not_refresh_clocks(self):
        now,bus,reader,clocks,*_=self.fixture();before=dict(clocks)
        self.assertFalse(reader.command_hold_ids(clocks,.8,1/60))
        self.assertIn(21,reader.command_hold_ids(clocks,.94,1/60))
        self.assertEqual(clocks,before);bus.read.assert_not_called()
        self.assertEqual(reader.max_feedback_age_s,1.)

    def test_two_bad_packets_recover_in_held_ticks_before_same_one_second_deadline(self):
        now,bus,reader,clocks,sample,samples,errors,clock=self.fixture()
        read=bus.read.side_effect;remaining=[2]
        def corrupt(sid,address,count):
            if sid==21 and remaining[0]:
                remaining[0]-=1;now[0]+=.003;raise CorruptResponse('captured two-bad-packet pattern')
            return read(sid,address,count)
        bus.read.side_effect=corrupt
        reader.last_failure[21]=.935
        with patch('vivis.runtime.feedback.time',clock):
            for tick in range(3):
                now[0]=.94+tick/60
                result=reader.step(min(now[0]+1/60-.001,1.),clocks,sample,lambda *e:errors.append(e),critical_ids=[21])
                self.assertLessEqual(result['reads'],8)
                if tick<2:self.assertEqual(sum(c.args[0]==21 for c in bus.read.call_args_list),tick+1)
                if tick<2:self.assertEqual(clocks[21],0.)
        self.assertEqual(len(errors),2);self.assertGreater(clocks[21],.97)
        self.assertLess(now[0],1.)
        self.assertEqual(next(state['temperature_c'] for sid,state in samples if sid==21),28)

    def test_partial_three_fields_cannot_refresh_age_but_read_only_slack_completes_fourth(self):
        now,bus,reader,clocks,sample,samples,errors,clock=self.fixture()
        clocks[22]=.94
        with patch('vivis.runtime.feedback.time',clock):
            first=reader.step(.9445,clocks,sample,lambda *e:errors.append(e),critical_ids=[21])
            self.assertEqual(first['reads'],3);self.assertEqual(reader.sid,21)
            self.assertEqual(clocks[21],0.)
            second=reader.step(.955,clocks,sample,lambda *e:errors.append(e),critical_ids=[21])
        self.assertIn(21,[sid for sid,_ in samples]);self.assertGreater(clocks[21],0.)
        self.assertEqual([c.args[1] for c in bus.read.call_args_list[:4]],[56,60,62,65])

    def test_actual_servo_alarm_is_not_swallowed_by_rescue(self):
        now,bus,reader,clocks,sample,samples,errors,clock=self.fixture()
        bus.read.side_effect=ServoStatusError(21,0x10)
        with patch('vivis.runtime.feedback.time',clock),self.assertRaises(ServoStatusError):
            reader.step(.955,clocks,sample,lambda *e:errors.append(e),critical_ids=[21])
        self.assertEqual(clocks[21],0.);self.assertFalse(samples)

    def test_playback_holds_one_command_tick_and_keeps_feedback_in_separate_log(self):
        from test_runtime import PlaybackTests
        from vivis.runtime.playback import play
        from vivis.interaction.follow import MonotonicClock
        profile,bus=PlaybackTests().continuous_fixture();source=Mock();source.sample.return_value=np.array([2047.])
        selected=[False]
        def critical(self,clocks,now,period):
            if selected[0]:return []
            selected[0]=True;return [2]
        with tempfile.TemporaryDirectory() as tmp,patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus), \
             patch.object(BudgetedFeedback,'command_hold_ids',critical):
            result=play(profile,[2],np.full((1,1),2047.),60,None,Path(tmp)/'log.json',Path(tmp)/'STOP',
                max_seconds=.1,frame_source=source,clock_factory=lambda:MonotonicClock(None))
        self.assertTrue(result['completed']);self.assertEqual(result['feedback_command_holds'],1)
        rescue=result['feedback_rescue_events'][0]
        self.assertFalse(rescue['command_sent']);self.assertGreater(rescue['feedback_schedule']['completed'],0)
        self.assertEqual(len(result['events']),result['sent_frames'])
        self.assertNotIn('goals',rescue)

    def test_persistent_missing_feedback_still_stops_at_original_watchdog(self):
        from test_runtime import PlaybackTests
        from vivis.runtime.playback import play
        from vivis.interaction.follow import MonotonicClock
        profile,bus=PlaybackTests().continuous_fixture();source=Mock();source.sample.return_value=np.array([2047.])
        bus.read.side_effect=ReadTimeout('persistent disconnection')
        with tempfile.TemporaryDirectory() as tmp,patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus):
            log=Path(tmp)/'log.json'
            with self.assertRaisesRegex(RuntimeError,'Feedback watchdog|Playback watchdog'):
                play(profile,[2],np.full((1,1),2047.),60,None,log,Path(tmp)/'STOP',
                    max_seconds=2,frame_source=source,clock_factory=lambda:MonotonicClock(None))
            result=json.loads(log.read_text())
        self.assertFalse(result['completed']);self.assertGreater(result.get('feedback_command_holds',0),0)
        self.assertEqual(profile['feedback_max_age_s'],1.)
        self.assertTrue(all(not r['command_sent'] for r in result['feedback_rescue_events']))
        bus.close.assert_called_once()


if __name__=='__main__':unittest.main()
