"""Opt-in homing reuses song sensor counters; never count damaged samples."""
import unittest
from unittest.mock import patch

import test_homing
from vivis.runtime.homing import home_all
from vivis.runtime.robot import Robot
from vivis.runtime.telemetry import ConfirmedSensorFaultError
from vivis.hardware.feetech.protocol import CorruptResponse,ReadTimeout,ProtocolError,ServoStatusError


class HomingSensorTests(unittest.TestCase):
    faults=({'packet_status_bits':1},{'voltage_raw':36},{'status_register':4},
            {'temperature_c':165},{'status_register':8},{'status_register':32})

    def fixture(self):
        axes,bus,clock,pos,speeds,enabled,writes,restores=test_homing.HomingTests().fixture()
        bus.servo_status_policy='raise'
        return axes,bus,clock,writes

    def run_home(self,axes,bus,clock,report,guard=lambda:None,enabled=True):
        with patch('vivis.runtime.homing.time',clock),patch('vivis.runtime.playback.time',clock):
            home_all(bus,axes,{},guard,report,max_speed=60,telemetry_policy=Robot().telemetry(),
                     sensor_fault_confirmation=enabled)

    def test_confirmation_is_explicitly_enabled_for_xiaoshuai_only(self):
        self.assertTrue(Robot('configs/robots/xiaoshuai/robot.json').home()['sensor_fault_confirmation'])
        self.assertFalse(Robot().home().get('sensor_fault_confirmation',False))

    def test_transient_faults_hold_goals_until_healthy_and_preserve_raw_samples(self):
        for fault in self.faults:
            with self.subTest(fault=fault):
                axes,bus,clock,writes=self.fixture();original=bus.feedback;seen=[];report={}
                def feedback(sid):
                    self.assertEqual(bus.servo_status_policy,'defer_sensor_faults')
                    state=original(sid)
                    if sid==4 and len(seen)<2:
                        seen.append(len(writes))
                        if len(seen)==1:return {**state,**fault}
                    return state
                bus.feedback=feedback
                self.run_home(axes,bus,clock,report)
                self.assertTrue(report['completed']);self.assertEqual(seen[0],seen[1])
                record=report['sensor_fault_confirmations'][0]
                self.assertTrue(record['recovered']);self.assertEqual(record['peak_consecutive_fault_samples'],1)
                for key,value in fault.items():self.assertEqual(record['samples'][0]['state'][key],value)
                self.assertEqual(bus.servo_status_policy,'raise')

    def test_each_persistent_fault_stops_at_four_whole_samples_without_fifth(self):
        for fault in self.faults:
            with self.subTest(fault=fault):
                axes,bus,clock,writes=self.fixture();original=bus.feedback;seen=[];report={}
                def feedback(sid):
                    seen.append((sid,len(writes)))
                    return {**original(sid),**fault}
                bus.feedback=feedback
                with self.assertRaisesRegex(ConfirmedSensorFaultError,':4'):
                    self.run_home(axes,bus,clock,report)
                self.assertEqual(len(seen),4);self.assertEqual(len(set(seen)),1)
                self.assertEqual(len(report['sensor_fault_confirmations'][0]['samples']),4)
                self.assertFalse(report['completed']);self.assertEqual(bus.servo_status_policy,'raise')

    def test_fault_kinds_never_add_together_during_a_hold(self):
        axes,bus,clock,writes=self.fixture();original=bus.feedback;report={};seen=[]
        faults=iter([{'packet_status_bits':1}]*3+[{'status_register':8}]*3+[{}])
        def feedback(sid):
            state=original(sid)
            if sid==4:
                fault=next(faults,None)
                if fault is not None:seen.append(len(writes));state.update(fault)
            return state
        bus.feedback=feedback;self.run_home(axes,bus,clock,report)
        self.assertTrue(report['completed']);self.assertEqual(len(seen),7);self.assertEqual(len(set(seen)),1)
        self.assertEqual([r['peak_consecutive_fault_samples'] for r in report['sensor_fault_confirmations']],[3,3])

    def test_missing_or_corrupt_sample_never_counts_as_a_sensor_vote(self):
        for error in (CorruptResponse('bad checksum'),ReadTimeout('missing reply')):
            with self.subTest(error=error):
                axes,bus,clock,writes=self.fixture();original=bus.feedback;attempts=[];report={}
                def feedback(sid):
                    if sid==4:
                        attempts.append(len(writes))
                        if len(attempts)==1:return {**original(sid),'packet_status_bits':1}
                        if len(attempts)==2:raise error
                    return original(sid)
                bus.feedback=feedback;self.run_home(axes,bus,clock,report)
                self.assertTrue(report['completed'])
                record=report['sensor_fault_confirmations'][0]
                self.assertEqual(record['peak_consecutive_fault_samples'],1);self.assertTrue(record['recovered'])
                self.assertTrue(report['communication_recoveries'][0]['recovered'])
                self.assertEqual(len(set(attempts[:4])),1)

    def test_position_unknown_alarm_and_transport_gate_still_fail_immediately(self):
        for fault,error in (({'position':2600,'packet_status_bits':1},None),
                            ({'packet_status_bits':2},None),({},ProtocolError('Damaged/missing feedback exceeds 5% of reads'))):
            with self.subTest(fault=fault,error=error):
                axes,bus,clock,writes=self.fixture();original=bus.feedback;calls=[];report={}
                def feedback(sid):
                    calls.append(sid)
                    if error:raise error
                    return {**original(sid),**fault}
                bus.feedback=feedback
                with self.assertRaises(RuntimeError):self.run_home(axes,bus,clock,report)
                self.assertEqual(len(calls),1);self.assertFalse(report['sensor_fault_confirmations'])
                self.assertEqual(bus.servo_status_policy,'raise')

    def test_operator_stop_during_hold_prevents_more_reads(self):
        axes,bus,clock,writes=self.fixture();original=bus.feedback;calls=[];report={};stopped=[False]
        sleep=clock.sleep
        def wait(delay):
            sleep(delay)
            if delay==Robot().telemetry()['sensor_fault_confirmation_period_s']:stopped[0]=True
        clock.sleep=wait
        def guard():
            if stopped[0]:raise RuntimeError('Operator stop')
        def feedback(sid):
            calls.append(sid);return {**original(sid),'packet_status_bits':1}
        bus.feedback=feedback
        with self.assertRaisesRegex(RuntimeError,'Operator stop'):self.run_home(axes,bus,clock,report,guard)
        self.assertEqual(len(calls),1);self.assertEqual(bus.servo_status_policy,'raise')

    def test_disabled_confirmation_preserves_strict_transport_alarm(self):
        axes,bus,clock,writes=self.fixture();calls=[]
        def feedback(sid):
            calls.append(sid);self.assertEqual(bus.servo_status_policy,'raise');raise ServoStatusError(sid,1)
        bus.feedback=feedback
        with self.assertRaises(ServoStatusError):self.run_home(axes,bus,clock,{},enabled=False)
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
