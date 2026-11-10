"""User-authorized encoder tolerance throughout startup/home/playback; no devices."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from test_cold_start import ColdAxis
from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.show import prepare_round
from vivis.runtime.homing import home_check,homing_feedback_motor
from vivis.runtime.playback import check_feedback
from vivis.runtime.trajectory import startup_target


class PositionToleranceTests(unittest.TestCase):
    def setUp(self):
        motor=Robot(ROOT/'configs/robots/xiaoshuai/robot.json').axes[34]
        self.low,self.high=motor['device_min'],motor['device_max']

    def fixture(self,position,enabled=1):
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        profile=robot.motion_profile();motor=next(m for m in profile['motors'] if m['id']==34)
        profile['motors']=[motor];profile['diagnostic_command_ids']=[34]
        axes=[a for a in robot.home('eyes-neck')['axes'] if a['id']==34]
        bus=ColdAxis(position=position,enabled=enabled,sid=34,low=motor['device_min'],high=motor['device_max'],model=11272)
        bus.close=Mock();bus.speed=motor['max_speed_counts_s']
        clock=SimpleNamespace(now=10.)
        clock.monotonic=lambda:clock.now
        clock.sleep=lambda duration:setattr(clock,'now',clock.now+max(duration,.00001))
        args=SimpleNamespace(usb_volume_times_three=False,home_only=True,home_speed=60,authorized_tongue_observation=False)
        return robot,profile,motor,axes,bus,clock,args

    def test_all_xiaoshuai_axes_use_the_canonical_fifty_count_policy(self):
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        self.assertEqual({a['position_feedback_tolerance_counts'] for a in robot.axes.values()},{50})
        self.assertEqual(robot.excluded_actuator_ids,{9,24})

    def test_real_prepare_pipeline_accepts_onsite2302_and_both_fifty_count_edges(self):
        for position,enabled in ((self.high+2,1),(self.high+50,1),(self.low-50,1),(self.high+50,0)):
            with self.subTest(position=position,enabled=enabled):
                robot,profile,motor,axes,bus,clock,args=self.fixture(position,enabled)
                report={}
                with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()), \
                     patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus), \
                     patch('vivis.runtime.homing.time',clock):
                    prepare_round(args,profile,[34],axes,None,None,robot.telemetry(),lambda:None,report,Path(tmp),lambda *_:None)
                self.assertEqual(report['self_check']['state'],'passed')
                self.assertTrue(report['preparation_completed']);self.assertTrue(report['homing']['completed'])
                self.assertEqual(bus.position,2047)
                self.assertTrue(all(self.low<=e[1]<=self.high for e in bus.events if e[0]=='goal'))

    def test_fifty_one_counts_fails_self_check_before_any_write(self):
        for position in (self.high+51,self.low-51,-1,4096):
            robot,profile,motor,axes,bus,clock,args=self.fixture(position);report={}
            with self.subTest(position=position),tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()), \
                 patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus), \
                 self.assertRaisesRegex(RuntimeError,'outside device limits'):
                prepare_round(args,profile,[34],axes,None,None,robot.telemetry(),lambda:None,report,Path(tmp),lambda *_:None)
            self.assertFalse(any(e[0] in ('goal','write') for e in bus.events))
            self.assertEqual(report['self_check']['state'],'failed')

    def test_startup_and_playback_agree_and_keep_alarm_and_encoder_checks(self):
        _,_,m,_,bus,_,_=self.fixture(2302)
        m.update(feedback_min=m['device_min'],feedback_max=m['device_max'])
        for position in (self.low-50,self.low,self.high,self.high+2,self.high+50):
            bus.position=position;s=bus.snapshot(34);f=bus.feedback(34)
            home_check(f,s,feedback_tolerance_counts=50)
            check_feedback(34,f,m)
            self.assertTrue(m['safe_min']<=startup_target(s,m,neck=True)<=m['safe_max'])
        for position in (self.low-51,self.high+51,-1,4096):
            bus.position=position;s=bus.snapshot(34);f=bus.feedback(34)
            with self.assertRaises(RuntimeError):home_check(f,s,feedback_tolerance_counts=50)
            with self.assertRaises(RuntimeError):check_feedback(34,f,m)
            with self.assertRaises(ValueError):startup_target(s,m,neck=True)
        bus.position=2302;f={**bus.feedback(34),'status_register':1}
        with self.assertRaises(RuntimeError):home_check(f,bus.snapshot(34),feedback_tolerance_counts=50)
        with self.assertRaises(RuntimeError):check_feedback(34,f,m)

    def test_all_active_homing_envelopes_admit_fifty_but_not_fifty_one(self):
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        for m in robot.motion_profile()['motors']:
            s=dict(configured_min=m['device_min'],configured_max=m['device_max'])
            hm=homing_feedback_motor(m,s)
            state=dict(temperature_c=35,load_raw=0,status_register=0,voltage_raw=238 if m['id']>=31 else 62)
            for position in (m['device_min']-50,m['device_max']+50):
                with self.subTest(id=m['id'],position=position):check_feedback(m['id'],dict(state,position=position),hm)
            for position in (m['device_min']-51,m['device_max']+51):
                with self.subTest(id=m['id'],position=position),self.assertRaises(RuntimeError):
                    check_feedback(m['id'],dict(state,position=position),hm)


if __name__=='__main__':unittest.main()
