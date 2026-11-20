"""Managed startup evidence, neutral returns and periodic fault recovery; no devices."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock,patch

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.motion_session import MotionSession,MotionSessionError,maintain_standby,preparation_action,for_profile
from vivis.runtime.show import prepare_round


class SessionTests(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        self.session=MotionSession.create(Path(folder.name)/'session.json',self.robot)
        self.fingerprint=patch('vivis.runtime.motion_session.port_identity',return_value={'port':'test-device'})
        self.fingerprint.start();self.addCleanup(self.fingerprint.stop)
        raw=bytearray(40);raw[18]=4;ram=bytearray(31)
        ram[6:8]=(600).to_bytes(2,'little')
        self.snapshot=dict(id=2,config_0_39_hex=raw.hex(),state_40_70_hex=ram.hex())
        self.report=dict(self_check=dict(state='passed'),homing=dict(completed=True),startup_snapshots=[self.snapshot])

    def test_ready_requires_successful_check_and_home(self):
        for key in ('self_check','homing'):
            bad=copy.deepcopy(self.report)
            bad[key]['state' if key=='self_check' else 'completed']='failed' if key=='self_check' else False
            with self.assertRaises(MotionSessionError):self.session.initialized(bad)
            self.assertFalse(self.session.read(check_port=False)['ready'])
        self.session.initialized(self.report);self.assertTrue(self.session.read(require_ready=True)['ready'])

    def test_evidence_rejects_other_owner_token_configuration_and_serial(self):
        self.session.initialized(self.report)
        original=json.loads(self.session.path.read_text())
        for field,value in [('token','wrong'),('supervisor_pid',-1),('robot_digest','wrong'),('port_identity',{'port':'new'})]:
            with self.subTest(field=field):
                changed=copy.deepcopy(original);changed[field]=value;self.session.write(changed)
                with self.assertRaises(MotionSessionError):self.session.read(require_ready=True)
        self.session.write(original)
        self.session.invalidate('lost feedback')
        with self.assertRaises(MotionSessionError):self.session.read(require_ready=True)

    def test_resume_cannot_be_enabled_by_a_public_flag_or_stale_environment(self):
        env=self.session.environment('resume')
        with patch.dict(os.environ,env),self.assertRaises(MotionSessionError):MotionSession.current(self.robot)
        self.session.initialized(self.report)
        with patch.dict(os.environ,{**env,'VIVIS_MANAGED_SESSION':'1'}),patch('os.getppid',return_value=os.getpid()):
            context,action=preparation_action(self.robot);self.assertEqual(action,'resume')
            context.validate_snapshot(self.snapshot)
            for_profile(self.robot.motion_profile())
            wrong=dict(self.snapshot);wrong['config_0_39_hex']='01'+wrong['config_0_39_hex'][2:]
            with self.assertRaises(MotionSessionError):context.validate_snapshot(wrong)
            for offset,value in ((1,b'\x80'),(8,b'\x01\x00'),(6,b'\x00\x00')):
                with self.subTest(ram_offset=offset):
                    ram=bytearray.fromhex(self.snapshot['state_40_70_hex']);ram[offset:offset+len(value)]=value
                    wrong=dict(self.snapshot,state_40_70_hex=ram.hex())
                    with self.assertRaises(MotionSessionError):context.validate_snapshot(wrong)

    def test_resume_does_not_open_bus_or_home_and_requires_initialized_evidence(self):
        args=SimpleNamespace(preparation_action='resume',motion_session=self.session,authorized_tongue_observation=False)
        with patch('vivis.hardware.feetech.sdk.OfficialSDKBus') as bus,patch('vivis.runtime.show.home_all') as home:
            with self.assertRaises(MotionSessionError):prepare_round(args,{},[],[],None,None,None,lambda:None,{},self.session.path.parent,lambda *_:None)
            self.session.initialized(self.report);report={}
            prepare_round(args,{},[],[],None,None,None,lambda:None,report,self.session.path.parent,lambda *_:None)
            bus.assert_not_called();home.assert_not_called()
            self.assertEqual(report['self_check']['state'],'reused');self.assertFalse(report['homing']['required'])

    def test_mojia_resume_keeps_jaw_tongue_interlock_and_end_unload_obligation(self):
        robot=Robot(ROOT/'configs/robots/mojia/robot.json');profile=robot.motion_profile();inventory=robot.home('eyes-neck')
        for enabled in (0,1):
            with self.subTest(initial_torque=enabled):
                calls=[];bus=Mock()
                bus.snapshot.side_effect=[dict(id=29,torque_enable=enabled,position_raw=2047),dict(id=31)]
                args=SimpleNamespace(preparation_action='resume',motion_session=Mock(),authorized_tongue_observation=True,home_speed=60)
                report={}
                with patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus),\
                     patch('vivis.runtime.show.prepare_recovery_clearance',side_effect=lambda *a:calls.append('open_jaw')),\
                     patch('vivis.runtime.show.recover_tongue',side_effect=lambda *a,**kw:calls.append('center_tongue')),\
                     patch('vivis.runtime.show.home_all',side_effect=lambda *a,**kw:calls.append(('home',[x['id'] for x in a[1]]))):
                    prepare_round(args,profile,list(profile['diagnostic_command_ids']),inventory['axes'],None,{},None,
                        lambda:None,report,self.session.path.parent,lambda *_:None)
                self.assertTrue(report['tongue_cleanup_required']);bus.close.assert_called_once()
                self.assertEqual(calls,[] if enabled else ['open_jaw','center_tongue',('home',[31])])

    def test_standby_inspection_preserves_mojia_tongue_unload_after_report_reset(self):
        from vivis.interaction.standby import main
        def prepare(*args):
            report=args[8]
            if args[0].preparation_action=='inspect':self.assertTrue(report['tongue_cleanup_required'])
            else:report['tongue_cleanup_required']=True
        def maintain(session,action,prepare_action,play,guard):
            prepare_action('startup');prepare_action('inspect');raise RuntimeError('Operator stop')
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()),\
             patch('vivis.interaction.standby.fcntl.flock'),patch('vivis.interaction.standby.check_competitors'),\
             patch('vivis.interaction.standby.PreviewPublisher',return_value=Mock(finish=Mock(return_value=None))),\
             patch('vivis.runtime.motion_session.preparation_action',return_value=(Mock(),'startup')),\
             patch('vivis.runtime.motion_session.maintain_standby',side_effect=maintain),\
             patch('vivis.interaction.standby.prepare_round',side_effect=prepare),\
             patch('vivis.interaction.standby.unload_tongue') as unload:
            self.assertEqual(main(['--robot',str(ROOT/'configs/robots/mojia/robot.json'),
                '--config',str(ROOT/'configs/interaction/standby.json'),
                '--continuous','--output',str(Path(tmp)/'standby')]),130)
            unload.assert_called_once();self.assertTrue(unload.call_args.args[1]['tongue_cleanup_required'])


class PreparationTests(unittest.TestCase):
    def fixture(self,action):
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json');profile=robot.motion_profile()
        motor=next(m for m in profile['motors'] if m['id']==2);profile['motors']=[motor]
        raw=bytearray(40);raw[18]=4;ram=bytearray(31);ram[0]=1;ram[6:8]=(600).to_bytes(2,'little')
        state=dict(id=2,model_raw=521,mode=0,torque_enable=1,configured_min=motor['device_min'],configured_max=motor['device_max'],
            configured_max_temperature=70,configured_min_voltage_raw=40,configured_max_voltage_raw=90,
            position_raw=2047,temperature_c=30,voltage_raw=60,load_raw=0,status_register=0,
            config_0_39_hex=raw.hex(),state_40_70_hex=ram.hex())
        bus=Mock();bus.snapshot.return_value=state
        bus.feedback.return_value=dict(position=2047,temperature_c=30,voltage_raw=60,load_raw=0,status_register=0)
        session=Mock();session.read.return_value={'ready':True}
        args=SimpleNamespace(preparation_action=action,motion_session=session,home_speed=60,home_only=True,
            usb_volume_times_three=False,authorized_tongue_observation=False)
        home=[dict(id=2,neutral=2047,neutral_confirmed=True,model_raw_expected=521)]
        return robot,profile,bus,args,home

    def test_periodic_inspection_is_read_only_and_has_no_homing(self):
        robot,profile,bus,args,home=self.fixture('inspect');report={}
        with tempfile.TemporaryDirectory() as tmp,patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus),\
             patch('vivis.runtime.show.home_all') as move,patch('vivis.runtime.show.restore_startup_settings') as restore:
            prepare_round(args,profile,[2],home,robot.startup([2],[2]),None,robot.telemetry(),lambda:None,report,Path(tmp),lambda *_:None)
            move.assert_not_called();restore.assert_not_called();bus.goals.assert_not_called()
            bus.sdk.write1ByteTxRx.assert_not_called();bus.sdk.write2ByteTxRx.assert_not_called()
            self.assertEqual(report['self_check']['state'],'passed');self.assertFalse(report['homing'])

    def test_warm_standby_returns_neutral_without_recommissioning(self):
        robot,profile,bus,args,home=self.fixture('standby');report={}
        def moved(*values,**kw):values[4]['completed']=True
        with tempfile.TemporaryDirectory() as tmp,patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus),\
             patch('vivis.runtime.show.home_all',side_effect=moved) as move,patch('vivis.runtime.show.restore_startup_settings') as restore:
            prepare_round(args,profile,[2],home,robot.startup([2],[2]),None,robot.telemetry(),lambda:None,report,Path(tmp),lambda *_:None)
            move.assert_called_once();restore.assert_not_called();args.motion_session.initialized.assert_not_called()
            self.assertEqual(move.call_args.args[6],{2:600})


class MaintenanceTests(unittest.TestCase):
    def fixture(self,stop_at=125):
        clock=SimpleNamespace(now=0.);clock.monotonic=lambda:clock.now
        clock.sleep=lambda n:setattr(clock,'now',clock.now+n)
        session=Mock();session.read.return_value={'policy':{'standby_check_interval_s':60}}
        def guard():
            if clock.now>=stop_at:raise RuntimeError('Operator stop')
        return clock,session,guard

    def test_cold_start_failure_is_startup_failure_not_ready_or_retry(self):
        clock,session,guard=self.fixture();prepare=Mock(side_effect=RuntimeError('home failed'));play=Mock()
        with self.assertRaisesRegex(RuntimeError,'home failed'):
            maintain_standby(session,'startup',prepare,play,guard,clock=clock.monotonic,wait=clock.sleep)
        play.assert_not_called();session.invalidate.assert_not_called();self.assertEqual(clock.now,0)

    def test_one_boot_then_read_only_inspection_each_minute_without_repeat_home(self):
        clock,session,guard=self.fixture();steps=[]
        def prepare(action):steps.append((clock.now,action))
        def play(seconds,number):clock.sleep(seconds)
        with self.assertRaisesRegex(RuntimeError,'Operator stop'):
            maintain_standby(session,'startup',prepare,play,guard,clock=clock.monotonic,wait=clock.sleep)
        self.assertEqual(steps,[(0,'startup'),(60,'inspect'),(120,'inspect')])

    def test_fault_notice_each_minute_and_recovery_needs_check_plus_home(self):
        clock,session,guard=self.fixture(stop_at=121);steps=[];plays=[]
        def prepare(action):
            steps.append((clock.now,action))
            if action=='standby' or action=='startup' and clock.now<120:raise RuntimeError('home failed')
        def play(seconds,number):plays.append(clock.now);clock.sleep(seconds)
        console=io.StringIO()
        with redirect_stdout(console),self.assertRaisesRegex(RuntimeError,'Operator stop'):
            maintain_standby(session,'standby',prepare,play,guard,clock=clock.monotonic,wait=clock.sleep)
        self.assertEqual(steps,[(0,'standby'),(60,'inspect'),(60,'startup'),(120,'inspect'),(120,'startup')])
        self.assertEqual(plays,[120]);self.assertEqual(console.getvalue().count('STANDBY FAULT '),2)
        self.assertEqual(console.getvalue().count('STANDBY RECOVERED '),1)

    def test_runtime_fault_holds_until_next_minute_and_does_not_replay_program(self):
        clock,session,guard=self.fixture(stop_at=71);steps=[];plays=[]
        def prepare(action):steps.append((clock.now,action))
        def play(seconds,number):
            plays.append(clock.now)
            if len(plays)==1:clock.sleep(10);raise RuntimeError('Feedback watchdog')
            clock.sleep(seconds)
        with redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'Operator stop'):
            maintain_standby(session,'standby',prepare,play,guard,clock=clock.monotonic,wait=clock.sleep)
        self.assertEqual(steps,[(0,'standby'),(70,'inspect'),(70,'startup')]);self.assertEqual(plays,[0,70])

    def test_failed_resource_cleanup_result_is_fault_not_successful_standby(self):
        clock,session,guard=self.fixture(stop_at=61);steps=[]
        def prepare(action):steps.append((clock.now,action))
        play=Mock(return_value=dict(completed=False,error='Playback resource cleanup failed'))
        with redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'Operator stop'):
            maintain_standby(session,'standby',prepare,play,guard,clock=clock.monotonic,wait=clock.sleep)
        self.assertEqual(steps,[(0,'standby'),(60,'inspect'),(60,'startup')])
        session.invalidate.assert_called_with(unittest.mock.ANY)
        self.assertIn('cleanup failed',str(session.invalidate.call_args.args[0]))


if __name__=='__main__':unittest.main()
