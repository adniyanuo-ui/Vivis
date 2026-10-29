"""Replay the cold-start ID21=2106 case using an unpowered in-memory servo."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vivis.runtime.homing import home_all,confirm_home_recoveries,home_check,recovery_policy
from vivis.runtime.robot import Robot
from vivis.runtime.playback import check_feedback


class ColdAxis:
    def __init__(self,position=2106,enabled=0,jammed=False,sid=21,low=1943,high=2100,model=521):
        self.sid,self.low,self.high,self.model=sid,low,high,model
        self.position=position;self.enabled=enabled;self.jammed=jammed
        self.target=0;self.speed=0;self.events=[];self.bad_prime=False
        self.read_positions=None;self.snapshot_position=None
        self.auto_enable=False;self.settle_offset=0
        self.sdk=SimpleNamespace(write1ByteTxRx=self.write1,write2ByteTxRx=self.write2)

    def snapshot(self,sid):
        config=bytearray(40);config[18]=4
        ram=bytearray(31);ram[0]=self.enabled;ram[6:8]=self.speed.to_bytes(2,'little')
        return dict(id=self.sid,model_raw=self.model,mode=0,torque_enable=self.enabled,
                    configured_min=self.low,configured_max=self.high,configured_max_temperature=80,
                    configured_min_voltage_raw=40 if self.model==521 else 90,
                    configured_max_voltage_raw=90 if self.model==521 else 254,
                    position_raw=self.position if self.snapshot_position is None else self.snapshot_position,
                    load_raw=0,temperature_c=35,voltage_raw=62 if self.model==521 else 238,status_register=0,
                    config_0_39_hex=config.hex(),state_40_70_hex=ram.hex())

    def feedback(self,sid):
        position=next(self.read_positions) if self.read_positions is not None else self.position
        return dict(position=position,load_raw=0,temperature_c=35,voltage_raw=62 if self.model==521 else 238,status_register=0)

    def read(self,sid,address,width):
        self.events.append(('read',address))
        values={40:self.enabled,42:0 if self.bad_prime else self.target,46:self.speed}
        return values[address].to_bytes(width,'little')

    def goals(self,goals,speeds):
        self.target=goals[self.sid];self.speed=speeds[self.sid]
        if not self.low<=self.target<=self.high:raise AssertionError('out-of-range goal sent')
        if not 1<=self.speed<=60:raise AssertionError('uncapped homing speed sent')
        self.events.append(('goal',self.target,self.speed))
        if self.auto_enable:self.enabled=1
        if self.enabled and not self.jammed:self.position=self.target+self.settle_offset

    def write1(self,sid,address,value):
        self.events.append(('write',address,value))
        if address!=40:raise AssertionError('unexpected configuration write')
        self.enabled=value
        if value and not self.jammed:self.position=self.target+self.settle_offset
        return 0,0

    def write2(self,sid,address,value):
        self.events.append(('write',address,value))
        if address!=46:raise AssertionError('device limit was changed')
        self.speed=value
        return 0,0


class ColdStartTests(unittest.TestCase):
    def fixture(self,**kwargs):
        robot=Robot();sid=kwargs.get('sid',21)
        axis=next(a for a in robot.home()['axes'] if a['id']==sid)
        motor=next(m for m in robot.profile('wz2')['motors'] if m['id']==sid)
        policy=copy.deepcopy(robot.home()['disabled_position_recovery'])
        clock=SimpleNamespace(now=10.)
        clock.monotonic=lambda:clock.now
        clock.sleep=lambda duration:setattr(clock,'now',clock.now+max(duration,.00001))
        return ColdAxis(**kwargs),axis,motor,policy,clock,robot.telemetry()

    def run_home(self,bus,axis,motor,policy,clock,telemetry):
        report={}
        with patch('vivis.runtime.homing.time',clock):
            home_all(bus,[axis],{bus.sid:motor},lambda:None,report,max_speed=60,
                     restore_speeds={bus.sid:motor['max_speed_counts_s']},keep_enabled=[bus.sid],telemetry_policy=telemetry,
                     position_recovery=policy)
        return report

    def test_disabled_six_count_offset_primes_at_2100_before_enable_then_homes(self):
        bus,axis,motor,policy,clock,telemetry=self.fixture()
        with self.assertRaisesRegex(RuntimeError,'position=2106'):
            home_check(bus.feedback(21),bus.snapshot(21),2047,20,telemetry)
        report=self.run_home(bus,axis,motor,policy,clock,telemetry)
        self.assertTrue(report['completed']);self.assertEqual(bus.position,2047)
        self.assertEqual(bus.speed,1000);self.assertEqual(bus.enabled,1)
        self.assertEqual(report['startup_reentered_ids'],[21])
        first=next(e for e in bus.events if e[0]=='goal')
        self.assertEqual(first,('goal',2100,60))
        enable=bus.events.index(('write',40,1));prime=bus.events.index(first)
        self.assertLess(prime,enable)
        self.assertIn(('read',42),bus.events[prime:enable])
        self.assertIn(('read',46),bus.events[prime:enable])

    def test_already_enabled_outside_position_is_rejected_before_any_goal(self):
        bus,axis,motor,policy,clock,telemetry=self.fixture(enabled=1)
        with self.assertRaises(RuntimeError):self.run_home(bus,axis,motor,policy,clock,telemetry)
        self.assertFalse(any(e[0]=='goal' for e in bus.events))

    def test_goal_update_auto_enable_is_accepted_for_both_models(self):
        for kwargs in ({},{'sid':35,'low':2040,'high':2247,'model':10760,'position':1904}):
            with self.subTest(kwargs=kwargs):
                bus,axis,motor,policy,clock,telemetry=self.fixture(**kwargs);bus.auto_enable=True
                report=self.run_home(bus,axis,motor,policy,clock,telemetry)
                self.assertTrue(report['completed']);self.assertEqual(bus.position,2047)
                self.assertEqual(report['startup_primes'][0]['enable_source'],'goal_update')
                self.assertNotIn(('write',40,1),bus.events)
                first_goal=next(i for i,e in enumerate(bus.events) if e[0]=='goal')
                self.assertIn(('read',40),bus.events[:first_goal])
                self.assertIn(('read',42),bus.events[first_goal:])
                self.assertIn(('read',46),bus.events[first_goal:])

    def test_external_enable_before_prime_still_prevents_any_goal(self):
        bus,axis,motor,policy,clock,telemetry=self.fixture()
        read=bus.read;torque_reads=0
        def external_enable(sid,address,width):
            nonlocal torque_reads
            if address==40:
                torque_reads+=1
                if torque_reads>policy['confirmation_samples']:bus.enabled=1
            return read(sid,address,width)
        bus.read=external_enable
        with self.assertRaisesRegex(RuntimeError,'before startup target was sent'):
            self.run_home(bus,axis,motor,policy,clock,telemetry)
        self.assertFalse(any(e[0]=='goal' for e in bus.events))

    def test_large_offset_and_invalid_encoder_are_not_recovered(self):
        for position in (2111,5000,-1):
            with self.subTest(position=position):
                bus,axis,motor,policy,clock,telemetry=self.fixture(position=position)
                with self.assertRaises(RuntimeError):self.run_home(bus,axis,motor,policy,clock,telemetry)
                self.assertFalse(any(e[0]=='goal' for e in bus.events))

    def test_disabling_the_policy_keeps_the_original_strict_startup(self):
        bus,axis,motor,policy,clock,telemetry=self.fixture()
        with self.assertRaises(RuntimeError):self.run_home(bus,axis,motor,None,clock,telemetry)
        self.assertFalse(any(e[0]=='goal' for e in bus.events))

    def test_inconsistent_rechecks_do_not_arm(self):
        bus,axis,motor,policy,clock,telemetry=self.fixture()
        bus.read_positions=iter([2106,2090])
        with self.assertRaisesRegex(RuntimeError,'inconsistent'):
            self.run_home(bus,axis,motor,policy,clock,telemetry)
        self.assertFalse(any(e[0] in ('goal','write') for e in bus.events))

    def test_stable_small_recheck_jitter_uses_only_the_observed_corridor(self):
        for positions in ([2106,2107],[2105,2106]):
            with self.subTest(positions=positions):
                bus,axis,motor,policy,clock,telemetry=self.fixture()
                snapshot=bus.snapshot(21);bus.read_positions=iter(positions)
                plans,_=confirm_home_recoveries(bus,[snapshot],policy,lambda:None,[],telemetry)
                self.assertEqual(plans[21]['feedback_max'],max(2100,*positions))
                self.assertEqual(plans[21]['start_position'],positions[-1])
                self.assertFalse(any(e[0] in ('goal','write') for e in bus.events))

    def test_stale_goal_readback_never_enables_torque(self):
        bus,axis,motor,policy,clock,telemetry=self.fixture();bus.bad_prime=True
        with self.assertRaisesRegex(RuntimeError,'readback failed'):
            self.run_home(bus,axis,motor,policy,clock,telemetry)
        self.assertNotIn(('write',40,1),bus.events)

    def test_jammed_axis_has_a_bounded_reentry_window(self):
        bus,axis,motor,policy,clock,telemetry=self.fixture(jammed=True);policy['timeout_s']=1.
        with self.assertRaisesRegex(RuntimeError,'did not reenter|did not settle at neutral'):
            self.run_home(bus,axis,motor,policy,clock,telemetry)
        self.assertLess(clock.now,10+1.875*59/60+1.1)
        self.assertNotIn(('write',46,1000),bus.events)

    def test_unused_id9_can_finish_with_existing_neutral_encoder_tolerance(self):
        robot=Robot();axis=next(a for a in robot.home()['axes'] if a['id']==9)
        bus=ColdAxis(sid=9,position=2056,low=1900,high=2050)
        bus.auto_enable=True;bus.settle_offset=5
        clock=SimpleNamespace(now=10.)
        clock.monotonic=lambda:clock.now
        clock.sleep=lambda delay:setattr(clock,'now',clock.now+max(delay,.00001))
        report={}
        with patch('vivis.runtime.homing.time',clock):
            home_all(bus,[axis],{},lambda:None,report,max_speed=60,
                restore_speeds={9:0},telemetry_policy=robot.telemetry(),
                position_recovery=robot.home()['disabled_position_recovery'])
        self.assertTrue(report['completed']);self.assertEqual(bus.position,2052)
        self.assertEqual(bus.target,2047);self.assertEqual(bus.enabled,0)
        self.assertEqual(report['startup_neutral_residuals'][0]['id'],9)
        self.assertTrue(all(1900<=e[1]<=2050 for e in bus.events if e[0]=='goal'))

    def test_reentry_grace_includes_the_common_smooth_home_duration(self):
        bus,axis,motor,policy,clock,telemetry=self.fixture()
        bus.settle_offset=5
        # A long shared trajectory may still be holding the boundary target
        # when the old one-second-from-enable timer would have expired.
        from vivis.runtime.homing import smooth_home_targets
        duration,_=smooth_home_targets({21:2106},{21:2047},60)
        policy['timeout_s']=1.
        report=self.run_home(bus,axis,motor,policy,clock,telemetry)
        self.assertTrue(report['completed']);self.assertGreater(clock.now,10+duration)

    def test_snapshot_outlier_uses_confirmed_actual_position_for_priming(self):
        bus,axis,motor,policy,clock,telemetry=self.fixture(position=2047);bus.snapshot_position=2106
        report=self.run_home(bus,axis,motor,policy,clock,telemetry)
        self.assertTrue(report['completed'])
        self.assertEqual(next(e for e in bus.events if e[0]=='goal'),('goal',2047,60))

    def test_playback_feedback_limits_are_not_relaxed(self):
        bus,axis,motor,policy,clock,telemetry=self.fixture()
        with self.assertRaises(RuntimeError):check_feedback(21,bus.feedback(21),motor)

    def test_policy_is_validated_without_hardware_access(self):
        for policy in (True,{},dict(max_outside_counts=4095)):
            with self.assertRaises(ValueError):recovery_policy(policy)

    def test_explicit_larger_offset_is_scoped_to_the_configured_axis_only(self):
        bus,axis,motor,policy,clock,telemetry=self.fixture(sid=35,low=2040,high=2247,model=10760,position=1904)
        policy['axis_overrides']={'35':{'max_outside_counts':160,'timeout_s':5.}}
        report=self.run_home(bus,axis,motor,policy,clock,telemetry)
        self.assertTrue(report['completed']);self.assertEqual(bus.position,2047)
        self.assertEqual(next(e for e in bus.events if e[0]=='goal'),('goal',2040,60))
        other,axis,motor,_,clock,telemetry=self.fixture(position=2111)
        with self.assertRaises(RuntimeError):self.run_home(other,axis,motor,policy,clock,telemetry)
        self.assertFalse(any(e[0]=='goal' for e in other.events))

    def test_full_35_axis_preparation_recovers_the_observed_cold_start(self):
        self.full_preparation(set())

    def test_full_preparation_with_goal_enabled_id9_and_mixed_firmware(self):
        self.full_preparation({9,21,24,35})

    def test_enabled_off_center_tongue_is_centered_before_jaw_closes(self):
        self.full_preparation(set(),tongue_enabled=True)

    def test_failed_tongue_recovery_never_closes_jaw(self):
        with self.assertRaisesRegex(RuntimeError,'ID29 did not reach its authorized startup corridor'):
            self.full_preparation(set(),tongue_enabled=True,tongue_jammed=True)

    def full_preparation(self,auto_enable_ids,tongue_enabled=False,tongue_jammed=False):
        import numpy as np
        from vivis.runtime import show
        from vivis.hardware.feetech.protocol import Bus
        from unittest.mock import Mock
        robot=Robot();original=robot.profile('wz2');ids=original['diagnostic_command_ids']
        config=robot.tracking();profile,ids,trajectory=show.prepare(config,original,ids,
            np.full((2,len(ids)),2047.),True,1.,robot.telemetry())
        motors={m['id']:m for m in profile['motors']}
        inventory=robot.home();memory={};positions={};events=[]
        cold={9:2056,21:2106,29:2273,35:1904}
        if tongue_enabled:cold[29]=1800
        for sid in range(2,37):
            m=motors.get(sid,{})
            c=bytearray(40);r=bytearray(31)
            c[3:5]=(10760 if 31<=sid<=35 else 521).to_bytes(2,'little');c[5]=sid;c[18]=4
            c[9:11]=m.get('device_min',1900).to_bytes(2,'little')
            c[11:13]=m.get('device_max',2050).to_bytes(2,'little')
            c[13]=m.get('device_max_temperature_c',80)
            c[14]=m.get('max_voltage_raw',90);c[15]=m.get('min_voltage_raw',40)
            c[21:24]=bytes([32,32,0]);c[16:18]=(1000).to_bytes(2,'little')
            r[1]=10;r[8:10]=(1000).to_bytes(2,'little');r[15]=1
            if sid==29 and tongue_enabled:
                r[0]=1;r[2:4]=cold[29].to_bytes(2,'little')
            r[22]=238 if 31<=sid<=35 else 62;r[23]=35
            memory[sid]=(c,r);positions[sid]=cold.get(sid,2047)
        def read(sid,address,width):
            c,r=memory[sid];r[16:18]=positions[sid].to_bytes(2,'little')
            return bytes((c+r)[address:address+width])
        def write(sid,address,value,width):
            events.append(('write',sid,address,value));c,r=memory[sid]
            block,offset=(c,address) if address<40 else (r,address-40)
            block[offset:offset+width]=value.to_bytes(width,'little')
            if address==40 and value==1:positions[sid]=int.from_bytes(r[2:4],'little')
            return 0,0
        def goals(targets,speeds):
            for sid,target in targets.items():
                if sid==29 and target==2047 and abs(positions[29]-2047)>40:
                    self.assertLessEqual(abs(positions[31]-1900),20)
                if sid==31 and target==2047 and any(e[:3]==('goal',31,1900) for e in events):
                    self.assertLessEqual(abs(positions[29]-2047),40)
                c,r=memory[sid];lo=int.from_bytes(c[9:11],'little');hi=int.from_bytes(c[11:13],'little')
                self.assertTrue(lo<=target<=hi);self.assertLessEqual(speeds[sid],60)
                events.append(('goal',sid,target,speeds[sid]))
                r[2:4]=target.to_bytes(2,'little');r[6:8]=speeds[sid].to_bytes(2,'little')
                if sid in auto_enable_ids:r[0]=1
                if r[0] and not (sid==29 and tongue_jammed):positions[sid]=target
        bus=SimpleNamespace(read=read,goals=goals,close=Mock())
        bus.snapshot=lambda sid:Bus.snapshot(bus,sid)
        bus.feedback=lambda sid:dict(position=positions[sid],load_raw=0,temperature_c=35,
            voltage_raw=238 if 31<=sid<=35 else 62,status_register=0)
        bus.sdk=SimpleNamespace(write1ByteTxRx=lambda sid,a,v:write(sid,a,v,1),
                                write2ByteTxRx=lambda sid,a,v:write(sid,a,v,2))
        clock=SimpleNamespace(now=10.)
        clock.monotonic=lambda:clock.now
        clock.sleep=lambda duration:setattr(clock,'now',clock.now+max(duration,.00001))
        args=SimpleNamespace(usb_volume_times_three=False,home_only=False,home_speed=60,
            authorized_tongue_observation=True,home_position_recovery=inventory['disabled_position_recovery'])
        report={}
        with tempfile.TemporaryDirectory() as directory,patch('vivis.runtime.show.time',clock),\
             patch('vivis.runtime.homing.time',clock),\
             patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus):
            show.prepare_round(args,profile,ids,inventory['axes'],robot.startup(),
                show.recovery_clearance_plan(inventory,motors),robot.telemetry(),lambda:None,
                report,Path(directory),lambda *_:None)
        self.assertTrue(report['homing']['completed'])
        self.assertEqual({r['id'] for r in report['disabled_position_recoveries']},{9,21,35})
        self.assertTrue(report['recovery_clearance']['required'])
        self.assertTrue(all(p==2047 for p in positions.values()))
        # The simulated mechanism rejects tongue recovery before jaw clearance,
        # and jaw closure before the tongue has returned. Check the actual goals.
        opened=next(i for i,e in enumerate(events) if e[:3]==('goal',31,1900))
        tongue=next(i for i,e in enumerate(events) if e[:3]==('goal',29,2047))
        closed=next(i for i,e in enumerate(events) if i>opened and e[:3]==('goal',31,2047))
        self.assertLess(opened,tongue);self.assertLess(tongue,closed)
        for sid,first_goal in ((21,2100),(35,robot.axes[35]['device_min'])):
            self.assertEqual(next(e for e in events if e[:2]==('goal',sid)),('goal',sid,first_goal,60))
        self.assertFalse(any(e[0]=='write' and e[2] in (9,10,11,12,18) for e in events))
        for row in report['homing']['startup_primes']:
            self.assertEqual(row['enable_source'],'goal_update' if row['id'] in auto_enable_ids else 'explicit_write')
        for sid in (9,24):self.assertEqual(memory[sid][1][0],0)


if __name__=='__main__':unittest.main()
