"""Startup, homing and telemetry regressions; no real actuator access."""
import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
from vivis.runtime.robot import Robot
from vivis.runtime.homing import home_all,smooth_home_targets,validate_home_axes
from vivis.runtime.startup import startup_plan,restore_startup_settings

class StartupSettingsTests(unittest.TestCase):
    def fixture(self):
        import json
        from vivis.motion.mapping import ROOT
        preset=Robot().startup()
        inventory=Robot().home()['axes']
        config={};ram={};writes=[]
        for a in inventory:
            sid=a['id'];c=bytearray(40);r=bytearray(31)
            c[3:5]=a['model_raw_expected'].to_bytes(2,'little');c[18]=4
            c[9:11]=(1900).to_bytes(2,'little');c[11:13]=(2100).to_bytes(2,'little')
            c[21:24]=bytes([32,32,0]);r[0]=int(sid not in (9,24,29));r[1]=10
            r[2:4]=(2047).to_bytes(2,'little')
            config[sid]=c;ram[sid]=r
        for a in preset['axes']:
            c=config[a['id']];lo,hi=a['device_limits']
            c[9:11]=lo.to_bytes(2,'little');c[11:13]=hi.to_bytes(2,'little')
        def snapshot(sid):
            c=config[sid];r=ram[sid]
            return dict(id=sid,model_raw=int.from_bytes(c[3:5],'little'),mode=0,status_register=0,
                configured_min=int.from_bytes(c[9:11],'little'),configured_max=int.from_bytes(c[11:13],'little'),
                torque_enable=r[0],goal_position_raw=int.from_bytes(r[2:4],'little'),
                config_0_39_hex=c.hex(),state_40_70_hex=r.hex())
        def read(sid,address,width):
            block,offset=(config[sid],address) if address<40 else (ram[sid],address-40)
            return bytes(block[offset:offset+width])
        def write(sid,address,value,width):
            writes.append((sid,address,value,width))
            block,offset=(config[sid],address) if address<40 else (ram[sid],address-40)
            block[offset:offset+width]=value.to_bytes(width,'little')
            return 0,0
        bus=SimpleNamespace(read=read,snapshot=snapshot,sdk=SimpleNamespace(
            write1ByteTxRx=lambda sid,addr,value:write(sid,addr,value,1),
            write2ByteTxRx=lambda sid,addr,value:write(sid,addr,value,2)))
        return preset,bus,[snapshot(a['id']) for a in inventory],writes,config,ram

    def test_powercycle_restores_both_models_neck_and_stages_low_speed_without_motion(self):
        preset,bus,before,writes,config,ram=self.fixture();report={}
        speeds=restore_startup_settings(bus,preset,before,60,lambda:None,report)
        self.assertTrue(report['completed']);self.assertEqual(len(speeds),35)
        for sid in range(2,37):
            if sid in (9,24):
                self.assertEqual(ram[sid][1],10);self.assertEqual(speeds[sid],0)
                self.assertFalse(any(row[0]==sid for row in writes));continue
            self.assertEqual(ram[sid][1],0)
            self.assertEqual(int.from_bytes(ram[sid][6:8],'little'),60)
            self.assertEqual(speeds[sid],1200 if sid in (10,11,12,13,15,25,26,27,28,29,30,31,35,36) else 1000)
        self.assertEqual(list(config[32][21:24]),[16,26,0])
        self.assertEqual(list(config[32][26:28]),[5,5])
        self.assertTrue(all(addr in (21,22,23,26,27,41,46) for _,addr,_,_ in writes))
        self.assertTrue(all(bus.snapshot(s['id'])['torque_enable']==s['torque_enable']
                            and bus.snapshot(s['id'])['goal_position_raw']==s['goal_position_raw'] for s in before))
        self.assertEqual(writes[0][1],46)

    def test_repeated_startup_does_not_repeat_matching_writes(self):
        preset,bus,before,writes,_,_=self.fixture()
        restore_startup_settings(bus,preset,before,60,lambda:None,{})
        writes.clear()
        after=[bus.snapshot(s['id']) for s in before]
        restore_startup_settings(bus,preset,after,60,lambda:None,{})
        self.assertEqual(writes,[])

    def test_later_axis_mismatch_or_bad_setting_prevents_all_writes(self):
        for key,value in [('model_raw',999),('speed_raw',0),('pid_raw',[999,0,0])]:
            with self.subTest(key=key):
                preset,bus,before,writes,_,_=self.fixture()
                preset['axes'][-1][key]=value
                with self.assertRaises((RuntimeError,ValueError)):
                    restore_startup_settings(bus,preset,before,60,lambda:None,{})
                self.assertEqual(writes,[])

    def test_readback_mismatch_aborts_without_repeating_write(self):
        preset,bus,before,writes,_,_=self.fixture()
        bus.read=lambda sid,address,count:bytes(count)
        with self.assertRaisesRegex(RuntimeError,'could not be verified'):
            restore_startup_settings(bus,preset,before,60,lambda:None,{})
        self.assertEqual(len(writes),1)

    def test_operator_stop_prevents_first_write(self):
        preset,bus,before,writes,_,_=self.fixture()
        def stop():raise RuntimeError('Operator stop')
        with self.assertRaisesRegex(RuntimeError,'Operator stop'):
            restore_startup_settings(bus,preset,before,60,stop,{})
        self.assertEqual(writes,[])

    def test_missing_inventory_rejected(self):
        preset,bus,before,writes,_,_=self.fixture();preset['axes'].pop()
        with self.assertRaisesRegex(ValueError,'inventory'):
            restore_startup_settings(bus,preset,before,60,lambda:None,{})
        self.assertEqual(writes,[])

    def test_unexpected_goal_change_is_rejected(self):
        preset,bus,before,writes,_,ram=self.fixture()
        original=bus.snapshot
        def snapshot(sid):
            ram[sid][2:4]=(2050).to_bytes(2,'little')
            return original(sid)
        bus.snapshot=snapshot
        with self.assertRaisesRegex(RuntimeError,'unexpected torque/goal/config change'):
            restore_startup_settings(bus,preset,before,60,lambda:None,{})


class RecoveryClearanceTests(unittest.TestCase):
    def fixture(self):
        from unittest.mock import Mock
        from vivis.runtime.show import recovery_clearance_plan
        motors={29:dict(id=29,neutral=2047,safe_min=1447,safe_max=2047,device_min=1447,
                       device_max=2147,max_speed_counts_s=900,model_raw_expected=521,initial_settle_tolerance_counts=40),
                31:dict(id=31,neutral=2047,safe_min=1380,safe_max=2047,device_min=1380,
                       device_max=2050,max_speed_counts_s=900,model_raw_expected=10760)}
        inventory=dict(recovery_clearance=dict(recovering_id=29,moving_id=31,position=1900,
                       max_speed_counts_s=60,settle_tolerance_counts=20))
        states=[dict(id=29,torque_enable=0,position_raw=2230),dict(id=31,torque_enable=1,position_raw=2030)]
        return inventory,motors,states,Mock(),recovery_clearance_plan(inventory,motors)

    def test_disabled_tongue_outside_corridor_first_opens_jaw_at_small_speed(self):
        from vivis.runtime.show import prepare_recovery_clearance
        _,motors,states,bus,plan=self.fixture();report={}
        with patch('vivis.runtime.show.home_all') as home:
            prepare_recovery_clearance(bus,plan,states,motors,lambda:None,report,60)
            home.assert_called_once();args=home.call_args.args
            self.assertEqual(args[1][0]['id'],31);self.assertEqual(args[1][0]['neutral'],1900)
            self.assertEqual(args[2][31]['initial_settle_tolerance_counts'],20)
            self.assertEqual(args[5],60);self.assertEqual(home.call_args.kwargs['keep_enabled'],[31])
        self.assertTrue(report['required']);self.assertEqual(motors[31]['neutral'],2047)
        self.assertEqual(motors[29]['initial_settle_tolerance_counts'],40)

    def test_no_clearance_motion_for_enabled_or_already_centered_tongue(self):
        from vivis.runtime.show import prepare_recovery_clearance
        for torque,position in [(1,2050),(0,2047),(0,2087)]:
            _,motors,states,bus,plan=self.fixture();states[0].update(torque_enable=torque,position_raw=position)
            with patch('vivis.runtime.show.home_all') as home:
                prepare_recovery_clearance(bus,plan,states,motors,lambda:None,{},60)
                home.assert_not_called()

    def test_clearance_limits_are_validated_before_motion(self):
        from vivis.runtime.show import recovery_clearance_plan
        for field,value in [('position',1379),('position',2051),('max_speed_counts_s',901),
                            ('settle_tolerance_counts',41),('moving_id',29)]:
            with self.subTest(field=field,value=value):
                inventory,motors,*_=self.fixture();inventory['recovery_clearance'][field]=value
                with self.assertRaises(ValueError):recovery_clearance_plan(inventory,motors)


class TelemetryPolicyTests(unittest.TestCase):
    def setUp(self):
        from vivis.motion.mapping import ROOT
        import json
        self.policy=json.loads((ROOT/'configs/robots/mojia/telemetry.json').read_text())
        self.motor=dict(safe_min=1900,safe_max=2100,max_load_raw=800,max_temperature_c=100,
                        min_voltage_raw=40,max_voltage_raw=90,telemetry_policy=self.policy)
        self.state=dict(position=2047,load_raw=0,temperature_c=40,voltage_raw=60,status_register=0)

    def test_strict_threshold_boundaries_and_load_direction_bit(self):
        from vivis.runtime.playback import check_feedback
        for load,temp,kinds in [(800,80,set()),(801,80,{'load'}),(1023,80,{'load'}),
                                (0,81,{'temperature'}),(900,100,{'load','temperature'}),
                                (1024+800,80,set()),(1024+801,80,{'load'})]:
            with self.subTest(load=load,temp=temp):
                warnings=check_feedback(5,{**self.state,'load_raw':load,'temperature_c':temp},self.motor)
                self.assertEqual({w['kind'] for w in warnings},kinds)

    def test_single_165_degree_sample_recovers_without_motion_writes(self):
        from unittest.mock import Mock,patch
        from vivis.runtime.playback import check_feedback
        from vivis.runtime.telemetry import verify_sample
        hot={**self.state,'temperature_c':165};cold={**self.state,'temperature_c':43}
        reader=Mock(return_value=cold);records=[]
        with patch('vivis.runtime.telemetry.time.sleep'):
            state,warnings=verify_sample(reader,28,hot,lambda f:check_feedback(28,f,self.motor),self.policy,records)
        self.assertEqual(state,cold);self.assertEqual(warnings,[]);reader.assert_called_once_with(28)
        self.assertTrue(records[0]['recovered']);self.assertEqual(records[0]['consecutive_high_samples'],0)
        self.assertEqual(records[0]['samples'][0]['state']['temperature_c'],165)

    def test_four_consecutive_high_samples_stop_and_no_fifth_read(self):
        from unittest.mock import Mock,patch
        from vivis.runtime.playback import check_feedback
        from vivis.runtime.telemetry import verify_sample,CriticalTemperatureError
        hot={**self.state,'temperature_c':101};reader=Mock(return_value=hot);records=[]
        with patch('vivis.runtime.telemetry.time.sleep'),self.assertRaisesRegex(CriticalTemperatureError,'4 consecutive'):
            verify_sample(reader,28,hot,lambda f:check_feedback(28,f,self.motor),self.policy,records)
        self.assertEqual(reader.call_count,3);self.assertEqual(len(records[0]['samples']),4)
        self.assertEqual(records[0]['consecutive_high_samples'],4);self.assertFalse(records[0]['recovered'])

    def test_100_degrees_resets_streak_after_three_high_samples(self):
        from unittest.mock import Mock,patch
        from vivis.runtime.playback import check_feedback
        from vivis.runtime.telemetry import verify_sample
        hot={**self.state,'temperature_c':165};boundary={**self.state,'temperature_c':100}
        reader=Mock(side_effect=[hot,hot,boundary]);records=[]
        with patch('vivis.runtime.telemetry.time.sleep'):
            state,warnings=verify_sample(reader,28,hot,lambda f:check_feedback(28,f,self.motor),self.policy,records)
        self.assertEqual(state['temperature_c'],100);self.assertEqual(warnings[0]['kind'],'temperature')
        self.assertTrue(records[0]['recovered']);self.assertEqual(records[0]['consecutive_high_samples'],0)

    def test_corrupt_timeout_and_servo_alarm_are_not_counted_as_hot_or_healthy(self):
        from unittest.mock import Mock,patch
        from vivis.runtime.playback import check_feedback
        from vivis.hardware.feetech.protocol import CorruptResponse,ReadTimeout,ServoStatusError
        from vivis.runtime.telemetry import verify_sample
        hot={**self.state,'temperature_c':165}
        for fault in [CorruptResponse('bad'),ReadTimeout('missing'),ServoStatusError(28,4)]:
            reader=Mock(side_effect=fault);records=[]
            with patch('vivis.runtime.telemetry.time.sleep'),self.assertRaises(type(fault)):
                verify_sample(reader,28,hot,lambda f:check_feedback(28,f,self.motor),self.policy,records)
            self.assertEqual(reader.call_count,1);self.assertEqual(records[0]['consecutive_high_samples'],1)
            self.assertFalse(records[0]['recovered'])
        reader=Mock(return_value={**hot,'status_register':4});records=[]
        with patch('vivis.runtime.telemetry.time.sleep'),self.assertRaises(ServoStatusError):
            verify_sample(reader,28,hot,lambda f:check_feedback(28,f,self.motor),self.policy,records)
        self.assertEqual(records[0]['consecutive_high_samples'],1)

    def test_stop_during_verification_prevents_followup_reads(self):
        from unittest.mock import Mock,patch
        from vivis.runtime.playback import check_feedback
        from vivis.runtime.telemetry import verify_sample
        reader=Mock();stop=Mock(side_effect=[False,True]);records=[]
        with patch('vivis.runtime.telemetry.time.sleep'),self.assertRaisesRegex(RuntimeError,'Operator stop'):
            verify_sample(reader,28,{**self.state,'temperature_c':165},lambda f:check_feedback(28,f,self.motor),
                          self.policy,records,should_stop=stop)
        reader.assert_not_called()

    def test_critical_temperature_and_other_faults_still_stop(self):
        from vivis.runtime.playback import check_feedback
        from vivis.runtime.telemetry import CriticalTemperatureError
        with self.assertRaises(CriticalTemperatureError):
            check_feedback(5,{**self.state,'temperature_c':101},self.motor)
        for change in ({'position':4096},{'position':1800},{'status_register':32},{'voltage_raw':100}):
            with self.subTest(change=change),self.assertRaises(RuntimeError):
                check_feedback(5,{**self.state,**change},self.motor)

    def test_homing_uses_same_temperature_policy_including_unmapped_axes(self):
        from vivis.runtime.homing import home_check
        from vivis.runtime.telemetry import CriticalTemperatureError
        snapshot=dict(id=9,configured_min=1900,configured_max=2100,configured_max_temperature=80,
                      configured_min_voltage_raw=40,configured_max_voltage_raw=90)
        warnings=home_check({**self.state,'temperature_c':100},snapshot,2047,20,self.policy)
        self.assertEqual(warnings[0]['kind'],'temperature')
        with self.assertRaises(CriticalTemperatureError):
            home_check({**self.state,'temperature_c':101},snapshot,2047,20,self.policy)
        with self.assertRaisesRegex(RuntimeError,'hardware temperature'):
            home_check({**self.state,'temperature_c':81},snapshot,2047,20)

    def test_warning_recorder_coalesces_persistent_warnings_and_records_clearance(self):
        from unittest.mock import patch
        from vivis.runtime.telemetry import WarningRecorder,evaluate
        records=[];recorder=WarningRecorder(records);high={**self.state,'load_raw':900}
        with patch('builtins.print') as output:
            for _ in range(5):recorder.observe(5,high,evaluate(5,high,self.policy))
            self.assertEqual(output.call_count,1)
            self.assertEqual(len(records),1);self.assertEqual(records[0]['count'],5)
            lower={**self.state,'load_raw':801}
            recorder.observe(5,lower,evaluate(5,lower,self.policy))
            self.assertEqual(records[0]['first_value'],900)
            self.assertEqual(records[0]['peak_value'],900);self.assertEqual(records[0]['value'],801)
            recorder.observe(5,self.state,[])
            self.assertTrue(records[0]['cleared']);self.assertFalse(recorder.active)
            recorder.observe(5,high,evaluate(5,high,self.policy))
            self.assertEqual(len(records),2)


class HomingTests(unittest.TestCase):
    def test_homing_rechecks_single_hot_sample_for_inventory_axes(self):
        import json
        from vivis.motion.mapping import ROOT
        axes,bus,clock,pos,speeds,enabled,writes,restores=self.fixture()
        original=bus.feedback;injected=False
        def feedback(sid):
            nonlocal injected
            state=original(sid)
            if sid==4 and not injected:
                injected=True;return {**state,'temperature_c':165}
            return state
        bus.feedback=feedback;report={}
        policy=json.loads((ROOT/'configs/robots/mojia/telemetry.json').read_text())
        with patch('vivis.runtime.homing.time',clock),patch('vivis.runtime.telemetry.time.sleep'):
            home_all(bus,axes,{},lambda:None,report,max_speed=60,telemetry_policy=policy)
        self.assertTrue(report['completed'])
        self.assertEqual(len(report['sensor_confirmations']),1)
        self.assertTrue(report['sensor_confirmations'][0]['recovered'])
        self.assertEqual(report['sensor_confirmations'][0]['samples'][0]['state']['temperature_c'],165)

    def test_home_only_covers_all35_without_starting_camera_audio_or_volume(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import Mock
        import numpy as np
        import vivis.runtime.show as entry
        from vivis.motion.mapping import ROOT
        c=Robot().tracking();p=Robot().profile('wz2')
        inventory=Robot().home()
        self.assertEqual({a['id'] for a in inventory['axes']},set(range(2,37)))
        self.assertTrue(all(a['neutral']==2047 for a in inventory['axes']))
        motors={m['id']:m for m in p['motors']}
        bus=Mock();bus.sdk.write1ByteTxRx.return_value=(0,0);bus.read.return_value=b'\x00'
        def snapshot(sid):
            m=motors.get(sid,{});raw=bytearray(40);raw[18]=4;ram=bytearray(31)
            # Deliberately non-playback speeds: home-only has no preset900 prerequisite.
            ram[6:8]=(100).to_bytes(2,'little')
            return dict(id=sid,model_raw=m.get('model_raw_expected',521),mode=0,
                torque_enable=0 if sid in (9,24,29) else 1,position_raw=2047,goal_position_raw=2047,
                configured_min=m.get('device_min',1900),configured_max=m.get('device_max',2050),
                configured_max_temperature=80,configured_min_voltage_raw=0,configured_max_voltage_raw=254,
                temperature_c=30,voltage_raw=120 if 31<=sid<=35 else 60,load_raw=0,status_register=0,
                config_0_39_hex=raw.hex(),state_40_70_hex=ram.hex())
        bus.snapshot.side_effect=snapshot
        ids=[m['id'] for m in p['motors'] if m['id'] not in (32,33,34)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            output=root/'run'
            args=['show','--robot',str(Robot().path),'--work','wz2',
                  '--output',str(output),'--home-only','--authorized-tongue-observation','--usb-volume-times-three']
            with patch.object(entry,'ROOT',root),\
                    patch.object(entry,'load_targets',return_value=(ids,np.full((3,len(ids)),2047.),60)),\
                    patch.object(entry,'home_all') as home,patch.object(entry,'recover_tongue') as tongue,\
                    patch.object(entry,'TargetReceiver') as camera,patch.object(entry,'configure_show_volume') as volume,\
                    patch.object(entry,'play') as play,patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus),\
                    patch('sys.argv',args):
                entry.main()
                home.assert_called_once();self.assertEqual(len(home.call_args.args[1]),35)
                self.assertEqual(home.call_args.args[5],60)
                self.assertEqual(tongue.call_args.kwargs['speed'],60)
                camera.assert_not_called();volume.assert_not_called();play.assert_not_called()
            result=json.loads((output/'summary.json').read_text());self.assertTrue(result['completed'])

    def fixture(self):
        clock=SimpleNamespace(now=10.)
        clock.monotonic=lambda:clock.now
        clock.sleep=lambda delay:setattr(clock,'now',clock.now+max(delay,.000001))
        axes=[dict(id=4,neutral=1800,neutral_confirmed=True,model_raw_expected=521),
              dict(id=18,neutral=2150,neutral_confirmed=True,model_raw_expected=521)]
        positions={4:1940,18:2100};speeds={4:900,18:0};enabled={4:1,18:0};writes=[];restores=[]
        bus=SimpleNamespace()
        def snapshot(sid):
            raw=bytearray(40);raw[18]=4;ram=bytearray(31);ram[6:8]=speeds[sid].to_bytes(2,'little')
            return dict(id=sid,model_raw=521,mode=0,torque_enable=enabled[sid],position_raw=positions[sid],
                configured_min=1600,configured_max=2300,configured_max_temperature=80,
                configured_min_voltage_raw=40,configured_max_voltage_raw=90,
                temperature_c=30,voltage_raw=60,load_raw=0,status_register=0,
                config_0_39_hex=raw.hex(),state_40_70_hex=ram.hex())
        bus.snapshot=snapshot
        def goals(targets,rates):
            writes.append((clock.now,copy.deepcopy(targets),copy.deepcopy(rates)))
            for sid,goal in targets.items():
                speeds[sid]=rates[sid]
                if enabled[sid]:positions[sid]=goal
        bus.goals=goals
        bus.feedback=lambda sid:dict(position=positions[sid],temperature_c=30,voltage_raw=60,load_raw=0,status_register=0)
        # All completed fixture motions follow their goal immediately.
        bus.read=lambda sid,address,count:(bytes([enabled[sid]]) if address==40 else
            positions[sid].to_bytes(2,'little') if address==42 else speeds[sid].to_bytes(2,'little'))
        def torque(sid,address,value):enabled[sid]=value;return 0,0
        def speed(sid,address,value):speeds[sid]=value;restores.append((sid,address,value));return 0,0
        bus.sdk=SimpleNamespace(write1ByteTxRx=torque,write2ByteTxRx=speed)
        return axes,bus,clock,positions,speeds,enabled,writes,restores

    def test_curve_is_smooth_bounded_and_supports_different_neutrals(self):
        start={4:1900.,18:2100.};neutral={4:1800.,18:2150.}
        duration,at=smooth_home_targets(start,neutral,60)
        self.assertEqual(at(0),start);self.assertEqual(at(duration),neutral)
        dt=.001
        for i in range(int(duration/dt)):
            a,b=at(i*dt),at((i+1)*dt)
            for sid in start:
                self.assertLessEqual(abs(b[sid]-a[sid])/dt,60.001)
        self.assertLess(abs(at(.001)[4]-start[4])/.001,.001)
        self.assertLess(abs(at(duration-.001)[4]-neutral[4])/.001,.001)

    def test_lost_write_ack_is_verified_without_repeating_write(self):
        from unittest.mock import Mock
        from vivis.runtime.homing import write_ram_verified
        bus=Mock();bus.sdk.write2ByteTxRx.return_value=(-6,0);bus.read.return_value=b'\x84\x03'
        row=write_ram_verified(bus,4,46,900,2)
        bus.sdk.write2ByteTxRx.assert_called_once_with(4,46,900)
        self.assertEqual(bus.read.call_count,2);self.assertFalse(row['write_retried'])
        bus.read.return_value=b'\x2c\x01'
        with self.assertRaisesRegex(RuntimeError,'could not be verified'):
            write_ram_verified(bus,4,46,900,2)
        bus.read.return_value=b'\x84\x03';bus.sdk.write2ByteTxRx.return_value=(0,4)
        with self.assertRaisesRegex(RuntimeError,'could not be verified'):
            write_ram_verified(bus,4,46,900,2)

    def test_home_uses_inventory_and_restores_speed_and_disabled_state(self):
        axes,bus,clock,pos,speeds,enabled,writes,restores=self.fixture();report={}
        with patch('vivis.runtime.homing.time',clock):
            home_all(bus,axes,{},lambda:None,report,max_speed=60)
        self.assertTrue(report['completed']);self.assertTrue(report['restored_speeds'])
        self.assertEqual(pos,{4:1800,18:2150});self.assertEqual(enabled,{4:1,18:0})
        self.assertEqual(speeds,{4:900,18:0})
        self.assertTrue(all(v==60 for _,_,rates in writes for v in rates.values()))
        self.assertTrue(all(address==46 for _,address,_ in restores))
        self.assertTrue(all(1600<=v<=2300 for _,goals,_ in writes for v in goals.values()))

    def test_stop_keeps_low_speed_and_restores_temporary_enable(self):
        axes,bus,clock,pos,speeds,enabled,writes,restores=self.fixture();report={}
        def guard():
            if len(writes)>=5:raise RuntimeError('Operator stop')
        with patch('vivis.runtime.homing.time',clock),self.assertRaisesRegex(RuntimeError,'Operator stop'):
            home_all(bus,axes,{},guard,report,max_speed=60)
        self.assertFalse(report['completed']);self.assertFalse(restores)
        self.assertEqual(enabled[18],0);self.assertEqual(speeds[4],60)

    def test_home_can_start_outside_tracking_stroke_but_inside_device_limits(self):
        from vivis.runtime.playback import check_feedback
        axes,bus,clock,pos,speeds,enabled,writes,restores=self.fixture();pos[4]=2240
        motor=dict(id=4,neutral=1800,safe_min=1780,safe_max=1820,max_temperature_c=80,max_load_raw=350,
                   min_voltage_raw=40,max_voltage_raw=90)
        with self.assertRaisesRegex(RuntimeError,'position'):
            check_feedback(4,bus.feedback(4),motor)
        report={}
        with patch('vivis.runtime.homing.time',clock):
            home_all(bus,axes,{4:motor},lambda:None,report,max_speed=60)
        self.assertTrue(report['completed']);self.assertEqual(pos[4],1800)
        first=next(goals[4] for _,goals,_ in writes if 4 in goals)
        self.assertGreater(first,2230)  # No jump/clipping to the 1820 tracking endpoint.
        self.assertEqual(motor['safe_max'],1820)

    def test_near_home_encoder_residual_never_widens_command_limits(self):
        from vivis.runtime.homing import home_check
        axes,bus,clock,pos,speeds,enabled,writes,restores=self.fixture()
        axes[0]['neutral']=2297;pos[4]=2302  # Device target maximum remains 2300.
        with self.assertRaisesRegex(RuntimeError,'outside device limits'):
            home_check(bus.feedback(4),bus.snapshot(4))
        report={}
        with patch('vivis.runtime.homing.time',clock):
            home_all(bus,axes,{},lambda:None,report,max_speed=60)
        self.assertTrue(report['completed']);self.assertEqual(pos[4],2297)
        self.assertTrue(all(1600<=p<=2300 for _,goals,_ in writes for p in goals.values()))
        state=bus.feedback(4);state['position']=2400
        with self.assertRaisesRegex(RuntimeError,'outside device limits'):
            home_check(state,bus.snapshot(4),2297,20)

    def test_invalid_inventory_or_actual_alarm_stops_before_motion(self):
        axes,bus,clock,pos,speeds,enabled,writes,restores=self.fixture()
        wrong=copy.deepcopy(axes);wrong[1]['neutral']=2400
        with self.assertRaisesRegex(RuntimeError,'outside device limits'):
            validate_home_axes(wrong,[bus.snapshot(a['id']) for a in axes])
        before=bus.snapshot
        def alarm(sid):return {**before(sid),'status_register':4}
        bus.snapshot=alarm
        with patch('vivis.runtime.homing.time',clock),self.assertRaisesRegex(RuntimeError,'alarm'):
            home_all(bus,axes,{},lambda:None,{})
        self.assertFalse(writes)
