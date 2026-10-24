"""Neutral startup motion, temporary enables, limits and speed restoration."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from xiaobai_runtime.homing import home_all,smooth_home_targets,validate_home_axes


class HomingTests(unittest.TestCase):
    def test_home_only_covers_all35_without_starting_camera_audio_or_volume(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import Mock
        import numpy as np
        import mojia_performance as entry
        from test_mojia_gaze import profile
        from xiaobai_runtime.mapping import ROOT
        c=json.loads((ROOT/'mojia/person_gaze.json').read_text());p=profile()
        inventory=json.loads((ROOT/'mojia/homing_axes.json').read_text())
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
        def read_json(path):
            return copy.deepcopy(c if str(path)=='config' else p if str(path)=='profile' else inventory)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            output=root/'run'
            args=['show','--config','config','--profile','profile','--targets','targets',
                  '--output',str(output),'--home-only','--authorized-tongue-observation','--usb-volume-times-three']
            with patch.object(entry,'ROOT',root),patch.object(entry,'read_json',side_effect=read_json),\
                    patch.object(entry,'load_targets',return_value=(ids,np.full((3,len(ids)),2047.),60)),\
                    patch.object(entry,'home_all') as home,patch.object(entry,'recover_tongue') as tongue,\
                    patch.object(entry,'TargetReceiver') as camera,patch.object(entry,'configure_show_volume') as volume,\
                    patch.object(entry,'play') as play,patch('xiaobai_runtime.feetech_sdk.OfficialSDKBus',return_value=bus),\
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
        from xiaobai_runtime.homing import write_ram_verified
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
        with patch('xiaobai_runtime.homing.time',clock):
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
        with patch('xiaobai_runtime.homing.time',clock),self.assertRaisesRegex(RuntimeError,'Operator stop'):
            home_all(bus,axes,{},guard,report,max_speed=60)
        self.assertFalse(report['completed']);self.assertFalse(restores)
        self.assertEqual(enabled[18],0);self.assertEqual(speeds[4],60)

    def test_home_can_start_outside_tracking_stroke_but_inside_device_limits(self):
        from xiaobai_runtime.playback import check_feedback
        axes,bus,clock,pos,speeds,enabled,writes,restores=self.fixture();pos[4]=2240
        motor=dict(id=4,neutral=1800,safe_min=1780,safe_max=1820,max_temperature_c=80,max_load_raw=350,
                   min_voltage_raw=40,max_voltage_raw=90)
        with self.assertRaisesRegex(RuntimeError,'position'):
            check_feedback(4,bus.feedback(4),motor)
        report={}
        with patch('xiaobai_runtime.homing.time',clock):
            home_all(bus,axes,{4:motor},lambda:None,report,max_speed=60)
        self.assertTrue(report['completed']);self.assertEqual(pos[4],1800)
        first=next(goals[4] for _,goals,_ in writes if 4 in goals)
        self.assertGreater(first,2230)  # No jump/clipping to the 1820 tracking endpoint.
        self.assertEqual(motor['safe_max'],1820)

    def test_near_home_encoder_residual_never_widens_command_limits(self):
        from xiaobai_runtime.homing import home_check
        axes,bus,clock,pos,speeds,enabled,writes,restores=self.fixture()
        axes[0]['neutral']=2297;pos[4]=2302  # Device target maximum remains 2300.
        with self.assertRaisesRegex(RuntimeError,'outside device limits'):
            home_check(bus.feedback(4),bus.snapshot(4))
        report={}
        with patch('xiaobai_runtime.homing.time',clock):
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
        with patch('xiaobai_runtime.homing.time',clock),self.assertRaisesRegex(RuntimeError,'alarm'):
            home_all(bus,axes,{},lambda:None,{})
        self.assertFalse(writes)


if __name__=='__main__':unittest.main()
