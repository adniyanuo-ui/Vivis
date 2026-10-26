"""Developer modes, canonical configuration and owned-resource regression tests."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
from xiaobai_runtime.mapping import ROOT  # Legacy imports must share module identity.
from vivis.runtime.robot import Robot
from vivis.runtime.show import prepare
from vivis.motion.expression import ExpressionMapper, adapt_channels, fuse_layers


class DeveloperConfigurationTests(unittest.TestCase):
    def test_canonical_robot_preserves_accepted_motors_and_startup(self):
        robot = Robot()
        old = json.loads((ROOT/'mojia/0001_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1000_PROFILE.json').read_text())
        from vivis.runtime.telemetry import apply_policy
        apply_policy(old,robot.telemetry())
        self.assertEqual({m['id']:m for m in robot.profile('0001')['motors']}, {m['id']:m for m in old['motors']})
        self.assertEqual(robot.home()['axes'], json.loads((ROOT/'mojia/homing_axes.json').read_text())['axes'])
        self.assertEqual(robot.startup()['axes'], json.loads((ROOT/'mojia/startup_hardware.json').read_text())['axes'])
        self.assertEqual(robot.home()['recovery_clearance'], json.loads((ROOT/'mojia/homing_axes.json').read_text())['recovery_clearance'])
        self.assertEqual(robot.calibration.read_bytes(), (ROOT/'mojia/墨甲9.23.json').read_bytes())

    def test_legacy_and_new_imports_are_the_same_modules(self):
        import xiaobai_runtime.playback as legacy
        import vivis.runtime.playback as current
        import mojia_performance as old_show
        import vivis.runtime.show as new_show
        self.assertIs(legacy, current)
        self.assertIs(old_show, new_show)

    def test_new_hardware_value_is_used_by_all_runtime_views(self):
        robot = Robot()
        robot.axes[32]['max_speed_counts_s'] = 850
        robot.axes[32]['max_acceleration_counts_s2'] = 70
        robot.axes[32]['pid_raw'] = [16, 27, 0]
        self.assertEqual(next(m for m in robot.profile('wz2')['motors'] if m['id']==32)['max_speed_counts_s'],850)
        self.assertEqual(next(m for m in robot.startup()['axes'] if m['id']==32)['speed_raw'],850)
        self.assertEqual(robot.tracking()['neck']['speed_counts_s'],850)
        self.assertEqual(robot.tracking()['neck']['acceleration_counts_s2'],70)
        self.assertEqual(robot.tracking()['neck']['observed_pid_raw'],[16,27,0])

    def test_neutral_and_named_channels_match_the_final_calibration(self):
        mapper=ExpressionMapper(Robot())
        neutral=mapper.frame({})
        self.assertEqual(len(neutral),30)
        self.assertEqual(set(neutral.values()),{2047.})
        brow=mapper.frame({'BrowDownLeft':1.})
        self.assertEqual(brow[2],2300.)
        self.assertEqual(brow[3],2098.)
        self.assertEqual(brow[4],1954.)
        self.assertEqual(brow[31],2047.)
        self.assertEqual(neutral,mapper.frame({}))
        with self.assertRaises(ValueError):mapper.frame({'Unknown':1.})
        with self.assertRaises(ValueError):mapper.frame({'BrowDownLeft':float('nan')})

    def test_semantic_adaptation_rejects_collisions_and_fusion_is_explicit(self):
        names=['BrowDownLeft','JawOpen']
        self.assertEqual(adapt_channels({'jaw_open':.5},{'jaw_open':'JawOpen'},names),{'JawOpen':.5})
        with self.assertRaises(ValueError):adapt_channels({'a':.2,'b':.3},{'a':'JawOpen','b':'JawOpen'},names)
        base={'JawOpen':.4,'BrowDownLeft':.1}
        self.assertEqual(fuse_layers(base,[({'BrowDownLeft':.9},.5)],names),{'JawOpen':.4,'BrowDownLeft':.5})
        self.assertEqual(base,{'JawOpen':.4,'BrowDownLeft':.1})

    def test_all_modes_and_both_works_check_without_opening_devices(self):
        from vivis.runtime.developer import run_example
        with patch('vivis.hardware.feetech.sdk.OfficialSDKBus',side_effect=AssertionError('serial opened')), \
             patch('vivis.runtime.developer.subprocess.Popen',side_effect=AssertionError('camera started')):
            for mode in ('face','eyes','eyes-neck'):
                for work in ('0001','wz2'):
                    with self.subTest(mode=mode,work=work),redirect_stdout(io.StringIO()) as buffer:
                        run_example(mode,['--check','--work',work])
                        result=json.loads(buffer.getvalue())
                        self.assertEqual(len(result['home_ids']),35)
                        expected=[] if mode=='face' else [8,23] if mode=='eyes' else [8,23,32]
                        self.assertEqual(result['gaze_owner_ids'],expected)
                        self.assertEqual(set(result['commanded']) & {7,8,22,23,32,33,34},set(expected))


class DeveloperPlaybackTests(unittest.TestCase):
    def test_each_mode_sends_only_owned_axes_and_keeps_the_same_face_timeline(self):
        from test_xiaobai import PlaybackTests
        from vivis.motion.ownership import MojiaPlaybackGaze
        from vivis.runtime.playback import play
        from vivis.vision.policy import VERSION
        for mode in ('face','eyes','eyes-neck'):
            with self.subTest(mode=mode):
                robot=Robot();c=robot.tracking();c['control']['vertical_enabled']=False
                c['control']['axes']=[a for a in c['control']['axes'] if a['id'] not in (7,22)]
                c['control']['held_eye_ids']=[7,22]
                original=robot.profile('wz2');source_ids=original['diagnostic_command_ids']
                data=np.full((18,len(source_ids)),2047.)
                data[:,source_ids.index(31)]=2047-np.arange(18)*2
                p,ids,trajectory=prepare(c,original,source_ids,data,True,1.,robot.telemetry(),mode)
                p.update(control_hz=60,face_update_hz=60,changed_goals_only=False)
                _,bus,clock=PlaybackTests().fixture();clock.duration=.3
                motors={m['id']:m for m in p['motors']};positions={sid:2047 for sid in ids}
                def state(sid):
                    return dict(position=positions[sid],temperature_c=25,load_raw=0,status_register=0,
                                voltage_raw=120 if 31<=sid<=35 else 60)
                def snapshot(sid):
                    raw=bytearray(40);raw[18]=4;raw[21:24]=bytes(c['neck']['observed_pid_raw'])
                    raw[26:28]=bytes(c['neck']['observed_deadband_counts']);ram=bytearray(31)
                    return dict(id=sid,model_raw=motors[sid]['model_raw_expected'],mode=0,torque_enable=1,
                                configured_min=motors[sid]['device_min'],configured_max=motors[sid]['device_max'],
                                position_raw=2047,config_0_39_hex=raw.hex(),state_40_70_hex=ram.hex(),
                                **{k:v for k,v in state(sid).items() if k!='position'})
                def read(sid,address,count):
                    s=state(sid)
                    return {56:s['position'].to_bytes(2,'little'),60:b'\0\0',
                            62:bytes([s['voltage_raw'],25]),65:b'\0'}[address]
                bus.snapshot.side_effect=snapshot;bus.feedback.side_effect=state;bus.read.side_effect=read
                bus.goals.side_effect=lambda goals,speeds:positions.update(goals)
                receiver=SimpleNamespace(poll=lambda now:dict(schema_version=VERSION,sequence=1,
                    timestamp_monotonic=now,target=dict(x=.4,y=0.)))
                override=None if mode=='face' else MojiaPlaybackGaze(c,p,ids,receiver,neck=mode=='eyes-neck')
                with tempfile.TemporaryDirectory() as directory, \
                     patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus), \
                     patch('vivis.runtime.playback.AudioClock',return_value=clock):
                    result=play(p,ids,trajectory,60,'fake',Path(directory)/'log.json',Path(directory)/'STOP',gaze_override=override)
                self.assertTrue(result['completed'])
                allowed=set(p['diagnostic_command_ids'])
                for event in result['events']:
                    self.assertEqual(set(event['goals']),allowed)
                    self.assertEqual(event['goals'][31],round(trajectory[event['frame'],ids.index(31)]))
                self.assertEqual(result['runtime_feedback_schedule']['priority_ids'],
                                 [] if mode=='face' else [23] if mode=='eyes' else [23,32])
                bus.close.assert_called_once();clock.close.assert_called_once()


class DeveloperResourcesTests(unittest.TestCase):
    def test_face_mode_never_starts_or_connects_to_vision(self):
        from vivis.runtime.developer import vision_service
        with patch('vivis.runtime.developer.preview_state',side_effect=AssertionError('preview accessed')), \
             patch('vivis.runtime.developer.subprocess.Popen',side_effect=AssertionError('camera started')):
            with vision_service(Path('not-required'),Path('/tmp'),False):pass

    def test_owned_camera_child_is_cleaned_up_when_motion_raises(self):
        import hashlib
        from vivis.runtime.developer import vision_service
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json';config.write_text('{}')
            digest=hashlib.sha256(config.read_bytes()).hexdigest()
            child=Mock();child.poll.return_value=None
            with patch('vivis.runtime.developer.preview_state',side_effect=[None,{'config_sha256':digest}]), \
                 patch('vivis.runtime.developer.subprocess.Popen',return_value=child):
                with self.assertRaisesRegex(RuntimeError,'motion failed'):
                    with vision_service(config,root,True):raise RuntimeError('motion failed')
            child.terminate.assert_called_once();child.wait.assert_called_once_with(timeout=4)

    def test_different_existing_camera_config_is_rejected_without_stopping_it(self):
        from vivis.runtime.developer import vision_service
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json';config.write_text('{}')
            with patch('vivis.runtime.developer.preview_state',return_value={'config_sha256':'other'}), \
                 patch('vivis.runtime.developer.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError,'不同配置'):
                    with vision_service(config,root,True):pass
                spawn.assert_not_called()


class DeveloperImportTests(unittest.TestCase):
    def test_customer_30fps_import_preserves_source_and_matches_saved_mapping(self):
        import wave
        from vivis.motion.export import export_work
        from vivis.runtime.player import load_targets
        robot=Robot()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'works.json'
            manifest.write_text(json.dumps(dict(schema_version='vivis-works-1.0',works={})))
            robot.definition['works']=str(manifest);robot.works=json.loads(manifest.read_text())
            values=np.zeros((4,61));values[:,ExpressionMapper(robot).channels.index('BrowDownLeft')]=[0.,.25,.5,1.]
            clip=root/'source.json';clip.write_text(json.dumps(dict(name='test',fps={'num':30,'den':1},blendShapes=values.tolist())))
            before=clip.read_bytes();audio=root/'source.wav'
            with wave.open(str(audio),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(48000);wav.writeframes(bytes(6400*2))
            with patch('vivis.motion.export.ROOT',root),patch('vivis.runtime.robot.ROOT',root):
                entry=export_work(robot,'new_work',clip,audio)
                robot.works=json.loads(manifest.read_text())
                work=robot.work('new_work')
                ids,positions,fps=load_targets(work['targets'],robot.profile('new_work'),work['audio'],robot.calibration)
                self.assertEqual(fps,60);self.assertEqual(len(positions),8)
                np.testing.assert_array_equal(positions[::2],positions[1::2])
                self.assertEqual(positions[-1,ids.index(2)],2300)
                self.assertEqual(entry['frames'],8)
                self.assertEqual(clip.read_bytes(),before)
                with self.assertRaisesRegex(ValueError,'already exists'):export_work(robot,'new_work',clip,audio)
                with self.assertRaises(ValueError):export_work(robot,'../bad',clip,audio)

    def test_asset_import_rejects_corrupt_bundle_without_copying_it(self):
        from vivis.runtime.assets import verify_assets
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'bundle';source.mkdir()
            (source/'audio.wav').write_bytes(b'bad')
            target=root/'deploy/audio.wav'
            robot=SimpleNamespace(works={'works':{'x':{}}},work=lambda name:dict(audio=target,
                targets=root/'deploy/targets.json',audio_sha256='0'*64,targets_sha256='1'*64))
            with patch('vivis.runtime.assets.ROOT',root):
                with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):verify_assets(robot,source)
            self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()
