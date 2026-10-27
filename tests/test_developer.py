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
from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.show import prepare
from vivis.motion.expression import ExpressionMapper, adapt_channels, fuse_layers


class DeveloperConfigurationTests(unittest.TestCase):


    def test_clean_package_preserves_accepted_motion_baseline(self):
        import hashlib
        robot=Robot()
        baseline=json.loads((ROOT/'tests/fixtures/baseline.json').read_text())
        digest=lambda data:hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        motors={m['id']:m for m in robot.profile('0001')['motors']}
        # The cold-start readback is 0..4095. Only device identity metadata
        # changed; the accepted tracking stroke remains 1600..2600.
        self.assertEqual((motors[32]['device_min'],motors[32]['device_max']),(0,4095))
        self.assertEqual((motors[32].pop('tracking_min'),motors[32].pop('tracking_max')),(1600,2600))
        motors[32].update(device_min=1600,device_max=2600)
        for sid in (32,33,34):
            self.assertNotIn('max_acceleration_counts_s2',motors[sid])
            motors[sid]['max_acceleration_counts_s2']=64  # Historical baseline only.
        self.assertEqual(digest(motors),baseline['motors_sha256'])
        startup=robot.startup()['axes']
        neck=next(a for a in startup if a['id']==32)
        self.assertEqual(neck['device_limits'],[0,4095])
        # User replaced ID32 and requested the same PID/deadband as its peers.
        self.assertEqual(neck['pid_raw'],[32,32,0])
        self.assertEqual(neck['deadband_raw'],[0,0])
        self.assertEqual(neck['acceleration_raw'],128)
        neck['device_limits']=[1600,2600]
        neck.update(pid_raw=[16,26,0],deadband_raw=[5,5],acceleration_raw=0)
        self.assertEqual(digest(startup),baseline['startup_sha256'])
        self.assertEqual(digest(robot.home()['axes']),baseline['home_sha256'])
        self.assertEqual(robot.calibration_sha256,baseline['calibration_sha256'])

    def test_new_hardware_value_is_used_by_all_runtime_views(self):
        robot = Robot()
        robot.axes[32]['max_speed_counts_s'] = 850
        robot.axes[32]['acceleration_raw'] = 10
        robot.axes[32]['pid_raw'] = [16, 27, 0]
        self.assertEqual(next(m for m in robot.profile('wz2')['motors'] if m['id']==32)['max_speed_counts_s'],850)
        self.assertEqual(next(m for m in robot.startup()['axes'] if m['id']==32)['speed_raw'],850)
        self.assertEqual(robot.tracking()['neck']['speed_counts_s'],850)
        self.assertEqual(robot.tracking()['neck']['observed_acceleration_raw'],10)
        self.assertEqual(next(a for a in robot.startup()['axes'] if a['id']==32)['acceleration_raw'],10)
        self.assertNotIn('acceleration_counts_s2',robot.tracking()['neck'])
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
        from test_runtime import PlaybackTests
        from vivis.motion.ownership import MojiaPlaybackGaze
        from vivis.runtime.playback import play
        from vivis.vision.policy import VERSION
        for mode,adaptive in (('face',False),('eyes',False),('eyes-neck',False),('eyes-neck',True)):
            with self.subTest(mode=mode,adaptive=adaptive):
                robot=Robot();c=robot.tracking();c['control']['vertical_enabled']=False
                c['neck']['adaptive_speed']=adaptive
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
                    ram[1]=c['neck']['observed_acceleration_raw'] if sid==32 else 0
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
                if mode=='eyes-neck':
                    sent=bus.goals.call_args_list[-len(result['events']):]
                    for event,call in zip(result['events'],sent):
                        speed=call.args[1][32]
                        self.assertEqual(speed,event['person_gaze']['neck_speed_command'])
                        self.assertEqual(speed,event['speed_overrides'][32])
                        self.assertTrue(1<=speed<=1000)
                    self.assertEqual(any(call.args[1][32]<1000 for call in sent),adaptive)
                bus.close.assert_called_once();clock.close.assert_called_once()


class DeveloperResourcesTests(unittest.TestCase):
    def ready_state(self,config,digest):
        import time
        from vivis.vision.policy import VERSION
        return dict(config_sha256=digest,target_protocol_version=VERSION,
            camera_device=config['camera']['device'],camera_eye=config['control']['camera_eye'],
            camera_transform={k:config['camera'][k] for k in ('rotation_deg','mirror_x')},
            service_age_s=0.,frame_age_ms=10,status='idle')

    def test_vision_identity_rejects_old_protocol_stale_frames_and_orientation_mismatch(self):
        import time
        from vivis.runtime.developer import compatible_vision
        config=Robot().tracking();good=self.ready_state(config,'hash')
        now=time.monotonic()
        self.assertTrue(compatible_vision(good,config,'hash'))
        for patch_values in (dict(target_protocol_version='old'),dict(camera_transform={}),
                dict(camera_device='/dev/wrong'),dict(camera_eye='left'),dict(status='camera_stale'),
                dict(frame_age_ms=400),dict(service_age_s=2),dict(service_age_s=-1),
                dict(service_age_s=.2,frame_age_ms=200)):
            with self.subTest(patch_values=patch_values):
                self.assertFalse(compatible_vision({**good,**patch_values},config,'hash'))

    def test_launcher_accepts_the_real_public_http_state_without_internal_timestamp(self):
        from urllib.request import urlopen
        from vivis.vision.preview import Preview
        from vivis.runtime.developer import compatible_vision
        config=Robot().tracking();preview=Preview('127.0.0.1',0)
        try:
            preview.update(self.ready_state(config,'hash'))
            with urlopen(f'http://127.0.0.1:{preview.server.server_address[1]}/state',timeout=1) as response:
                state=json.load(response)
            self.assertNotIn('_updated_at',state)
            self.assertIn('service_age_s',state)
            self.assertTrue(compatible_vision(state,config,'hash'))
        finally:preview.close()

    def test_camera_only_service_publishes_the_identity_expected_by_the_launcher(self):
        import hashlib,time
        from vivis.vision import service
        from vivis.vision.policy import VERSION
        from vivis.runtime.developer import compatible_vision
        config=Robot().tracking();states=[]
        camera=Mock();camera.actual={};camera.get.side_effect=lambda:(1,time.monotonic(),None)
        detector=Mock();detector.detect.side_effect=lambda sample:(dict(schema_version=VERSION,
            sequence=1,timestamp_monotonic=sample[1],target=None),b'jpeg',dict(processing_ms=1.))
        preview=Mock();preview.update.side_effect=lambda state,jpeg=None:states.append({**state,'service_age_s':0.})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'config.json';path.write_text(json.dumps(config))
            with patch.object(service,'ROOT',root),patch.object(service,'LatestCamera',return_value=camera),\
                    patch.object(service,'FaceVision',return_value=detector),patch.object(service,'Preview',return_value=preview),\
                    patch.object(service.socket,'socket'),redirect_stdout(io.StringIO()):
                service.main(['--config',str(path),'--output',str(root/'vision'),'--seconds','.04'])
            self.assertTrue(states)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertTrue(compatible_vision(states[0],config,digest))
        camera.close.assert_called_once();preview.close.assert_called_once()
    def test_face_mode_never_starts_or_connects_to_vision(self):
        from vivis.runtime.developer import vision_service
        with patch('vivis.runtime.developer.preview_state',side_effect=AssertionError('preview accessed')), \
             patch('vivis.runtime.developer.subprocess.Popen',side_effect=AssertionError('camera started')):
            with vision_service(Path('not-required'),Path('/tmp'),False):pass

    def test_owned_camera_child_is_cleaned_up_when_motion_raises(self):
        import hashlib
        from vivis.runtime.developer import vision_service
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json';data=Robot().tracking();config.write_text(json.dumps(data))
            digest=hashlib.sha256(config.read_bytes()).hexdigest()
            child=Mock();child.poll.return_value=None
            with patch('vivis.runtime.developer.preview_state',side_effect=[None,self.ready_state(data,digest)]), \
                 patch('vivis.runtime.developer.subprocess.Popen',return_value=child):
                with self.assertRaisesRegex(RuntimeError,'motion failed'):
                    with vision_service(config,root,True):raise RuntimeError('motion failed')
            child.terminate.assert_called_once();child.wait.assert_called_once_with(timeout=4)

    def test_different_existing_camera_config_is_rejected_without_stopping_it(self):
        from vivis.runtime.developer import vision_service
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json';config.write_text(json.dumps(Robot().tracking()))
            with patch('vivis.runtime.developer.preview_state',return_value={'config_sha256':'other'}), \
                 patch('vivis.runtime.developer.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError,'不同配置'):
                    with vision_service(config,root,True):pass
                spawn.assert_not_called()


class DeveloperImportTests(unittest.TestCase):
    def test_customer_30fps_import_preserves_source_and_matches_saved_mapping(self):
        import wave
        from vivis.motion.export import export_work
        from vivis.runtime.trajectory import load_targets
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
