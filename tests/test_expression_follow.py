"""Face Landmarker semantics, held-axis ownership, realtime loss and clock guards."""
from contextlib import redirect_stdout
import csv
import fcntl
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vivis.paths import ROOT
from vivis.interaction.expression import MediaPipeAdapter,ExpressionFrames
from vivis.interaction.follow import plan,main,MonotonicClock
from vivis.runtime.robot import Robot


def names():
    with (ROOT/'configs/schemas/blendshape_channels_61.csv').open() as f:return next(csv.reader(f))[2:]


def settings():return json.loads((ROOT/'configs/interaction/expression_follow.json').read_text())


def observation(sequence=1,stamp=1.,**weights):
    raw={n[0].lower()+n[1:]:0. for n in names()[:52] if n!='TongueOut'}
    raw.update(_neutral=.9,**weights)
    return dict(schema_version='vivis-face-expression-1.0',valid=True,landmark_count=478,
                sequence=sequence,timestamp_monotonic=stamp,weights=raw)


class AdapterTests(unittest.TestCase):
    def test_neutral_is_not_tongue_and_named_coefficients_are_not_indices(self):
        raw=observation(eyeBlinkLeft=.8,jawOpen=.4,mouthPucker=.7)
        a=MediaPipeAdapter(names()[:52]);mapped=a.adapt(raw)
        self.assertEqual(mapped[names().index('TongueOut')],0.)
        for n,v in [('EyeBlinkLeft',.8),('JawOpen',.4),('MouthPucker',.7)]:self.assertEqual(mapped[names().index(n)],v)
        raw['weights']=dict(reversed(list(raw['weights'].items())))
        np.testing.assert_array_equal(mapped,a.adapt(raw))

    def test_optional_mirror_swaps_pairs_but_not_central_channels(self):
        result=MediaPipeAdapter(names()[:52],True).adapt(observation(eyeBlinkLeft=.8,jawOpen=.4,mouthLeft=.2))
        self.assertEqual(result[names().index('EyeBlinkRight')],.8)
        self.assertEqual(result[names().index('EyeBlinkLeft')],0.)
        self.assertEqual(result[names().index('MouthRight')],.2)
        self.assertEqual(result[names().index('JawOpen')],.4)

    def test_bad_binding_and_nonfinite_values_rejected(self):
        for key,value in [('_neutral',float('nan')),('jawOpen',1.01),('eyeBlinkLeft',True)]:
            raw=observation();raw['weights'][key]=value
            with self.assertRaises(ValueError):MediaPipeAdapter(names()[:52]).adapt(raw)
        for key in ('_neutral','mouthPucker'):
            raw=observation();del raw['weights'][key]
            with self.assertRaises(ValueError):MediaPipeAdapter(names()[:52]).adapt(raw)
        for delta in [dict(valid=False),dict(landmark_count=477)]:
            with self.assertRaises(ValueError):MediaPipeAdapter(names()[:52]).adapt({**observation(),**delta})


class SourceTests(unittest.TestCase):
    def fixture(self):
        mapper=SimpleNamespace(order=names(),ids=list(range(52)),map_weights=lambda w:np.array(w))
        receiver=SimpleNamespace(max_age=.3,packet=None)
        receiver.poll=lambda now:receiver.packet
        return receiver,ExpressionFrames(receiver,mapper,settings())

    def packet(self,raw):
        return dict(sequence=raw['sequence'],timestamp_monotonic=raw['timestamp_monotonic'],
                    target=dict(x=0.,y=0.),face_expression=raw)

    def test_latest_only_filters_without_new_packets_and_stale_capture_cannot_refresh(self):
        receiver,source=self.fixture();receiver.packet=self.packet(observation(jawOpen=1.))
        source.sample(0.,1.)
        first=source.sample(.05,1.05)[names().index('JawOpen')]
        second=source.sample(.1,1.1)[names().index('JawOpen')]
        self.assertTrue(0<first<second<1.)
        for t in np.arange(1.2,2.5,.05):last=source.sample(t-1,t)
        self.assertLess(last.max(),1e-4);self.assertEqual(source.valid_observations,1)
        self.assertEqual(source.state,'neutral')

    def test_future_ambiguous_malformed_or_mismatched_observation_cannot_move_face(self):
        cases=[]
        for stamp in (2.,0.):cases.append(self.packet(observation(stamp=stamp,jawOpen=1.)))
        packet=self.packet(observation(jawOpen=1.));packet['target']=None;cases.append(packet)
        packet=self.packet(observation(jawOpen=1.));packet['face_expression']['landmark_count']=0;cases.append(packet)
        packet=self.packet(observation(jawOpen=1.));packet['face_expression']['timestamp_monotonic']=1.01;cases.append(packet)
        for packet in cases:
            receiver,source=self.fixture();receiver.packet=packet
            np.testing.assert_array_equal(source.sample(0.,1.),np.zeros(52))
            np.testing.assert_array_equal(source.sample(.05,1.05),np.zeros(52))
            self.assertEqual(source.valid_observations,0)

    def test_reacquisition_and_gain_remain_bounded(self):
        receiver,source=self.fixture();source.gain=2.
        receiver.packet=self.packet(observation(jawOpen=.8));source.sample(0.,1.)
        self.assertLessEqual(source.sample(.1,1.1).max(),1.)
        receiver.packet=self.packet({**observation(sequence=2,stamp=1.15), 'valid':False})
        source.sample(.15,1.15)
        receiver.packet=self.packet(observation(sequence=3,stamp=1.2,eyeBlinkRight=.8))
        result=source.sample(.2,1.2)
        self.assertGreater(result[names().index('EyeBlinkRight')],0.)
        self.assertEqual(source.valid_observations,2)
        self.assertGreater(source.rejected_observations,0)


class OwnershipTests(unittest.TestCase):
    def test_camera_handshake_requires_the_expression_feature(self):
        from vivis.runtime.developer import compatible_vision
        from vivis.vision.policy import VERSION
        config=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),settings(),1.)['tracking']
        state=dict(config_sha256='digest',target_protocol_version=VERSION,
            camera_transform={k:config['camera'][k] for k in ('rotation_deg','mirror_x')},
            camera_device=config['camera']['device'],camera_eye=config['control'].get('camera_eye'),
            service_age_s=.01,frame_age_ms=20,status='tracking',target_features=['face_xy','face_expression'])
        self.assertTrue(compatible_vision(state,config,'digest'))
        state['target_features']=['face_xy']
        self.assertFalse(compatible_vision(state,config,'digest'))

    def test_both_heads_prepare_and_command_only_face_eyelid_axes(self):
        for head,count in [('xiaoshuai',25),('mojia',26)]:
            robot=Robot(ROOT/f'configs/robots/{head}/robot.json')
            files={p:p.read_bytes() for p in robot.directory.glob('*.json')}
            with patch.object(Robot,'work',side_effect=AssertionError('music lookup')):
                p=plan(robot,settings(),1.)
            self.assertEqual(len(p['ids']),count)
            self.assertFalse(set(p['ids'])&{7,8,22,23,32,33,34})
            self.assertTrue({5,6,20,21}<=set(p['ids']))
            self.assertEqual(p['ids'],p['profile']['diagnostic_command_ids'])
            self.assertEqual(set(p['ids']),{m['id'] for m in p['profile']['motors']})
            self.assertEqual(set(p['ids']),{a['id'] for a in p['home']['axes']})
            preset=robot.startup(active_ids=p['ids'],inventory_ids=p['ids'])
            self.assertEqual(set(p['ids']),{a['id'] for a in preset['axes']})
            for file,raw in files.items():self.assertEqual(file.read_bytes(),raw)

    def test_offline_no_hardware_audio_media_or_model(self):
        original=Path.read_bytes
        def guarded(path,*args,**kwargs):
            if path.suffix.lower() in ('.wav','.task') or path.name in ('source.json','targets.json'):
                raise AssertionError('media/model read')
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_bytes',guarded),patch('vivis.runtime.developer.subprocess.Popen',side_effect=AssertionError('device/model opened')), \
             patch('vivis.hardware.feetech.sdk.OfficialSDKBus',side_effect=AssertionError('serial opened')), \
             redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['--check']),0)
        self.assertTrue(json.loads(out.getvalue())['offline_valid'])

    def test_invalid_settings_fail_before_devices(self):
        for key,values in [('gain',[float('nan'),0.,2.1,True]),('filter_tau_s',[0.,-1.]),
                           ('release_s',[0.]),('mirror_expression',[1,'on'])]:
            for value in values:
                config=settings();config[key]=value
                with self.assertRaises(ValueError):plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),config,1.)

    def test_busy_control_lock_never_opens_camera_bus_or_overwrites_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'outputs/developer').mkdir(parents=True)
            (root/'outputs/runtime').mkdir();status=root/'outputs/runtime/control_status.json'
            status.write_text('original')
            config=settings();config['perception']=str(ROOT/config['perception'])
            p=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),config,1.)
            with (root/'outputs/developer/control.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch('vivis.interaction.follow.ROOT',root),patch('vivis.interaction.follow.plan',return_value=p), \
                     patch('vivis.interaction.follow.check_competitors'),patch('vivis.interaction.follow.vision_service') as camera:
                    # Runtime deps are already checked elsewhere; provide harmless placeholders.
                    for key in ('python','model'):
                        f=root/p['perception'][key];f.parent.mkdir(parents=True,exist_ok=True);f.touch()
                    with self.assertRaises(BlockingIOError):main(['--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),
                        '--config',str(ROOT/'configs/interaction/expression_follow.json')])
                    camera.assert_not_called()
                self.assertEqual(status.read_text(),'original')


class ClockTests(unittest.TestCase):
    def test_monotonic_clock_has_no_audio_lifecycle(self):
        clock=MonotonicClock(1.)
        self.assertIsNone(clock.time())
        with patch('vivis.interaction.follow.time.monotonic',side_effect=[10.,10.2]):
            clock.start();self.assertAlmostEqual(clock.time(),.2)
        clock.close();self.assertIsNone(clock.fault)

    def test_shared_playback_without_audio_still_checks_goals_and_closes_bus(self):
        from test_runtime import PlaybackTests
        from vivis.runtime.playback import play
        for goals,good in [([2050.],True),([float('nan')],False),([5000.],False)]:
            profile,bus,clock=PlaybackTests().fixture()
            profile.pop('audio_device');profile.pop('physical_audio_output_confirmed')
            source=SimpleNamespace(sample=lambda t,now:np.array(goals))
            with tempfile.TemporaryDirectory() as folder,patch('vivis.runtime.playback.Bus',return_value=bus), \
                 patch('vivis.runtime.playback.AudioClock',side_effect=AssertionError('audio opened')):
                log=Path(folder)/'log.json'
                def run():return play(profile,[2],np.full((3,1),2047.),60,None,log,Path(folder)/'STOP',
                                      frame_source=source,clock_factory=lambda:clock)
                if good:
                    result=run();self.assertTrue(result['completed'])
                    self.assertNotIn('audio_seconds_at_stop',result)
                    self.assertTrue(all('timeline_seconds' in e for e in result['events']))
                else:
                    with self.assertRaises(ValueError):run()
                bus.close.assert_called_once();clock.close.assert_called_once()


if __name__=='__main__':unittest.main()
