"""P0 protocol, social ownership, cancellation and offline/physical separation."""
from collections import deque
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.interaction.animation import A2FAdapter,FrameMapper
from vivis.interaction.smile import SmileGate
from vivis.interaction.singing import plan,main
from vivis.interaction.stream import LiveFrames


def names():
    with (ROOT/'configs/schemas/blendshape_channels_61.csv').open() as f:return next(csv.reader(f))[2:]


def settings():return json.loads((ROOT/'configs/interaction/p0.json').read_text())


class AnimationTests(unittest.TestCase):
    def test_precompiled_mapping_matches_saved_original_rules_for_both_heads(self):
        import hashlib
        from vivis.motion.mapping import compile_trajectory,CompiledFaceMapping
        frozen=json.loads((ROOT/'tests/fixtures/p0_mapping_fingerprints.json').read_text())
        rng=np.random.default_rng(frozen['seed']);values=np.zeros((70,61))
        values[1,:52]=1.;values[2:54,:52]=np.eye(52);values[54:,:52]=rng.random((16,52))
        values[:,55:]=rng.uniform(-.3,.3,(70,6))
        for record in frozen['records']:
            robot=Robot(ROOT/f'configs/robots/{record["head"]}/robot.json')
            # Frozen fingerprints cover the original full mapping; temporary
            # actuator exclusions are separately checked against source columns.
            robot.excluded_actuator_ids.clear()
            shapes={s['shape']:{p['id']:p['position'] for p in s['positions']} for s in json.loads(robot.calibration.read_text())['blendShapes']}
            profile=robot.motion_profile()
            for key in ('mixing','jaw_owner','paired_eye_look_is_already_in_lids'):profile[key]=record[key]
            ids,positions,_=compile_trajectory(values,names(),shapes,profile,scope=record['scope'])
            self.assertEqual(ids,record['ids'])
            digest=hashlib.sha256(np.round(positions,6).astype('<f8').tobytes()).hexdigest()
            self.assertEqual(digest,record['sha256_rounded_6_decimals'],str(record))
            prepared=CompiledFaceMapping(names(),shapes,profile,scope=record['scope'])
            np.testing.assert_allclose(np.vstack([prepared.map(v[None,:])[0] for v in values]),positions,atol=1e-10,rtol=0)

    def adapter(self):
        canonical=names()[:52];raw=list(reversed(canonical))
        a=A2FAdapter(canonical,'current')
        a.bind(dict(schema_version='vivis-a2f-bindings-1.0',playback_id='current',
                    names=dict(skin=[n[0].lower()+n[1:] for n in raw],tongue=['tongueExtra'])))
        return a,raw

    def packet(self):
        return dict(schema_version='vivis-a2f-raw-1.0',playback_id='current',frame_seq=0,
                    audio_sample_index=0,sample_rate_hz=16000,weights=[i/100 for i in range(53)])

    def test_names_reorder_not_dimensions_or_indices(self):
        a,raw=self.adapter();p=self.packet();frame=a.adapt(p)
        for name,value in zip(names()[:52],frame['weights']):self.assertEqual(value,raw.index(name)/100)
        self.assertEqual(len(frame['weights']),52)
        self.assertEqual(frame['channel_mapping_version'],'a2f-skin52-v1')

    def test_old_instances_bad_clock_invalid_weights_and_sequence_fail(self):
        changes=[{'playback_id':'cancelled'},{'audio_sample_index':-1},{'sample_rate_hz':48000},
                 {'frame_seq':1},{'weights':[True]*53},{'weights':[float('nan')]*53},{'weights':[1.01]*53}]
        for delta in changes:
            a,_=self.adapter();p=self.packet();p.update(delta)
            with self.assertRaises(ValueError):a.adapt(p)
        a,_=self.adapter();a.adapt(self.packet())
        with self.assertRaises(ValueError):a.adapt(self.packet())

    def test_missing_channel_binding_never_silently_pads(self):
        a=A2FAdapter(names()[:52],'id')
        raw=[n[0].lower()+n[1:] for n in names()[:52]];raw[-1]='_neutral'
        with self.assertRaises(ValueError):a.bind(dict(schema_version='vivis-a2f-bindings-1.0',playback_id='id',names=dict(skin=raw,tongue=[])))

    def test_social_does_not_override_critical_calibrated_joint_targets(self):
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        p=plan(robot,'0001','eyes-neck',settings())
        m=FrameMapper(p['order'],p['shapes'],p['profile'],p['ids'])
        weights=[0.]*52
        weights[p['order'].index('JawOpen')]=.4
        weights[p['order'].index('MouthFunnel')]=.5
        base=m.map_weights(weights);smiling=m.map_weights(weights,1.)
        self.assertTrue(np.any(base!=smiling))
        for sid in m.protected_ids:
            if sid in p['ids']:self.assertEqual(base[p['ids'].index(sid)],smiling[p['ids'].index(sid)])
        self.assertTrue(np.array_equal(m.map_weights(weights,0.),base))
        for sid,x in zip(p['ids'],smiling):
            self.assertGreaterEqual(x,robot.axes[sid]['safe_min']);self.assertLessEqual(x,robot.axes[sid]['safe_max'])

    def test_social_response_can_increase_an_already_smiling_audio_frame(self):
        p=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),'0001','eyes-neck',settings())
        m=FrameMapper(p['order'],p['shapes'],p['profile'],p['ids'])
        weights=[0.]*52
        for name in ('MouthSmileLeft','MouthSmileRight'):weights[p['order'].index(name)]=.4
        before=m.map_weights(weights);after=m.map_weights(weights,1.)
        self.assertTrue(np.any(before!=after))
        self.assertAlmostEqual(m.last_fused[p['order'].index('MouthSmileLeft')],.55)

    def test_social_ownership_does_not_jump_at_small_round_lip_coefficients(self):
        # The old .01 switch caused tens of counts of smile delta to appear
        # or vanish while a smooth source crossed that coefficient.
        for name in ('xiaoshuai','mojia'):
            p=plan(Robot(ROOT/f'configs/robots/{name}/robot.json'),'0001','eyes-neck',settings())
            m=FrameMapper(p['order'],p['shapes'],p['profile'],p['ids'],.3)
            previous=None;previous_weight=None
            for weight in (.0,.009,.0099,.0101,.011,.02):
                raw=[0.]*52;raw[p['order'].index('MouthFunnel')]=weight
                base=m.map_weights(raw).copy();response=m.map_weights(raw,1.).copy()
                for sid in m.speech_owned_ids:
                    self.assertEqual(base[p['ids'].index(sid)],response[p['ids'].index(sid)])
                if previous is not None and weight-previous_weight<.00021:
                    self.assertLess(float(max(abs(response-previous))),1.)
                previous=response;previous_weight=weight

    def test_eye_corner_smile_is_visible_in_targets_without_touching_articulation(self):
        p=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),'0001','eyes-neck',settings())
        m=FrameMapper(p['order'],p['shapes'],p['profile'],p['ids'],.3,social_channels=settings()['social_channels'])
        raw=[0.]*52;raw[p['order'].index('JawOpen')]=.4;raw[p['order'].index('MouthFunnel')]=.2
        base=m.map_weights(raw).copy();response=m.map_weights(raw,1.)
        self.assertGreater(response[p['ids'].index(6)]-base[p['ids'].index(6)],40.)
        self.assertLess(response[p['ids'].index(21)]-base[p['ids'].index(21)],-50.)
        for sid in m.speech_owned_ids:self.assertEqual(base[p['ids'].index(sid)],response[p['ids'].index(sid)])
        for name in ('EyeBlinkLeft','EyeBlinkRight'):raw[p['order'].index(name)]=1.
        closed=m.map_weights(raw).copy();responding=m.map_weights(raw,1.)
        for sid in (5,6,20,21):self.assertEqual(closed[p['ids'].index(sid)],responding[p['ids'].index(sid)])
        with self.assertRaises(ValueError):
            FrameMapper(p['order'],p['shapes'],p['profile'],p['ids'],social_channels=['JawOpen'])


class SmileTests(unittest.TestCase):
    def test_single_response_no_continuous_smile_and_rearm(self):
        gate=SmileGate(settings()['social_smile']['gate']);values=[]
        for i in range(200):values.append(gate.step(.9,.9,i*.02)['strength'])
        self.assertGreater(max(values),.9);self.assertEqual(values[-1],0.)
        first=gate.last_trigger
        for i in range(200,250):gate.step(0.,0.,i*.02)
        for i in range(250,350):gate.step(.9,.9,i*.02)
        self.assertGreater(gate.last_trigger,first)

    def test_short_spike_and_one_sided_smile_do_not_trigger(self):
        for unilateral in (False,True):
            gate=SmileGate(settings()['social_smile']['gate']);maximum=0.
            for i in range(100):
                left=.9 if unilateral or 10<=i<14 else 0.
                right=0. if unilateral else left
                maximum=max(maximum,gate.step(left,right,i*.02)['strength'])
            self.assertEqual(maximum,0.)

    def test_lost_face_fades_instead_of_latching(self):
        gate=SmileGate(settings()['social_smile']['gate'])
        for i in range(36):last=gate.step(.9,.9,i*.02)['strength']
        self.assertGreater(last,0.)
        first=gate.step(0.,0.,.72,False)['strength']
        self.assertGreater(first,0.);self.assertLess(first,last)
        for i in range(37,80):final=gate.step(0.,0.,i*.02,False)['strength']
        self.assertEqual(final,0.)


class OfflineP0Tests(unittest.TestCase):
    def test_plan_never_reads_source_expression_or_old_target_json(self):
        original_text=Path.read_text;original_bytes=Path.read_bytes
        def guarded_text(path,*args,**kwargs):
            if path.name in ('source.json','targets.json'):raise AssertionError('prerecorded animation read')
            return original_text(path,*args,**kwargs)
        def guarded_bytes(path,*args,**kwargs):
            if path.name in ('source.json','targets.json'):raise AssertionError('prerecorded animation read')
            return original_bytes(path,*args,**kwargs)
        with patch.object(Path,'read_text',guarded_text),patch.object(Path,'read_bytes',guarded_bytes), \
             patch('vivis.hardware.feetech.sdk.OfficialSDKBus',side_effect=AssertionError('serial opened')), \
             patch('vivis.interaction.stream.subprocess.Popen',side_effect=AssertionError('model or camera opened')):
            with redirect_stdout(io.StringIO()) as out:main(['--check'])
        report=json.loads(out.getvalue());self.assertTrue(report['offline_valid'])
        self.assertFalse(report['prerecorded_animation_read']);self.assertFalse(report['motion_hardware_opened'])

    def test_both_head_plans_share_code_and_leave_configuration_bytes_intact(self):
        for name in ('xiaoshuai','mojia'):
            path=ROOT/f'configs/robots/{name}/robot.json';robot=Robot(path)
            before={p:p.read_bytes() for p in path.parent.glob('*.json')}
            p=plan(robot,'0001','eyes-neck',settings())
            self.assertEqual(p['tracking']['control']['camera_mount'],'fixed' if name=='xiaoshuai' else 'eye')
            self.assertEqual(len(p['neutral']),6774)
            self.assertTrue(np.all(p['neutral']==p['neutral'][0]))
            for file,raw in before.items():self.assertEqual(file.read_bytes(),raw)


class LiveQueueTests(unittest.TestCase):
    def fake(self,folder,ramp=False):
        import sys
        path=Path(folder)/'worker.py'
        header=dict(schema_version='vivis-a2f-bindings-1.0',names=dict(skin=[n[0].lower()+n[1:] for n in names()[:52]],tongue=[]))
        text='#!'+sys.executable+'\nimport json,sys\npid=sys.argv[-1]\nheader='+repr(header)+"\nheader['playback_id']=pid\nprint(json.dumps(header),flush=True)\n"
        text+="for i in range(20):\n print(json.dumps(dict(schema_version='vivis-a2f-raw-1.0',playback_id=pid,frame_seq=i,audio_sample_index=int(i*16000/60),sample_rate_hz=16000,weights=[.1]*52)),flush=True)\n"
        if ramp:text=text.replace('weights=[.1]*52','weights=[i/20]*52')
        path.write_text(text);path.chmod(0o755)
        class Mapper:
            order=names();ids=[1];last_fused=[.1]*52;protected_ids=[]
            def map_weights(self,w,smile):return np.array([w[0]+smile])
        return path,Mapper()

    def test_sampling_interpolates_at_audio_time_and_keeps_original_knots(self):
        with tempfile.TemporaryDirectory() as folder:
            worker,mapper=self.fake(folder,ramp=True)
            source=LiveFrames(worker,'model','audio',1,Path(folder)/'stream',mapper,capacity=20,prebuffer=20)
            try:
                source.wait_ready(timeout=3)
                self.assertEqual(source.sample(0.,time.monotonic())[0],0.)
                self.assertEqual(source.sample(266/16000,time.monotonic())[0],.05)
                midpoint=(266+533)/2/16000
                self.assertAlmostEqual(source.sample(midpoint,time.monotonic())[0],.075)
                self.assertEqual(source.current['frame_seq'],1)
                self.assertEqual(source.frames[0]['frame_seq'],2)
                self.assertEqual(source.sample(533/16000,time.monotonic())[0],.1)
                self.assertEqual(source.underflows,0)
                self.assertGreater(source.summary()['interpolated_ticks'],0)
            finally:source.close()

    def test_queue_is_bounded_and_model_pipe_does_not_block_control(self):
        with tempfile.TemporaryDirectory() as folder:
            worker,mapper=self.fake(folder)
            source=LiveFrames(worker,'model','audio',1,Path(folder)/'stream',mapper,capacity=5,prebuffer=3)
            try:
                source.wait_ready(timeout=3)
                time.sleep(.03)
                self.assertLessEqual(source.max_queued,5)
                self.assertEqual(source.sample(0.,time.monotonic())[0],.1)
                with self.assertRaisesRegex(RuntimeError,'underflow'):source.sample(1.,time.monotonic())
            finally:source.close()
            self.assertFalse(source.thread.is_alive());self.assertIsNotNone(source.process.poll())

    def test_stale_social_event_cannot_override_live_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            worker,mapper=self.fake(folder)
            source=LiveFrames(worker,'model','audio',1,Path(folder)/'stream',mapper,
                social=lambda now:dict(schema_version='vivis-smile-event-1.0',timestamp_monotonic=now-1,strength=1.),capacity=5,prebuffer=3)
            try:
                source.wait_ready(timeout=3)
                self.assertEqual(source.sample(0.,time.monotonic())[0],.1)
            finally:source.close()


class LivePlaybackTests(unittest.TestCase):
    def test_real_feedback_is_refreshed_during_existing_dac_preroll(self):
        import test_mouth_commands as helpers
        from types import SimpleNamespace
        from vivis.runtime.playback import play
        profile,bus,clock=helpers.MouthCommandTests().fixture(True)
        profile.update(schema_version='vivis-feetech-playback-1.0',developer_mode='face',scheduled_feedback=True,
            serial_backend='feetech_python',feedback_mode='short_registers',feedback_max_age_s=1.,control_hz=60)
        epoch=[None]
        clock.start.side_effect=lambda:epoch.__setitem__(0,time.monotonic())
        clock.time.side_effect=lambda:time.monotonic()-epoch[0]-.06
        payload={56:(2047).to_bytes(2,'little'),60:bytes(2),62:bytes([60,25]),65:bytes(1)}
        bus.read.side_effect=lambda sid,address,count:payload[address]
        source=SimpleNamespace(sample=lambda t,now:np.array([2047.,2047.]))
        with tempfile.TemporaryDirectory() as folder,patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus), \
             patch('vivis.runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2,12],np.full((9,2),2047.),60,'fake',Path(folder)/'log.json',Path(folder)/'STOP',frame_source=source)
        self.assertTrue(result['completed'])
        self.assertGreater(result['preroll_feedback_samples'],0)
        self.assertTrue(all(event['audio_seconds']>=0 for event in result['events']))

    def test_live_face_uses_existing_speed_and_direct_mouth_guards(self):
        import test_mouth_commands as helpers
        from types import SimpleNamespace
        from vivis.runtime.playback import play
        profile,bus,clock=helpers.MouthCommandTests().fixture(True)
        source=SimpleNamespace(sample=lambda t,now:np.array([2250.,2250.]))
        with tempfile.TemporaryDirectory() as folder,patch('vivis.runtime.playback.Bus',return_value=bus), \
             patch('vivis.runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2,12],np.full((9,2),2047.),60,'fake',Path(folder)/'log.json',Path(folder)/'STOP',frame_source=source)
        self.assertTrue(result['completed'])
        for event in result['events']:self.assertEqual(event['goals'][12],2250)
        for call in bus.goals.call_args_list:self.assertEqual(call.args[1],{2:300,12:1200})

    def test_out_of_bound_or_invalid_live_face_is_stopped_before_transmission(self):
        import test_mouth_commands as helpers
        from types import SimpleNamespace
        from vivis.runtime.playback import play
        for value in ([2047.,2301.],[float('nan'),2047.],[2047.]):
            profile,bus,clock=helpers.MouthCommandTests().fixture(True)
            source=SimpleNamespace(sample=lambda t,now:np.array(value))
            with tempfile.TemporaryDirectory() as folder,patch('vivis.runtime.playback.Bus',return_value=bus), \
                 patch('vivis.runtime.playback.AudioClock',return_value=clock):
                with self.assertRaises(ValueError):
                    play(profile,[2,12],np.full((9,2),2047.),60,'fake',Path(folder)/'log.json',Path(folder)/'STOP',frame_source=source)
            for call in bus.goals.call_args_list:self.assertLessEqual(call.args[0][12],2300)
            clock.close.assert_called_once();bus.close.assert_called_once()


if __name__=='__main__':unittest.main()
