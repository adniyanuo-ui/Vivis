"""Optional trajectory engine, state continuity and launcher isolation; no I/O."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from vivis.motion.upstream import configure,settings,require_engine,UpstreamTrajectory,position_extrema
from vivis.runtime.robot import Robot
from vivis.vision.pursuit import MojiaPursuit
from test_gaze import config,packet

HAS_ENGINE=importlib.util.find_spec('ruckig') is not None


class OptionalPlanningTests(unittest.TestCase):
    def test_disabled_config_and_eyes_only_do_not_load_optional_engine(self):
        c=config()
        with patch('vivis.motion.upstream.require_engine',side_effect=AssertionError('unexpected import')):
            controller=MojiaPursuit(c,[8,23,32],[300,300,600],True)
            self.assertIsNone(controller.upstream)
            c['neck']['upstream_trajectory']['enabled']=True
            eyes=MojiaPursuit(c,[8,23],[300,300],False)
            self.assertIsNone(eyes.upstream)

    def test_off_output_is_identical_with_or_without_new_configuration(self):
        c=config();c['neck'].update(speed_counts_s=600,target_filter_tau_s=.15)
        old=copy.deepcopy(c);old['neck'].pop('upstream_trajectory',None)
        a=MojiaPursuit(c,[8,23,32],[300,300,600],True)
        b=MojiaPursuit(old,[8,23,32],[300,300,600],True)
        x=np.full(3,2047.);y=x.copy()
        for tick in range(600):
            now=10+(tick+1)/60;sample=packet(now,.7*math.sin(tick/35))
            if 200<tick<300:sample['target']=None
            x,dx=a.step(sample,x,1/60,now,returning=tick>500)
            y,dy=b.step(sample,y,1/60,now,returning=tick>500)
            np.testing.assert_array_equal(x,y)
            self.assertEqual(dx,dy)

    def test_explicit_off_overrides_saved_on_and_other_modes_cannot_enable(self):
        c=Robot().tracking()
        self.assertTrue(configure(c,'eyes-neck')['enabled'])
        self.assertFalse(configure(c,'eyes-neck','off')['enabled'])
        for mode,home in [('face',False),('eyes',False),('eyes-neck',True)]:
            with self.subTest(mode=mode,home=home):
                c=Robot().tracking();c['neck']['upstream_trajectory']['enabled']=True
                self.assertFalse(configure(c,mode,home_only=home)['enabled'])
                with self.assertRaises(ValueError):configure(c,mode,'on',home)

    def test_invalid_limits_and_missing_engine_fail_before_hardware(self):
        for key,value in [('enabled',1),('max_acceleration_counts_s2',True),
                          ('max_acceleration_counts_s2',0),('max_jerk_counts_s3',float('nan'))]:
            with self.assertRaises(ValueError):settings({'upstream_trajectory':{'enabled':False,key:value}})
        with patch('vivis.motion.upstream.importlib.import_module',side_effect=ModuleNotFoundError('ruckig')):
            with self.assertRaisesRegex(RuntimeError,'--upstream-trajectory off'):require_engine()
        from types import SimpleNamespace
        with patch('vivis.motion.upstream.importlib.import_module',return_value=SimpleNamespace(__version__='0.15.3')):
            with self.assertRaisesRegex(RuntimeError,'ruckig==0.19.4'):require_engine()

    def test_launcher_forwards_switch_without_opening_camera_or_serial(self):
        from vivis.runtime.developer import run_example
        for option in ('on','off'):
            with patch('vivis.runtime.show.main',return_value=0) as show:
                self.assertEqual(run_example('eyes-neck',['--check','--upstream-trajectory',option]),0)
            command=show.call_args.args[0]
            self.assertEqual(command[command.index('--upstream-trajectory')+1],option)
            self.assertIn('--check',command)

    def test_enabled_check_with_missing_dependency_opens_no_hardware(self):
        from vivis.runtime.show import main
        with patch('serial.Serial') as serial,patch('vivis.runtime.playback.AudioClock') as audio, \
                patch('vivis.motion.upstream.importlib.import_module',side_effect=ModuleNotFoundError('ruckig')):
            with self.assertRaisesRegex(RuntimeError,'--upstream-trajectory off'):
                main(['--output','unused-upstream-check','--check','--upstream-trajectory','on'])
            serial.assert_not_called();audio.assert_not_called()


@unittest.skipUnless(HAS_ENGINE,"Install '.[trajectory]' to verify the optional native engine")
class TrajectoryTests(unittest.TestCase):
    def planner(self,position=2047.,low=1600,high=2600):
        plan=settings({'upstream_trajectory':{'enabled':True}})
        p=UpstreamTrajectory(low,high,600,plan);p.reset(position)
        return p

    def assert_motion(self,p,old,dt):
        self.assertTrue(p.low-1e-7<=p.position<=p.high+1e-7)
        self.assertLessEqual(abs(p.velocity),p.maximum_speed+1e-6)
        self.assertLessEqual(abs(p.acceleration),p.maximum_acceleration+1e-6)
        self.assertLessEqual(abs(p.acceleration-old[2]),p.maximum_jerk*dt+1e-5)
        self.assertLessEqual(abs(p.position-old[0]),p.maximum_speed*dt+1e-7)

    def test_velocity_mode_has_no_fictitious_zero_position_or_forced_endpoint_motion(self):
        p=self.planner();profile=p._profile(p.state,velocity=100.)
        low,high=position_extrema(profile)
        self.assertAlmostEqual(low,2047.);self.assertAlmostEqual(high,2058.25)
        for tick in range(60):p.step(100.,1/60)
        self.assertAlmostEqual(p.position,2135.75,places=7)
        self.assertAlmostEqual(p.velocity,100.)
        self.assertEqual(p.last_diagnostic['mode'],'velocity')

    def test_start_reversal_stop_and_irregular_control_periods_preserve_state_and_limits(self):
        p=self.planner()
        for tick in range(1200):
            dt=(.005,.017,.033,.08,.15)[tick%5]
            requested=(100.,-100.,0.,600.,-600.)[(tick//60)%5]
            old=p.state;p.step(requested,dt,returning=(tick//37)%9==4)
            self.assert_motion(p,old,dt)
        p=self.planner()
        for _ in range(60):p.step(100.,1/60)
        before=p.state;p.step(-100.,1/60)
        self.assertGreater(p.velocity,0.)  # Reverse by braking, never reset v to zero.
        self.assert_motion(p,before,1/60)
        for _ in range(30):p.step(-100.,1/60)
        self.assertAlmostEqual(p.velocity,-100.)

    def test_mirrored_commands_have_mirrored_references(self):
        left=self.planner();right=self.planner()
        for tick in range(240):
            velocity=100*math.sin(tick/30)
            left.step(-velocity,1/60);right.step(velocity,1/60)
            self.assertAlmostEqual(left.position+right.position,4094.,places=7)
            self.assertAlmostEqual(left.velocity+right.velocity,0.,places=7)
            self.assertAlmostEqual(left.acceleration+right.acceleration,0.,places=6)

    def test_small_motion_is_not_held_until_sixteen_counts(self):
        p=self.planner();goals=[]
        for _ in range(120):goals.append(round(p.step(2.,1/60)))
        changed=[b-a for a,b in zip(goals,goals[1:]) if b!=a]
        self.assertTrue(changed);self.assertTrue(all(step==1 for step in changed))
        self.assertGreater(p.position,2050.)

    def test_both_boundaries_brake_before_limits_and_allow_immediate_inward_reversal(self):
        for sign in (-1,1):
            p=self.planner()
            for _ in range(1800):
                old=p.state;p.step(sign*100.,1/60);self.assert_motion(p,old,1/60)
            bound=p.high if sign>0 else p.low
            self.assertAlmostEqual(p.position,bound,places=6)
            self.assertAlmostEqual(p.velocity,0.,places=6)
            for _ in range(6):p.step(-sign*100.,1/60)
            self.assertLess((p.position-bound)*sign,0.)

    def test_cached_three_second_return_and_recapture_have_no_state_reset(self):
        p=self.planner(2350.)
        for _ in range(60):p.step(100.,1/60)
        old=p.state;p.step(0.,1/60,True)
        self.assert_motion(p,old,1/60)
        self.assertGreater(p.velocity,0.)
        profile=p.return_profile;duration=profile.duration
        for _ in range(30):
            old=p.state;p.step(0.,1/60,True);self.assert_motion(p,old,1/60)
            self.assertIs(p.return_profile,profile)
        old=p.state;p.step(-50.,1/60,False);self.assert_motion(p,old,1/60)
        self.assertIsNone(p.return_profile)
        for _ in range(360):p.step(0.,1/60,True)
        self.assertGreaterEqual(duration,3.)
        np.testing.assert_allclose(p.state,(2047.,0.,0.),atol=1e-7)

    def test_planner_changes_neck_only_and_does_not_invoke_position_lowpass(self):
        from types import SimpleNamespace
        from vivis.motion.ownership import MojiaPlaybackGaze
        from vivis.runtime.show import prepare
        robot=Robot();c=robot.tracking();c['neck']['upstream_trajectory']['enabled']=True
        c['control']['vertical_enabled']=False
        c['control']['axes']=[a for a in c['control']['axes'] if a['id'] in (8,23)]
        c['control']['held_eye_ids']=[7,22]
        work=robot.work('wz2');raw=json_load(work['targets']);ids=raw['motor_ids']
        positions=np.asarray(raw['positions'],dtype=float)
        p,ids,trajectory=prepare(c,robot.profile('wz2'),ids,positions,mode='eyes-neck')
        old=copy.deepcopy(c);old['neck']['upstream_trajectory']['enabled']=False
        receiver=SimpleNamespace(poll=lambda now:packet(now,.7*math.sin((now-10)*2)))
        on=MojiaPlaybackGaze(c,p,ids,receiver);off=MojiaPlaybackGaze(old,p,ids,receiver)
        last=np.full(len(ids),2047.)
        on.prepare(trajectory,last,[],10.);off.prepare(trajectory,last,[],10.)
        self.assertIsNone(on.scheduler);self.assertIsNotNone(off.scheduler)
        on.set_audio_time(0,113);off.set_audio_time(0,113)
        with patch('vivis.vision.pursuit.NeckCommandScheduler.step',side_effect=AssertionError('double smoothing')):
            for tick in range(240):
                now=10+(tick+1)/60;base=trajectory[tick%len(trajectory)]
                on.observe(23,{'position':2047+round(30*math.sin(tick/30))},now)
                actual=on.apply(base,last,1/60,now)
                for j,sid in enumerate(ids):
                    if sid not in on.owned:self.assertEqual(actual[j],base[j])
                last=actual
        # With identical image and encoder input, eye commands are unaffected.
        a=MojiaPursuit(c,[8,23,32],[1000,1000,600],True,neck_bounds=(1600,2600))
        b=MojiaPursuit(old,[8,23,32],[1000,1000,600],True,neck_bounds=(1600,2600))
        x=np.full(3,2047.);y=x.copy()
        for tick in range(240):
            now=10+(tick+1)/60;s=packet(now,.7*math.sin(tick/30))
            for ctl in (a,b):ctl.observe_eye(2047+round(30*math.sin(tick/30)),now)
            x,_=a.step(s,x,1/60,now);y,_=b.step(s,y,1/60,now)
            np.testing.assert_array_equal(x[:2],y[:2])


def json_load(path):
    import json
    return json.loads(Path(path).read_text())
