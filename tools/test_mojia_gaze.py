"""Mojia-only gaze limits, mounting modes, loss and eye/head feedback simulations."""
import copy
import json
import unittest
import numpy as np

from xiaobai_runtime.mapping import ROOT
from xiaobai_runtime.mojia_gaze import validate_mojia, MojiaPursuit
from xiaobai_runtime.person_gaze import VERSION, validate_config
from prepare_mojia import proposal, hardware_profile
from xiaobai_runtime.mapping import inputs
from mojia_track_person import tracking_start_position


def config():
    c=json.loads((ROOT/'mojia/person_gaze.json').read_text())
    c['control'].update(camera_mount='eye',camera_eye='right',mount_confirmed=True)
    for a in c['control']['axes']:a.update(negative=2007,positive=2087)
    c['neck']['offset_counts']=20
    return c


def profile():
    _,_,_,shapes=inputs(ROOT/'tests/fixtures/blendshapes61.json',ROOT/'mojia/墨甲9.23.json',
                       ROOT/'xiaobai/blendshape_channels_61.csv',allowed_motor_ids=set(range(2,32))|{35,36})
    states=[dict(id=sid,model_raw=521 if sid<=30 or sid==36 else 10760,mode=0,status_register=0,
                 configured_min=0,configured_max=4095,configured_max_temperature=80,
                 configured_min_voltage_raw=40,configured_max_voltage_raw=254) for sid in range(2,37)]
    p=hardware_profile(proposal(shapes,2047,'all_calibrated'),dict(port='fake',servos=states))
    p['calibration_sha256']=config()['calibration_sha256']
    return p


def packet(now,x):
    return dict(schema_version=VERSION,sequence=1,timestamp_monotonic=now,
                target=None if x is None else dict(x=x,y=0.))


class MojiaGazeTests(unittest.TestCase):
    def test_eye_only_resume_does_not_require_or_recenter_held_neck(self):
        held=dict(id=32,position_raw=2095,goal_position_raw=2095,configured_min=1600,configured_max=2600)
        self.assertIsNone(tracking_start_position(held))
        eye=dict(id=23,position_raw=2106,goal_position_raw=2106)
        self.assertEqual(tracking_start_position(eye,dict(safe_min=1987,safe_max=2107)),2106.)
        with self.assertRaises(RuntimeError):tracking_start_position({**eye,'goal_position_raw':2200},dict(safe_min=1987,safe_max=2107))
        with self.assertRaises(RuntimeError):tracking_start_position(held,dict(safe_min=1967,safe_max=2127))

    def test_requested_900_stage_bounds_loss_and_return(self):
        c=json.loads((ROOT/'mojia/person_gaze.json').read_text())
        validate_mojia(c,profile(),ROOT/'mojia/墨甲9.23.json',True,True)
        controller=MojiaPursuit(c,[8,23,32],[300,300,900],True)
        old=np.full(3,2047.)
        for i in range(2400):
            now=10+(i+1)/60
            x=.8 if i<900 else -.8 if i<1800 else None
            new,_=controller.step(packet(now,x),old,1/60,now,returning=i>=2100)
            self.assertTrue(np.isfinite(new).all())
            self.assertTrue(np.all(abs(new[:2]-2047)<=60))
            self.assertLessEqual(abs(new[2]-2047),80)
            self.assertTrue(np.all(abs(new-old)<=np.array([300,300,900])/60+1e-8))
            old=new
        np.testing.assert_allclose(old,[2047]*3,atol=.25)

    def test_xiaobai_speed_stage_requires_neck_confirmation(self):
        c,p=config(),profile()
        c['control'].update(calibration_confirmed=True,speed_counts_s=300)
        c['neck'].update(speed_counts_s=75,acceleration_counts_s2=70,offset_counts=160,
                         gain_counts_s=180,eye_lead_s=.08,deadband=.14,stop_deadband=.07,filter_tau_s=.12)
        with self.assertRaises(ValueError):validate_mojia(c,p,ROOT/'mojia/墨甲9.23.json',True,False)
        c['neck']['direction_confirmed']=True
        validate_mojia(c,p,ROOT/'mojia/墨甲9.23.json',True,False)
        with self.assertRaises(ValueError):validate_mojia(c,p,ROOT/'mojia/墨甲9.23.json',True,True)
        controller=MojiaPursuit(c,[8,23,32],[300,300,75],True)
        old=np.array([2047.,2047.,2207.])
        for i in range(300):
            old,_=controller.step(None,old,1/60,10+(i+1)/60,returning=True)
        np.testing.assert_allclose(old,[2047]*3,atol=.25)

    def test_calibration_and_commissioning_confirmations(self):
        c,p=config(),profile()
        validate_mojia(c,p,ROOT/'mojia/墨甲9.23.json',neck=True,trial=True)
        for changed in ['mount','calibration','profile']:
            a,b=copy.deepcopy(c),copy.deepcopy(p)
            if changed=='mount':a['control']['mount_confirmed']=False
            if changed=='calibration':a['calibration_sha256']='wrong'
            if changed=='profile':b['schema_version']='xiaobai'
            with self.assertRaises(ValueError):validate_mojia(a,b,ROOT/'mojia/墨甲9.23.json',True,True)
        with self.assertRaises(ValueError):validate_mojia(c,p,ROOT/'mojia/墨甲9.23.json',True,False)
        with self.assertRaises(ValueError):validate_config(c,live=True,profile=p,ids=[8,23])

    def test_trial_caps_and_endpoint_directions(self):
        c,p=config(),profile()
        for mutate in [lambda a:a['neck'].update(offset_counts=81),
                       lambda a:a['control']['axes'][0].update(positive=2110),
                       lambda a:a['control']['axes'][1].update(id=8),
                       lambda a:a['neck'].update(sign=True)]:
            a=copy.deepcopy(c);mutate(a)
            with self.assertRaises(ValueError):validate_mojia(a,p,ROOT/'mojia/墨甲9.23.json',True,True)

    def test_eye_and_neck_limits_speed_and_normal_return(self):
        c=config();controller=MojiaPursuit(c,[8,23,32],[60,60,25],True)
        old=np.full(3,2047.)
        for i in range(1200):
            now=10+(i+1)/60
            new,d=controller.step(packet(now,.8),old,1/60,now,returning=i>=900)
            self.assertTrue(np.all(abs(new[:2]-2047)<=40))
            self.assertLessEqual(abs(new[2]-2047),20)
            self.assertTrue(np.all(abs(new-old)<=np.array([60,60,25])/60+1e-8))
            old=new
        np.testing.assert_allclose(old,[2047]*3,atol=.25)

    def test_lost_face_releases_eyes_and_brakes_neck(self):
        c=config();controller=MojiaPursuit(c,[8,23,32],[60,60,25],True)
        old=np.full(3,2047.)
        for i in range(600):
            now=10+(i+1)/60
            sample=packet(now,.8) if i<180 else packet(now,None)
            old,d=controller.step(sample,old,1/60,now)
        np.testing.assert_allclose(old[:2],[2047]*2,atol=.1)
        self.assertAlmostEqual(controller.velocity,0.)

    def test_eye_mounted_visual_loop_converges_with_head_motion(self):
        c=config();controller=MojiaPursuit(c,[8,23,32],[60,60,25],True)
        old=np.full(3,2047.);initial=.6;errors=[]
        for i in range(1800):
            now=10+(i+1)/60
            # Assumed geometry for a regression, not a measured camera calibration.
            error=float(np.clip(initial-(old[1]-2047)/40-(old[2]-2047)/40,-1,1))
            errors.append(error)
            old,_=controller.step(packet(now,error),old,1/60,now)
        self.assertLess(abs(errors[-1]),.08)
        self.assertGreater(old[2],2047)

    def test_eyes_only_never_adds_neck_axis(self):
        c=config();controller=MojiaPursuit(c,[8,23],[60,60],False)
        new,_=controller.step(packet(10,.9),np.array([2047.,2047.]),1/60,10)
        self.assertEqual(new.shape,(2,))


if __name__=='__main__':unittest.main()
