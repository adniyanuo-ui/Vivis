"""CAD-derived pair, camera/yaw coupling and single-owner isolation; no hardware."""
import copy,math,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from vivis.runtime.robot import Robot
from vivis.motion.parallel_neck import RotaryParallelNeck
from vivis.vision.parallel_follow import camera_axes,configure,validate,CoupledJointReference,ParallelFaceFollower
from vivis.motion.ownership import tracking_profile,MojiaPlaybackGaze
from test_gaze import packet


def configuration():
    robot=Robot();c=robot.tracking();c['parallel_neck']['enabled']=True
    c['control']['vertical_enabled']=False;c['control']['held_eye_ids']=[7,22]
    c['control']['axes']=[a for a in c['control']['axes'] if a['id'] in (8,23)]
    return c,robot.motion_profile()


class ParallelFollowTests(unittest.TestCase):
    def test_cad_neutral_and_confirmed_encoder_directions(self):
        c,p=configuration();k=validate(c,p)
        pitch=k.inverse(.01,0.);roll=k.inverse(0.,.01)
        self.assertGreater(pitch[33],2047);self.assertLess(pitch[34],2047)
        self.assertGreater(roll[33],2047);self.assertGreater(roll[34],2047)
        self.assertIsNone(k.pose_limits)
        self.assertEqual([(b['low'],b['high']) for b in k.branches],[(1447,2647),(1647,2647)])
        for a in np.linspace(1447,2647,17):
            for b in np.linspace(1647,2647,17):
                pose=k.forward({33:float(a),34:float(b)})
                check=k.inverse(*pose)
                np.testing.assert_allclose([check[33],check[34]],[a,b],atol=1e-4)

    def test_above_platform_yaw_rotates_pitch_roll_actuation_directions(self):
        j,p,r=camera_axes((0.,0.),math.pi/2)
        np.testing.assert_allclose(np.linalg.solve(j,[1.,0.]),[0.,1.],atol=1e-12)
        self.assertEqual(p,0.);self.assertEqual(r,0.)
        np.testing.assert_allclose(camera_axes((0.,0.),0.)[0],np.eye(2),atol=1e-12)

    def test_rear_facing_camera_mount_reverses_both_platform_axes_and_feedback(self):
        c,p=configuration();mount=c['parallel_neck']['control']['camera_mount_yaw_rad']
        self.assertAlmostEqual(mount,math.pi)
        matrix,_,_=camera_axes((0.,0.),0.,mount)
        np.testing.assert_allclose(matrix,-np.eye(2),atol=1e-12)
        # With this mount, positive CAD pitch/roll is negative camera pitch/roll.
        self.assertAlmostEqual(camera_axes((.03,0.),0.,mount)[1],-.03)
        self.assertAlmostEqual(camera_axes((0.,.02),0.,mount)[2],-.02)
        # Field report: upward face must now decrease33/increase34, and a
        # positive eye-line tilt must decrease both, unlike the first trial.
        k=validate(c,p)
        for y,tilt,signs in ((-.6,0.,(-1,1)),(.6,0.,(1,-1)),
                             (0.,.12,(-1,-1)),(0.,-.12,(1,1))):
            f,measured=self.run_face(y=y,human_roll=tilt,ticks=60)
            for sid,sign in zip((33,34),signs):
                self.assertGreater(sign*(measured[sid]-2047),0)
            # The same CAD and confirmed encoder directions still apply.
            np.testing.assert_allclose(k.forward({33:measured[33],34:measured[34]}),f.measured_pose,atol=.003)

    def test_explicit_enable_is_isolated_from_other_modes_and_unconfirmed_data(self):
        c,p=configuration();self.assertFalse(configure(c,'eyes')['enabled'])
        with self.assertRaises(ValueError):configure(c,'eyes','on')
        c,p=configuration();c['parallel_neck']['directions_confirmed']=False
        with self.assertRaisesRegex(ValueError,'confirmed'):validate(c,p)
        c,p=configuration();c['parallel_neck']['geometry']['branches'][0]['minimum_count']=0
        with self.assertRaisesRegex(ValueError,'device'):validate(c,p)
        for offset in (math.inf,math.nan,True,4.):
            c,p=configuration();c['parallel_neck']['control']['camera_mount_yaw_rad']=offset
            with self.assertRaisesRegex(ValueError,'camera_mount_yaw_rad'):validate(c,p)

    def test_pair_motion_limits_braking_and_three_second_home(self):
        c,p=configuration();k=validate(c,p);r=CoupledJointReference(k,c['parallel_neck']['motion'])
        for tick in range(900):
            old=r.a.copy();goals=r.step([70.*math.sin(tick/30),-60.*math.cos(tick/40)],1/60)
            self.assertTrue(all(b['low']-1e-6<=goals[b['id']]<=b['high']+1e-6 for b in k.branches))
            self.assertLessEqual(np.max(np.abs(r.v)),100+1e-6)
            self.assertLessEqual(np.max(np.abs(r.a)),400+1e-6)
            self.assertLessEqual(np.max(np.abs(r.a-old)),4000/60+1e-5)
        r.step([0.,0.],1/60,returning=True);cached=r.return_profile
        for tick in range(360):
            r.step([0.,0.],1/60,returning=True)
            if tick<60:self.assertIs(r.return_profile,cached)
        np.testing.assert_allclose(r.p,[2047,2047],atol=1e-6)
        np.testing.assert_allclose(r.v,[0,0],atol=1e-6)

    def run_face(self,y=-.7,human_roll=0.,yaw=0.,ticks=360):
        c,p=configuration();f=ParallelFaceFollower(c,p);measured={32:2047-yaw*4096/(2*math.pi),33:2047.,34:2047.}
        f.reset(measured,10.)
        for i in range(ticks):
            now=10+(i+1)/60
            for sid in (32,33,34):f.observe(sid,measured[sid],now)
            _,_,camera_roll=camera_axes(f.measured_pose,yaw,c['parallel_neck']['control']['camera_mount_yaw_rad'])
            sample=packet(now,0.);sample['target'].update(y=y,eye_line_roll_rad=human_roll-camera_roll)
            goals=f.step(sample,'tracking',1/60,now)
            measured.update({sid:float(round(v)) for sid,v in goals.items()})
        return f,measured

    def test_up_down_follow_and_camera_roll_at_turned_head(self):
        for y in (-.7,.7):
            for yaw in (0.,math.pi/6,-math.pi/6):
                f,_=self.run_face(y=y,yaw=yaw)
                self.assertGreater(-math.copysign(1,y)*f.diagnostic['camera_pitch_rad'],math.radians(.7))
                self.assertGreater(abs(f.diagnostic['camera_pitch_rad']),math.radians(5.))
                # During continuous large pitch motion, the roll servo loop
                # may have transient cross-axis error; it must remain small.
                self.assertLess(abs(f.diagnostic['camera_roll_rad']),math.radians(1.))

    def test_roll_is_small_fraction_of_human_tilt_not_horizontal_face_error(self):
        f,_=self.run_face(y=0.,human_roll=math.radians(4.))
        self.assertAlmostEqual(math.degrees(f.diagnostic['camera_roll_rad']),1.,delta=.2)
        self.assertLess(abs(f.diagnostic['camera_pitch_rad']),math.radians(.2))

    def test_roll_target_has_no_two_degree_cap_and_uses_same_follow_ratio(self):
        f,_=self.run_face(y=0.,human_roll=math.radians(20.))
        self.assertAlmostEqual(math.degrees(f.diagnostic['roll_target_rad']),5.,delta=.2)
        self.assertAlmostEqual(math.degrees(f.diagnostic['camera_roll_rad']),5.,delta=.6)

    def test_device_limits_are_single_source_for_model_and_command_owner(self):
        c,p=configuration();motor=next(m for m in p['motors'] if m['id']==33)
        motor['device_max']=2500
        k=validate(c,p)
        self.assertEqual(k.branches[0]['high'],2500)
        active=tracking_profile(c,p,[m['id'] for m in p['motors']])
        for sid in (33,34):
            m=next(m for m in active['motors'] if m['id']==sid)
            self.assertEqual((m['safe_min'],m['safe_max']),(m['device_min'],m['device_max']))

    def test_full_range_brakes_at_device_stop_and_has_enough_time_to_return(self):
        c,p=configuration();k=validate(c,p);r=CoupledJointReference(k,c['parallel_neck']['motion'])
        for _ in range(480):r.step([100.,100.],1/60)
        self.assertGreater(r.p[0],2500)
        self.assertLessEqual(max(r.p),2647+1e-6)
        self.assertLess(np.max(np.abs(r.v)),1e-5)
        window=r.return_window()
        self.assertGreater(window,6.)
        for _ in range(math.ceil(window*60)):r.step([0.,0.],1/60,returning=True)
        np.testing.assert_allclose(r.p,[2047,2047],atol=1e-6)

    def test_live_nearly_symmetric_brake_preserves_continuity_and_stops(self):
        # Exact input from the173310 run: strict Time synchronization fails in
        # Ruckig0.19.4 at Step2, despite a feasible jerk-limited stop per axis.
        c,p=configuration();k=validate(c,p);r=CoupledJointReference(k,c['parallel_neck']['motion'])
        r.reset([1847.353933442777,2220.837926281496])
        r.v=np.array([-2.502828474255631,2.502828707989294])
        r.a=np.array([141.5013349549928,-141.5013415622423])
        profile=r.profile()
        np.testing.assert_allclose(r.sample(profile,0.),(r.p,r.v,r.a),atol=1e-10)
        self.assertTrue(r.bounded(profile))
        self.assertLess(profile.duration,.1)
        np.testing.assert_allclose(r.sample(profile,profile.duration)[1:],[np.zeros(2),np.zeros(2)],atol=1e-8)
        for _ in range(12):r.step([0.,0.],1/60)
        np.testing.assert_allclose(r.v,[0.,0.],atol=1e-8)
        np.testing.assert_allclose(r.a,[0.,0.],atol=1e-8)

    def test_near_brake_switching_surface_and_nonzero_velocity_targets(self):
        c,p=configuration();r=CoupledJointReference(validate(c,p),c['parallel_neck']['motion'])
        rng=np.random.default_rng(173310)
        for _ in range(128):
            r.reset([1847.,2221.])
            accelerations=rng.uniform(10,300,2)*np.array([1.,-1.])
            r.a=accelerations
            r.v=-np.sign(accelerations)*accelerations**2/(2*4000)+rng.uniform(-1e-6,1e-6,2)
            profile=r.profile()
            self.assertTrue(r.bounded(profile))
            np.testing.assert_allclose(r.sample(profile,profile.duration)[1:],[np.zeros(2),np.zeros(2)],atol=1e-7)
        r.reset([2047.,2047.]);moving=r.profile([20.,-30.])
        np.testing.assert_allclose(r.sample(moving,moving.duration)[1:],[np.array([20.,-30.]),np.zeros(2)],atol=1e-8)
        r.reset([1800.,2300.]);home=r.profile(home=True,minimum=3.)
        self.assertGreaterEqual(home.duration,3.)
        np.testing.assert_allclose(r.sample(home,home.duration)[0],[2047.,2047.],atol=1e-8)

    def test_platform_can_start_song_end_return_without_changing_yaw_return_time(self):
        c,p=configuration();ids=[m['id'] for m in p['motors']]
        active=tracking_profile(c,p,ids);f=MojiaPlaybackGaze(c,active,ids,SimpleNamespace(poll=lambda now:packet(now,0.)))
        measured=np.full(len(ids),2047.);base=measured.copy()
        f.prepare(np.array([base]),measured,[],10.)
        far={33:2600.,34:2600.}
        f.platform.reference.reset(list(far.values()))
        for sid,pos in {32:2047.,**far}.items():f.observe(sid,{'position':pos},10.)
        f.set_audio_time(106.5,113.)
        f.apply(base,measured,1/60,10.01)
        self.assertTrue(f.platform_returning)
        self.assertFalse(f.returning)

    def test_stale_feedback_brakes_and_target_loss_returns_both_axes(self):
        f,measured=self.run_face(ticks=90)
        for tick in range(60):f.step(packet(12.+tick/60,0.),'tracking',1/60,12.+tick/60)
        self.assertEqual(f.diagnostic['reason'],'feedback_stale')
        np.testing.assert_allclose(f.reference.v,[0,0],atol=1e-6)
        for tick in range(360):
            now=14+tick/60
            for sid in (32,33,34):f.observe(sid,measured[sid],now)
            goals=f.step(None,'idle',1/60,now)
            measured.update(goals)
        np.testing.assert_allclose(f.reference.p,[2047,2047],atol=1e-6)

    def test_single_owner_preserves_all_other_targets_and_old_yaw_eye_outputs(self):
        c,p=configuration();ids=[m['id'] for m in p['motors']]
        # Base profile excludes the unmapped platform axes from commanded IDs.
        on_profile=tracking_profile(c,p,ids);old=copy.deepcopy(c);old['parallel_neck']['enabled']=False
        off_profile=tracking_profile(old,p,ids)
        def sample(now):
            p=packet(now,.4);p['target'].update(y=-.5,eye_line_roll_rad=.12);return p
        r=SimpleNamespace(poll=sample)
        on=MojiaPlaybackGaze(c,on_profile,ids,r);off=MojiaPlaybackGaze(old,off_profile,ids,r)
        base=np.full((1,len(ids)),2047.);base[0,ids.index(31)]=1900.
        measured=np.full(len(ids),2047.)
        on.prepare(base,measured,[],10.);off.prepare(base,measured,[],10.)
        on.set_audio_time(0,113);off.set_audio_time(0,113)
        self.assertEqual(on.owned,[8,23,32,33,34]);self.assertEqual(on_profile['held_neck_ids'],[])
        for tick in range(120):
            now=10+(tick+1)/60
            for sid in (23,32,33,34):on.observe(sid,{'position':2047},now);off.observe(sid,{'position':2047},now)
            a=on.apply(base[0],measured,1/60,now);b=off.apply(base[0],measured,1/60,now)
            for j,sid in enumerate(ids):
                if sid not in (33,34):self.assertEqual(a[j],b[j])
        self.assertEqual(a[ids.index(31)],1900.)
