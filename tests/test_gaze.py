"""Mojia eye/head coordination and bounded feedback scheduler regressions."""
import copy,json,unittest
import numpy as np
from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.vision.pursuit import validate_mojia,MojiaPursuit,NeckCommandScheduler
from vivis.vision.policy import VERSION,validate_config
from vivis.motion.mapping import inputs
from support import proposal,hardware_profile

def baseline_config():
    c=Robot().tracking()
    for axis in c['control']['axes']:axis.pop('speed_counts_s',None)
    c['neck'].update(range_mode='offset',offset_counts=80)
    c['neck']['upstream_trajectory']['enabled']=False  # Preserve legacy controller fixtures.
    # Retain the previously accepted eye-speed fixtures for legacy regressions.
    c['control']['speed_counts_s']=300
    for axis in c['control']['axes']:
        if axis['coordinate']=='y':axis['speed_counts_s']=100
    return c


def config():
    c=baseline_config()
    c['control'].update(camera_mount='eye',camera_eye='right',mount_confirmed=True,vertical_enabled=False)
    c['control']['axes']=[a for a in c['control']['axes'] if a['coordinate']=='x']
    for a in c['control']['axes']:a.update(negative=2007,positive=2087)
    c['neck']['offset_counts']=20
    return c


def profile():
    _,_,_,shapes=inputs(ROOT/'tests/fixtures/blendshapes61.json',ROOT/'configs/robots/mojia/calibration.json',
                       ROOT/'configs/schemas/blendshape_channels_61.csv',allowed_motor_ids=set(range(2,32))|{35,36})
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
    def test_falling_camera_error_recruits_neck_symmetrically_before_eye_saturation(self):
        for side in ('left','right'):
            starts=[];paths=[]
            for sign in (-1,1):
                c=config();c['control']['camera_eye']=side
                for a in c['control']['axes']:a.update(negative=1987,positive=2107)
                controller=MojiaPursuit(c,[8,23,32],[300,300,600],True)
                old=np.full(3,2047.);start=None;path=[]
                for tick in range(16):
                    now=10+tick*.05
                    controller.observe_eye(2047+sign*min(60,tick*3),now)
                    old,d=controller.step(packet(now,sign*(.9-.03*tick)),old,.05,now)
                    path.append(old[2]-2047)
                    if start is None and abs(d['neck_velocity'])>0:
                        start=tick
                        self.assertEqual(d['neck_assist_reason'],'eye_reserve')
                        self.assertLess(abs(d['eye_offset_measured']),1.)
                self.assertIsNotNone(start);starts.append(start);paths.append(path)
            self.assertEqual(starts,[5,5])
            np.testing.assert_allclose(paths[0],-np.array(paths[1]),atol=1e-8)

    def test_reversal_clears_old_image_error_polarity_in_both_directions(self):
        from vivis.vision.pursuit import VisualNeckAssist
        for sign in (-1,1):
            assist=VisualNeckAssist(config())
            self.assertGreater(sign*assist.step(packet(10.,sign*.9),'tracking',sign,sign,60),0)
            v=assist.step(packet(10.05,-sign*.3),'tracking',-sign*.4,-sign,60)
            self.assertLess(sign*v,0)
            self.assertEqual(assist.reason,'eye_reserve')

    def test_neck_assistance_uses_encoder_reserve_not_a_saturated_eye_command(self):
        c=config();controller=MojiaPursuit(c,[8,23,32],[300,300,600],True)
        controller.observe_eye(2047,10.)
        previous=np.array([2087.,2087.,2047.])
        previous,d=controller.step(packet(10.01,.4),previous,.01,10.01)
        self.assertEqual(d['eye_offset_command'],1.)
        self.assertEqual(d['eye_offset_measured'],0.)
        self.assertEqual(d['neck_velocity'],0.)
        controller.observe_eye(2060,10.05)
        previous,d=controller.step(packet(10.06,.3),previous,.05,10.06)
        self.assertGreater(d['neck_velocity'],0.)
        self.assertEqual(d['neck_assist_reason'],'eye_reserve')

    def test_stale_camera_eye_feedback_does_not_recruit_the_neck(self):
        controller=MojiaPursuit(config(),[8,23,32],[300,300,600],True)
        controller.observe_eye(2047,9.99);controller.observe_eye(2087,10.)
        previous=np.array([2087.,2087.,2047.])
        previous,d=controller.step(packet(10.02,.4),previous,.02,10.02)
        self.assertGreater(d['neck_velocity'],0.)
        previous,d=controller.step(packet(10.4,.4),previous,.02,10.4)
        self.assertEqual(d['neck_velocity'],0.)
        self.assertEqual(d['neck_assist_reason'],'eye_feedback_stale')

    def test_filtered_fractional_boundary_noise_does_not_toggle_integer_neck_goals(self):
        scheduler=NeckCommandScheduler(2047,10,600,False,filter_tau_s=.15)
        for tick in range(1,241):
            goal,_=scheduler.step(2047.5+(.02 if tick%2 else -.02),10+tick/60)
            self.assertEqual(goal,2047)
        previous=2047
        for tick in range(1,121):
            goal,_=scheduler.step(2047+tick*.1,14+tick/60)
            self.assertTrue(0<=goal-previous<=1)
            self.assertLessEqual(abs(scheduler.filtered_position-goal),1.)
            previous=goal
        self.assertGreater(previous,2055)

    def test_neck_target_filter_rejects_steps_and_is_consistent_across_tick_rates(self):
        positions=[]
        for hz in (60,100):
            scheduler=NeckCommandScheduler(2047,10,1000,False,filter_tau_s=.15)
            first,_=scheduler.step(2147,10+1/hz)
            self.assertLess(first,2070)
            for tick in range(2,int(hz*.3)+1):scheduler.step(2147,10+tick/hz)
            positions.append(scheduler.filtered_position)
        self.assertAlmostEqual(positions[0],positions[1],places=8)
        scheduler=NeckCommandScheduler(2047,10,1000,False,filter_tau_s=.15)
        goals=[]
        for tick in range(1,121):
            goal,_=scheduler.step(2047+(20 if tick%2 else -20),10+tick/60)
            if tick>30:goals.append(goal)
        self.assertLessEqual(max(goals)-min(goals),4)

    def test_neck_return_starts_at_applied_position_and_takes_three_seconds(self):
        c=config();c['neck'].update(return_duration_s=3.,target_filter_tau_s=.15)
        controller=MojiaPursuit(c,[8,23,32],[300,300,1000],True,neck_bounds=(1600,2600))
        previous=np.array([2047.,2047.,2600.])
        scheduler=NeckCommandScheduler(2500,10,1000,False,filter_tau_s=.15)
        raw=[];sent=[]
        for tick in range(1,241):
            now=10+tick/60
            previous,d=controller.step(None,previous,1/60,now,returning=True,
                                        applied_neck=scheduler.filtered_position)
            goal,_=scheduler.step(previous[-1],now,returning=True)
            raw.append(previous[-1]);sent.append(goal)
        self.assertEqual(raw[0],2500.)
        self.assertAlmostEqual(d['neck_return_duration_s'],3.)
        self.assertGreater(raw[60],2300)
        self.assertTrue(all(b<=a+1e-8 for a,b in zip(raw,raw[1:])))
        self.assertLessEqual(max(abs(b-a) for a,b in zip(sent,sent[1:])),7)
        self.assertEqual(sent[-1],2047)
        self.assertTrue(all(2047<=p<=2500 for p in raw))
        previous,d=controller.step(packet(15.,.2),previous,1/60,15.,applied_neck=scheduler.filtered_position)
        self.assertFalse(d['returning_to_center']);self.assertIsNone(controller.return_started)

    def test_bad_target_smoothing_parameters_are_rejected_before_devices_open(self):
        for values in (dict(target_filter_tau_s=-1),dict(target_filter_tau_s=float('nan')),
                       dict(return_duration_s=0),dict(return_duration_s=float('inf'))):
            c=config();c['neck'].update(values)
            with self.assertRaises(ValueError):validate_mojia(c,profile(),ROOT/'configs/robots/mojia/calibration.json',True,True)

    def test_lost_face_returns_eyes_and_neck_with_bounded_motion_and_can_reacquire(self):
        for stale in (False,True):
            with self.subTest(stale=stale):
                c=config();c['neck'].update(range_mode='device');c['neck'].pop('offset_counts')
                controller=MojiaPursuit(c,[8,23,32],[300,300,900],True,neck_bounds=(1600,2600))
                old=np.array([2070.,2070.,2150.]);controller.policy.weight=1.
                controller.policy.xy=np.array([23/40,0.])
                controller.policy.last_seen=10.;controller.velocity=30.
                scheduler=NeckCommandScheduler(old[-1],10,900,False,endpoints=(1600,2600))
                saw_hold=saw_return=False;packet0=packet(10.,.5)
                for i in range(1200):
                    now=10+(i+1)/60;before=controller.velocity
                    target=packet0 if stale else packet(now,None)
                    new,d=controller.step(target,old,1/60,now)
                    self.assertTrue(np.all(abs(new-old)<=np.array([300,300,900])/60+1e-8))
                    if d['state']=='holding':saw_hold=True;self.assertFalse(d['returning_to_center'])
                    if d['state'] in ('releasing','idle'):
                        saw_return=True;self.assertTrue(d['returning_to_center'])
                        self.assertEqual(d['neck_assist_reason'],'lost_face_return')
                    goal,_=scheduler.step(new[-1],now,returning=d['returning_to_center'])
                    old=new
                self.assertTrue(saw_hold);self.assertTrue(saw_return)
                np.testing.assert_allclose(old,[2047]*3,atol=.25);self.assertEqual(goal,2047)
                new,d=controller.step(packet(now+1/60,.4),old,1/60,now+1/60)
                self.assertEqual(d['state'],'tracking');self.assertFalse(d['returning_to_center'])

    def test_neck_direction_change_is_not_delayed_by_a_software_acceleration_ramp(self):
        c=config();controller=MojiaPursuit(c,[8,23,32],[300,300,900],True)
        controller.assist.step=lambda *args:90.
        old=np.array([2047.,2047.,2047.])
        new,d=controller.step(packet(10.,.4),old,1/60,10.)
        self.assertAlmostEqual(d['neck_velocity'],90.)
        controller.assist.step=lambda *args:-90.
        back,d=controller.step(packet(10.+1/60,-.4),new,1/60,10.+1/60)
        self.assertAlmostEqual(d['neck_velocity'],-90.)
        self.assertAlmostEqual(back[-1],2047.)

    def test_current_speed_policy_only_raises_calibrated_mouth_axes(self):
        from vivis.motion.ownership import tracking_profile
        c=Robot().tracking()
        p=Robot().profile('0001')
        validate_mojia(c,p,ROOT/'configs/robots/mojia/calibration.json',True,True)
        self.assertEqual(c['control']['speed_counts_s'],1000)
        self.assertTrue(all(a['speed_counts_s']==1000 for a in c['control']['axes']))
        runtime=tracking_profile(c,p,[m['id'] for m in p['motors']])
        self.assertEqual({m['id']:m['max_acceleration_counts_s2'] for m in runtime['motors']
                          if 'max_acceleration_counts_s2' in m},{})
        for m in runtime['motors']:
            self.assertNotIn(m['id'],(9,24))
            mouth={10,11,12,13,15,25,26,27,28,29,30,31,35,36}
            self.assertEqual(m['max_speed_counts_s'],600 if m['id']==32 else 1200 if m['id'] in mouth else 1000)
        from vivis.runtime.show import prepare
        source_ids=p['diagnostic_command_ids']
        policy=json.loads((ROOT/'configs/robots/mojia/telemetry.json').read_text())
        prepared,_,_=prepare(c,p,source_ids,np.full((2,len(source_ids)),2047.),True,1.5,policy)
        self.assertTrue(all(m['max_speed_counts_s']==(600 if m['id']==32 else 1200 if m['id'] in mouth else 1000) for m in prepared['motors']))

    def test_device_range_uses_both_asymmetric_hardware_endpoints(self):
        from vivis.vision.pursuit import neck_command_limits
        from vivis.motion.ownership import tracking_profile
        c=config();c['neck'].update(range_mode='device');c['neck'].pop('offset_counts')
        p=profile();motor=next(m for m in p['motors'] if m['id']==32)
        motor.update(device_min=1600,device_max=2600)
        validate_mojia(c,p,ROOT/'configs/robots/mojia/calibration.json',True,True)
        self.assertEqual(neck_command_limits(c,p),(1600,2600))
        runtime=tracking_profile(c,p,[m['id'] for m in p['motors']])
        neck=next(m for m in runtime['motors'] if m['id']==32)
        self.assertEqual((neck['safe_min'],neck['safe_max']),(1600,2600))
        for direction,initial,endpoint in [(1,2599.5,2600),(-1,1600.5,1600)]:
            controller=MojiaPursuit(c,[8,23,32],[300,300,900],True,neck_bounds=(1600,2600))
            eye=2087 if direction>0 else 2007
            old=np.array([eye,eye,initial],dtype=float)
            for i in range(120):
                now=10+(i+1)/60
                old,_=controller.step(packet(now,float(direction)),old,1/60,now)
                self.assertTrue(1600<=old[-1]<=2600)
            self.assertEqual(old[-1],endpoint)
        with self.assertRaisesRegex(ValueError,'requires the hardware profile'):
            MojiaPursuit(c,[8,23,32],[300,300,900],True)

    def test_cold_start_device_range_does_not_expand_tracking_or_change_other_modes(self):
        from vivis.vision.pursuit import neck_command_limits
        from vivis.runtime.show import prepare
        from vivis.runtime.startup import startup_plan
        robot=Robot();c=robot.tracking();p=robot.profile('wz2')
        motor=next(m for m in p['motors'] if m['id']==32)
        self.assertEqual((motor['device_min'],motor['device_max']),(0,4095))
        self.assertEqual(neck_command_limits(c,p),(1600,2600))
        ids=p['diagnostic_command_ids']
        for mode in ('face','eyes','eyes-neck'):
            runtime,_,_=prepare(c,p,ids,np.full((2,len(ids)),2047.),True,1.,robot.telemetry(),mode)
            neck=next(m for m in runtime['motors'] if m['id']==32)
            expected=(2027,2067) if mode=='face' else (1600,2600)
            self.assertEqual((neck['safe_min'],neck['safe_max']),expected)
            self.assertEqual(32 in runtime['gaze_owner_ids'],mode=='eyes-neck')
        # The old device-limit mismatch is accepted only after explicitly
        # updating the hardware definition, never by disabling verification.
        preset=robot.startup();axis=next(a for a in preset['axes'] if a['id']==32)
        preset.update(axes=[axis],excluded_ids=[])
        raw=bytearray(40);raw[18]=4;ram=bytearray(31)
        snapshot=dict(id=32,model_raw=10760,mode=0,status_register=0,
            configured_min=0,configured_max=4095,config_0_39_hex=raw.hex(),state_40_70_hex=ram.hex())
        _,plan=startup_plan(preset,[snapshot],60)
        self.assertFalse(any(row['address'] in (9,10,11,12) for row in plan))
        snapshot['configured_max']=3000
        with self.assertRaisesRegex(RuntimeError,'limits/status mismatch'):
            startup_plan(preset,[snapshot],60)

    def test_configured_tracking_range_cannot_escape_device_bounds(self):
        robot=Robot();c=robot.tracking();p=robot.profile('wz2')
        motor=next(m for m in p['motors'] if m['id']==32)
        for low,high in ((-1,2600),(1600,4096),(2050,2600),(2600,1600),(1600.5,2600)):
            with self.subTest(low=low,high=high):
                motor.update(tracking_min=low,tracking_max=high)
                with self.assertRaises(ValueError):
                    validate_mojia(c,p,robot.calibration,True,True)

    def test_final_count_at_device_limit_is_sent_and_return_window_scales(self):
        scheduler=NeckCommandScheduler(1601,10,900,False,endpoints=(1600,2600))
        self.assertEqual(scheduler.step(1600,10.05),(1600,900))
        c=config();c['neck'].update(range_mode='device');c['neck'].pop('offset_counts')
        controller=MojiaPursuit(c,[8,23,32],[300,300,900],True,neck_bounds=(1600,2600))
        window=controller.return_window(2600)
        self.assertGreaterEqual(window,5.)
        slower=copy.deepcopy(c);slower['neck']['speed_counts_s']=75
        slow=MojiaPursuit(slower,[8,23,32],[300,300,75],True,neck_bounds=(1600,2600))
        self.assertGreater(slow.return_window(2600),window)
        old=np.array([2047.,2047.,2600.])
        for i in range(int(np.ceil(window*60))):
            now=10+(i+1)/60
            old,_=controller.step(packet(now,None),old,1/60,now,returning=True)
        self.assertLess(abs(old[-1]-2047),1.)

    def test_left_camera_uses_left_yaw_and_pitch_as_its_motion_reference(self):
        c=baseline_config();c['control']['camera_eye']='left'
        controller=MojiaPursuit(c,[8,23,7,22,32],[300,300,100,100,900],True)
        controller.policy.weight=1.
        old=np.array([2055.,2037.,2057.,2057.,2047.])
        controller.step(packet(10.,0.),old,1/60,10.)
        self.assertEqual(controller.camera_eye_id,8)
        self.assertGreater(controller.policy.xy[0],0.)
        self.assertLess(controller.policy.xy[1],0.)

    def test_growing_error_recruits_neck_after_eyes_and_preserves_eye_outputs(self):
        c=baseline_config()
        coupled=MojiaPursuit(c,[8,23,7,22,32],[300,300,100,100,900],True)
        eyes=MojiaPursuit(c,[8,23,7,22],[300,300,100,100])
        previous=np.full(5,2047.);flat=previous[:4].copy();eye_start=neck_start=None
        for i in range(90):
            now=10+(i+1)/60;p=packet(now,.16+i*.005)
            previous,d=coupled.step(p,previous,1/60,now)
            flat,_=eyes.step(p,flat,1/60,now)
            np.testing.assert_allclose(previous[:4],flat)
            if eye_start is None and abs(previous[1]-2047)>=1:eye_start=now
            if neck_start is None and d['neck_velocity']>0:neck_start=now
        self.assertIsNotNone(neck_start)
        self.assertGreater(neck_start,eye_start)
        self.assertLess(neck_start-eye_start,.3)

    def test_measured_eye_must_move_before_growth_recruits_neck(self):
        c=config();controller=MojiaPursuit(c,[8,23,32],[300,300,900],True)
        old=np.full(3,2047.);controller.observe_eye(2047,10.)
        for i in range(12):
            now=10+(i+1)/60
            old,d=controller.step(packet(now,.16+i*.01),old,1/60,now)
            self.assertEqual(d['neck_velocity'],0.)
        controller.observe_eye(2049,10.2)
        old,d=controller.step(packet(10.22,.4),old,.02,10.22)
        self.assertGreater(d['neck_velocity'],0.)
        self.assertEqual(d['eye_motion_source'],'encoder')

    def test_repeated_frame_cannot_invent_growing_error(self):
        c=config();controller=MojiaPursuit(c,[8,23,32],[300,300,900],True)
        old=np.full(3,2047.);p=packet(10.,.2)
        for i in range(15):
            old,d=controller.step(p,old,1/60,10+i/60)
            self.assertEqual(d['neck_velocity'],0.)
            self.assertEqual(d['image_error_growth'],0.)

    def test_past_growth_does_not_recruit_neck_after_error_starts_falling(self):
        c=config();controller=MojiaPursuit(c,[8,23,32],[300,300,900],True)
        old=np.full(3,2047.);controller.observe_eye(2047,10.)
        for i in range(12):
            now=10+(i+1)/60
            old,d=controller.step(packet(now,.16+i*.01),old,1/60,now)
        controller.observe_eye(2049,10.2)
        old,d=controller.step(packet(10.22,.25),old,.02,10.22)
        self.assertEqual(d['neck_velocity'],0.)

    def test_reversal_leaves_neck_limit_while_eyes_still_right_of_centre(self):
        c=config();c['neck']['offset_counts']=80
        for a in c['control']['axes']:a.update(negative=1987,positive=2107)
        controller=MojiaPursuit(c,[8,23,32],[300,300,900],True)
        old=np.array([2107.,2107.,2127.]);first=None
        for i in range(50):
            now=10+(i+1)/60
            old,d=controller.step(packet(now,-.16-i*.006),old,1/60,now)
            if first is None and d['neck_velocity']<0:first=old.copy()
        self.assertIsNotNone(first)
        self.assertGreater(first[1],2047)
        self.assertLess(first[2],2127)
        self.assertLess(old[2],2120)

    def test_decreasing_error_keeps_eyes_in_charge_without_measured_eye_excursion(self):
        controller=MojiaPursuit(config(),[8,23,32],[300,300,900],True)
        old=np.full(3,2047.)
        for i in range(45):
            now=10+(i+1)/60
            controller.observe_eye(2047,now)
            old,d=controller.step(packet(now,.25*(1-i/45)),old,1/60,now)
            self.assertEqual(d['neck_velocity'],0.)
        self.assertGreater(old[1],2047)

    def test_saturated_eye_with_constant_error_can_request_help(self):
        c=config();controller=MojiaPursuit(c,[8,23,32],[300,300,900],True)
        old=np.array([2087.,2087.,2047.]);controller.observe_eye(2087,10.)
        old,d=controller.step(packet(10.02,.3),old,.02,10.02)
        self.assertGreater(d['neck_velocity'],0.)
        self.assertEqual(d['neck_assist_reason'],'eye_limit')

    def test_no_fixed_delay_or_old_coordination_config_is_accepted(self):
        c=config();c['neck']['eye_lead_s']=.04
        with self.assertRaisesRegex(ValueError,'without a fixed eye delay'):
            validate_mojia(c,profile(),ROOT/'configs/robots/mojia/calibration.json',True,True)

    def test_neck_fixed_speed_900_does_not_get_overwritten_by_coalescing(self):
        scheduler=NeckCommandScheduler(2047,10,900,adaptive_speed=False)
        for i in range(120):
            now=10+(i+1)/60
            goal,speed=scheduler.step(2047+20*(now-10),now)
            self.assertEqual(speed,900)
        self.assertLessEqual(abs(goal-2087),1)

    def test_neck_scheduler_suppresses_one_count_hunting_without_losing_home(self):
        scheduler=NeckCommandScheduler(2095,10,900)
        for i in range(120):
            goal,speed=scheduler.step(2095+(1 if i%2 else -1),10+(i+1)/60)
            self.assertEqual(goal,2095)
            self.assertLessEqual(speed,900)
        home=NeckCommandScheduler(2048,10,900)
        self.assertEqual(home.step(2047,10.1,returning=True)[0],2047)

    def test_neck_scheduler_matches_slow_trajectory_instead_of_always_using_900(self):
        scheduler=NeckCommandScheduler(2047,10,900)
        goals=[];speeds=[]
        for i in range(120):
            now=10+(i+1)/60
            goal,speed=scheduler.step(2047+20*(now-10),now)
            goals.append(goal);speeds.append(speed)
        self.assertLessEqual(abs(goals[-1]-2087),1)
        self.assertLessEqual(max(speeds),100)
        changes=sum(a!=b for a,b in zip(goals,goals[1:]))
        self.assertLess(changes,40)

    def test_independent_vertical_speed_and_bad_axis_limit(self):
        c=baseline_config()
        validate_mojia(c,profile(),ROOT/'configs/robots/mojia/calibration.json',False,True)
        for value in [True,0,-1,1001,float('nan')]:
            invalid=copy.deepcopy(c);invalid['control']['axes'][-1]['speed_counts_s']=value
            with self.assertRaises(ValueError):validate_mojia(invalid,profile(),ROOT/'configs/robots/mojia/calibration.json',False,True)
        controller=MojiaPursuit(c,[8,23,7,22],[300,300,100,100])
        previous=np.full(4,2047.)
        for i in range(600):
            now=10+(i+1)/60;p=packet(now,1.);p['target']['y']=1. if i<300 else -1.
            new,_=controller.step(p,previous,1/60,now)
            self.assertTrue(np.all(abs(new[2:]-2047)<=20))
            self.assertTrue(np.all(abs(new-previous)<=np.array([300,300,100,100])/60+1e-8))
            previous=new

    def test_vertical_pair_opposite_encoders_and_horizontal_independence(self):
        c=baseline_config()
        validate_mojia(c,profile(),ROOT/'configs/robots/mojia/calibration.json',False,True)
        horizontal=config()
        for a in horizontal['control']['axes']:a.update(negative=1987,positive=2107)
        full=MojiaPursuit(c,[8,23,7,22],[300]*4)
        flat=MojiaPursuit(horizontal,[8,23],[300]*2)
        previous=np.full(4,2047.);old_flat=previous[:2].copy()
        for i in range(600):
            now=10+(i+1)/60;p=packet(now,.5);p['target']['y']=.6
            previous,_=full.step(p,previous,1/60,now)
            old_flat,_=flat.step(p,old_flat,1/60,now)
            np.testing.assert_allclose(previous[:2],old_flat)
            self.assertTrue(np.all(abs(previous[2:]-2047)<=30))
        self.assertLess(previous[2],2047)
        self.assertGreater(previous[3],2047)
        for i in range(300):
            now=20+(i+1)/60
            previous,_=full.step(None,previous,1/60,now,returning=True)
        np.testing.assert_allclose(previous,[2047]*4,atol=.25)
        c['control']['vertical_direction_confirmed']=False
        with self.assertRaises(ValueError):validate_mojia(c,profile(),ROOT/'configs/robots/mojia/calibration.json',False,False)


    def test_requested_900_stage_bounds_loss_and_return(self):
        c=config()
        for a in c['control']['axes']:a.update(negative=1987,positive=2107)
        c['neck']['offset_counts']=80
        validate_mojia(c,profile(),ROOT/'configs/robots/mojia/calibration.json',True,True)
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

    def test_extended_speed_stage_requires_neck_confirmation(self):
        c,p=config(),profile()
        c['control'].update(calibration_confirmed=True,speed_counts_s=300)
        c['neck'].update(speed_counts_s=75,offset_counts=160,
                         gain_counts_s=180,deadband=.14,stop_deadband=.07,filter_tau_s=.12)
        with self.assertRaises(ValueError):validate_mojia(c,p,ROOT/'configs/robots/mojia/calibration.json',True,False)
        c['neck']['direction_confirmed']=True
        validate_mojia(c,p,ROOT/'configs/robots/mojia/calibration.json',True,False)
        with self.assertRaises(ValueError):validate_mojia(c,p,ROOT/'configs/robots/mojia/calibration.json',True,True)
        controller=MojiaPursuit(c,[8,23,32],[300,300,75],True)
        old=np.array([2047.,2047.,2207.])
        for i in range(300):
            old,_=controller.step(None,old,1/60,10+(i+1)/60,returning=True)
        np.testing.assert_allclose(old,[2047]*3,atol=.25)

    def test_calibration_and_commissioning_confirmations(self):
        c,p=config(),profile()
        validate_mojia(c,p,ROOT/'configs/robots/mojia/calibration.json',neck=True,trial=True)
        for changed in ['mount','calibration','profile']:
            a,b=copy.deepcopy(c),copy.deepcopy(p)
            if changed=='mount':a['control']['mount_confirmed']=False
            if changed=='calibration':a['calibration_sha256']='wrong'
            if changed=='profile':b['schema_version']='unsupported'
            with self.assertRaises(ValueError):validate_mojia(a,b,ROOT/'configs/robots/mojia/calibration.json',True,True)
        with self.assertRaises(ValueError):validate_mojia(c,p,ROOT/'configs/robots/mojia/calibration.json',True,False)
        with self.assertRaises(ValueError):validate_config({**c,'schema_version':'unsupported'})

    def test_trial_caps_and_endpoint_directions(self):
        c,p=config(),profile()
        for mutate in [lambda a:a['neck'].update(offset_counts=81),
                       lambda a:a['control']['axes'][0].update(positive=2110),
                       lambda a:a['control']['axes'][1].update(id=8),
                       lambda a:a['neck'].update(sign=True)]:
            a=copy.deepcopy(c);mutate(a)
            with self.assertRaises(ValueError):validate_mojia(a,p,ROOT/'configs/robots/mojia/calibration.json',True,True)

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
        # Reserve assistance may share a stationary offset with the neck;
        # the image loop must converge while reducing the eye's excursion.
        self.assertGreater(old[2],2047)
        self.assertLess(abs(old[1]-2047),initial*40)

    def test_eyes_only_never_adds_neck_axis(self):
        c=config();controller=MojiaPursuit(c,[8,23],[60,60],False)
        new,_=controller.step(packet(10,.9),np.array([2047.,2047.]),1/60,10)
        self.assertEqual(new.shape,(2,))


class MojiaPerformanceTests(unittest.TestCase):
    def test_disabled_vertical_axes_are_not_owned_by_song_or_tracking(self):
        from vivis.motion.ownership import tracking_profile,MojiaPlaybackGaze
        c,p,ids,r=self.setup_performance()
        c['control']['vertical_enabled']=False
        c['control']['axes']=[a for a in c['control']['axes'] if a['id'] not in (7,22)]
        c['control']['held_eye_ids']=[7,22]
        p=tracking_profile(c,p,ids)
        self.assertTrue({7,22}.isdisjoint(p['diagnostic_command_ids']))
        self.assertTrue({7,22}<=set(ids))
        a=MojiaPlaybackGaze(c,p,ids,r)
        self.assertEqual(a.owned,[8,23,32])
        self.assertTrue({7,22}.isdisjoint(a.command_ids))

    def test_slow_preview_disk_does_not_block_control_and_final_state_wins(self):
        import tempfile,time,threading
        from pathlib import Path
        from unittest.mock import patch
        from vivis.motion.ownership import PreviewPublisher
        started=threading.Event();release=threading.Event();original=Path.write_text
        def delayed(path,data,*args,**kwargs):
            started.set();release.wait(timeout=2)
            return original(path,data,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory,patch.object(Path,'write_text',delayed):
            path=Path(directory)/'status.json';publisher=PreviewPublisher(path)
            try:
                publisher.submit(dict(active=True,frame=0))
                self.assertTrue(started.wait(timeout=1))
                begin=time.perf_counter()
                for i in range(100):publisher.submit(dict(active=True,frame=i))
                self.assertLess(time.perf_counter()-begin,.05)
            finally:
                release.set();error=publisher.finish(dict(active=False,state='completed'))
            self.assertIsNone(error)
            self.assertEqual(json.loads(path.read_text()),dict(active=False,state='completed'))

    def test_usb_volume_three_times_is_fixed_reference_and_not_cumulative(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from vivis.runtime.show import configure_show_volume
        sd=Mock();sd.query_devices.return_value={'name':'USB Audio Device: - (hw:0,0)'}
        before='Limits: Playback 0 - 30'
        after='Front Left: Playback 23 [77%] [-10.50dB] [on]\\nFront Right: Playback 23 [77%] [-10.50dB] [on]'
        with patch.dict(sys.modules,{'sounddevice':sd}),patch('vivis.runtime.show.subprocess.run') as run:
            run.side_effect=[SimpleNamespace(stdout=x) for x in [before,after,before,after]]
            gain,record=configure_show_volume('USB Audio Device')
            again,_=configure_show_volume('USB Audio Device')
        self.assertAlmostEqual(gain*10**(12/20),3.)
        self.assertEqual(gain,again);self.assertLess(gain,1.)
        self.assertEqual(run.call_args_list[1].args[0],run.call_args_list[3].args[0])

    def setup_performance(self):
        from vivis.motion.ownership import tracking_profile
        from unittest.mock import Mock
        c=baseline_config();c['control']['camera_eye']='right';p=profile()
        ids=[m['id'] for m in p['motors']]
        p=tracking_profile(c,p,ids)
        receiver=Mock();receiver.poll.side_effect=lambda now:packet(now,.3)
        return c,p,ids,receiver

    def test_left_camera_adapter_observes_only_its_own_eye_encoder(self):
        from vivis.motion.ownership import MojiaPlaybackGaze
        c,p,ids,r=self.setup_performance();c['control']['camera_eye']='left'
        adapter=MojiaPlaybackGaze(c,p,ids,r)
        initial=np.full(len(ids),2047.);initial[ids.index(8)]=2050.;initial[ids.index(23)]=2030.
        adapter.prepare(np.array([initial]),initial,[],10.)
        self.assertEqual(adapter.controller.eye_observation,2050.)
        adapter.observe(23,dict(position=2070),10.1)
        self.assertEqual(adapter.controller.eye_observation,2050.)
        adapter.observe(8,dict(position=2053),10.2)
        self.assertEqual(adapter.controller.eye_observation,2053.)
        self.assertEqual(adapter.controller.eye_motion,1.)

    def test_tracking_changes_only_owned_columns_and_preserves_song(self):
        from vivis.motion.ownership import MojiaPlaybackGaze
        c,p,ids,receiver=self.setup_performance()
        a=MojiaPlaybackGaze(c,p,ids,receiver)
        base=np.full(len(ids),2047.);base[ids.index(31)]=1800.;base[ids.index(29)]=1900.
        initial=np.full(len(ids),2047.);initial[ids.index(32)]=2100.
        ramp=a.prepare(np.array([base]),initial,[],10.)
        self.assertEqual(ramp[0,ids.index(32)],2100.)
        a.set_audio_time(0,113)
        for i in range(120):
            result=a.apply(base,initial,1/60,10+(i+1)/60)
            for j,sid in enumerate(ids):
                if sid not in a.owned:self.assertEqual(result[j],base[j])
            self.assertTrue(1967<=result[ids.index(32)]<=2127)
            initial=result

    def test_wrong_tracking_speed_or_changed_pid_rejected(self):
        from vivis.motion.ownership import MojiaPlaybackGaze
        c,p,ids,r=self.setup_performance();bad=copy.deepcopy(p)
        next(m for m in bad['motors'] if m['id']==22)['max_speed_counts_s']=300
        with self.assertRaisesRegex(ValueError,'preset'):MojiaPlaybackGaze(c,bad,ids,r)
        a=MojiaPlaybackGaze(c,p,ids,r);raw=bytearray(40);raw[18]=4;ram=bytearray(31)
        with self.assertRaisesRegex(RuntimeError,'hardware preset'):
            a.prepare(np.full((3,len(ids)),2047.),np.full(len(ids),2047.),
                [dict(id=32,config_0_39_hex=raw.hex(),state_40_70_hex=ram.hex())],10.)

    def test_combined_matches_standalone_for_same_camera_feedback_and_clock(self):
        """Arbitration must preserve solo eye/neck output and the song's other axes."""
        from vivis.motion.ownership import MojiaPlaybackGaze, tracking_profile
        from vivis.motion.mapping import limit_step
        for held_vertical in (False, True):
            with self.subTest(held_vertical=held_vertical):
                c,p,ids,receiver=self.setup_performance()
                if held_vertical:
                    c['control']['vertical_enabled']=False
                    c['control']['held_eye_ids']=[7,22]
                    c['control']['axes']=[a for a in c['control']['axes'] if a['id'] not in (7,22)]
                    p=tracking_profile(c,p,ids)
                combined=MojiaPlaybackGaze(c,p,ids,receiver)
                owned=combined.owned
                solo=MojiaPursuit(c,owned,combined.speeds,True)
                now=10.;elapsed=0.;duration=15.
                previous=np.full(len(owned),2047.)
                last=np.full(len(ids),2047.)
                combined.prepare(np.array([last]),last,[],now)
                solo.observe_eye(2047,now)
                scheduler=NeckCommandScheduler(2047,now,c['neck']['speed_counts_s'],
                                               adaptive_speed=c['neck']['adaptive_speed'])
                limits=np.array([next(m['max_speed_counts_s'] for m in p['motors'] if m['id']==sid) for sid in ids])
                for i in range(900):
                    # Uneven ticks, reversals, target loss, encoder observations and return.
                    dt=(.008,.025,.017)[i%3];now+=dt;elapsed+=dt
                    target=packet(now,.65*np.sin(elapsed*1.4))
                    if 4<elapsed<6:target['target']=None
                    receiver.poll.side_effect=None;receiver.poll.return_value=target
                    if i%6==0:
                        observed=round(previous[owned.index(23)])
                        solo.observe_eye(observed,now)
                        combined.observe(23,dict(position=observed),now)
                    previous,_=solo.step(target,previous,dt,now,returning=elapsed>=duration-5)
                    expected=previous.copy()
                    expected[-1],_=scheduler.step(previous[-1],now,returning=elapsed>=duration-5)
                    base=np.full(len(ids),2047.)
                    base[ids.index(31)]=1900+20*np.sin(elapsed)
                    combined.set_audio_time(elapsed,duration)
                    actual=combined.apply(base,last,dt,now)
                    np.testing.assert_allclose(actual[combined.columns],expected,atol=1e-9)
                    for j,sid in enumerate(ids):
                        if sid not in owned:self.assertEqual(actual[j],base[j])
                    last=limit_step(last,actual,limits,dt)
                    np.testing.assert_allclose(last[combined.columns],expected,atol=1e-9)

    def test_budgeted_audio_loop_recovers_read_loss_and_samples_face_at_30_or_60(self):
        import tempfile,time
        from pathlib import Path
        from unittest.mock import patch
        from test_runtime import PlaybackTests
        from vivis.motion.ownership import MojiaPlaybackGaze
        from vivis.runtime.playback import play
        from vivis.hardware.feetech.protocol import ReadTimeout,CorruptResponse
        import importlib.util
        cases=[(30,'left',60,False,False),(60,'right',60,False,False),(30,'right',30,False,False)]
        if importlib.util.find_spec('ruckig') is not None:cases.extend([(60,'right',60,True,False),(60,'right',60,True,True)])
        for face_hz,eye,control_hz,upstream,parallel in cases:
            with self.subTest(face_hz=face_hz,camera_eye=eye,control_hz=control_hz,upstream=upstream,parallel=parallel):
                c,p,ids,r=self.setup_performance();fixture,bus,clock=PlaybackTests().fixture()
                c['neck']['upstream_trajectory']['enabled']=upstream
                c['parallel_neck']['enabled']=parallel
                if parallel:
                    from vivis.motion.ownership import tracking_profile
                    canonical=Robot()
                    for motor in p['motors']:
                        if motor['id'] in (33,34):motor['max_speed_counts_s']=canonical.axes[motor['id']]['max_speed_counts_s']
                    p=tracking_profile(c,p,ids)
                c['control']['camera_eye']=eye
                for key in ['audio_device','hardware_stop_description','physical_audio_output_confirmed','actuator_latency_ms','stop_method']:
                    p[key]=fixture[key]
                p.update(serial_backend='feetech_python',feedback_mode='short_registers',
                         scheduled_feedback=True,face_update_hz=face_hz,control_hz=control_hz,
                         feedback_max_age_s=1.,changed_goals_only=False)
                clock.duration=.4
                positions={sid:2047 for sid in ids};motors={m['id']:m for m in p['motors']}
                def state(sid):
                    return dict(position=positions[sid],temperature_c=25,load_raw=0,status_register=0,
                                voltage_raw=120 if 31<=sid<=35 else 60)
                snapshot_loss=[False]
                def snapshot(sid):
                    if sid==2 and not snapshot_loss[0]:
                        snapshot_loss[0]=True
                        raise CorruptResponse('captured ID2 preflight snapshot failure')
                    raw=bytearray(40);raw[18]=4;raw[21:24]=bytes(c['neck']['observed_pid_raw'])
                    raw[26:28]=bytes(c['neck']['observed_deadband_counts'])
                    ram=bytearray(31);ram[1]=c['neck']['observed_acceleration_raw']
                    return dict(id=sid,model_raw=motors[sid]['model_raw_expected'],mode=0,torque_enable=1,
                        configured_min=0,configured_max=4095,position_raw=2047,**{k:v for k,v in state(sid).items() if k!='position'},
                        config_0_39_hex=raw.hex(),state_40_70_hex=ram.hex())
                startup_loss=[False]
                def startup_state(sid):
                    if sid==4 and not startup_loss[0]:
                        startup_loss[0]=True
                        time.sleep(.11)  # Recovery must reset the ramp's step clock.
                        raise CorruptResponse('captured ID4 startup failure')
                    return state(sid)
                bus.snapshot.side_effect=snapshot;bus.feedback.side_effect=startup_state
                bus.goals.side_effect=lambda goals,speeds:positions.update(goals)
                attempts=[0]
                def read(sid,address,count):
                    attempts[0]+=1
                    if attempts[0]==23:raise ReadTimeout('injected one lost field')
                    s=state(sid)
                    return {56:s['position'].to_bytes(2,'little'),60:b'\x00\x00',
                            62:bytes([s['voltage_raw'],s['temperature_c']]),65:b'\x00'}[address]
                bus.read.side_effect=read
                trajectory=np.full((24,len(ids)),2047.)
                trajectory[:,ids.index(31)]=2000-np.arange(24)
                a=MojiaPlaybackGaze(c,p,ids,r)
                with tempfile.TemporaryDirectory() as directory,patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus),patch('vivis.runtime.playback.AudioClock',return_value=clock):
                    result=play(p,ids,trajectory,60,'fake',Path(directory)/'log.json',Path(directory)/'STOP',gaze_override=a)
                self.assertTrue(result['completed'])
                self.assertTrue(all(e['person_gaze']['upstream_trajectory']['enabled']==upstream
                                    for e in result['events']))
                self.assertEqual(result['control_hz'],control_hz)
                self.assertEqual(result['playback_fps'],60)
                self.assertEqual(result['runtime_feedback_schedule']['budget_ms'],1000/control_hz)
                self.assertEqual(result['runtime_feedback_schedule']['budget_mode'],'remaining_control_period')
                self.assertEqual(result['runtime_feedback_schedule']['priority_ids'],[a.camera_eye_id,32]+([33,34] if parallel else []))
                self.assertTrue(startup_loss[0])
                self.assertEqual({r['id'] for r in result['startup_communication_recoveries']},{2,4})
                self.assertTrue(all(r['recovered'] for r in result['startup_communication_recoveries']))
                bus.set_read_policy.assert_called_once_with(3,retry=False)
                self.assertEqual(sum(len(e.get('feedback_errors',[])) for e in result['events']),1)
                for e in result['events']:
                    self.assertLessEqual(e['feedback_schedule']['reads'],8*60/control_hz)
                    source=e.get('face_frame',e['frame'])
                    if face_hz==30:self.assertEqual(source%2,0)
                    self.assertEqual(e['goals'][31],trajectory[source,ids.index(31)])
                self.assertGreaterEqual(len(result['events']),int(control_hz*.33))
                stamps=[e['sent_monotonic'] for e in result['events']]
                self.assertAlmostEqual(float(np.median(np.diff(stamps))),1/control_hz,delta=.004)
                samples=[s for e in result['events'] for s in e.get('feedback_samples',[])]
                for sid in (a.camera_eye_id,32):self.assertGreaterEqual(sum(s['id']==sid for s in samples),3)

    def test_audio_loop_keeps_jaw_and_prioritizes_eye_feedback(self):
        import tempfile,time,gc
        from pathlib import Path
        from unittest.mock import patch
        from test_runtime import PlaybackTests
        from vivis.motion.ownership import MojiaPlaybackGaze
        from vivis.runtime.playback import play
        c,p,ids,r=self.setup_performance();fixture,bus,clock=PlaybackTests().fixture()
        for key in ['audio_device','hardware_stop_description','physical_audio_output_confirmed','actuator_latency_ms','stop_method']:
            p[key]=fixture[key]
        p.pop('serial_backend',None);p['changed_goals_only']=False;clock.duration=.3
        positions={sid:2047 for sid in ids};reads=[];motors={m['id']:m for m in p['motors']}
        def snapshot(sid):
            raw=bytearray(40);raw[18]=4;raw[21:24]=bytes(c['neck']['observed_pid_raw']);raw[26:28]=bytes(c['neck']['observed_deadband_counts'])
            ram=bytearray(31);ram[1]=c['neck']['observed_acceleration_raw']
            return dict(id=sid,model_raw=motors[sid]['model_raw_expected'],mode=0,torque_enable=1,
                configured_min=0,configured_max=4095,position_raw=2047,temperature_c=25,load_raw=0,status_register=0,
                voltage_raw=120 if sid in [31,32,33,34,35] else 60,
                config_0_39_hex=raw.hex(),state_40_70_hex=ram.hex())
        bus.snapshot.side_effect=snapshot
        bus.goals.side_effect=lambda goals,speeds:positions.update(goals)
        def feedback(sid):
            reads.append((sid,time.monotonic()))
            return dict(position=positions[sid],temperature_c=25,load_raw=0,status_register=0,
                        voltage_raw=120 if sid in [31,32,33,34,35] else 60)
        bus.feedback.side_effect=feedback
        a=MojiaPlaybackGaze(c,p,ids,r)
        gc_before=gc.isenabled();original_clock=clock.time.side_effect
        def audio_time():
            self.assertFalse(gc.isenabled())
            return original_clock()
        clock.time.side_effect=audio_time
        trajectory=np.full((18,len(ids)),2047.)
        trajectory[:,ids.index(31)]=2047-np.arange(18)
        with tempfile.TemporaryDirectory() as directory,patch('vivis.runtime.playback.Bus',return_value=bus),patch('vivis.runtime.playback.AudioClock',return_value=clock):
            result=play(p,ids,trajectory,60,'fake',Path(directory)/'log.json',Path(directory)/'STOP',gaze_override=a)
        self.assertTrue(result['completed'])
        self.assertEqual(gc.isenabled(),gc_before)
        self.assertTrue(result['cyclic_gc_suspended_during_audio'])
        for event in result['events']:
            self.assertEqual(event['goals'][31],trajectory[event['frame'],ids.index(31)])
            self.assertNotIn(33,event['goals']);self.assertNotIn(34,event['goals'])
            self.assertEqual(event['person_gaze']['eye_motion_source'],'encoder')
        stamps=[t for sid,t in reads if sid==23]
        self.assertGreaterEqual(len(stamps),3)
        self.assertLess(max(np.diff(stamps)),.16)


class BudgetedFeedbackTests(unittest.TestCase):
    def setup_reader(self,latency=.0002):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from vivis.runtime.feedback import BudgetedFeedback
        now=[1.];samples=[];errors=[];bus=Mock()
        payload={56:b'\xff\x07',60:b'\x00\x00',62:b'\x78\x28',65:b'\x00'}
        def read(sid,address,count):
            now[0]+=latency
            return payload[address]
        bus.read.side_effect=read
        reader=BudgetedFeedback(bus,[2,23,32]);clocks={sid:.9 for sid in reader.ids}
        def sample(sid,state,started):
            samples.append((sid,state,started));clocks[sid]=now[0]
        return reader,now,SimpleNamespace(monotonic=lambda:now[0]),clocks,sample,samples,errors

    def test_priorities_and_partial_samples_preserve_full_feedback(self):
        from unittest.mock import patch
        reader,now,clock,clocks,sample,samples,errors=self.setup_reader()
        with patch('vivis.runtime.feedback.time',clock):
            result=reader.step(1.007,clocks,sample,lambda *a:errors.append(a))
        self.assertEqual([s[0] for s in samples],[23,32])
        self.assertEqual(result['reads'],8)
        self.assertEqual(samples[0][1],dict(position=2047,load_raw=0,voltage_raw=120,temperature_c=40,status_register=0))
        reader,now,clock,clocks,sample,samples,errors=self.setup_reader(.001)
        with patch('vivis.runtime.feedback.time',clock):
            reader.step(1.005,clocks,sample,lambda *a:errors.append(a))
            self.assertFalse(samples);self.assertEqual(clocks[23],.9)
            now[0]=1.0167
            reader.step(now[0]+.007,clocks,sample,lambda *a:errors.append(a))
        self.assertEqual(samples[0][0],23)
        self.assertLess(samples[0][2],1.0167)

    def test_timeout_discards_partial_and_services_others_without_same_tick_retry(self):
        from unittest.mock import patch
        from vivis.hardware.feetech.protocol import ReadTimeout,ProtocolError
        reader,now,clock,clocks,sample,samples,errors=self.setup_reader()
        original=reader.bus.read.side_effect;failed=False
        def read(sid,address,count):
            nonlocal failed
            if sid==23 and address==60 and not failed:
                failed=True;now[0]+=.003
                raise ReadTimeout('injected')
            return original(sid,address,count)
        reader.bus.read.side_effect=read
        with patch('vivis.runtime.feedback.time',clock):
            result=reader.step(1.007,clocks,sample,lambda *a:errors.append(a))
        self.assertLessEqual(result['reads'],8)
        self.assertGreater(result['reads'],2);self.assertNotIn(23,[s[0] for s in samples])
        self.assertEqual(result['failed_ids'],[23]);self.assertNotEqual(reader.sid,23)
        self.assertEqual([c.args[1] for c in reader.bus.read.call_args_list if c.args[0]==23],[56,60])
        self.assertEqual(clocks[23],.9);self.assertEqual(len(errors),1)
        reader.bus.read.side_effect=ProtocolError('servo alarm')
        now[0]=1.0167
        with patch('vivis.runtime.feedback.time',clock),self.assertRaises(ProtocolError):
            reader.step(now[0]+.007,clocks,sample,lambda *a:errors.append(a))

    def test_deadline_does_not_start_an_unbudgeted_transaction(self):
        from unittest.mock import patch
        reader,now,clock,clocks,sample,samples,errors=self.setup_reader()
        with patch('vivis.runtime.feedback.time',clock):
            reader.step(1.003,clocks,sample,lambda *a:errors.append(a))
        reader.bus.read.assert_not_called()

    def test_30hz_slot_can_service_more_axes_when_slack_exists_without_exceeding_cap(self):
        from unittest.mock import patch
        reader,now,clock,clocks,sample,samples,errors=self.setup_reader(.001)
        reader.max_requests_per_step=16
        with patch('vivis.runtime.feedback.time',clock):
            result=reader.step(1.022,clocks,sample,lambda *a:errors.append(a))
        self.assertEqual(result['reads'],16)
        self.assertEqual(len(samples),4)
        self.assertLessEqual(now[0],1.022)

    def test_failed_priority_axis_does_not_starve_healthy_axes_or_refresh_valid_clock(self):
        from unittest.mock import patch
        from vivis.hardware.feetech.protocol import CorruptResponse
        reader,now,clock,clocks,sample,samples,errors=self.setup_reader()
        original=reader.bus.read.side_effect
        def read(sid,address,count):
            if sid==23:
                now[0]+=.003
                raise CorruptResponse('persistent priority-axis damage')
            return original(sid,address,count)
        reader.bus.read.side_effect=read
        with patch('vivis.runtime.feedback.time',clock):
            reader.step(now[0]+.007,clocks,sample,lambda *a:errors.append(a))
            self.assertEqual(clocks[23],.9)
            self.assertGreater(reader.last_attempt[23],.9)
            now[0]=1.0167
            reader.step(now[0]+.007,clocks,sample,lambda *a:errors.append(a))
        self.assertIn(32,[s[0] for s in samples]);self.assertIn(2,[s[0] for s in samples])
        self.assertEqual(clocks[23],.9)

    def test_approaching_watchdog_deadline_overrides_eye_and_neck_priority(self):
        from unittest.mock import patch
        reader,now,clock,clocks,sample,samples,errors=self.setup_reader()
        clocks[2]=.2  # 0.8s old; priority axes are also due but only0.1s old.
        with patch('vivis.runtime.feedback.time',clock):
            result=reader.step(now[0]+.007,clocks,sample,lambda *a:errors.append(a))
        self.assertEqual(samples[0][0],2);self.assertEqual(result['urgent_ids'],[2])
        self.assertLess(now[0]-.2,1.)

    def test_urgent_retry_discards_damaged_whole_sample_and_recovers_on_next_tick(self):
        from unittest.mock import patch
        from vivis.hardware.feetech.protocol import CorruptResponse
        reader,now,clock,clocks,sample,samples,errors=self.setup_reader()
        clocks[2]=.2;original=reader.bus.read.side_effect;failed=False
        def read(sid,address,count):
            nonlocal failed
            if sid==2 and address==65 and not failed:
                failed=True;now[0]+=.003
                raise CorruptResponse('captured failure at final field')
            return original(sid,address,count)
        reader.bus.read.side_effect=read
        with patch('vivis.runtime.feedback.time',clock):
            reader.step(now[0]+.007,clocks,sample,lambda *a:errors.append(a))
            self.assertNotIn(2,[s[0] for s in samples]);self.assertEqual(clocks[2],.2)
            self.assertNotEqual(reader.sid,2)
            now[0]=1.0167
            reader.step(now[0]+.007,clocks,sample,lambda *a:errors.append(a))
        self.assertIn(2,[s[0] for s in samples]);self.assertGreater(clocks[2],1.)
        self.assertEqual([c.args[1] for c in reader.bus.read.call_args_list if c.args[0]==2],[56,60,62,65]*2)

    def test_failed_urgent_axis_rotates_to_other_urgent_axes_without_refreshing_validity(self):
        from unittest.mock import patch
        from vivis.hardware.feetech.protocol import CorruptResponse
        reader,now,clock,clocks,sample,samples,errors=self.setup_reader()
        clocks.update({2:.1,23:.15,32:.2});original=reader.bus.read.side_effect
        def read(sid,address,count):
            if sid==2:
                now[0]+=.003
                raise CorruptResponse('oldest urgent axis repeatedly damaged')
            return original(sid,address,count)
        reader.bus.read.side_effect=read
        with patch('vivis.runtime.feedback.time',clock):
            first=reader.step(now[0]+.007,clocks,sample,lambda *a:errors.append(a))
            now[0]=1.+1/60
            second=reader.step(now[0]+.007,clocks,sample,lambda *a:errors.append(a))
        self.assertEqual(clocks[2],.1)
        self.assertTrue({23,32}<={s[0] for s in samples})
        self.assertLessEqual(first['reads'],8);self.assertLessEqual(second['reads'],8)
        self.assertLessEqual(len([c for c in reader.bus.read.call_args_list if c.args[0]==2]),2)

    def test_persistently_dead_axis_remains_stale_while_healthy_axes_are_serviced(self):
        from unittest.mock import patch
        from vivis.hardware.feetech.protocol import ReadTimeout
        reader,now,clock,clocks,sample,samples,errors=self.setup_reader()
        original=reader.bus.read.side_effect
        def read(sid,address,count):
            if sid==23:
                now[0]+=.003
                raise ReadTimeout('axis disconnected')
            return original(sid,address,count)
        reader.bus.read.side_effect=read
        with patch('vivis.runtime.feedback.time',clock):
            for tick in range(56):
                now[0]=1.+tick/60
                reader.step(now[0]+.007,clocks,sample,lambda *a:errors.append(a))
        self.assertEqual(clocks[23],.9)
        self.assertEqual([sid for sid in reader.ids if now[0]-clocks[sid]>1.],[23])
