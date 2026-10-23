"""Mojia-only gaze limits, mounting modes, loss and eye/head feedback simulations."""
import copy
import json
import unittest
import numpy as np

from xiaobai_runtime.mapping import ROOT
from xiaobai_runtime.mojia_gaze import validate_mojia, MojiaPursuit, NeckCommandScheduler
from xiaobai_runtime.person_gaze import VERSION, validate_config
from prepare_mojia import proposal, hardware_profile
from xiaobai_runtime.mapping import inputs
from mojia_track_person import tracking_start_position


def config():
    c=json.loads((ROOT/'mojia/person_gaze.json').read_text())
    c['control'].update(camera_mount='eye',camera_eye='right',mount_confirmed=True,vertical_enabled=False)
    c['control']['axes']=[a for a in c['control']['axes'] if a['coordinate']=='x']
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
    def test_growing_error_recruits_neck_after_eyes_and_preserves_eye_outputs(self):
        c=json.loads((ROOT/'mojia/person_gaze.json').read_text())
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

    def test_decreasing_error_keeps_eyes_in_charge(self):
        controller=MojiaPursuit(config(),[8,23,32],[300,300,900],True)
        old=np.full(3,2047.)
        for i in range(45):
            now=10+(i+1)/60
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
            validate_mojia(c,profile(),ROOT/'mojia/墨甲9.23.json',True,True)

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
        c=json.loads((ROOT/'mojia/person_gaze.json').read_text())
        validate_mojia(c,profile(),ROOT/'mojia/墨甲9.23.json',False,True)
        for value in [True,0,-1,301,float('nan')]:
            invalid=copy.deepcopy(c);invalid['control']['axes'][-1]['speed_counts_s']=value
            with self.assertRaises(ValueError):validate_mojia(invalid,profile(),ROOT/'mojia/墨甲9.23.json',False,True)
        controller=MojiaPursuit(c,[8,23,7,22],[300,300,100,100])
        previous=np.full(4,2047.)
        for i in range(600):
            now=10+(i+1)/60;p=packet(now,1.);p['target']['y']=1. if i<300 else -1.
            new,_=controller.step(p,previous,1/60,now)
            self.assertTrue(np.all(abs(new[2:]-2047)<=20))
            self.assertTrue(np.all(abs(new-previous)<=np.array([300,300,100,100])/60+1e-8))
            previous=new

    def test_vertical_pair_opposite_encoders_and_horizontal_independence(self):
        c=json.loads((ROOT/'mojia/person_gaze.json').read_text())
        validate_mojia(c,profile(),ROOT/'mojia/墨甲9.23.json',False,True)
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
        with self.assertRaises(ValueError):validate_mojia(c,profile(),ROOT/'mojia/墨甲9.23.json',False,False)

    def test_eye_only_resume_does_not_require_or_recenter_held_neck(self):
        held=dict(id=32,position_raw=2095,goal_position_raw=2095,configured_min=1600,configured_max=2600)
        self.assertIsNone(tracking_start_position(held))
        eye=dict(id=23,position_raw=2106,goal_position_raw=2106)
        self.assertEqual(tracking_start_position(eye,dict(safe_min=1987,safe_max=2107)),2106.)
        with self.assertRaises(RuntimeError):tracking_start_position({**eye,'goal_position_raw':2200},dict(safe_min=1987,safe_max=2107))
        self.assertEqual(tracking_start_position(held,dict(safe_min=1967,safe_max=2127)),2095.)

    def test_requested_900_stage_bounds_loss_and_return(self):
        c=config()
        for a in c['control']['axes']:a.update(negative=1987,positive=2107)
        c['neck']['offset_counts']=80
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
                         gain_counts_s=180,deadband=.14,stop_deadband=.07,filter_tau_s=.12)
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
        # A stationary target that the eyes can acquire needs no neck motion.
        self.assertAlmostEqual(old[2],2047)

    def test_eyes_only_never_adds_neck_axis(self):
        c=config();controller=MojiaPursuit(c,[8,23],[60,60],False)
        new,_=controller.step(packet(10,.9),np.array([2047.,2047.]),1/60,10)
        self.assertEqual(new.shape,(2,))


class MojiaPerformanceTests(unittest.TestCase):
    def setup_performance(self):
        from xiaobai_runtime.mojia_performance import tracking_profile
        from unittest.mock import Mock
        c=json.loads((ROOT/'mojia/person_gaze.json').read_text());p=profile()
        ids=[m['id'] for m in p['motors']]
        p=tracking_profile(c,p,ids)
        receiver=Mock();receiver.poll.side_effect=lambda now:packet(now,.3)
        return c,p,ids,receiver

    def test_tracking_changes_only_owned_columns_and_preserves_song(self):
        from xiaobai_runtime.mojia_performance import MojiaPlaybackGaze
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
        from xiaobai_runtime.mojia_performance import MojiaPlaybackGaze
        c,p,ids,r=self.setup_performance();bad=copy.deepcopy(p)
        next(m for m in bad['motors'] if m['id']==22)['max_speed_counts_s']=300
        with self.assertRaisesRegex(ValueError,'preset'):MojiaPlaybackGaze(c,bad,ids,r)
        a=MojiaPlaybackGaze(c,p,ids,r);raw=bytearray(40);raw[18]=4;ram=bytearray(31)
        with self.assertRaisesRegex(RuntimeError,'hardware preset'):
            a.prepare(np.full((3,len(ids)),2047.),np.full(len(ids),2047.),
                [dict(id=32,config_0_39_hex=raw.hex(),state_40_70_hex=ram.hex())],10.)

    def test_audio_loop_keeps_jaw_and_prioritizes_eye_feedback(self):
        import tempfile,time,gc
        from pathlib import Path
        from unittest.mock import patch
        from test_xiaobai import PlaybackTests
        from xiaobai_runtime.mojia_performance import MojiaPlaybackGaze
        from xiaobai_runtime.playback import play
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
        with tempfile.TemporaryDirectory() as directory,patch('xiaobai_runtime.playback.Bus',return_value=bus),patch('xiaobai_runtime.playback.AudioClock',return_value=clock):
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


if __name__=='__main__':unittest.main()
