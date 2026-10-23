"""Largest-face selection, stale input, horizontal control and playback integration."""
import copy
import json
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from urllib.request import urlopen

import numpy as np

from xiaobai_runtime.mapping import ROOT, limit_step
from xiaobai_runtime.person_gaze import (GazeOverride, GazePolicy, TargetReceiver, VERSION,
                                        axis_normalized, axis_position, validate_config, validate_packet,
                                        prepare_direction_trial, prepare_neck_follow_trial, NeckFollowTrial)
from xiaobai_runtime.person_vision import select_largest


def configuration():
    config = json.loads((ROOT / 'xiaobai/person_gaze.json').read_text())
    config['control'].update(camera_mount='fixed', mount_confirmed=True, calibration_confirmed=True)
    return config


def packet(stamp, x=.5, y=0., sequence=1):
    return {'schema_version': VERSION, 'sequence': sequence, 'timestamp_monotonic': stamp,
            'target': None if x is None else {'x': x, 'y': y},
            'camera_transform': {'rotation_deg': 0, 'mirror_x': False}}


def profile():
    return json.loads((ROOT / 'xiaobai/playback_profile_0001_observation.json').read_text())


def face(x, y, width, height, score=.95):
    return [x, y, width, height, x + width * .3, y + height * .35,
            x + width * .7, y + height * .35, x + width * .5, y + height * .5,
            x + width * .3, y + height * .8, x + width * .7, y + height * .8, score]


class SelectionTests(unittest.TestCase):
    def test_area_wins_over_confidence_and_distance_to_center(self):
        selected, faces = select_largest([face(300, 200, 40, 50, .99), face(10, 10, 150, 180, .85)], 640, 480)
        self.assertEqual(len(faces), 2)
        self.assertEqual(selected['bbox'], [10, 10, 150, 180])
        self.assertAlmostEqual(selected['area_ratio'], 150 * 180 / (640 * 480))
        self.assertAlmostEqual(selected['point'][1], 10 + 180 * .35)

    def test_clipped_area_and_landmark_fallback(self):
        outside = face(-190, 10, 200, 300)
        inside = face(200, 100, 100, 100)
        inside[4:8] = [-10, -10, -10, -10]
        selected, _ = select_largest([outside, inside], 640, 480)
        self.assertEqual(selected['bbox'], [200, 100, 100, 100])
        self.assertEqual(selected['point'], [250, 150])

    def test_bad_tiny_low_score_and_offscreen_faces_do_not_select(self):
        for faces in [None, [], [face(0, 0, 1, 1)], [face(0, 0, 100, 100, .5)],
                      [face(700, 10, 100, 100)], [[float('nan')] * 15]]:
            self.assertEqual(select_largest(faces, 640, 480), (None, []))


class TransportTests(unittest.TestCase):
    def test_explicit_status_recording_retains_only_valid_sdk_payloads(self):
        from xiaobai_runtime.feetech_sdk import OfficialSDKBus,COMM_SUCCESS,COMM_RX_CORRUPT,COMM_RX_TIMEOUT
        from xiaobai_runtime.protocol import ServoStatusError,CorruptResponse,ReadTimeout
        bus=OfficialSDKBus.__new__(OfficialSDKBus);bus.sdk=Mock()
        bus.servo_status_notifications=[]
        bus.sdk.readTxRx.return_value=([255,7],COMM_SUCCESS,1)
        with self.assertRaises(ServoStatusError):bus._read_once(4,56,2)
        bus.servo_status_policy='record_only'
        self.assertEqual(bus._read_once(4,56,2),b'\xff\x07')
        self.assertEqual(bus.servo_status_notifications[0]['servo_status'],1)
        for result,exc in [(COMM_RX_CORRUPT,CorruptResponse),(COMM_RX_TIMEOUT,ReadTimeout)]:
            bus.sdk.readTxRx.return_value=([],result,1)
            with self.assertRaises(exc):bus._read_once(4,56,2)
        bus.sdk.readTxRx.return_value=([255],COMM_SUCCESS,1)
        with self.assertRaises(CorruptResponse):bus._read_once(4,56,2)
        self.assertEqual(len(bus.servo_status_notifications),1)

    def test_packet_rejects_old_future_nonfinite_or_non_numeric_targets(self):
        for sample in [packet(9.), packet(10.01), packet(float('nan')), packet(10., float('nan')),
                       packet(10., True), packet(10., 1.1), packet(10., .5, float('inf'))]:
            with self.assertRaises(ValueError):
                validate_packet(sample, 10., .3)
        validate_packet(packet(9.9, None), 10., .3)

    def test_real_udp_ignores_malformed_and_out_of_order_and_has_one_receiver(self):
        config = configuration()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', 0))
            config['transport']['port'] = probe.getsockname()[1]
        receiver = TargetReceiver(config)
        try:
            with self.assertRaises(OSError):
                TargetReceiver(config)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                now = time.monotonic()
                address = ('127.0.0.1', config['transport']['port'])
                for data in [b'not-json', b'null', json.dumps(packet(now - .01, .8, sequence=2)).encode(),
                             json.dumps(packet(now - .02, -.5)).encode()]:
                    sender.sendto(data, address)
                latest = receiver.poll(now)
                self.assertEqual(latest['target']['x'], .8)
                self.assertEqual(receiver.rejected, 3)
                begin = time.monotonic()
                self.assertIs(receiver.poll(now), latest)
                self.assertLess(time.monotonic() - begin, .05)
        finally:
            receiver.close()

    def test_no_service_readiness_fails_without_opening_actuators(self):
        config = configuration()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', 0))
            config['transport']['port'] = probe.getsockname()[1]
        receiver = TargetReceiver(config)
        try:
            with self.assertRaisesRegex(RuntimeError, 'No fresh camera packets'):
                receiver.wait_ready(seconds=.02)
        finally:
            receiver.close()

    def test_orientation_mismatch_cannot_drive_a_reversed_eye_command(self):
        config = configuration()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', 0))
            config['transport']['port'] = probe.getsockname()[1]
        receiver = TargetReceiver(config)
        try:
            data = packet(time.monotonic())
            data['camera_transform']['mirror_x'] = True
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(json.dumps(data).encode(), ('127.0.0.1', config['transport']['port']))
                self.assertIsNone(receiver.poll())
                self.assertEqual(receiver.rejected, 1)
        finally:
            receiver.close()


class ControlTests(unittest.TestCase):
    def test_xiaobai_tracking_and_neck_setup_reject_mojia_profile(self):
        p=profile();p['schema_version']='mojia-playback-0.1'
        with self.assertRaisesRegex(ValueError,'Mojia'):
            validate_config(configuration(),live=True,profile=p,ids=[8,23])
        with self.assertRaisesRegex(ValueError,'Mojia'):
            prepare_neck_follow_trial(p,32)

    def test_expanded_trial_is_bounded_and_keeps_confirmation_pending(self):
        c=configuration();c['control']['calibration_confirmed']=False
        c,p=prepare_direction_trial(c,profile(),80)
        validate_config(c,live=True,profile=p,ids=[8,23],direction_trial=True)
        self.assertFalse(c['control']['calibration_confirmed'])
        for a in c['control']['axes']:
            self.assertEqual([a['negative'],a['neutral'],a['positive']],[1967.,2047,2127.])
            m=next(m for m in p['motors'] if m['id']==a['id'])
            self.assertEqual(m['max_speed_counts_s'],120)
        for offset in [0,120,True,80.0]:
            with self.assertRaises(ValueError):prepare_direction_trial(c,p,offset)
        next(m for m in p['motors'] if m['id']==8)['max_speed_counts_s']=121
        with self.assertRaisesRegex(ValueError,'Direction trial'):
            validate_config(c,live=True,profile=p,ids=[8,23],direction_trial=True)

    def test_direction_trial_does_not_claim_confirmed_calibration_or_change_other_axes(self):
        original_config=configuration();original_config['control']['calibration_confirmed']=False
        original_profile=profile();c,p=prepare_direction_trial(original_config,original_profile)
        self.assertFalse(c['control']['calibration_confirmed'])
        validate_config(c,live=True,profile=p,ids=[8,23],direction_trial=True)
        with self.assertRaises(ValueError):validate_config(c,live=True,profile=p,ids=[8,23])
        for before,after in zip(original_profile['motors'],p['motors']):
            if before['id'] not in [8,23]:self.assertEqual(before,after)
        self.assertEqual(original_profile,profile())
        self.assertFalse(original_config['control']['calibration_confirmed'])

    def test_direction_trial_rejects_wide_positions_or_speed(self):
        c,p=prepare_direction_trial(configuration(),profile())
        for key,value in [('safe_max',2100),('max_speed_counts_s',61)]:
            changed=copy.deepcopy(p)
            next(m for m in changed['motors'] if m['id']==8)[key]=value
            with self.assertRaisesRegex(ValueError,'Direction trial'):
                validate_config(c,live=True,profile=changed,ids=[8,23],direction_trial=True)

    def test_live_rejects_unconfirmed_mount_bad_axis_and_limits(self):
        config = configuration()
        validate_config(config, live=True, profile=profile(), ids=[8, 23])
        mutations = [lambda c: c['control'].update(mount_confirmed=False),
                     lambda c: c['control'].update(calibration_confirmed=False),
                     lambda c: c['control'].update(camera_mount='eye', camera_eye='unconfirmed'),
                     lambda c: c['control']['axes'][1].update(id=8),
                     lambda c: c['control']['axes'][0].update(id=9),
                     lambda c: c['control']['axes'][0].update(negative=0),
                     lambda c: c['control']['axes'][0].update(negative=2047),
                     lambda c: c['control']['axes'][0].update(positive=float('nan'))]
        for mutation in mutations:
            c = copy.deepcopy(config)
            mutation(c)
            with self.assertRaises(ValueError):
                validate_config(c, live=True, profile=profile(), ids=[8, 9, 23])

    def test_asymmetric_and_reversed_encoder_roundtrip(self):
        for axis in [{'neutral': 2047, 'negative': 1900, 'positive': 2400},
                     {'neutral': 2047, 'negative': 2300, 'positive': 1800}]:
            for u in [-1., -.5, 0., .1, 1.]:
                self.assertAlmostEqual(axis_normalized(axis, axis_position(axis, u)), u)

    def test_deadband_and_vertical_movement_do_not_move_horizontal_eyes(self):
        policy = GazePolicy(configuration())
        for step in range(60):
            now = 10 + step / 60
            xy, _, _ = policy.step(packet(now, .02, .9), now, 1 / 60, np.zeros(2))
        np.testing.assert_equal(xy, [0., 0.])

    def test_static_camera_converges_and_loss_smoothly_releases(self):
        config = configuration()
        policy = GazePolicy(config)
        for step in range(120):
            now = 10 + step / 60
            xy, weight, state = policy.step(packet(now, .8), now, 1 / 60, policy.xy)
        self.assertEqual(weight, 1.)
        self.assertEqual(state, 'tracking')
        self.assertAlmostEqual(xy[0], (.8 - config['control']['deadband']) * config['control']['fixed_gain'], places=5)
        old_packet = packet(now, .8)
        held = policy.step(old_packet, now + .4, 1 / 60, policy.xy)
        self.assertEqual(held[2], 'holding')
        weights = []
        for step in range(120):
            result = policy.step(old_packet, now + 1 + step / 60, 1 / 60, policy.xy)
            weights.append(result[1])
        self.assertTrue(all(a >= b for a, b in zip(weights, weights[1:])))
        self.assertEqual(weights[-1], 0.)

    def test_fixed_camera_response_in_seconds_is_consistent_at_30_and_60_fps(self):
        responses = {}
        config = configuration()
        axes = config['control']['axes']
        destination = np.array([axis_position(a, .8 - config['control']['deadband'])
                                for a in axes])
        for fps in (30, 60):
            receiver = Mock(rejected=0)
            override = GazeOverride(config, [8, 23], profile(), receiver)
            previous = np.full(2, 2047.)
            reached = None
            for step in range(fps * 2):
                now = 10 + (step + 1) / fps
                receiver.poll.return_value = packet(now, .8)
                goal = override.apply(previous, previous, 1 / fps, now)
                current = limit_step(previous, goal, np.full(2, 300.), 1 / fps)
                self.assertTrue(np.all(np.abs(current - previous) <= 300 / fps + 1e-9))
                if reached is None and np.all(current >= 2047 + .9 * (destination - 2047)):
                    reached = (step + 1) / fps
                previous = current
            self.assertIsNotNone(reached)
            np.testing.assert_allclose(previous, destination, atol=.01)
            responses[fps] = reached
            override.close()
        self.assertLessEqual(abs(responses[30] - responses[60]), 1 / 30 + 1e-9)

    def test_override_changes_only_8_and_23_and_does_not_mutate_the_clip(self):
        config = configuration()
        ids = list(range(2, 35))
        source = np.full(len(ids), 2047.)
        receiver = Mock(rejected=0)
        override = GazeOverride(config, ids, profile(), receiver)
        for step in range(90):
            now = 10 + step / 60
            receiver.poll.return_value = packet(now, .8, -.9)
            result = override.apply(source, source, 1 / 60, now)
        changed = [sid for sid, before, after in zip(ids, source, result) if before != after]
        self.assertEqual(changed, [8, 23])
        np.testing.assert_equal(source, np.full(len(ids), 2047.))
        override.close()
        receiver.close.assert_called_once()

    def test_eye_mounted_camera_keeps_centered_face_at_nonzero_eye_position(self):
        config = configuration()
        config['control'].update(camera_mount='eye', camera_eye='left')
        receiver = Mock(rejected=0)
        override = GazeOverride(config, [8, 23], profile(), receiver)
        previous = np.array([axis_position(a, .4) for a in config['control']['axes']])
        for step in range(120):
            now = 10 + step / 60
            receiver.poll.return_value = packet(now, 0.)
            result = override.apply(previous, previous, 1 / 60, now)
            np.testing.assert_allclose(result, previous, atol=1e-9)

    def test_eye_mounted_visual_loop_converges_with_common_rate_limiter(self):
        config = configuration()
        config['control'].update(camera_mount='eye', camera_eye='left')
        receiver = Mock(rejected=0)
        override = GazeOverride(config, [8, 23], profile(), receiver)
        axes = config['control']['axes']
        previous = np.array([2047., 2047.])
        for step in range(600):
            now = 10 + step / 60
            actual = axis_normalized(axes[0], previous[0])
            receiver.poll.return_value = packet(now, .65 - actual)
            goal = override.apply(previous, previous, 1 / 60, now)
            previous = limit_step(previous, goal, np.full(2, 300.), 1 / 60)
        self.assertAlmostEqual(axis_normalized(axes[0], previous[0]), .65, delta=.05)
        self.assertTrue(np.all(previous < [a['positive'] for a in axes]))


class NeckFollowTests(unittest.TestCase):
    def test_triple_neck_speed_keeps_double_acceleration_eyes_and_range(self):
        runs={}
        for multiplier in (2,3):
            c,r,p,ids=self.setup_trial(offset_counts=160,duration=20,full_song=True,speed_multiplier=multiplier)
            previous=np.full(len(ids),2047.);source=previous.copy();source[2]=2090.
            speeds=np.array([next(m['max_speed_counts_s'] for m in p['motors'] if m['id']==i) for i in ids])
            eyes=[];peak_speed=0.;cross80=None;neck=ids.index(32)
            for step in range(20*60):
                now=10+(step+1)/60;r.poll.return_value=packet(now,.8)
                old_velocity=c.velocity
                result=c.apply(source,previous,1/60,now)
                self.assertEqual(result[2],2090.)
                np.testing.assert_equal(result[-2:],[2047.,2047.])
                self.assertTrue(1887<=result[neck]<=2207)
                if min(abs(previous[neck]-1887),abs(previous[neck]-2207))>1:
                    self.assertLessEqual(abs(c.velocity-old_velocity),70/60+1e-9)
                previous=limit_step(previous,result,speeds,1/60)
                peak_speed=max(peak_speed,abs(c.velocity));eyes.append(previous[:2].copy())
                if cross80 is None and previous[neck]>=2127:cross80=(step+1)/60
            self.assertAlmostEqual(previous[neck],2047.)
            self.assertAlmostEqual(peak_speed,25*multiplier)
            self.assertEqual(c.settings['max_acceleration_counts_s2'],70)
            self.assertEqual(c.settings['eye_lead_s'],.08)
            self.assertEqual(c.settings['filter_tau_s'],.12)
            runs[multiplier]=(cross80,np.array(eyes));c.close()
        self.assertLess(runs[3][0],runs[2][0])
        np.testing.assert_allclose(runs[2][1],runs[3][1],atol=1e-9)

    def test_double_neck_speed_preserves_eye_motion_range_and_return_home(self):
        runs={}
        for multiplier in (1,2):
            c,r,p,ids=self.setup_trial(offset_counts=160,duration=20,full_song=True,speed_multiplier=multiplier)
            previous=np.full(len(ids),2047.);source=previous.copy();source[2]=2090.
            speeds=np.array([next(m['max_speed_counts_s'] for m in p['motors'] if m['id']==i) for i in ids])
            eyes=[];peak_speed=0.;cross80=None;neck=ids.index(32)
            for step in range(20*60):
                now=10+(step+1)/60;r.poll.return_value=packet(now,.8)
                result=c.apply(source,previous,1/60,now)
                self.assertEqual(result[2],2090.)
                np.testing.assert_equal(result[-2:],[2047.,2047.])
                self.assertTrue(1887<=result[neck]<=2207)
                old=previous;previous=limit_step(previous,result,speeds,1/60)
                self.assertLessEqual(abs(previous[neck]-old[neck]),25*multiplier/60+1e-9)
                peak_speed=max(peak_speed,abs(c.velocity));eyes.append(previous[:2].copy())
                if cross80 is None and previous[neck]>=2127:cross80=(step+1)/60
            self.assertAlmostEqual(previous[neck],2047.)
            self.assertAlmostEqual(peak_speed,25*multiplier)
            self.assertEqual(c.settings['max_acceleration_counts_s2'],35*multiplier)
            runs[multiplier]=(cross80,np.array(eyes));c.close()
        self.assertLess(runs[2][0],runs[1][0])
        np.testing.assert_allclose(runs[1][1],runs[2][1],atol=1e-9)

    def test_neck_speed_multiplier_is_bounded_by_existing_profile(self):
        original=profile()
        for value in [0,1.5,4,True,float('nan')]:
            with self.assertRaises(ValueError):prepare_neck_follow_trial(original,32,speed_multiplier=value)
        p,settings=prepare_neck_follow_trial(original,32,offset_counts=160,speed_multiplier=2)
        self.assertEqual(original,profile())
        for before,after in zip(original['motors'],p['motors']):
            if before['id']!=32:self.assertEqual(before,after)
        next(m for m in original['motors'] if m['id']==32)['max_speed_counts_s']=30
        limited,_=prepare_neck_follow_trial(original,32,offset_counts=160,speed_multiplier=2)
        self.assertEqual(next(m for m in limited['motors'] if m['id']==32)['max_speed_counts_s'],30)

    def test_neck32_diagnostic_requires_homed_sms_axes_and_current_device_bounds(self):
        from xiaobai_neck_sweep import sweep_plan
        config = bytearray(40);config[:3] = bytes([2,53,0]);config[18] = 44;config[30] = 1
        snapshots = [dict(id=sid, model_raw=11272, mode=0, torque_enable=1,
                          goal_position_raw=2047, position_raw=2048, configured_min=1547,
                          configured_max=2544, config_0_39_hex=config.hex()) for sid in (32,33,34)]
        plan = sweep_plan(snapshots)
        self.assertEqual(plan['positions'], [2047,2127,2047,1967,2047])
        self.assertEqual(plan['id'], 32)
        expanded = sweep_plan(snapshots, 160)
        self.assertEqual(expanded['positions'], [2047,2207,2047,1887,2047])
        self.assertEqual(expanded['step_counts'], plan['step_counts'])
        self.assertEqual(expanded['speed_raw'], plan['speed_raw'])
        for change in [dict(torque_enable=0), dict(mode=1), dict(goal_position_raw=2100),
                       dict(configured_max=2100), dict(model_raw=521)]:
            changed = copy.deepcopy(snapshots);changed[0].update(change)
            with self.assertRaises(ValueError):sweep_plan(changed)
        for offset in (0,161,320,True):
            with self.assertRaises(ValueError):sweep_plan(snapshots, offset)
        changed = copy.deepcopy(snapshots);changed[0]['configured_min'] = 1900
        with self.assertRaises(ValueError):sweep_plan(changed, 160)

    def test_neck32_diagnostic_verifies_unicast_goal_without_enabling_torque(self):
        import struct
        from xiaobai_neck_sweep import write_verified_goal
        bus=Mock();bus.acceleration={32:0};bus.sdk.WritePosEx.return_value=(0,1)
        plan=dict(id=32,safe_min=1967,safe_max=2127,speed_raw=25)
        bus.read.return_value=struct.pack('<HHH',2127,0,25);events=[]
        write_verified_goal(bus,2127,plan,events)
        bus.sdk.WritePosEx.assert_called_once_with(32,2127,25,0)
        bus.sdk.write1ByteTxRx.assert_not_called()
        self.assertEqual(events[0]['servo_status'],1)
        bus.reset_mock()
        for position in [1966,2128,True]:
            with self.assertRaises(ValueError):write_verified_goal(bus,position,plan,events)
        bus.sdk.WritePosEx.assert_not_called()
        bus.read.return_value=struct.pack('<HHH',2047,0,25)
        with self.assertRaisesRegex(RuntimeError,'readback'):
            write_verified_goal(bus,2127,plan,events)

    def setup_trial(self, sid=32, sign=1, offset_counts=20, duration=10, full_song=False, speed_multiplier=1):
        p, settings = prepare_neck_follow_trial(profile(), sid, sign, offset_counts, speed_multiplier)
        ids = [8, 23, 31, 32, 33, 34]
        receiver = Mock(rejected=0)
        gaze = GazeOverride(configuration(), ids, p, receiver)
        return NeckFollowTrial(gaze, p, ids, settings, duration, full_song=full_song), receiver, p, ids

    def test_eyes_start_first_and_neck_begins_before_eye_limit_then_returns_home(self):
        controller, receiver, p, ids = self.setup_trial()
        previous = np.full(len(ids), 2047.)
        source = previous.copy();source[ids.index(31)] = 2090
        speeds = np.array([next(m['max_speed_counts_s'] for m in p['motors'] if m['id'] == i) for i in ids])
        neck = ids.index(32)
        reached = False
        first_eye = first_neck = None
        for step in range(600):
            elapsed = (step + 1) / 60;now = 10 + elapsed
            receiver.poll.return_value = packet(now, .8)
            goal = controller.apply(source, previous, 1/60, now)
            if elapsed <= controller.settings['eye_lead_s']:
                self.assertEqual(goal[neck], 2047)
            if first_eye is None and abs(goal[0]-2047)>.01:first_eye=elapsed
            if first_neck is None and abs(goal[neck]-2047)>.01:
                first_neck=elapsed
                self.assertLess(abs(axis_normalized(controller.gaze.axes[0],previous[0])),.6)
            if elapsed >= .2 and elapsed <= .3:
                self.assertGreater(goal[0], 2047)
            self.assertEqual(goal[ids.index(31)], 2090)
            np.testing.assert_equal(goal[-2:], [2047, 2047])
            current = limit_step(previous, goal, speeds, 1/60)
            self.assertLessEqual(abs(current[neck] - previous[neck]), 25/60 + 1e-9)
            self.assertTrue(2027 <= current[neck] <= 2067)
            reached |= current[neck] > 2066
            previous = current
        self.assertTrue(reached)
        self.assertLess(first_eye,first_neck)
        self.assertLess(first_neck,.35)
        self.assertAlmostEqual(previous[neck], 2047)
        controller.close()

    def test_real_head_motion_reduces_eye_offset_for_both_target_directions(self):
        # A deliberately simple simulated plant, not robot angle calibration:
        # the camera turns with the head, with 0.004 image units per count and
        # 120ms actuator lag. Eye compensation must follow the resulting image.
        import math
        for world_target in (.6,-.6):
            controller,receiver,p,ids=self.setup_trial(offset_counts=160,duration=60,full_song=True)
            previous=np.full(len(ids),2047.);actual_head=2047.
            speeds=np.array([next(m['max_speed_counts_s'] for m in p['motors'] if m['id']==i) for i in ids])
            peak_eye=0.;final_eye=0.
            for step in range(12*60):
                now=10+(step+1)/60
                image_error=float(world_target-.004*(actual_head-2047))
                receiver.poll.return_value=packet(now,image_error)
                result=controller.apply(np.full(len(ids),2047.),previous,1/60,now)
                self.assertEqual(controller.last_diagnostic['state'],'tracking')
                previous=limit_step(previous,result,speeds,1/60)
                actual_head=float(actual_head-math.expm1(-1/60/.12)*(previous[ids.index(32)]-actual_head))
                final_eye=axis_normalized(controller.gaze.axes[0],previous[0])
                peak_eye=max(peak_eye,abs(final_eye))
            self.assertGreater(peak_eye,.4)
            self.assertLess(abs(final_eye),peak_eye*.35)
            self.assertGreater(np.sign(world_target)*(actual_head-2047),70)
            self.assertAlmostEqual(.004*(actual_head-2047)+final_eye,world_target,delta=.06)
            controller.close()

    def test_commanded_neck_motion_alone_must_not_artificially_center_eyes(self):
        controller,receiver,p,ids=self.setup_trial(offset_counts=160,duration=60,full_song=True)
        previous=np.full(len(ids),2047.)
        speeds=np.array([next(m['max_speed_counts_s'] for m in p['motors'] if m['id']==i) for i in ids])
        for step in range(12*60):
            now=10+(step+1)/60
            receiver.poll.return_value=packet(now,.6)  # Camera never actually turns.
            result=controller.apply(np.full(len(ids),2047.),previous,1/60,now)
            previous=limit_step(previous,result,speeds,1/60)
        self.assertGreater(previous[ids.index(32)],2206)
        self.assertGreater(axis_normalized(controller.gaze.axes[0],previous[0]),.5)
        controller.close()

    def test_reversing_target_does_not_drive_in_old_direction_due_to_filter_memory(self):
        controller,receiver,p,ids=self.setup_trial(offset_counts=160,duration=60,full_song=True)
        previous=np.full(len(ids),2047.)
        speeds=np.array([next(m['max_speed_counts_s'] for m in p['motors'] if m['id']==i) for i in ids])
        negative_follow_samples=0
        for step in range(4*60):
            now=10+(step+1)/60;receiver.poll.return_value=packet(now,.8 if step<60 else -.3)
            result=controller.apply(np.full(len(ids),2047.),previous,1/60,now)
            neck=controller.last_diagnostic['neck']
            if neck['state']=='following' and neck['control_offset_x']<-.14:
                negative_follow_samples+=1
                self.assertLessEqual(neck['requested_velocity_counts_s'],0.)
            previous=limit_step(previous,result,speeds,1/60)
        self.assertGreater(negative_follow_samples,60)
        controller.close()

    def test_neck_brakes_on_loss_and_can_reverse_encoder_direction(self):
        for sid, sign in [(32, 1), (32, -1)]:
            controller, receiver, p, ids = self.setup_trial(sid, sign)
            previous = np.full(len(ids), 2047.)
            for step in range(90):
                now = 10 + (step + 1)/60
                receiver.poll.return_value = packet(now, .8)
                previous = controller.apply(previous, previous, 1/60, now)
            self.assertGreater(sign * (previous[ids.index(sid)] - 2047), 0)
            neck_before = previous[ids.index(sid)]
            receiver.poll.return_value = packet(now + .1, None)
            for tick in range(60):
                goal = controller.apply(previous, previous, 1/60, now + .1 + tick/60)
                self.assertLessEqual(abs(goal[ids.index(sid)]-previous[ids.index(sid)]),25/60+1e-9)
                previous = goal
            self.assertLessEqual(abs(previous[ids.index(sid)]-neck_before), 25**2/(2*35)+1)
            self.assertAlmostEqual(controller.velocity,0.)
            self.assertEqual(controller.last_diagnostic['neck']['state'], 'holding')
            controller.close()

    def test_trial_cannot_enable_other_neck_axes_or_expand_existing_envelope(self):
        before = profile();p, settings = prepare_neck_follow_trial(before, 32)
        self.assertEqual(before, profile())
        self.assertEqual(set(p['diagnostic_command_ids']) & {32,33,34}, {32})
        for old, new in zip(before['motors'], p['motors']):
            if old['id'] != 32:self.assertEqual(old, new)
        for sid in [8,33,34,True]:
            with self.assertRaises(ValueError):prepare_neck_follow_trial(before, sid)
        ids=[8,23,31,32,33,34];gaze=GazeOverride(configuration(),ids,p,Mock(rejected=0))
        for duration in [0,4,61,float('nan')]:
            with self.assertRaises(ValueError):NeckFollowTrial(gaze,p,ids,settings,duration)
        p['diagnostic_command_ids'].append(34)
        with self.assertRaises(ValueError):NeckFollowTrial(gaze,p,ids,settings,10)
        gaze.close()

    def test_explicit_expanded_trial_clips_to_device_limits_and_finishes_return(self):
        original = profile()
        p, settings = prepare_neck_follow_trial(original, 32, offset_counts=80)
        self.assertEqual(original, profile())
        self.assertEqual([settings['safe_min'], settings['safe_max']], [1967, 2127])
        self.assertGreaterEqual(settings['return_home_s'], 80/25 + 1)
        device_limited = copy.deepcopy(original)
        next(m for m in device_limited['motors'] if m['id'] == 32)['device_max'] = 2080
        _, clipped = prepare_neck_follow_trial(device_limited, 32, offset_counts=80)
        self.assertEqual(clipped['safe_max'], 2080)
        for value in [0, 81, 161, True, 80.0]:
            with self.assertRaises(ValueError):prepare_neck_follow_trial(original, 32, offset_counts=value)
        controller, receiver, p, ids = self.setup_trial(32, offset_counts=80, duration=14)
        previous = np.full(len(ids), 2047.)
        speeds = np.array([next(m['max_speed_counts_s'] for m in p['motors'] if m['id'] == i) for i in ids])
        neck = ids.index(32); peak = 2047.
        for step in range(14*60):
            now = 10 + (step+1)/60
            receiver.poll.return_value = packet(now, 1.)
            goal = controller.apply(previous, previous, 1/60, now)
            current = limit_step(previous, goal, speeds, 1/60)
            self.assertLessEqual(abs(current[neck]-previous[neck]), 25/60+1e-9)
            peak = max(peak, current[neck]);previous = current
        self.assertGreaterEqual(peak, 2126)
        self.assertAlmostEqual(previous[neck], 2047)
        controller.close()

    def test_full_song_neck_smooths_velocity_preserves_face_and_returns_home(self):
        controller, receiver, p, ids = self.setup_trial(offset_counts=160,duration=120,full_song=True)
        previous=np.full(len(ids),2047.);source=previous.copy();source[2]=2090.
        speed=np.array([next(m['max_speed_counts_s'] for m in p['motors'] if m['id']==i) for i in ids])
        neck=ids.index(32);peak=2047.;low=2047.
        for step in range(120*60):
            elapsed=(step+1)/60;now=10+elapsed
            x=.8 if elapsed<35 else -.8 if elapsed<70 else .5
            if 80<elapsed<82:x=None
            receiver.poll.return_value=packet(now,x)
            old_velocity=controller.velocity
            goal=controller.apply(source,previous,1/60,now)
            self.assertEqual(goal[2],2090.)
            np.testing.assert_equal(goal[-2:],[2047.,2047.])
            self.assertTrue(1887<=goal[neck]<=2207)
            self.assertLessEqual(abs(controller.velocity),25+1e-9)
            if min(abs(previous[neck]-1887),abs(previous[neck]-2207))>1:
                self.assertLessEqual(abs(controller.velocity-old_velocity),35/60+1e-9)
            current=limit_step(previous,goal,speed,1/60)
            self.assertLessEqual(abs(current[neck]-previous[neck]),25/60+1e-9)
            peak=max(peak,current[neck]);low=min(low,current[neck]);previous=current
        self.assertGreaterEqual(peak,2206)
        self.assertLessEqual(low,1888)
        self.assertAlmostEqual(previous[neck],2047.)
        controller.close()

    def test_small_image_noise_does_not_start_neck_and_full_mode_needs_axis_confirmation(self):
        controller,receiver,_,ids=self.setup_trial()
        previous=np.full(len(ids),2047.)
        for step in range(5*60):
            now=10+(step+1)/60;receiver.poll.return_value=packet(now,.1 if step%2 else -.1)
            result=controller.apply(previous,previous,1/60,now)
            self.assertEqual(result[ids.index(32)],2047.)
            previous=result
        p,settings=prepare_neck_follow_trial(profile(),32,offset_counts=160)
        c=configuration();c['neck_observations']['horizontal_motion_confirmed']=False
        gaze=GazeOverride(c,ids,p,Mock(rejected=0))
        with self.assertRaisesRegex(ValueError,'onsite confirmation'):
            NeckFollowTrial(gaze,p,ids,settings,120,full_song=True)
        gaze.close();controller.close()


class PreviewTests(unittest.TestCase):
    def test_local_http_state_is_available_without_any_actuator(self):
        from xiaobai_runtime.gaze_preview import Preview
        preview = Preview('127.0.0.1', 0)
        port = preview.server.server_address[1]
        try:
            preview.update({'status': 'tracking', 'actuators_opened': False})
            with urlopen(f'http://127.0.0.1:{port}/state', timeout=1) as response:
                state = json.load(response)
            self.assertEqual(state['status'], 'tracking')
            self.assertIs(state['actuators_opened'], False)
            with urlopen(f'http://127.0.0.1:{port}/', timeout=1) as response:
                self.assertIn('仅左右跟随', response.read().decode())
        finally:
            preview.close()


class PlaybackIntegrationTests(unittest.TestCase):
    def test_invalid_requested_duration_fails_before_input_or_device_access(self):
        import xiaobai_observation_player as player
        for duration in ['nan', 'inf', '-inf', '0', '-1']:
            with self.subTest(duration=duration), \
                    patch('sys.argv', ['player', '--profile', 'unused.json', '--output', 'unused',
                                       '--max-seconds='+duration, '--check']), \
                    patch.object(player, 'read_json') as read, \
                    patch.object(player, 'play') as playback, \
                    patch('sys.stderr'):
                with self.assertRaises(SystemExit) as raised:
                    player.main()
                self.assertEqual(raised.exception.code, 2)
                read.assert_not_called()
                playback.assert_not_called()

    def test_audio_index_and_jaw_survive_camera_loss(self):
        # Exercise the actual playback loop with an actual eye controller;
        # only the physical audio and serial transports are replaced.
        from test_xiaobai import PlaybackTests
        from xiaobai_runtime.playback import play
        p, bus, clock = PlaybackTests().fixture()
        p['motors'] = [{**p['motors'][0], 'id': sid} for sid in [8, 23, 31]]
        config = configuration()
        for axis in config['control']['axes']:
            axis.update(negative=1900, neutral=2047, positive=2200)
        previous = {sid: 2047 for sid in [8, 23, 31]}
        bus.goals.side_effect = lambda goals, speeds: previous.update(goals)
        bus.feedback.side_effect = lambda sid: dict(position=previous[sid], temperature_c=25,
                                                   load_raw=0, status_register=0)
        receiver = Mock(rejected=0)
        receiver.poll.side_effect = [packet(time.monotonic(), .8), None, None, None]
        override = GazeOverride(config, [8, 23, 31], p, receiver)
        trajectory = np.array([[2047., 2047., 2047.], [2047., 2047., 2050.], [2047., 2047., 2055.]])
        with tempfile.TemporaryDirectory() as temp, patch('xiaobai_runtime.playback.Bus', return_value=bus), \
                patch('xiaobai_runtime.playback.AudioClock', return_value=clock):
            result = play(p, [8, 23, 31], trajectory, 60, 'fake', Path(temp) / 'log.json',
                          Path(temp) / 'STOP', gaze_override=override)
        self.assertTrue(result['completed'])
        self.assertEqual([e['frame'] for e in result['events']], sorted(e['frame'] for e in result['events']))
        for event in result['events']:
            frame = event['frame']
            self.assertEqual(event['goals'][31], trajectory[frame, 2])
            self.assertIn('person_gaze', event)

    def test_bad_dynamic_target_is_rejected_before_any_runtime_goal(self):
        from test_xiaobai import PlaybackTests
        from xiaobai_runtime.playback import play
        p, bus, clock = PlaybackTests().fixture()
        override = Mock(axes=[{'id': 2}])
        override.apply.return_value = np.array([5000.])
        with tempfile.TemporaryDirectory() as temp, patch('xiaobai_runtime.playback.Bus', return_value=bus), \
                patch('xiaobai_runtime.playback.AudioClock', return_value=clock):
            with self.assertRaisesRegex(ValueError, 'Live gaze target'):
                play(p, [2], np.full((3, 1), 2047.), 60, 'fake', Path(temp) / 'log.json',
                     Path(temp) / 'STOP', gaze_override=override)
        bus.goals.assert_not_called()


if __name__ == '__main__':
    unittest.main()
