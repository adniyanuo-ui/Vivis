"""No-face acoustic admission and safe handoff using the real motion profile."""
import asyncio
import json
from pathlib import Path
import time
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.interaction.sound_attention import configuration,SideSpeechGate,NoFaceSpeechGate,bearing
from vivis.paths import ROOT
from test_sound_attention import MotionFixture


class SearchGateTests(unittest.TestCase):
    def setUp(self):
        self.c=configuration();self.gate=NoFaceSpeechGate(self.c,lambda _:True)

    def observe(self,now,**changes):
        direction=dict(updated_monotonic=now,frame_valid=True,sector=5,sequence=1)
        feedback=dict(updated_monotonic=now,visual_fresh=True,face_count=0,phase='idle',owner='owner')
        metrics=dict(clean_rms=.01,reference_rms=0.)
        available=changes.pop('available',True)
        for k,v in changes.items():
            if k in feedback:feedback[k]=v
            elif k in metrics:metrics[k]=v
            else:direction[k]=v
        return self.gate.observe(now,b'voice',metrics,direction,feedback,available=available,now=now)

    def test_valid_voice_without_face_or_identity_requests_seek(self):
        for n in range(4):request=self.observe(10.+n*.02)
        self.assertEqual(request['action'],'seek');self.assertEqual(request['bearing_deg'],-90)
        self.assertNotIn('person',request)

    def test_echo_noise_stale_camera_faces_and_busy_never_admit(self):
        for changes in (dict(reference_rms=.1),dict(clean_rms=0.),dict(updated_monotonic=1.),
                dict(visual_fresh=False),dict(face_count=None),dict(face_count=1),dict(face_count=2),
                dict(available=False),dict(frame_valid=False),dict(phase='out')):
            self.gate=NoFaceSpeechGate(self.c,lambda _:True)
            for n in range(10):self.assertIsNone(self.observe(10.+n*.02,**changes),changes)

    def test_direction_packet_is_not_replayed_after_cooldown(self):
        direction=dict(updated_monotonic=10.,frame_valid=True,sector=5,sequence=1)
        def frame(now):
            return self.gate.observe(now,b'voice',dict(clean_rms=.01,reference_rms=0.),direction,
                dict(updated_monotonic=now,visual_fresh=True,face_count=0,phase='idle',owner='owner'),available=True,now=now)
        for n in range(4):request=frame(10.+n*.02)
        self.assertIsNotNone(request)
        self.gate.last_episode=0.
        for n in range(4):self.assertIsNone(frame(10.2+n*.02))

    def test_capture_gap_does_not_join_independent_voice_fragments(self):
        self.observe(10.);self.observe(10.02)
        self.assertIsNone(self.observe(10.2))


class SearchMotionTests(MotionFixture):
    def request(self,now,azimuth=90):
        return dict(super().request(now,azimuth),action='seek')

    def replay(self,*,found=False,lease=True):
        request=self.request(11);trace=[];hold=[];status=None
        for n in range(900):
            now=11+n/60;request['updated_monotonic']=now
            visible=found and n>180
            output,status=self.step(now,request if lease or n<20 else None,visible=visible)
            trace.append(self.current.copy())
            if status.get('phase')=='seek_hold':hold.append(now)
            if status.get('completed') and status['phase']=='idle':break
        return np.asarray(trace),hold,status

    def test_not_found_moves_holds_and_centers_in_six_seconds_with_original_bounds(self):
        trace,hold,status=self.replay()
        self.assertEqual(status['result'],'empty');self.assertGreater(hold[-1]-hold[0],1.4)
        self.assertTrue(5.7<=status['completed']['elapsed_s']<=6.2)
        np.testing.assert_allclose(self.current,self.neutral)
        self.assertGreater(np.ptp(trace[:,self.ids.index(32)]),100)
        for sid in self.axes:
            x=trace[:,self.ids.index(sid)]
            self.assertGreaterEqual(x.min(),self.motion.low[sid]);self.assertLessEqual(x.max(),self.motion.high[sid])
            self.assertLessEqual(np.max(abs(np.diff(x)))*60,self.motion.speeds[sid]+.01)
            if sid in (32,33,34):
                self.assertLess(np.max(abs(np.diff(x,2)))*3600,1600)
                self.assertLess(np.max(abs(np.diff(x,3)))*216000,16000)

    def test_real_face_at_stopped_pose_hands_off_without_returning_away(self):
        trace,hold,status=self.replay(found=True)
        self.assertEqual(status['result'],'found');self.assertIsNone(self.motion.episode)
        self.assertGreater(abs(self.current[self.ids.index(32)]-2047),100)
        self.assertAlmostEqual(trace[-1,self.ids.index(32)],trace[-2,self.ids.index(32)],places=5)

    def test_lost_lease_returns_to_center_instead_of_looking_indefinitely(self):
        trace,hold,status=self.replay(lease=False)
        self.assertEqual(status['result'],'empty');np.testing.assert_allclose(self.current,self.neutral)

    def test_face_already_found_before_admission_does_not_turn_away(self):
        output,status=self.step(11,self.request(11),visible=True)
        self.assertIsNone(output);self.assertEqual(status['result'],'found')

    def test_mirrored_board_mapping_drives_the_requested_yaw_sign(self):
        request=self.request(11,bearing(5,self.c))
        for n in range(150):
            now=11+n/60;request['updated_monotonic']=now
            self.step(now,request,visible=False)
        self.assertLess(self.current[self.ids.index(32)],2047)

    def test_existing_gaze_owner_accepts_seek_then_tracks_found_face(self):
        from vivis.interaction.standby import StandbyGaze,plan
        from vivis.runtime.robot import Robot
        settings=json.loads((ROOT/'configs/interaction/interaction.json').read_text())
        profile,ids,_,mapper,source,trajectory=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),settings,1.)
        source.acoustic_path=Path('/not-opened/sound_attention.json')
        receiver=SimpleNamespace(max_age=.3,event=None,poll=lambda now:receiver.event);source.receiver=receiver
        mailbox=SimpleNamespace(latest=None,start=lambda:mailbox,submit=Mock(),close=lambda:None)
        with patch('vivis.interaction.sound_attention.Mailbox',return_value=mailbox):
            gaze=StandbyGaze(source.tracking,profile,ids,receiver,source)
        self.addCleanup(gaze.finish,True,None)
        gaze.prepare(trajectory,mapper.neutral,[],0.);gaze.set_audio_time(0.,None)
        last=mapper.neutral.copy();request=None;active=False;resumed=False;found=False
        for n in range(480):
            now=10+n/60;visible=n>=240
            receiver.event=dict(schema_version='vivis-person-target-1.0',timestamp_monotonic=now,
                sequence=n,faces=int(visible),target=dict(x=0.,y=0.,eye_line_roll_rad=0.) if visible else None)
            if n==60:request=dict(self.request(now),owner=gaze.acoustic.owner)
            if request:request['updated_monotonic']=now
            mailbox.latest=request
            for sid,value in zip(ids,last):gaze.observe(sid,dict(position=float(value)),now)
            last=gaze.apply(source.sample(now,now),last,1/60,now)
            active|=gaze.last_diagnostic['state']=='acoustic_attention'
            found|=mailbox.submit.call_args.args[0].get('result')=='found'
            if found:resumed|=gaze.last_diagnostic['state']=='tracking'
            for motor in profile['motors']:
                self.assertTrue(motor['safe_min']<=last[ids.index(motor['id'])]<=motor['safe_max'])
        self.assertTrue(active);self.assertTrue(found);self.assertTrue(resumed)


class SearchVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from test_cloud_control import CloudBackendTests
        from vivis.speech.realtime import VoiceSession
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.events=[];self.v=CloudBackendTests().service(self.tmp.name,self.events)
        self.addCleanup(self.v.log.close);self.addCleanup(self.v.recorders.close);self.addCleanup(self.v.conversation_animation.close)
        v=self.v;v.enabled=True;v.health_available=True;v.sound_started=True;v.side_suppressed=False
        v.context=dict(revision=1,mode='interaction',ready=True)
        v.cloud_state=VoiceSession(v.player,v.log)
        v.sound_gate=SideSpeechGate(v.sound_config,lambda _:True)
        v.search_gate=NoFaceSpeechGate(v.sound_config,lambda _:True)
        v.direction_observer=SimpleNamespace(latest=None)
        v.sound_mailbox=SimpleNamespace(latest=None,submit=Mock())
        v.presence.single_visible=lambda:False

    def frame(self,now,**changes):
        v=self.v
        v.direction_observer.latest=dict(updated_monotonic=now,frame_valid=True,sector=5,sequence=1)
        v.sound_mailbox.latest=dict(updated_monotonic=now,phase='idle',owner='owner',face_count=0,visual_fresh=True)
        v.sound_mailbox.latest.update(changes)
        return v.acoustic_microphone(now,b'original_audio',dict(clean_rms=.01,reference_rms=0.))

    async def test_no_face_seek_keeps_normal_audio_identity_and_turn_ownership(self):
        v=self.v;before=(v.turn_person,v.turn_item_id,v.memory_session,v.input_generation)
        v.in_speech=True  # Seeking must not wait for this ordinary main utterance to end.
        start=time.monotonic()-.1
        for n in range(4):self.assertFalse(self.frame(start+n*.02))
        request=v.sound_search;self.assertIsNotNone(request);self.assertIsNone(v.sound_attention)
        self.assertFalse(v.sound_frozen());self.assertFalse(v.side_suppressed)
        self.assertEqual(before,(v.turn_person,v.turn_item_id,v.memory_session,v.input_generation))
        self.assertIsNone(v.side_task)
        v.sound_search_tick();self.assertEqual(v.sound_mailbox.submit.call_args.args[0]['action'],'seek')
        v.sound_mailbox.latest.update(completed=dict(id=request['id'],elapsed_s=5.9),result='empty')
        v.sound_search_tick();self.assertIsNone(v.sound_search)
        self.assertEqual(before,(v.turn_person,v.turn_item_id,v.memory_session,v.input_generation))

    async def test_robot_playback_never_starts_search(self):
        start=time.monotonic()-.1
        self.frame(start)
        self.v.search_gate=NoFaceSpeechGate(self.v.sound_config,lambda _:True)
        for n in range(4):
            now=start+n*.02
            self.v.direction_observer.latest=dict(updated_monotonic=now,frame_valid=True,sector=5,sequence=2)
            self.v.sound_mailbox.latest.update(updated_monotonic=now)
            self.assertFalse(self.v.acoustic_microphone(now,b'echo',dict(clean_rms=.01,reference_rms=.1)))
        self.assertIsNone(self.v.sound_search)

    async def test_standby_retains_three_original_wake_rules_and_cannot_seek(self):
        self.v.context['mode']='standby';start=time.monotonic()-.1
        for n in range(5):self.assertFalse(self.frame(start+n*.02))
        self.assertIsNone(self.v.sound_search)

    async def test_mode_change_and_owner_change_cancel_request(self):
        start=time.monotonic()-.1
        for n in range(4):self.frame(start+n*.02)
        self.assertIsNotNone(self.v.sound_search)
        self.v.sound_mailbox.latest['owner']='new-owner'
        self.v.sound_search_tick();self.assertIsNone(self.v.sound_search)
