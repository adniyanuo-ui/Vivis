"""Replay real UART anomalies, PCM attribution and finite gaze without devices."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.hardware.haokai import DirectionParser,DirectionState
from vivis.interaction.sound_attention import configuration,bearing,SideSpeechGate,GlanceMotion,Mailbox
from vivis.paths import ROOT
from vivis.speech.realtime_audio import VoiceBuffer


class AcousticProtocolTests(unittest.TestCase):
    def test_all_sectors_and_real_zero_anomalies_across_every_byte_split(self):
        raw=bytes.fromhex('55 AA 08 F7 00 DD 7E 55 AA 00 02 7E 55 AA 01 FE 00 83 7E 55 AA 00 9E 7E')
        for cut in range(len(raw)+1):
            p=DirectionParser(False)
            frames=p.feed(raw[:cut])+p.feed(raw[cut:])
            self.assertEqual([f.sector for f in frames],[8,0,1,0])
            self.assertTrue(all(len(f.raw)==3 for f in frames))
        for i in range(12):
            self.assertEqual(DirectionParser(False).feed(bytes([85,170,i]))[0].sector,i)
        self.assertEqual(DirectionParser(False).feed(bytes([85,170,12])),[])

    def test_unchecked_observation_never_claims_verified_checksum_or_speech(self):
        coordinates=json.loads((ROOT/'configs/perception/sound_direction.json').read_text())['coordinates']
        frame=DirectionParser(False).feed(bytes.fromhex('55 AA 00 2C 7E'))[0]
        state=DirectionState(coordinates,.75,False);state.observe(frame,1.)
        value=state.snapshot(1.1)
        self.assertEqual(value['sector'],0);self.assertFalse(value['checksum_verified'])
        self.assertFalse(value['motion_authorized']);self.assertIsNone(value['speech_active'])
        self.assertFalse(state.snapshot(2.)['frame_valid'])
        with self.assertRaises(ValueError):DirectionState(coordinates,.75).observe(frame,1.)


class SideAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.c=configuration();self.g=SideSpeechGate(self.c,lambda pcm:pcm==b'voice')

    def frame(self,now,**changes):
        direction=dict(updated_monotonic=now,frame_valid=True,sector=0,sequence=1)
        feedback=dict(updated_monotonic=now,phase='idle',single_face=True,settled=True,owner='owner',main_bearing_deg=0.)
        metrics=dict(clean_rms=.01,reference_rms=0.)
        for key,value in changes.items():
            if key in feedback:feedback[key]=value
            elif key in metrics:metrics[key]=value
            else:direction[key]=value
        return self.g.observe(now,b'voice',metrics,direction,feedback,available=True,now=now)

    def test_mounting_is_explicit_and_zero_not_automatically_forward(self):
        self.assertEqual(bearing(8,self.c),0)
        self.assertEqual(bearing(0,self.c),-120)
        self.assertEqual(bearing(5,self.c),90)

    def test_head_cooldown_does_not_discard_a_new_side_utterance(self):
        for n in range(4):_,first=self.frame(10+n*.02)
        self.assertTrue(first['glance_permitted'])
        self.g.reset(keep_history=True,now=10.4)
        for n in range(4):_,second=self.frame(10.5+n*.02)
        self.assertIsNotNone(second);self.assertFalse(second['glance_permitted'])

    def test_one_clean_side_sentence_quarantined_then_admitted_once(self):
        requests=[]
        for n in range(30):
            suppressed,request=self.frame(10+n*.02)
            self.assertTrue(suppressed)
            if request:requests.append(request)
        self.assertEqual(len(requests),1);self.assertEqual(requests[0]['sector'],0)
        self.assertTrue(self.g.contains(10.01));self.assertFalse(self.g.contains(9.))

    def test_echo_same_direction_multiple_people_and_stale_never_trigger(self):
        for change in ({'reference_rms':.1},{'sector':8},{'single_face':False},
                       {'frame_valid':False},{'updated_monotonic':1.}):
            self.g=SideSpeechGate(self.c,lambda _:True)
            for n in range(12):self.assertEqual(self.frame(10+n*.02,**change),(False,None),change)

    def test_noise_not_speech_and_capture_gap_does_not_join_runs(self):
        self.g.vad=lambda _:False
        self.assertEqual(self.frame(10.),(False,None))
        self.g.vad=lambda _:True
        self.frame(11.);self.frame(11.02)
        suppressed,request=self.frame(12.)
        self.assertTrue(suppressed);self.assertIsNone(request)

    def test_new_main_direction_ends_side_mask_without_swallowing_main_word(self):
        for n in range(4):self.frame(10+n*.02)
        suppressed,request=self.frame(10.1,sector=8,sequence=2)
        self.assertFalse(suppressed);self.assertIsNone(request)
        self.assertFalse(self.g.contains(10.12))

    def test_busy_main_still_excludes_qualified_side_pcm_without_starting_motion(self):
        for n in range(5):
            now=10+n*.02
            suppressed,request=self.g.observe(now,b'voice',dict(clean_rms=.01,reference_rms=0.),
                dict(updated_monotonic=now,frame_valid=True,sector=0,sequence=1),
                dict(updated_monotonic=now,phase='idle',single_face=True,settled=True,owner='owner',main_bearing_deg=0.),
                available=False,now=now)
            self.assertTrue(suppressed);self.assertIsNone(request)
        self.assertTrue(self.g.contains(10.02))


class MotionEpisodeTests(unittest.TestCase):
    def setUp(self):
        from vivis.interaction.standby import plan
        from vivis.runtime.robot import Robot
        settings=json.loads((ROOT/'configs/interaction/interaction.json').read_text())
        profile,self.ids,_,mapper,source,_=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),settings,1.)
        self.neutral=mapper.neutral.copy();self.current=self.neutral.copy();self.c=configuration()
        motors={m['id']:m for m in profile['motors']}
        axes={a['id']:a for a in source.tracking['control']['axes']};axes.update({s:{} for s in (32,33,34)})
        self.motion=GlanceMotion(self.c,self.ids,{s:motors[s]['safe_min'] for s in axes},
            {s:motors[s]['safe_max'] for s in axes},{s:motors[s]['max_speed_counts_s'] for s in axes},
            self.neutral,axes,source.tracking['neck']['upstream_trajectory'])
        self.axes=axes
        for n in range(20):self.step(10+n/60)

    def step(self,now,request=None,visible=True):
        packet=dict(timestamp_monotonic=now,target=dict(x=0.,y=0.)) if visible else dict(timestamp_monotonic=now,target=None)
        values,status=self.motion.step(request,self.current,packet,now)
        if values:
            for sid,value in values.items():self.current[self.ids.index(sid)]=value
        return values,status

    def request(self,now,azimuth=90):
        return dict(id='one',owner=self.motion.owner,action='glance',bearing_deg=azimuth,
            heard_monotonic=now-.08,created_monotonic=now,updated_monotonic=now,main_present=True)

    def test_large_angle_returns_within_budget_and_never_changes_mouth_pitch_roll(self):
        request=self.request(11,120);trace=[];status=None
        for n in range(240):
            now=11+n/60;request['updated_monotonic']=now
            values,status=self.step(now,request)
            trace.append(self.current.copy())
            if status.get('completed') and status['phase']=='idle':break
        self.assertLess(status['completed']['elapsed_s'],2.15)
        self.assertLess(status['completed']['excursion_fraction'],1.)
        np.testing.assert_allclose(self.current,self.neutral)
        self.assertGreater(np.ptp(np.asarray(trace)[:,self.ids.index(32)]),1)
        for sid in set(self.ids)-{8,23,32}:
            np.testing.assert_allclose(np.asarray(trace)[:,self.ids.index(sid)],self.neutral[self.ids.index(sid)])
        for sid in self.axes:
            x=np.asarray(trace)[:,self.ids.index(sid)]
            self.assertTrue(np.all(x>=self.motion.low[sid]));self.assertTrue(np.all(x<=self.motion.high[sid]))
            self.assertLessEqual(np.max(abs(np.diff(x)))*60,self.motion.speeds[sid]+.01)
        yaw=np.asarray(trace)[:,self.ids.index(32)]
        self.assertLess(np.max(abs(np.diff(yaw,2)))*3600,1600)
        self.assertLess(np.max(abs(np.diff(yaw,3)))*216000,16000)

    def test_expired_or_wrong_owner_request_cannot_move(self):
        request=self.request(11);request['owner']='other'
        for now in (11,12,13):self.assertIsNone(self.step(now,request)[0])

    def test_lease_loss_finishes_safe_leg_and_returns_not_stuck_looking_sideways(self):
        request=self.request(11)
        for n in range(240):
            now=11+n/60
            values,status=self.step(now,request if n<10 else None)
        np.testing.assert_allclose(self.current,self.neutral)
        self.assertIsNone(self.motion.episode)

    def test_no_main_and_no_side_face_search_is_finite_and_ends_neutral(self):
        request=self.request(11);request['main_present']=False
        for n in range(1800):
            now=11+n/60;request['updated_monotonic']=now
            values,status=self.step(now,request,visible=False)
            if status.get('result')=='empty':break
        self.assertEqual(status.get('result'),'empty')
        np.testing.assert_allclose(self.current,self.neutral)

    def test_duplicate_request_never_runs_twice(self):
        request=self.request(11);completions=[]
        for n in range(360):
            now=11+n/60;request['updated_monotonic']=now
            values,status=self.step(now,request)
            if status.get('completed'):completions.append(status['completed']['elapsed_s'])
        self.assertEqual(len(set(completions)),1)

    def test_corrupt_nan_request_has_no_motion_authority(self):
        for key,value in [('bearing_deg',float('nan')),('bearing_deg',181),('id',[]),('heard_monotonic',float('inf'))]:
            request=self.request(11);request[key]=value
            self.assertIsNone(self.step(11,request)[0])

    def test_real_gaze_owner_reseeds_after_return_without_old_side_target(self):
        from vivis.interaction.standby import StandbyGaze,plan
        from vivis.runtime.robot import Robot
        settings=json.loads((ROOT/'configs/interaction/interaction.json').read_text())
        profile,ids,_,mapper,source,trajectory=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),settings,1.)
        source.acoustic_path=Path('/not-opened/sound_attention.json')
        receiver=SimpleNamespace(max_age=.3,event=None,poll=lambda now:receiver.event);source.receiver=receiver
        mailbox=SimpleNamespace(latest=None,start=lambda:mailbox,submit=lambda p:None,close=lambda:None)
        with patch('vivis.interaction.sound_attention.Mailbox',return_value=mailbox):
            gaze=StandbyGaze(source.tracking,profile,ids,receiver,source)
        gaze.prepare(trajectory,mapper.neutral,[],0.);gaze.set_audio_time(0.,None)
        last=mapper.neutral.copy();request=None;active=False;resumed=False
        for n in range(360):
            now=10+n/60
            receiver.event=dict(schema_version='vivis-person-target-1.0',timestamp_monotonic=now,
                sequence=n,target=dict(x=0.,y=0.,eye_line_roll_rad=0.))
            if n==60:request=dict(self.request(now),owner=gaze.acoustic.owner)
            if request:request['updated_monotonic']=now
            mailbox.latest=request
            for sid,value in zip(ids,last):gaze.observe(sid,dict(position=float(value)),now)
            last=gaze.apply(source.sample(now,now),last,1/60,now)
            active|=gaze.last_diagnostic['state']=='acoustic_attention'
            if n>210:resumed|=gaze.last_diagnostic['state']=='tracking'
            for motor in profile['motors']:
                self.assertTrue(motor['safe_min']<=last[ids.index(motor['id'])]<=motor['safe_max'])
        self.assertTrue(active);self.assertTrue(resumed)
        self.assertAlmostEqual(last[ids.index(32)],2047.,places=1)
        gaze.finish(True,None)


class AudioOwnershipTests(unittest.TestCase):
    def test_cached_priority_clip_passes_paused_pcm_then_original_can_resume(self):
        player=VoiceBuffer(1.,10,10.)
        a=np.full(2400,1000,dtype='<i2').tobytes();b=np.full(2400,-1000,dtype='<i2').tobytes()
        player.add('main',a);player.finish('main');player.render(100,1.)
        player.pause('main');player.begin_priority('side');player.add('side',b);player.finish('side')
        heard=player.render(4800,1.01)
        self.assertLess(np.min(heard),-.02);self.assertIn('side',player.started_responses)
        player.cancel('side');player.end_priority('side');player.resume(['main'])
        self.assertGreater(np.max(player.render(4800,1.2)),.02)
        self.assertEqual(player.response_ids(),[])

    def test_no_second_priority_owner_and_no_pcm_exceeds_cap(self):
        player=VoiceBuffer(1.,10,1.)
        player.begin_priority('a')
        with self.assertRaises(ValueError):player.begin_priority('b')
        player.end_priority('b');self.assertEqual(player.priority_response,'a')


class SideSourceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from test_cloud_control import CloudBackendTests
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.s=CloudBackendTests().service(self.tmp.name,[])
        self.addCleanup(self.s.log.close);self.addCleanup(self.s.recorders.close)
        self.addCleanup(self.s.conversation_animation.close)
        self.s.sound_gate=SideSpeechGate(configuration(),lambda _:True)
        self.s.sound_gate.intervals.append([10.,12.])

    async def test_late_asr_and_tool_cannot_become_main_turn_or_business(self):
        sent=[]
        async def send(s):sent.append(json.loads(s))
        ws=SimpleNamespace(send=send);timeline=SimpleNamespace(at=lambda _:11.)
        old=(self.s.input_generation,self.s.turn_item_id,self.s.memory_session)
        for event in [dict(type='input_audio_buffer.speech_started',item_id='side',audio_start_ms=100),
            dict(type='input_audio_buffer.speech_stopped',item_id='side'),
            dict(type='conversation.item.input_audio_transcription.completed',item_id='side',transcript='把何工的留言给我'),
            dict(type='response.created',response=dict(id='side_reply')),
            dict(type='response.function_call_arguments.done',response_id='side_reply',call_id='call',name='courier_pickup'),
            dict(type='response.done',response=dict(id='side_reply',status='completed'))]:
            self.assertTrue(await self.s.filter_side_event(event,timeline,ws))
        self.assertEqual((self.s.input_generation,self.s.turn_item_id,self.s.memory_session),old)
        self.assertIsNone(self.s.side_wire_item)
        self.assertEqual(sent[0]['type'],'conversation.item.delete')
        self.assertIn('false',sent[1]['item']['output'])
        timeline.at=lambda _:14.
        self.assertFalse(await self.s.filter_side_event(dict(type='input_audio_buffer.speech_started',item_id='main'),timeline,ws))

    async def test_business_identity_is_frozen_but_raw_recognition_not_relabelled(self):
        stable=dict(person_id='heyude',held=False,face={'state':'unavailable'})
        self.s.sound_attention=dict(stable=stable)
        self.assertIs(self.s.stable_identity(),stable)
        self.s.identity_fusion=Mock()
        self.s.accept_speaker_identity(dict(turn_id='side'))
        self.s.identity_fusion.stage_voice.assert_not_called()
        self.assertTrue(self.s.collection_visual_evidence(10.)['blocked'])

    async def test_vad_before_local_direction_confirmation_is_retracted_before_business(self):
        s=self.s;s.sound_gate.intervals.clear();s.context.update(mode='interaction',ready=True)
        ws=SimpleNamespace(send=Mock());timeline=SimpleNamespace(at=lambda _:11.)
        self.assertFalse(await s.filter_side_event(dict(type='input_audio_buffer.speech_started',item_id='early'),timeline,ws))
        s.begin_user_turn('early');s.in_speech=True
        s.sound_gate.intervals.append([10.,12.])
        self.assertTrue(await s.filter_side_event(dict(type='input_audio_buffer.speech_stopped',item_id='early'),timeline,ws))
        self.assertFalse(s.memory_turns['early']['valid']);self.assertFalse(s.in_speech)
        self.assertIsNone(s.turn_item_id)

    async def test_delayed_main_vad_uses_its_adc_time_not_present_side_activity(self):
        s=self.s;s.side_suppressed=True
        timeline=SimpleNamespace(at=lambda _:9.)
        self.assertFalse(await s.filter_side_event(dict(type='input_audio_buffer.speech_started',item_id='main'),
            timeline,SimpleNamespace(send=Mock())))
        self.assertNotIn('main',s.side_items)

    async def test_shutdown_never_resumes_paused_old_audio(self):
        from vivis.speech.realtime import VoiceSession
        s=self.s;s.context.update(revision=1)
        s.player.add('main',np.full(2400,1000,dtype='<i2').tobytes());s.player.finish('main')
        s.player.render(100,time.monotonic());state=VoiceSession(s.player,s.log);s.cloud_state=state
        request=dict(id='shutdown',owner='owner',connection=s.message_connection,session=s.memory_session,
            revision=1,heard_monotonic=time.monotonic(),audio=SimpleNamespace(pcm24_bytes=bytes(34560)))
        s.sound_attention=request;s.sound_mailbox=SimpleNamespace(submit=Mock())
        s.side_task=asyncio.create_task(s.side_acknowledgement(request,state));await asyncio.sleep(0)
        s.shutdown.set();s.cancel_sound_attention('shutdown')
        await asyncio.gather(s.side_task,return_exceptions=True)
        self.assertIn('main',s.player.paused);self.assertIsNone(s.player.priority_response)

    async def test_task_and_standby_do_not_inherit_side_pcm_mute(self):
        s=self.s;s.sound_started=True;s.enabled=True;s.sound_gate=Mock()
        for mode in ('standby','task'):
            s.context.update(mode=mode,ready=True)
            self.assertFalse(s.acoustic_microphone(10.,b'voice',dict(clean_rms=.01,reference_rms=0.)))
        s.sound_gate.observe.assert_not_called()
        self.assertEqual(s.sound_gate.reset.call_count,2)

    async def test_old_side_response_done_cannot_clear_new_side_source(self):
        s=self.s;ws=SimpleNamespace(send=Mock());timeline=SimpleNamespace(at=lambda _:11.)
        for item,rid in [('first','r1'),('second','r2')]:
            self.assertTrue(await s.filter_side_event(dict(type='input_audio_buffer.speech_started',item_id=item),timeline,ws))
            self.assertTrue(await s.filter_side_event(dict(type='response.created',response=dict(id=rid)),timeline,ws))
        self.assertTrue(await s.filter_side_event(dict(type='response.done',response=dict(id='r1')),timeline,ws))
        self.assertEqual(s.side_wire_item,'second')
        self.assertTrue(await s.filter_side_event(dict(type='response.done',response=dict(id='r2')),timeline,ws))
        self.assertIsNone(s.side_wire_item)

    async def test_mic_admission_preserves_original_side_pcm_after_motion_returns(self):
        from vivis.speech.realtime import VoiceSession
        s=self.s;s.sound_config=configuration();s.sound_started=True
        s.context.update(mode='interaction',target='interaction',ready=True,revision=1)
        s.enabled=True;s.turn_reply_pending=lambda:False;s.presence.single_visible=lambda:True
        stable=dict(person_id='heyude',held=False,face={'state':'confirmed','person_id':'heyude'})
        s.stable_identity=lambda:stable
        now=time.monotonic()
        s.direction_observer=SimpleNamespace(latest=dict(updated_monotonic=now,frame_valid=True,sector=0,sequence=1))
        s.sound_mailbox=SimpleNamespace(latest=dict(updated_monotonic=now,phase='idle',single_face=True,settled=True,
            owner='owner',main_bearing_deg=0.),submit=Mock())
        s.side_audio=SimpleNamespace(pcm24_bytes=np.full(17280,1000,dtype='<i2').tobytes())
        state=VoiceSession(s.player,s.log);s.cloud_state=state
        s.public_prompt_task=Mock();s.public_prompt_task.done.return_value=False
        s.reception.conversation.quiet_since=now-10
        session=s.memory_session
        for n in range(4):
            self.assertTrue(s.acoustic_microphone(now+n*.02,b'voice',dict(clean_rms=.01,reference_rms=0.)))
        request=s.sound_attention;self.assertIsNotNone(request);self.assertEqual(request['person'],'heyude')
        s.public_prompt_task.cancel.assert_called_once();self.assertIsNone(s.reception.conversation.quiet_since)
        self.assertEqual(b''.join(frame[1] for frame in s.side_dialogue.frames),b'voice'*4)
        self.assertEqual(s.memory_session,session)
        s.sound_mailbox.latest=dict(updated_monotonic=time.monotonic(),owner='owner',phase='idle',result='returned',
            completed=dict(id=request['id'],elapsed_s=1.8,excursion_fraction=.2))
        s.acoustic_tick();await asyncio.sleep(0)
        self.assertIs(s.sound_attention,request);self.assertTrue(request['motion_returned'])
        self.assertFalse(s.side_dialogue.capture_done.is_set())
        s.cancel_sound_attention('shutdown');await asyncio.gather(s.side_task,return_exceptions=True)
        self.assertIsNone(s.sound_attention);self.assertEqual(s.memory_session,session)
        self.assertIsNone(s.player.priority_response)


class MailboxTests(unittest.TestCase):
    def test_deleted_corrupt_and_future_files_do_not_supply_authority(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a';b=Path(folder)/'b';m=Mailbox(a,b).start()
            try:
                a.write_text('{')
                time.sleep(.05);self.assertIsNone(m.latest)
                a.write_text('{"owner":"one"}')
                m.submit(dict(owner='two'))
                deadline=time.monotonic()+1
                while time.monotonic()<deadline and (m.latest is None or not b.exists()):time.sleep(.01)
                self.assertEqual(m.latest['owner'],'one');self.assertEqual(json.loads(b.read_text())['owner'],'two')
                a.unlink();time.sleep(.05);self.assertIsNone(m.latest)
            finally:m.close()
            self.assertFalse(m.thread.is_alive())


if __name__=='__main__':unittest.main()
