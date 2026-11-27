"""Changed-direction state is retained; new voice attribution is still scoped."""
import asyncio
import base64
from contextlib import asynccontextmanager
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,Mock,patch

from vivis.hardware.haokai import DirectionParser,DirectionState
from vivis.interaction.sound_attention import configuration,NoFaceSpeechGate,SideSpeechGate,direction_available
from vivis.speech.sound_direction import DirectionObserver,configuration as receiver_configuration


def state_packet(now,*,event=1.,sector=5,generation=1,sequence=1,**changes):
    result=dict(update_mode='on_change',event_window_s=.75,frame_valid=True,
        updated_monotonic=event,received_monotonic=event,observed_monotonic=now,
        observer_lease_s=1.,serial_open=True,generation=generation,sector=sector,sequence=sequence)
    result.update(changes);return result


class DirectionStateTests(unittest.TestCase):
    def setUp(self):self.c=receiver_configuration()

    def test_unchanged_bearing_survives_without_rewriting_its_event_time(self):
        s=DirectionState(self.c['coordinates'],.75,False,'on_change')
        s.observe(DirectionParser(False).feed(bytes.fromhex('55 AA 05 FA'))[0],10.)
        old=s.snapshot(10.1);held=s.snapshot(100.)
        self.assertEqual(held['status'],'held');self.assertEqual(held['sector'],5)
        self.assertTrue(held['frame_valid']);self.assertFalse(held['event_fresh'])
        self.assertEqual(held['received_monotonic'],old['received_monotonic'])
        self.assertEqual(held['sequence'],old['sequence']);self.assertIsNone(held['speech_active'])
        self.assertFalse(held['motion_authorized'])

    def test_no_first_event_and_future_clock_never_invent_a_direction(self):
        s=DirectionState(self.c['coordinates'],.75,False,'on_change')
        self.assertIsNone(s.snapshot(10.)['sector'])
        s.observe(DirectionParser(False).feed(bytes.fromhex('55 AA 00'))[0],10.)
        self.assertFalse(s.snapshot(9.)['frame_valid']);self.assertIsNone(s.snapshot(9.)['sector'])

    def test_legacy_periodic_source_still_expires(self):
        s=DirectionState(self.c['coordinates'],.75,False)
        s.observe(DirectionParser(False).feed(bytes.fromhex('55 AA 05'))[0],10.)
        self.assertEqual(s.snapshot(11.)['status'],'stale')

    def test_empty_reads_preserve_state_but_refresh_only_observer_health(self):
        o=DirectionObserver(self.c,'/not/the/array');now=[10.];results=[];n=[0]
        port=SimpleNamespace(device='/dev/ttyUSB0',vid=0x1a86,pid=0x7523)
        receiver=Mock(in_waiting=0)
        def read(_):
            n[0]+=1
            if n[0]==1:return bytes.fromhex('55 AA 05 FA')
            if n[0]==2:now[0]=30.;return b''
            results.append((o.status(30.),o.attention_snapshot(30.)));o.stop.set();return b''
        receiver.read.side_effect=read
        with patch('serial.tools.list_ports.comports',return_value=[port]), \
                patch('vivis.speech.sound_direction.open_receiver',return_value=receiver), \
                patch('vivis.speech.sound_direction.time',SimpleNamespace(monotonic=lambda:now[0])):
            o.run()
        status,packet=results[0]
        self.assertEqual(status['status'],'held');self.assertEqual(packet['sector'],5)
        self.assertEqual(packet['updated_monotonic'],10.);self.assertEqual(packet['observed_monotonic'],30.)
        self.assertEqual(packet['sequence'],1);self.assertIsNone(o.latest)
        self.assertFalse(o.opened);receiver.write.assert_not_called();receiver.close.assert_called_once()

    def test_reopen_discards_old_coordinate_before_first_read(self):
        o=DirectionObserver(self.c,'/not/the/array');o.opened=True;o.generation=1
        o.latest=state_packet(10.);states=[];receiver=Mock(in_waiting=0)
        def read(_):
            states.append((o.status(10.),o.attention_snapshot(10.)));o.stop.set();return b''
        receiver.read.side_effect=read
        with patch('serial.tools.list_ports.comports',return_value=[SimpleNamespace(device='/dev/ttyUSB0',vid=0x1a86,pid=0x7523)]), \
                patch('vivis.speech.sound_direction.open_receiver',return_value=receiver):o.run()
        self.assertEqual(o.generation,2);self.assertEqual(states[0][0]['status'],'waiting')
        self.assertIsNone(states[0][1]);self.assertIsNone(o.latest)

    def test_hung_reader_error_and_close_do_not_expose_held_coordinates(self):
        o=DirectionObserver(self.c,'/not/the/array');o.opened=True;o.generation=1
        o.latest=state_packet(10.)
        self.assertEqual(o.status(12.)['status'],'unresponsive');self.assertIsNone(o.attention_snapshot(12.))
        o.error='USB removed';self.assertIsNone(o.attention_snapshot(10.))
        o.error=None;o.close();self.assertIsNone(o.attention_snapshot(10.));self.assertIsNone(o.latest)


class EventGateTests(unittest.TestCase):
    def setUp(self):
        self.c=configuration();self.g=NoFaceSpeechGate(self.c,lambda pcm:pcm==b'voice')

    def frame(self,now,*,pcm=b'voice',available=True,direction=None,**changes):
        feedback=dict(updated_monotonic=now,visual_fresh=True,face_count=0,phase='idle',owner='owner')
        feedback.update(changes)
        return self.g.observe(now,pcm,dict(clean_rms=.01 if pcm==b'voice' else 0.,reference_rms=0.),
            state_packet(now) if direction is None else direction,feedback,available=available,now=now)

    def test_old_direction_with_live_receiver_can_only_offer_a_seek_hint(self):
        for n in range(4):request=self.frame(30.+n*.02)
        self.assertIsNotNone(request);self.assertTrue(request['last_known_direction_only'])
        self.assertEqual(request['direction_generation'],1);self.assertNotIn('person',request)

    def test_one_continuous_utterance_does_not_repeat_the_unchanged_seek(self):
        requests=[]
        for n in range(1500):
            r=self.frame(10+n*.02)
            if r:requests.append(r)
        self.assertEqual(len(requests),1)

    def test_duplicate_packets_do_not_turn_continuous_speech_into_repeated_searches(self):
        requests=[]
        for n in range(1500):
            now=10+n*.02
            r=self.frame(now,direction=state_packet(now,event=now,sequence=n+1))
            if r:requests.append(r)
        self.assertEqual(len(requests),1)

    def test_next_utterance_after_silence_and_cooldown_reuses_same_bearing(self):
        for n in range(4):first=self.frame(10+n*.02)
        self.assertIsNotNone(first)
        for n in range(30):self.frame(10.08+n*.02,pcm=b'quiet',available=False,phase='seek_hold')
        for n in range(4):second=self.frame(16+n*.02)
        self.assertIsNotNone(second);self.assertEqual(first['sector'],second['sector'])
        self.assertNotEqual(first['id'],second['id'])

    def test_current_direction_change_can_be_used_after_motion_cooldown(self):
        for n in range(4):self.frame(10+n*.02)
        for n in range(4):request=self.frame(16+n*.02,direction=state_packet(16+n*.02,event=16.,sector=0,sequence=2))
        self.assertIsNotNone(request);self.assertEqual(request['sector'],0)

    def test_no_current_human_voice_means_no_motion_despite_retained_state(self):
        for n in range(100):self.assertIsNone(self.frame(30+n*.02,pcm=b'quiet'))
        self.assertEqual(len(self.g.used),0)

    def test_missing_health_or_future_times_cannot_validate_a_retained_bearing(self):
        for change in (dict(serial_open=False),dict(observed_monotonic=1.),dict(generation=0),
                dict(updated_monotonic=31.),dict(frame_valid=False),dict(observer_lease_s=float('nan'))):
            self.assertFalse(direction_available(state_packet(30.,**change),30.),change)

    def test_side_attribution_does_not_use_a_held_direction_alone(self):
        vad=Mock(return_value=True);gate=SideSpeechGate(self.c,vad)
        for n in range(4):
            now=30+n*.02
            masked,request=gate.observe(now,b'voice',dict(clean_rms=.01,reference_rms=0.),state_packet(now,sector=0),
                dict(updated_monotonic=now,phase='idle',single_face=True,owner='owner',main_bearing_deg=0.),available=True,now=now)
            self.assertFalse(masked);self.assertIsNone(request)
        self.assertEqual(gate.diagnostic['reason'],'direction_held_unattributed')
        self.assertEqual(vad.call_count,4)  # Current audio still updates the stateful VAD.

    def test_admitted_side_utterance_keeps_listening_without_coordinate_repeats(self):
        gate=SideSpeechGate(self.c,lambda _:True);requests=[]
        for n in range(1000):
            now=10+n*.02
            masked,request=gate.observe(now,b'voice',dict(clean_rms=.01,reference_rms=0.),state_packet(now,event=10.,sector=0),
                dict(updated_monotonic=now,phase='idle',single_face=True,owner='owner',main_bearing_deg=0.),available=True,now=now)
            self.assertTrue(masked)
            if request:requests.append(request)
        self.assertEqual(len(requests),1);self.assertEqual(requests[0]['direction_generation'],1)

    def test_new_receiver_generation_ends_old_side_audio_scope(self):
        gate=SideSpeechGate(self.c,lambda _:True)
        def frame(now,generation):
            return gate.observe(now,b'voice',dict(clean_rms=.01,reference_rms=0.),state_packet(now,event=now,sector=0,generation=generation),
                dict(updated_monotonic=now,phase='idle',single_face=True,owner='owner',main_bearing_deg=0.),available=True,now=now)
        for n in range(4):_,request=frame(10+n*.02,1)
        self.assertIsNotNone(request)
        self.assertEqual(frame(10.08,2),(False,None))
        self.assertEqual(gate.diagnostic['reason'],'direction_source_changed')

    def test_short_breath_during_introduction_does_not_require_a_new_direction(self):
        gate=SideSpeechGate(self.c,lambda pcm:pcm==b'voice')
        def frame(now,pcm):
            return gate.observe(now,pcm,dict(clean_rms=.01 if pcm==b'voice' else 0.,reference_rms=0.),
                state_packet(now,event=10.,sector=0),dict(updated_monotonic=now,phase='idle',
                    single_face=True,owner='owner',main_bearing_deg=0.),available=True,now=now)
        for n in range(4):frame(10+n*.02,b'voice')
        self.assertEqual(frame(10.5,b'quiet'),(True,None))
        self.assertEqual(frame(10.58,b'voice'),(True,None))
        self.assertEqual(frame(11.3,b'quiet'),(False,None))

    def test_held_introduction_slot_only_creates_an_unverified_capture(self):
        gate=SideSpeechGate(self.c,lambda _:True)
        for n in range(4):
            now=30+n*.02
            masked,request=gate.observe(now,b'voice',dict(clean_rms=.01,reference_rms=0.),state_packet(now,sector=0),
                dict(updated_monotonic=now,phase='idle',single_face=True,owner='owner',main_bearing_deg=0.),
                available=True,now=now,allow_held_followup=True)
            self.assertTrue(masked)
        self.assertEqual(request['source_relation'],'introduction_followup_unverified')
        self.assertNotIn('actor',request);self.assertNotIn('person',request)


class VoiceScopeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from test_cloud_control import CloudBackendTests
        from vivis.speech.realtime import VoiceSession
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        v=CloudBackendTests().service(self.tmp.name,[]);self.v=v
        self.addCleanup(v.log.close);self.addCleanup(v.recorders.close);self.addCleanup(v.conversation_animation.close)
        v.context=dict(revision=1,mode='interaction',ready=True);v.enabled=True;v.health_available=True
        v.sound_gate=SideSpeechGate(configuration(),lambda _:True)
        v.search_gate=NoFaceSpeechGate(configuration(),lambda pcm:pcm==b'voice')
        v.sound_started=True;v.side_suppressed=False;v.cloud_state=VoiceSession(v.player,v.log)
        event=time.monotonic()-30.
        v.direction_observer=SimpleNamespace(attention_snapshot=lambda now:state_packet(now,event=event))
        v.sound_mailbox=SimpleNamespace(latest=None,submit=Mock())
        self.identity=(v.turn_person,v.memory_session,v.input_generation)

    def frame(self,stamp,pcm=b'voice',phase='idle'):
        self.v.sound_mailbox.latest=dict(updated_monotonic=time.monotonic(),owner='owner',phase=phase,face_count=0,visual_fresh=True)
        return self.v.acoustic_microphone(stamp,pcm,dict(clean_rms=.01 if pcm==b'voice' else 0.,reference_rms=0.))

    async def start_seek(self):
        start=time.monotonic()-.1
        for n in range(4):self.assertFalse(self.frame(start+n*.02))
        self.assertIsNotNone(self.v.sound_search);return start

    async def test_held_seek_never_masks_pcm_or_changes_business_actor(self):
        await self.start_seek();self.assertFalse(self.v.sound_frozen())
        self.assertTrue(self.v.sound_search['last_known_direction_only'])
        self.assertEqual(self.identity,(self.v.turn_person,self.v.memory_session,self.v.input_generation))
        self.v.sound_search_tick();self.assertEqual(self.v.sound_mailbox.submit.call_args.args[0]['action'],'seek')

    async def test_receiver_loss_cancels_inflight_seek_without_identity_mutation(self):
        await self.start_seek();self.v.direction_observer.attention_snapshot=lambda _:None
        self.v.sound_search_tick();self.assertIsNone(self.v.sound_search)
        self.assertEqual(self.v.sound_mailbox.submit.call_args.args[0]['action'],'cancel')
        self.assertEqual(self.identity,(self.v.turn_person,self.v.memory_session,self.v.input_generation))

    async def test_silence_is_observed_during_motion_to_release_utterance_latch(self):
        start=await self.start_seek()
        self.assertTrue(self.v.search_gate.used)
        for n in range(30):self.assertFalse(self.frame(start+.08+n*.02,b'quiet','seek_hold'))
        self.assertFalse(self.v.search_gate.used)

    async def test_receiver_reconnect_cannot_continue_old_seek(self):
        await self.start_seek()
        self.v.direction_observer.attention_snapshot=lambda now:state_packet(now,generation=2,event=now)
        self.v.sound_search_tick();self.assertIsNone(self.v.sound_search)

    async def test_held_followup_requires_same_guest_owner_visit_and_deadline(self):
        now=time.monotonic();v=self.v;direction=state_packet(now,event=now-30.)
        v.introduction_draft=dict(introducer='heyude',guest_key='a'*32,guest_owner='b'*32,
            session=v.memory_session,stamp=now)
        feedback=dict(visitor=dict(token='a'*32,owner='b'*32))
        self.assertTrue(v.held_introduction_followup(direction,feedback,now))
        for guest in (dict(token='x'*32,owner='b'*32),dict(token='a'*32,owner='x'*32)):
            self.assertFalse(v.held_introduction_followup(direction,dict(visitor=guest),now))
        self.assertFalse(v.held_introduction_followup(state_packet(now+121.,event=now-30.),feedback,now+121.))
        v.introduction_draft['session']='old-visit'
        self.assertFalse(v.held_introduction_followup(direction,feedback,now))

    async def test_held_followup_routes_without_reusing_the_previous_introducer_identity(self):
        v=self.v;start=time.monotonic()-.1
        v.introduction_draft=dict(introducer='heyude',guest_key='a'*32,guest_owner='b'*32,
            session=v.memory_session,stamp=start)
        v.presence.single_visible=lambda:True;v.turn_reply_pending=lambda:False
        v.side_audio=SimpleNamespace(pcm24_bytes=bytes(640));v.sound_mailbox.latest=dict(
            updated_monotonic=time.monotonic(),owner='owner',phase='idle',single_face=True,
            face_count=1,main_bearing_deg=0.,visitor=dict(token='a'*32,owner='b'*32))
        with patch('vivis.speech.side_dialogue.SideDialogue.run',new=AsyncMock(return_value=None)) as runner:
            for n in range(4):self.assertTrue(v.acoustic_microphone(start+n*.02,b'voice',dict(clean_rms=.01,reference_rms=0.)))
            task=v.side_task
            self.assertEqual(v.sound_attention['source_relation'],'introduction_followup_unverified')
            self.assertIsNone(v.side_dialogue.actor);self.assertIsNone(v.side_dialogue.draft())
            self.assertEqual(self.identity,(v.turn_person,v.memory_session,v.input_generation))
            v.cancel_sound_attention('shutdown');await asyncio.gather(task,return_exceptions=True)
            runner.assert_not_called()  # Cancelled before scheduling; no stale session/coroutine created.

    async def test_missing_name_followup_rechecks_current_voice_before_registering(self):
        from vivis.speech.side_dialogue import SideDialogue
        from vivis.interaction.introductions import Introductions
        v=self.v;v.knowledge.root=Path(self.tmp.name);now=time.monotonic()
        request=dict(id='2'*32,person=None,guest_key='a'*32,guest=dict(token='a'*32,owner='b'*32,stamp=now),
            connection=v.message_connection,session=v.memory_session,revision=v.context['revision'],generation=v.input_generation)
        first=SideDialogue(v,request);first.actor='heyude';first.transcript='这位是我们的软件工程师'
        output=await first.tool('introduce_visitor',dict(background='软件工程师'))
        self.assertEqual(output['status'],'needs_address')
        second=SideDialogue(v,dict(request,id='3'*32,source_relation='introduction_followup_unverified'))
        second.transcript='这位叫王教授'
        self.assertIsNone(second.draft())
        with self.assertRaises(ValueError):await second.tool('introduce_visitor',dict(preferred_address='王教授'))
        self.assertFalse(Introductions(Path(self.tmp.name)).statuses())
        second.actor='wangrunzhang';self.assertIsNone(second.draft())
        second.actor='heyude'  # Simulated independent confirmation for THIS utterance.
        output=await second.tool('introduce_visitor',dict(preferred_address='王教授'))
        self.assertTrue(output['accepted']);saved=Introductions(Path(self.tmp.name)).statuses()[0]
        self.assertIn('软件工程师',saved['source']['text']);self.assertIn('王教授',saved['source']['text'])
        self.assertIsNone(v.introduction_draft)
        self.assertEqual(self.identity,(v.turn_person,v.memory_session,v.input_generation))

    async def test_long_introduction_finishes_after_direction_is_held_without_borrowing_main_identity(self):
        import numpy as np
        from vivis.speech.side_dialogue import SideDialogue
        from vivis.interaction.introductions import Introductions
        v=self.v;now=time.monotonic();v.knowledge.root=Path(self.tmp.name)
        v.direction_observer.attention_snapshot=lambda t:state_packet(t,event=now-20.02)
        request=dict(id='1'*32,owner='owner',person=None,guest_key='a'*32,
            guest=dict(token='a'*32,owner='b'*32,stamp=now),connection=v.message_connection,
            session=v.memory_session,revision=v.context['revision'],generation=v.input_generation,
            heard_monotonic=now-20.,created_monotonic=now-19.92,
            direction_update_mode='on_change',direction_generation=1,motion_returned=True)
        lane=SideDialogue(v,request);v.side_dialogue=lane;v.sound_attention=request
        lane.transcript='这是王教授，研究灵巧手。'
        with self.assertRaises(ValueError):
            await lane.tool('introduce_visitor',dict(preferred_address='王教授',background='研究灵巧手'))
        self.assertFalse(Introductions(Path(self.tmp.name)).statuses())
        async def confirmed_current_voice(*_):lane.actor='heyude'
        lane.identify=confirmed_current_voice  # Simulated current B voice, not a direction-derived identity.
        audio=(np.sin(np.arange(320)*.12)*4000).astype('<i2').tobytes()
        for n in range(1000):lane.feed(now-20+n*.02,audio,bytes(len(audio)))
        lane.end();v.acoustic_tick();self.assertIs(v.sound_attention,request)
        packets=[dict(type='conversation.item.input_audio_transcription.completed',transcript=lane.transcript),
            dict(type='response.created',response=dict(id='intro_tool')),
            dict(type='response.function_call_arguments.done',response_id='intro_tool',call_id='intro',name='introduce_visitor',
                arguments=json.dumps(dict(display_name='',preferred_address='王教授',background='研究灵巧手'))),
            dict(type='response.done',response=dict(id='intro_tool',status='completed')),
            dict(type='response.created',response=dict(id='intro_answer')),
            dict(type='response.audio.delta',response_id='intro_answer',delta=base64.b64encode(audio).decode()),
            dict(type='response.audio_transcript.done',response_id='intro_answer',transcript='王教授，欢迎！'),
            dict(type='response.done',response=dict(id='intro_answer',status='completed'))]
        class WS:
            async def send(self,raw):
                event=json.loads(raw)
                if event['type']=='session.update':self.session=event['session']
            async def recv(self):return json.dumps(dict(type='session.updated',session=self.session))
            def __aiter__(self):return self
            async def __anext__(self):
                if not packets:raise StopAsyncIteration
                return json.dumps(packets.pop(0))
        @asynccontextmanager
        async def connection(*_):yield WS()
        heard=[]
        async def dac():
            while not task.done():
                heard.append(v.player.render(1920,time.monotonic()));await asyncio.sleep(.002)
        with patch('vivis.speech.side_dialogue.websocket_connection',connection):
            task=asyncio.create_task(lane.run());await asyncio.gather(task,dac())
        self.assertTrue(any(np.any(x) for x in heard));self.assertEqual(lane.actor,'heyude')
        saved=Introductions(Path(self.tmp.name)).statuses()
        self.assertEqual(len(saved),1);self.assertNotEqual(saved[0]['person_id'],'heyude')
        self.assertEqual(saved[0]['introducer'],'heyude')
        self.assertEqual(self.identity,(v.turn_person,v.memory_session,v.input_generation))
