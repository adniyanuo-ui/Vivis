"""Body presence/wake integration, with no microphone, serial or live camera."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock

import numpy as np

import test_cloud_control as fixtures
from vivis.interaction.cloud_control import SpeechPresence,PresencePublisher,ReceptionGate
from vivis.speech.realtime import VoiceSession
from vivis.vision.person import PersonDetector,PersonWorker,configuration,verify_model


class DetectorTests(unittest.TestCase):
    def test_actual_bundled_model_loads_and_rejects_blank_image(self):
        import cv2
        cv2.setNumThreads(1)
        self.assertTrue(verify_model(configuration()).is_file())
        self.assertEqual(PersonDetector(configuration()).detect(np.zeros((480,640,3),np.uint8))['persons'],0)

    def test_wrong_model_hash_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'model').write_bytes(b'not a model')
            (root/'manifest').write_text(json.dumps(dict(destination='model',size=11,sha256='wrong')))
            with self.assertRaisesRegex(ValueError,'missing or changed'):
                verify_model(dict(model_manifest='manifest'),root)

    def test_person_class_only_rgb_letterbox_and_original_coordinate_clipping(self):
        d=PersonDetector(configuration());network=Mock();d.net=network
        outputs=[]
        for stride in (8,16,32):
            n=(416//stride)**2
            outputs.extend([np.zeros((1,n,80),np.float32),np.zeros((1,n,32),np.float32)])
        # One central person, one high-scoring dog; padded/outside boxes clipped.
        center=26*52+26;outputs[0][0,center,0]=.9
        outputs[0][0,center+3,16]=.99
        network.forward.return_value=outputs
        frame=np.full((200,400,3),(10,20,30),np.uint8)
        result=d.detect(frame)
        self.assertEqual(result['persons'],1)
        x,y,w,h=result['boxes'][0]['bbox']
        self.assertTrue(0<=x<x+w<=400);self.assertTrue(0<=y<y+h<=200)
        blob=network.setInput.call_args.args[0]
        self.assertAlmostEqual(float(blob[0,0,208,208]),(30-103.53)/57.375,places=5)
        self.assertAlmostEqual(float(blob[0,0,0,208]),-103.53/57.375,places=5)

    def test_worker_replaces_pending_frame_and_keeps_capture_timestamp(self):
        entered=threading.Event();release=threading.Event();second=threading.Event();seen=[]
        def infer(frame):
            seen.append(frame)
            if frame=='one':entered.set();release.wait(2)
            else:second.set()
            return dict(persons=1,boxes=[],processing_ms=1)
        worker=PersonWorker(configuration(),detector=SimpleNamespace(detect=infer));self.addCleanup(worker.close)
        worker.submit((1,10,'one'),10);self.assertTrue(entered.wait(1))
        worker.submit((2,11,'two'),11);worker.submit((3,12,'three'),12)
        self.assertEqual(worker.pending[0],3);release.set();self.assertTrue(second.wait(1))
        worker.close();self.assertEqual(seen,['one','three'])
        self.assertEqual(worker.observation()['observed_monotonic'],12)

    def test_local_inference_failure_disables_body_only(self):
        failed=threading.Event()
        def infer(_):failed.set();raise ValueError('inference failed')
        worker=PersonWorker(configuration(),detector=SimpleNamespace(detect=infer))
        worker.submit((1,10,None),10);self.assertTrue(failed.wait(1));worker.thread.join(1)
        self.assertFalse(worker.observation()['available']);self.assertEqual(worker.observation()['persons'],0)
        worker.close()


class PresenceTests(unittest.TestCase):
    def test_turning_away_after_greeting_does_not_rearm_duplicate_welcome(self):
        gate=ReceptionGate(fixtures.config('configs/interaction/reception.json'))
        gate.dismiss_current_visit()
        for now in (10,14,20):self.assertFalse(gate.observe(False,'standby',now,occupied=True))
        self.assertFalse(gate.observe(True,'standby',21,occupied=True))
        gate.observe(False,'standby',22);gate.observe(False,'standby',26)
        self.assertFalse(gate.observe(True,'standby',27))
        self.assertTrue(gate.observe(True,'standby',27.5))

    def test_body_alone_never_proactively_wakes_or_speaks_and_is_not_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            c=fixtures.config('configs/perception/speech_presence.json');p=SpeechPresence(c,folder);pub=PresencePublisher(c,folder)
            pub.publish(0,10,now=10,person=dict(enabled=True,available=True,persons=1,observed_monotonic=10))
            self.assertTrue(p.greeting_visible(10));self.assertFalse(p.allowed(10));self.assertFalse(p.last_visible)
            gate=ReceptionGate(fixtures.config('configs/interaction/reception.json'))
            self.assertFalse(gate.observe(p.last_visible,'standby',10));self.assertFalse(gate.observe(p.last_visible,'standby',11))
            p.greet();self.assertTrue(p.allowed(10));self.assertEqual(p.wake_until,0)
            self.assertFalse(p.visible(10));self.assertFalse(p.single_visible(10))
            p.clear_greeting();self.assertFalse(p.allowed(10))

    def test_body_timestamp_is_not_refreshed_by_new_face_publications(self):
        with tempfile.TemporaryDirectory() as folder:
            c=fixtures.config('configs/perception/speech_presence.json');p=SpeechPresence(c,folder);pub=PresencePublisher(c,folder)
            for bad in (dict(persons=1,observed_monotonic=10),dict(persons=True,observed_monotonic=20),
                        dict(persons=1,observed_monotonic=21),dict(persons=1,observed_monotonic=float('nan'))):
                pub.last=-float('inf');pub.publish(0,20,now=20,person=dict(enabled=True,available=True,**bad))
                self.assertFalse(p.greeting_visible(20))


class ThreeWakeRulesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.events=[];self.s=fixtures.CloudBackendTests().service(folder.name,self.events);self.addCleanup(self.s.log.close)
        self.s.context.update(mode='standby',target='standby',ready=True,revision=1)
        self.s.identified_person=Mock(return_value=None)
        self.s.presence=SpeechPresence(self.s.presence.c,folder.name)
        self.pub=PresencePublisher(self.s.presence.c,folder.name);self.state=VoiceSession(self.s.player,self.s.log)

    def publish(self,faces=0,persons=0):
        now=time.monotonic();self.pub.last=-float('inf')
        self.pub.publish(faces,now,now,person=dict(enabled=True,available=True,persons=persons,observed_monotonic=now))

    def final(self,text):
        s=self.s;s.input_generation+=1;s.begin_user_turn('u');s.question_started('u',self.state)
        s.turn_ended_at=time.monotonic();s.memory_turns['u']['text']=text
        return dict(item_id='u',transcript=text)

    async def tool(self,args):
        sent=[]
        async def send(text):sent.append(json.loads(text))
        self.s.response_revisions['r']=1
        await self.s.tool(SimpleNamespace(send=send),dict(call_id='c-'+args.get('wake_reason',''),response_id='r',name='robot_program',arguments=json.dumps(args)))
        return json.loads(sent[-1]['item']['output'])

    async def test_face_arrival_without_identity_wakes_after_existing_hold(self):
        self.publish(faces=1);s=self.s;s.presence.allowed()
        self.assertFalse(s.reception.observe(s.presence.last_visible,'standby',10))
        self.assertTrue(s.reception.observe(s.presence.last_visible,'standby',10.5))
        self.assertIsNone(s.identified_person())

    async def test_body_greeting_wakes_and_survives_camera_handoff_until_ready(self):
        self.publish(persons=1);s=self.s
        self.assertTrue(s.admit_explicit_wake(self.final('hello，你好。'),self.state))
        self.assertEqual(self.events[-1]['arguments']['wake_reason'],'greeting')
        self.assertEqual(s.presence.wake_until,0)
        s.request(dict(kind='voice_context',revision=2,mode='interaction',target='interaction',ready=False))
        self.pub.close()  # Old camera owner releases presence during handoff.
        controller=SimpleNamespace(append=AsyncMock());audio=SimpleNamespace(enable=AsyncMock())
        self.assertFalse(await s.complete_explicit_wake(None,self.state,controller,audio,'instructions'))
        self.assertIsNotNone(s.explicit_wake_pending)
        self.publish(persons=1)
        s.request(dict(kind='voice_context',revision=3,mode='interaction',target='interaction',ready=True));s.enabled=True
        self.assertTrue(await s.complete_explicit_wake(None,self.state,controller,audio,'instructions'))
        self.assertTrue(s.can_speak());self.assertIsNone(s.identified_person())

    async def test_empty_frame_greeting_cannot_wake_even_if_model_forges_name_reason(self):
        self.publish();s=self.s
        self.assertFalse(s.admit_explicit_wake(self.final('hello，你好。'),self.state))
        for reason,explicit in (('greeting',False),('name',True)):
            result=await self.tool(dict(action='wake',wake_reason=reason,explicit_wake=explicit))
            self.assertFalse(result['accepted'])
        self.assertFalse(self.events);self.assertEqual(s.presence.wake_until,0)

    async def test_empty_frame_exact_name_still_wakes(self):
        self.publish();s=self.s
        self.assertTrue(s.admit_explicit_wake(self.final('小帅你好'),self.state))
        self.assertTrue(self.events[-1]['arguments']['explicit_wake'])
        self.assertGreater(s.presence.wake_until,time.monotonic())

    async def test_body_quote_and_playback_echo_are_not_greetings(self):
        self.publish(persons=1);s=self.s
        self.assertFalse(s.admit_explicit_wake(self.final('他说你好'),self.state))
        self.state.generating='robot';self.state.responses['robot']=dict(transcript='你好')
        self.assertFalse(s.admit_explicit_wake(self.final('你好'),self.state))
        self.assertFalse(self.events)


if __name__=='__main__':unittest.main()
