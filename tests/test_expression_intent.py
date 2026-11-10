"""Intent arbitration and cloud control failures, without audio or hardware."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from vivis.paths import ROOT
from vivis.interaction.expression_intent import configuration,arguments,IntentWriter,IntentEnvelope,atomic_json
from vivis.interaction.cloud_control import controller_intent,normalize_program
from vivis.speech.cloud_managed import CloudManagedVoice,CloudIdleTimeout,ConversationReset


class IntentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'intent.json';self.c=configuration()

    def test_explicit_command_beats_social_response_until_expiry(self):
        writer=IntentWriter(self.path,self.c)
        result=writer.publish(dict(expression='neutral',source='command',duration_s=5),7,10)
        social=writer.publish(dict(expression='smile',source='social'),7,11)
        self.assertEqual(social['status'],'suppressed_by_explicit_command')
        self.assertEqual(social['intent_id'],result['intent_id'])
        next_result=writer.publish(dict(expression='smile',source='social'),7,16)
        self.assertEqual(next_result['status'],'pending_expression_target')
        self.assertNotEqual(next_result['intent_id'],result['intent_id'])

    def test_smooth_smile_retraction_expiry_and_target_ack(self):
        writer=IntentWriter(self.path,self.c);envelope=IntentEnvelope(self.path,self.c)
        result=writer.publish(dict(expression='smile',duration_s=8),1,10)
        values=[envelope.step(0.,10+i/60) for i in range(241)]
        self.assertGreater(values[-1],.95)
        self.assertLess(max(abs(b-a) for a,b in zip(values,values[1:])),.03)
        envelope.acknowledge()
        self.assertEqual(writer.acknowledgement(result['intent_id'])['status'],'target_accepted')
        writer.publish(dict(expression='neutral',duration_s=4),1,14.1)
        closed=[envelope.step(1.,14.1+i/60) for i in range(241)]
        self.assertLess(closed[180],.1)
        self.assertLess(max(abs(b-a) for a,b in zip(closed,closed[1:])),.04)
        recovered=[envelope.step(.7,18.2+i/60) for i in range(301)]
        self.assertAlmostEqual(recovered[-1],.7,delta=.02)

    def test_clear_releases_override_and_stale_or_invalid_commands_are_not_accepted(self):
        writer=IntentWriter(self.path,self.c);envelope=IntentEnvelope(self.path,self.c)
        writer.publish(dict(expression='neutral'),1,10)
        for i in range(240):envelope.step(1.,10+i/60)
        self.assertLess(envelope.step(1.,14),.03)
        writer.clear()
        for i in range(300):last=envelope.step(.5,14.1+i/60)
        self.assertAlmostEqual(last,.5,delta=.02)
        packet=dict(expression='smile',intensity=1.,duration_s=8.,source='command',schema_version='vivis-expression-intent-1.0',
                    id='a'*32,revision=1,created_monotonic=0.,expires_monotonic=8.)
        atomic_json(self.path,packet)
        self.assertIsNone(IntentWriter(self.path,self.c).acknowledgement(packet['id']))
        other=IntentEnvelope(self.path,self.c);self.assertEqual(other.step(0.,100),0.)
        for invalid in (dict(expression='cry'),dict(expression='smile',intensity=float('nan')),
                        dict(expression='smile',duration_s=100),dict(expression='smile',id=32)):
            with self.assertRaises(ValueError):arguments(invalid,self.c)

    def test_actual_interaction_mapper_uses_same_intent_without_servo_id_settings(self):
        from vivis.runtime.robot import Robot
        from vivis.interaction.standby import plan
        settings=json.loads((ROOT/'configs/interaction/interaction.json').read_text())
        _,_,_,mapper,source,_=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),settings,1.)
        source.expression_intent=IntentEnvelope(self.path,self.c)
        writer=IntentWriter(self.path,self.c)
        start=source.sample(0.,10)
        writer.publish(dict(expression='smile'),1,10)
        for i in range(240):smile=source.sample(0.,10+i/60)
        self.assertGreater(source.smile,.9)
        self.assertTrue(any(abs(float(a)-float(b))>1 for a,b in zip(start,smile)))
        writer.publish(dict(expression='neutral'),1,14.1)
        for i in range(300):source.sample(0.,14.1+i/60)
        self.assertLess(source.smile,.02)
        self.assertEqual(mapper.smile_strength,settings['social_response']['strength'])

    def test_stop_does_not_need_wake_face_or_unused_programme_fields(self):
        for request in (dict(action='stop'),dict(action='stop',program='song'),dict(action='stop',program='',explicit_wake=False)):
            self.assertEqual(controller_intent(request,{'song'},'task'),dict(kind='interaction'))
        self.assertEqual(controller_intent(dict(action='standby'),{'song'},'task'),dict(kind='standby'))
        self.assertEqual(controller_intent(dict(action='stop'),{'song'},'standby'),dict(kind='standby'))
        with self.assertRaises(ValueError):normalize_program(dict(action='play',program='missing'),{'song'})


class CloudIntentTests(unittest.IsolatedAsyncioTestCase):
    def service(self,folder,events):
        def read(name):return json.loads((ROOT/name).read_text())
        speech=read('configs/perception/speech.json');speech['managed_voice']=read('configs/interaction/p1.json')['realtime_voice']
        speech['managed_voice']['lip_sync']['enabled']=False;speech['programs']=str(ROOT/'configs/interaction/programs.json')
        with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'unit-placeholder'}),patch('vivis.interaction.memory.PersonMemory'):
            service=CloudManagedVoice(speech,ROOT/'configs/robots/xiaoshuai/robot.json',folder,events.append)
        service.memory=None;return service

    async def test_expression_tool_uses_file_and_waits_for_control_loop_not_voice_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            events=[];service=self.service(Path(tmp),events);sent=[]
            async def send(text):sent.append(json.loads(text))
            service.context.update(revision=4,mode='interaction',target='interaction',ready=True)
            service.response_revisions['r']=4;service.presence.wake()
            try:
                await service.tool(SimpleNamespace(send=send),dict(name='set_expression',call_id='one',response_id='r',arguments='{"expression":"smile"}'))
                result=json.loads(sent[-1]['item']['output'])
                self.assertEqual(result['status'],'pending_expression_target')
                self.assertIsNotNone(service.expression_pending)
                loop=IntentEnvelope(service.expression.path,service.expression.c);loop.step(0.,time.monotonic());loop.acknowledge()
                self.assertIsNotNone(service.expression.acknowledgement(result['intent_id']))
                service.context.update(mode='task',target='baigufuren')
                await service.tool(SimpleNamespace(send=send),dict(name='set_expression',call_id='two',response_id='r',arguments='{"expression":"smile"}'))
                self.assertFalse(json.loads(sent[-1]['item']['output'])['accepted'])
                self.assertIsNotNone(service.tool_repair)
            finally:service.expression.clear();service.log.close()

    async def test_bad_control_arguments_are_logged_and_can_be_corrected(self):
        with tempfile.TemporaryDirectory() as tmp:
            events=[];service=self.service(Path(tmp),events);sent=[]
            async def send(text):sent.append(json.loads(text))
            service.context.update(revision=2,mode='task',target='expression-follow',ready=True);service.response_revisions['r']=2
            ws=SimpleNamespace(send=send)
            try:
                await service.tool(ws,dict(name='robot_program',call_id='bad',response_id='r',arguments='{}'))
                self.assertFalse(json.loads(sent[-1]['item']['output'])['accepted']);self.assertIsNotNone(service.tool_repair)
                await service.tool(ws,dict(name='robot_program',call_id='good',response_id='r',arguments='{"action":"stop","program":"expression-follow"}'))
                self.assertTrue(json.loads(sent[-1]['item']['output'])['accepted']);self.assertIsNone(service.tool_repair)
                self.assertEqual(events[-1]['arguments']['program'],'')
                await service.tool(ws,dict(name='robot_program',call_id='invalid-json',response_id='r',arguments='{'))
                self.assertIn('error',json.loads(sent[-1]['item']['output']))
            finally:service.log.close()

    async def test_only_specific_idle_timeout_reconnects_and_keeps_microphone_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=self.service(Path(tmp),[]);opens=[]
            async def connection():
                opens.append(1)
                if len(opens)==1:raise CloudIdleTimeout()
                service.shutdown.set()
            try:
                with patch.object(service,'_connection',side_effect=connection):await service.cloud(0)
                self.assertEqual(len(opens),2);self.assertIsNone(service.decoder)
            finally:service.log.close()

    async def test_new_visitor_does_not_inherit_old_tool_failure_or_expression(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=self.service(Path(tmp),[]);opens=[]
            service.tool_repair=dict(error='Old visitor request rejected');service.control_pending=True
            service.resume_history.append(dict(role='user',text='Old visitor context'))
            service.expression.publish(dict(expression='smile'),1)
            async def connection():
                opens.append(1)
                if len(opens)==1:raise ConversationReset('confirmed_person_changed')
                self.assertIsNone(service.tool_repair);self.assertFalse(service.control_pending)
                self.assertEqual(list(service.resume_history),[]);self.assertFalse(service.expression.path.exists())
                service.shutdown.set()
            try:
                with patch.object(service,'_connection',side_effect=connection):await service.cloud(0)
            finally:service.log.close()
