"""Cold proactive greeting: no microphone speech, but confirmed controller input."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from vivis.paths import ROOT
from vivis.speech.cloud_managed import CloudManagedVoice,ControllerEventGate
from vivis.interaction.cloud_control import SpeechPresence
from test_managed_voice import FakeAudio


class ControllerEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_proactive_event_is_labeled_and_waits_for_matching_server_ack(self):
        sent=[];gate=ControllerEventGate()
        class WS:
            async def send(self,text):sent.append(json.loads(text))
        task=asyncio.create_task(gate.append(WS(),dict(mode='interaction',ready=True),'welcome','自然欢迎当前访客'))
        await asyncio.sleep(0)
        item=sent[0]['item'];self.assertEqual(item['role'],'user')
        self.assertIn('human_speech": false',item['content'][0]['text'])
        self.assertIn('不是用户说话',item['content'][0]['text'])
        gate.observe(dict(type='conversation.item.created',item=dict(id='unrelated')))
        await asyncio.sleep(0);self.assertFalse(task.done())
        gate.observe(dict(type='conversation.item.created',item=dict(id=item['id'])))
        self.assertEqual(await task,item['id']);self.assertFalse(gate.pending)

    async def test_cancelled_stage_leaves_no_pending_event(self):
        gate=ControllerEventGate()
        class WS:
            async def send(self,text):pass
        task=asyncio.create_task(gate.append(WS(),{},'return','控制器已回交互'))
        await asyncio.sleep(0);task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertFalse(gate.pending)

    async def test_cold_reception_generates_only_after_controller_event_and_never_fakes_user_transcript(self):
        messages=asyncio.Queue();sent=[];has_user=False;request_seen=False;events=[]
        class WS:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await messages.get())
            async def send(self,text):
                nonlocal has_user,request_seen
                packet=json.loads(text);sent.append(packet)
                if packet['type']=='conversation.item.create':
                    has_user=True
                    await messages.put(dict(type='conversation.item.created',item=packet['item']))
                if packet['type']=='response.create':
                    if not has_user:
                        await messages.put(dict(type='error',error=dict(code='invalid_value',message='conversation has no messages or no user message')))
                    request_seen=True
        async def connected(ws,session,log):return session
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{'DASHSCOPE_API_KEY':'unit-placeholder'}), \
             patch('vivis.speech.managed.AudioIO',FakeAudio),patch('vivis.interaction.memory.PersonMemory'), \
             patch('vivis.speech.cloud_managed.websocket_connection',return_value=WS()), \
             patch('vivis.speech.cloud_managed.connect_session',side_effect=connected):
            def read(name):return json.loads((ROOT/name).read_text())
            speech=read('configs/perception/speech.json');speech['managed_voice']=read('configs/interaction/p1.json')['realtime_voice']
            speech['managed_voice']['lip_sync']['enabled']=False;speech['programs']=str(ROOT/'configs/interaction/programs.json')
            service=CloudManagedVoice(speech,ROOT/'configs/robots/xiaoshuai/robot.json',Path(tmp),events.append)
            service.memory=None;service.presence=SpeechPresence(read('configs/perception/speech_presence.json'),tmp)
            service.reception.c=dict(service.reception.c,enabled=False)
            service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
            service.enabled=True;service.presence.wake();service.announcement_revision=1
            task=asyncio.create_task(service._connection())
            try:
                deadline=time.monotonic()+2
                while not request_seen and time.monotonic()<deadline:
                    if task.done():task.result()
                    await asyncio.sleep(.01)
                self.assertTrue(request_seen);self.assertTrue(has_user)
                self.assertFalse(any(e['kind']=='cloud_error' for e in events))
                self.assertFalse(any(e['kind']=='cloud_transcript' and e.get('role')=='user' for e in events))
                kinds=[p['type'] for p in sent]
                self.assertLess(kinds.index('conversation.item.create'),kinds.index('response.create'))
            finally:
                task.cancel();await asyncio.gather(task,return_exceptions=True);service.log.close()
