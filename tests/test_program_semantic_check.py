"""Tool-omission repair accepts only a complete, bounded semantic decision."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch

from vivis.speech.program_followup import classify


class ProgramSemanticCheckTests(unittest.IsolatedAsyncioTestCase):
    async def result(self,content,status='completed'):
        class Socket:
            def __init__(self):self.send=AsyncMock()
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            async def recv(self):return json.dumps(dict(type='session.updated',session=dict(modalities=['text'],tools=[])))
            def __aiter__(self):
                async def events():
                    yield json.dumps(dict(type='response.text.done',text=content))
                    yield json.dumps(dict(type='response.done',response=dict(status=status)))
                return events()
        socket=Socket()
        with patch('vivis.speech.program_followup.websocket_connection',return_value=socket):
            value=await classify('wss://test','placeholder',dict(voice='test'),'今天星期几？',SimpleNamespace(write=lambda *a,**k:None),1)
        packets=[json.loads(call.args[0]) for call in socket.send.await_args_list]
        return value,packets

    async def test_question_is_text_only_has_no_tools_and_keeps_exact_utterance(self):
        value,packets=await self.result('{"kind":"question"}')
        self.assertEqual(value,'question')
        config=packets[0]['session']
        self.assertEqual(config['modalities'],['text']);self.assertFalse(config['tools']);self.assertFalse(config['enable_search'])
        self.assertIsNone(config['turn_detection'])
        self.assertEqual(json.loads(packets[1]['item']['content'][0]['text']),dict(utterance='今天星期几？'))

    async def test_incomplete_or_unexpected_decisions_cannot_authorize_handoff(self):
        for raw,status in (('{"kind":"question"}','cancelled'),('{"kind":"question"','completed'),
                ('{"kind":"play"}','completed'),('{"kind":"question","program":"x"}','completed')):
            with self.subTest(raw=raw,status=status),self.assertRaises(ValueError):await self.result(raw,status)

    async def test_timeout_is_local_and_cancels_request(self):
        class Socket:
            async def __aenter__(self):await asyncio.Future()
            async def __aexit__(self,*_):return False
        with patch('vivis.speech.program_followup.websocket_connection',return_value=Socket()):
            with self.assertRaises(TimeoutError):
                await classify('wss://test','placeholder',dict(voice='test'),'今天星期几？',None,.01)


if __name__=='__main__':unittest.main()
