"""Optional text drafting cannot acquire devices, mutate business, or queue old speech."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from vivis.speech.social_topics import draft


class SocialTopicTests(unittest.IsolatedAsyncioTestCase):
    def socket(self,value=None,*,tool=False,status='completed',handshake_tools=False):
        self.sent=[];self.closed=False
        output=value or dict(topic='music',text='您平时喜欢听什么类型的音乐？',style='joy')
        events=[dict(type='response.created',response=dict(id='draft')),
            dict(type='response.text.done',response_id='draft',text=json.dumps(output,ensure_ascii=False)),
            dict(type='response.done',response=dict(id='draft',status=status))]
        if tool:events.insert(1,dict(type='response.function_call_arguments.done',name='robot_program'))
        outer=self
        class Socket:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):outer.closed=True
            async def send(self,data):outer.sent.append(json.loads(data))
            async def recv(self):
                return json.dumps(dict(type='session.updated',session=dict(modalities=['text'],
                    tools=[dict(name='forbidden')] if handshake_tools else [],enable_search=False)))
            def __aiter__(self):return self
            async def __anext__(self):
                if not events:raise StopAsyncIteration
                return json.dumps(events.pop(0))
        return Socket()

    async def generate(self,socket,timeout=1):
        with patch('vivis.speech.social_topics.websocket_connection',return_value=socket):
            return await draft('wss://same-model','test-secret',dict(voice='longanyang_v3.1'),
                dict(allowed_topics={'music':'音乐'}),Mock(),80,timeout)

    async def test_same_provider_text_only_without_search_tools_or_old_history(self):
        value=await self.generate(self.socket())
        self.assertEqual(value['topic'],'music');self.assertTrue(self.closed)
        session=self.sent[0]['session']
        self.assertEqual(session['voice'],'longanyang_v3.1');self.assertEqual(session['modalities'],['text'])
        self.assertFalse(session['enable_search']);self.assertEqual(session['tools'],[])
        self.assertEqual(self.sent[-1]['type'],'response.create')

    async def test_private_content_tool_and_failed_generation_never_become_speech(self):
        options=[dict(value=dict(topic='music',text='您的手机号12345678对吗？',style='joy')),
            dict(tool=True),dict(status='failed'),dict(handshake_tools=True),
            dict(value=dict(topic='unknown',text='您想聊什么？',style='joy'))]
        for args in options:
            with self.assertRaises(ValueError):await self.generate(self.socket(**args))
            self.assertTrue(self.closed)

    async def test_timeout_and_cancellation_close_only_the_optional_connection(self):
        for cancelled in (False,True):
            ws=self.socket();started=asyncio.Event()
            async def waiting():started.set();await asyncio.Future()
            ws.recv=waiting
            with patch('vivis.speech.social_topics.websocket_connection',return_value=ws):
                task=asyncio.create_task(draft('wss://same-model','test-secret',dict(voice='longanyang_v3.1'),
                    dict(allowed_topics={'music':'音乐'}),Mock(),80,.05))
                await started.wait()
                if cancelled:task.cancel()
                with self.assertRaises(asyncio.CancelledError if cancelled else TimeoutError):await task
            self.assertTrue(self.closed)


if __name__=='__main__':unittest.main()
