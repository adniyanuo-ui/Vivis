"""Program-owned reading uses the chat voice but cannot invent receipt text."""
import asyncio
import base64
import json
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock,Mock,patch

from vivis.speech.trusted_reading import RealtimeReader,ReadingMismatch
from vivis.speech.runtime_policy import policy


class TrustedReadingTests(IsolatedAsyncioTestCase):
    def reader(self,text='取件码三、三、八、八。',status='completed'):
        self.sent=[];self.closed=False
        events=[dict(type='response.created',response=dict(id='r')),
            dict(type='response.audio.delta',response_id='r',delta=base64.b64encode(b'\0'*4800).decode()),
            dict(type='response.audio_transcript.done',response_id='r',transcript=text),
            dict(type='response.done',response=dict(id='r',status=status))]
        outer=self
        class WS:
            async def __aenter__(self):
                if not hasattr(self,'original'):self.original=list(events)
                else:events[:]=self.original
                return self
            async def __aexit__(self,*_):outer.closed=True
            async def send(self,s):outer.sent.append(json.loads(s))
            def __aiter__(self):return self
            async def __anext__(self):
                if not events:raise StopAsyncIteration
                return json.dumps(events.pop(0))
        self.ws=WS();self.events=events
        self.log=Mock()
        return RealtimeReader('wss://test','test-secret',dict(voice='longanyang_v3.1',
            tools=[dict(name='must-not-be-in-reader')],instructions='private chat history',
            turn_detection=dict(type='smart_turn_v2')),self.log)

    async def test_same_voice_no_tools_no_history_and_verified_audio(self):
        reader=self.reader()
        with patch('vivis.speech.trusted_reading.websocket_connection',return_value=self.ws), \
             patch('vivis.speech.trusted_reading.connect_session',new_callable=AsyncMock) as connect:
            audio=await reader.generate('取件码3、3、8、8。')
        settings=connect.call_args.args[1]
        self.assertEqual(settings['voice'],'longanyang_v3.1');self.assertEqual(settings['tools'],[])
        self.assertNotIn('private chat history',settings['instructions'])
        self.assertEqual(len(audio.samples),2400);self.assertEqual(audio.sample_rate,24000)
        self.assertTrue(self.closed)
        self.assertEqual(json.loads(self.sent[0]['item']['content'][0]['text'])['text'],'取件码3、3、8、8。')

    async def test_wrong_code_or_false_commit_is_never_returned_for_playback(self):
        for expected,actual in [('取件码三三八八。','取件码三三八九。'),('尚未保存。','已保存。')]:
            reader=self.reader(actual)
            with patch('vivis.speech.trusted_reading.websocket_connection',return_value=self.ws), \
                 patch('vivis.speech.trusted_reading.connect_session',new_callable=AsyncMock):
                with self.assertRaisesRegex(ValueError,'differs'):await reader.generate(expected)
            self.assertTrue(self.closed)

    async def test_failed_response_and_tool_call_are_rejected(self):
        for tool in (False,True):
            reader=self.reader(status='failed')
            if tool:self.events.insert(1,dict(type='response.function_call_arguments.done',name='write_database'))
            with patch('vivis.speech.trusted_reading.websocket_connection',return_value=self.ws), \
                 patch('vivis.speech.trusted_reading.connect_session',new_callable=AsyncMock):
                with self.assertRaises(ValueError):await reader.generate('取件码三、三、八、八。')

    async def test_timeout_closes_connection_without_leaking_background_job(self):
        reader=self.reader()
        async def delayed(*_):await asyncio.sleep(1)
        with patch('vivis.speech.trusted_reading.websocket_connection',return_value=self.ws), \
             patch('vivis.speech.trusted_reading.connect_session',side_effect=delayed), \
             patch('vivis.speech.trusted_reading.policy',return_value={**policy(),'message_tts_operation_s':.02}):
            with self.assertRaises(TimeoutError):await reader.generate('测试')
        self.assertTrue(self.closed)

    async def test_mismatch_retry_returns_only_verified_audio_once(self):
        reader=self.reader();verified=object()
        reader._generate=AsyncMock(side_effect=[ReadingMismatch('wrong text'),verified])
        self.assertIs(await reader.generate('已保存，等转告。'),verified)
        self.assertEqual(reader._generate.await_count,2)
        reader._generate=AsyncMock(side_effect=ReadingMismatch('wrong again'))
        with self.assertRaises(ReadingMismatch):await reader.generate('已保存，等转告。')
        self.assertEqual(reader._generate.await_count,2)
