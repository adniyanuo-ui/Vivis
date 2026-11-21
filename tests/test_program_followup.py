"""Program handoff and proactive reception with a fake cloud and DAC."""
import asyncio
import base64
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import test_cloud_control as fixtures
from vivis.interaction.reception_conversation import ReceptionConversation, public_invitation
from vivis.speech.realtime import VoiceSession


class ProgramFollowupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.events = []
        self.service = service = fixtures.CloudBackendTests().service(directory.name, self.events)
        self.addCleanup(service.log.close)
        service.presence.single_visible = Mock(return_value=True)
        service.presence.allowed = Mock(return_value=True)
        service.recognized_person = Mock(return_value=None)
        service.identified_person = Mock(return_value=None)
        service.context.update(mode='task', target='baigufuren', ready=True, revision=2)
        self.state = service.cloud_state = VoiceSession(service.player, service.log)
        self.ws = SimpleNamespace(send=AsyncMock())

    def turn(self, item, rid):
        s = self.service
        s.input_generation += 1
        s.begin_user_turn(item)
        s.question_started(item, self.state)
        s.turn_ended_at = time.monotonic()
        self.response(item, rid)

    def response(self, item, rid):
        s = self.service
        self.state.responses[rid] = {'user_turn': {'item_id': item}, 'status': 'completed'}
        s.response_inputs[rid] = s.input_generation
        s.response_revisions[rid] = s.context['revision']

    async def tool(self, rid, name, **args):
        await self.service.tool(self.ws, dict(name=name, response_id=rid, call_id=str(self.ws.send.await_count),
            arguments=json.dumps(args)))
        return json.loads(json.loads(self.ws.send.await_args.args[0])['item']['output'])

    def ready_interaction(self):
        s = self.service
        s.request(dict(kind='voice_context', revision=3, mode='interaction', target='interaction', ready=False,
                       transition_reply_id=s.farewell['id']))
        s.request(dict(kind='voice_context', revision=4, mode='interaction', target='interaction', ready=True,
                       transition_reply_id=s.farewell['id']))
        s.finish_farewell(s.farewell, 'played')
        s.enabled = True

    async def test_question_waits_for_final_and_ready_keeps_original_once_and_cannot_select_song(self):
        s = self.service
        self.turn('original', 'intent')
        result = await self.tool('intent','program_followup',kind='question')
        self.assertTrue(result['accepted'])
        self.assertFalse(self.events)
        s.accept_program_followup('会唱白骨夫人吗？')
        self.assertEqual(self.events[0]['arguments']['action'], 'stop')
        self.assertEqual(s.farewell['text'], '稍等，我先停一下。')
        controller = SimpleNamespace(append=AsyncMock())
        await s.resume_program_followup(self.ws, controller, self.state)
        controller.append.assert_not_called()
        self.ready_interaction()
        await s.resume_program_followup(self.ws, controller, self.state)
        await s.resume_program_followup(self.ws, controller, self.state)
        controller.append.assert_awaited_once()
        self.assertIn('会唱白骨夫人吗', controller.append.await_args.args[3])
        self.assertIn('original', s.message_receipts.sources)
        self.response('original','continuation')
        self.assertFalse((await self.tool('continuation','robot_program',action='play',program='baigufuren'))['accepted'])
        self.assertEqual(len([e for e in self.events if e['kind']=='cloud_action']),1)

    async def test_short_negation_false_laughter_and_superseded_tool_do_not_stop_program(self):
        s = self.service
        for i, text in enumerate(('不要停','哈哈哈','嗯','歌词写的是你多大了。','剧本里说大笑一下。')):
            self.turn('item'+str(i),'rid'+str(i))
            s.question_onset['final_text'] = text
            self.assertFalse((await self.tool('rid'+str(i),'program_followup',kind='expression'))['accepted'])
        self.turn('new','latest')
        self.assertFalse((await self.tool('rid4','program_followup',kind='question'))['accepted'])
        self.assertFalse(self.events)

    async def test_new_request_replaces_previous_and_identity_change_discards_it(self):
        s = self.service
        self.turn('old','first')
        s.question_onset['final_text'] = '你多大了？'
        await self.tool('first','program_followup',kind='question')
        self.turn('new','second')
        s.question_onset['final_text'] = '大笑一下。'
        await self.tool('second','program_followup',kind='expression')
        self.assertEqual(s.program_followup['text'],'大笑一下。')
        self.assertEqual(len([e for e in self.events if e['kind']=='cloud_action']),1)
        s.recognized_person.return_value = 'other-visitor'  # Even while the UI still holds the previous name.
        controller = SimpleNamespace(append=AsyncMock())
        await s.resume_program_followup(self.ws,controller,self.state)
        controller.append.assert_not_called()
        self.assertIsNone(s.program_followup)

    async def test_new_input_during_network_ack_does_not_send_old_response(self):
        s = self.service
        self.turn('old','first')
        s.question_onset['final_text'] = '大笑一下。'
        await self.tool('first','program_followup',kind='expression')
        self.ready_interaction()
        def replace(*_):self.turn('new','later')
        controller = SimpleNamespace(append=AsyncMock(side_effect=replace))
        count = self.ws.send.await_count
        await s.resume_program_followup(self.ws,controller,self.state)
        self.assertEqual(self.ws.send.await_count,count)
        self.assertIsNone(s.followup_response)

    async def test_expression_cannot_act_until_ready_and_executes_once(self):
        s = self.service
        self.turn('original','first')
        s.question_onset['final_text'] = '大笑一下。'
        s.expression.publish = Mock(return_value=dict(status='pending_expression_target',intent_id='expression'))
        await self.tool('first','program_followup',kind='expression')
        self.assertFalse((await self.tool('first','set_expression',expression='smile',source='command'))['accepted'])
        self.ready_interaction()
        await s.resume_program_followup(self.ws,SimpleNamespace(append=AsyncMock()),self.state)
        self.response('original','continuation')
        self.assertFalse((await self.tool('continuation','set_expression',expression='smile',source='social'))['accepted'])
        self.assertTrue((await self.tool('continuation','set_expression',expression='smile',source='command'))['accepted'])
        self.assertFalse((await self.tool('continuation','set_expression',expression='smile',source='command'))['accepted'])
        s.expression.publish.assert_called_once()

    async def test_welcome_yields_if_user_starts_during_render_preparation(self):
        s = self.service
        s.enabled = True
        s.context.update(mode='interaction', target='interaction', ready=True)
        generating = asyncio.Event()
        async def delay(_):
            generating.set()
            await asyncio.Future()
        reader = SimpleNamespace(generate=delay)
        with patch('vivis.speech.trusted_reading.RealtimeReader', return_value=reader):
            s.schedule_public_prompt(self.state,welcome=True)
            await asyncio.wait_for(generating.wait(),1)
            s.cancel_public_prompt()
            await s.public_prompt_task
        self.assertFalse(self.events)
        self.assertEqual(s.player.count,0)

    async def test_receive_loop_routes_final_transcript_to_one_audible_answer(self):
        s = self.service
        s.audio = Mock(output_stream=None)
        s.reception.c = dict(s.reception.c,enabled=False)
        s.schedule_farewell = Mock()  # No cloud synthesis in the receive-loop test.
        inbox = asyncio.Queue()
        sent = []
        class Socket:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await inbox.get())
            async def send(self,data):
                event=json.loads(data);sent.append(event)
                item=event.get('item',{})
                if item.get('id','').startswith('controller_'):
                    inbox.put_nowait(dict(type='conversation.item.created',item=item))
                if event['type']=='response.create':
                    for packet in (
                        dict(type='response.created',response=dict(id='answer')),
                        dict(type='response.audio.delta',response_id='answer',delta=base64.b64encode(b'\x01\x00'*240).decode()),
                        dict(type='response.audio_transcript.done',response_id='answer',transcript='我二十岁。'),
                        dict(type='response.done',response=dict(id='answer',status='completed'))):
                        inbox.put_nowait(packet)
        async def until(predicate):
            deadline=time.monotonic()+1.5
            while not predicate():
                if connection.done():connection.result()
                if time.monotonic()>deadline:raise AssertionError('Cloud receive-loop test timed out')
                await asyncio.sleep(.005)
        async def connect(ws,session,log):return session
        with patch('vivis.speech.cloud_managed.websocket_connection',return_value=Socket()), \
             patch('vivis.speech.cloud_managed.connect_session',side_effect=connect):
            connection=asyncio.create_task(s._connection())
            try:
                await until(lambda:s.ws is not None)
                for packet in (
                    dict(type='input_audio_buffer.speech_started',item_id='spoken'),
                    dict(type='input_audio_buffer.speech_stopped',item_id='spoken'),
                    dict(type='response.created',response=dict(id='decide')),
                    dict(type='response.function_call_arguments.done',response_id='decide',call_id='control',
                         name='program_followup',arguments='{"kind":"question"}'),
                    dict(type='response.done',response=dict(id='decide',status='completed',output=[dict(type='function_call')]))):
                    inbox.put_nowait(packet)
                await until(lambda:s.program_followup is not None)
                self.assertFalse(any(e['kind']=='cloud_action' for e in self.events))
                inbox.put_nowait(dict(type='conversation.item.input_audio_transcription.completed',
                                      item_id='spoken',transcript='你多大了？'))
                await until(lambda:any(e['kind']=='cloud_action' for e in self.events))
                self.ready_interaction()
                await until(lambda:s.player.count>0)
                self.assertEqual(s.cloud_state.responses['answer']['user_turn']['item_id'],'spoken')
                self.assertEqual(s.message_receipts.responses['answer']['kind'],'ordinary')
                self.assertEqual(len([e for e in sent if e['type']=='response.create']),1)
                tool_updates = [e['session']['tools'] for e in sent if e['type']=='session.update' and 'tools' in e['session']]
                self.assertIn('set_expression',{tool['name'] for tool in tool_updates[-1]})
                self.assertTrue(any(e.get('text')=='我二十岁。' for e in self.events))
            finally:
                connection.cancel()
                await asyncio.gather(connection,return_exceptions=True)


class ReceptionConversationTests(unittest.TestCase):
    def test_timeout_limits_restarts_after_activity_and_respects_thinking_and_decline(self):
        r = ReceptionConversation(dict(enabled=True,first_prompt_s=6,min_interval_s=60,max_per_visit=3))
        self.assertFalse(r.due(0,available=True))
        self.assertFalse(r.due(5.99,available=True))
        self.assertFalse(r.due(6,available=False))
        self.assertFalse(r.due(7,available=True))
        self.assertTrue(r.due(13,available=True))
        r.played(13)
        self.assertFalse(r.due(72.9,available=True))
        r.defer('thinking')
        self.assertFalse(r.due(74,available=True))
        r.user_started();r.addressed_response()
        self.assertFalse(r.due(75,available=True))
        self.assertTrue(r.due(81,available=True));r.played(81)
        self.assertFalse(r.due(90,available=True))
        self.assertTrue(r.due(141,available=True));r.played(141)
        self.assertFalse(r.due(800,available=True))
        r.reset();r.defer('declined');r.addressed_response()
        self.assertFalse(r.due(900,available=True));self.assertFalse(r.due(1000,available=True))

    def test_public_welcome_only_uses_verified_address_and_public_subject_category(self):
        turns=[dict(role='assistant',text='剧本'),dict(role='user',text='歌曲以及我的手机号123456789')]
        self.assertEqual(public_invitation('王工',turns,welcome=True),'王工，您好！想看看有哪些歌曲吗？')
        self.assertEqual(public_invitation(None,turns,welcome=True),'您好！您想聊聊天，还是看看我会做什么？')
