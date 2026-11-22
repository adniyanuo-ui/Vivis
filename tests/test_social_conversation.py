"""Stable speech hints and optional public topics; virtual cloud/DAC, no devices."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,Mock,patch

import test_cloud_control as fixtures
from test_social_expression import face
from vivis.paths import ROOT
from vivis.interaction.social_conversation import (SocialConversation,configuration,public_context,validate_topic)
from vivis.interaction.social_expression import observation
from vivis.speech.realtime import VoiceSession


def packet(style,stamp,owner='camera'):
    return dict(schema_version='vivis-face-presence-1.0',faces=1,observed_monotonic=stamp,owner=owner,
        expression=dict(observation_only=True,bilateral_smile=.8 if style=='joy' else 0.,patterns=observation(face(style,stamp))))


class SocialContextTests(unittest.TestCase):
    def test_stable_patterns_not_old_frames_or_new_people_inform_speech(self):
        context=SocialConversation(configuration(),1.)
        first=packet('joy',0.)
        for i in range(100):cue=context.observe(first,i/60,'one',active=True)
        self.assertEqual(cue['response_style'],'neutral')
        for i in range(30):cue=context.observe(packet('joy',2+i/5),2+i/5,'one',active=True)
        self.assertEqual(cue['response_style'],'joy');self.assertTrue(cue['observation_only'])
        self.assertNotIn('age',cue);self.assertNotIn('gender',cue)
        cue=context.observe(packet('joy',8.),8.,'two',active=True)
        self.assertEqual(cue['response_style'],'neutral')
        for i in range(30):context.observe(packet('concern',9+i/5),9+i/5,'two',active=True)
        self.assertEqual(context.cue['response_style'],'concern')
        many=packet('concern',16.);many['faces']=2
        self.assertFalse(context.observe(many,16.,'two',active=True)['available'])
        self.assertFalse(context.observe(packet('joy',16.),18.,'two',active=True)['available'])
        self.assertFalse(context.observe(packet('joy',18.),18.,'two',active=False)['available'])

    def test_tone_switch_does_not_edit_accepted_motor_or_voice_parameters(self):
        c=configuration();context=SocialConversation(c,1.)
        self.assertIn('用户本句内容',context.instructions());self.assertIn('正常语速',context.instructions())
        context.c=dict(c,tone_enabled=False)
        self.assertEqual(context.instructions(),'')
        for switch in ('tone_enabled','topics_enabled','topic_expression_enabled'):
            self.assertFalse(configuration(None)[switch])

    def test_public_history_is_projected_and_private_fields_never_pass_through(self):
        knowledge=SimpleNamespace(persona='阳光、幽默的前台接待',business='仿生交互产品',people={
            'engineer':dict(role='算法工程师',professional_summary='研究机器人',phone='13812345678')})
        history=[dict(role='user',text='给小王留言，取件码7568，买摄像头，之后听歌曲'),
            dict(role='assistant',text='虚构的偏好：运动')]
        current=[dict(role='user',text='今天工作好累，想聊点轻松的。'),dict(role='user',text='手机号13812345678'),
            dict(role='user',text='快递取件码7568')]
        result=public_context(knowledge,'engineer',history,current,['music'])
        raw=json.dumps(result,ensure_ascii=False)
        self.assertNotIn('7568',raw);self.assertNotIn('13812345678',raw);self.assertNotIn('小王',raw)
        self.assertIn('music',result['previous_public_topic_hints']);self.assertIn('technology',result['previous_public_topic_hints'])
        self.assertNotIn('sport',result['previous_public_topic_hints']);self.assertNotIn('music',result['allowed_topics'])
        self.assertIn('工作好累',raw)
        unknown=public_context(knowledge,None)
        self.assertFalse(unknown['person_known']);self.assertEqual(unknown['previous_public_topic_hints'],[])

    def test_topic_validation_rejects_private_repeated_or_action_outputs(self):
        context=dict(allowed_topics={'relax':'轻松闲聊'})
        good=dict(topic='relax',style='concern',text='要不要先聊点轻松的？')
        self.assertEqual(validate_topic(good,context,80),good)
        for bad in ({**good,'topic':'music'},{**good,'text':'您的取件码7568对吗？'},
                {**good,'text':'今天累吗？需要帮忙吗？'},{**good,'action':'play'},{**good,'style':'angry'}):
            with self.assertRaises(ValueError):validate_topic(bad,context,80)
        skip=dict(topic='skip',style='neutral',text='')
        self.assertEqual(validate_topic(skip,context,80),skip)


class SocialManagedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.events=[];self.s=fixtures.CloudBackendTests().service(folder.name,self.events)
        self.addCleanup(self.s.log.close)
        self.s.social=SocialConversation(configuration(),1.)
        self.s.presence.single_visible=Mock(return_value=True);self.s.presence.allowed=Mock(return_value=True)
        self.s.recognized_person=Mock(return_value=None);self.s.identified_person=Mock(return_value=None)
        self.s.context.update(mode='interaction',target='interaction',ready=True,revision=2);self.s.enabled=True
        self.state=self.s.cloud_state=VoiceSession(self.s.player,self.s.log)
        self.audio=SimpleNamespace(pcm24_bytes=b'\x01\x00'*2400,samples=[],sample_rate=24000)
        self.proposal=dict(topic='music',style='joy',text='您平时更爱听轻快的歌，还是舒缓一点的？')

    async def until(self,predicate):
        deadline=time.monotonic()+2
        while not predicate():
            if time.monotonic()>deadline:raise AssertionError('Public speech timed out')
            await asyncio.sleep(.005)

    async def drain(self):
        await self.until(lambda:self.s.player.count>0)
        self.s.player.render(240,time.monotonic()+.01)
        await self.until(lambda:self.s.public_prompt_started)
        self.s.player.render(self.s.player.count,time.monotonic()+.01)
        await asyncio.wait_for(self.s.public_prompt_task,2)

    async def test_stable_cue_reaches_current_qwen_context_and_is_absent_in_tasks(self):
        s=self.s;stamp=time.monotonic()
        path=Path(s.folder)/'presence.json';s.presence.path=path
        for i in range(10):
            now=stamp+i/5
            path.write_text(json.dumps(packet('listening',now)))
            with patch('vivis.speech.cloud_managed.time.monotonic',return_value=now):text=s.context_snapshot()[0]
        self.assertIn('眉部收拢',text);self.assertIn('"response_style": "listening"',text)
        self.assertIn('当前辅助语气建议',text)
        s.social.c=dict(s.social.c,tone_enabled=False)
        text=s.context_snapshot()[0]
        self.assertNotIn('visual_social_cue',text);self.assertNotIn('当前辅助语气建议',text)
        s.context.update(mode='task',target=next(iter(s.program_ids)))
        text=s.context_snapshot()[0]
        self.assertNotIn('visual_social_cue',text);self.assertNotIn('当前辅助语气建议',text)

    async def test_public_topic_uses_same_voice_actual_question_and_social_pose_once_audible(self):
        s=self.s;s.fixed_audio=Mock(return_value=self.audio)
        s.social_turns.append(dict(role='user',text='这是上一位访客的话。',session='old-visitor'))
        s.social_turns.append(dict(role='user',text='今天有点累，想随便聊聊。',session=s.memory_session))
        reader=SimpleNamespace(generate=AsyncMock(return_value=self.audio))
        with patch('vivis.speech.social_topics.draft',new=AsyncMock(return_value=self.proposal)) as draft, \
                patch('vivis.speech.trusted_reading.RealtimeReader',return_value=reader) as factory:
            s.schedule_public_prompt(self.state,welcome=False)
            self.assertEqual(s.reception.conversation.count,0);self.assertIsNone(s.expression.active)
            await self.drain()
        self.assertEqual(factory.call_args.args[2]['voice'],s.active_voice)
        self.assertEqual(factory.call_args.kwargs['style'],'joy');reader.generate.assert_awaited_once_with(self.proposal['text'])
        drafted_context=draft.await_args.args[3]
        self.assertNotIn('上一位访客',json.dumps(drafted_context,ensure_ascii=False))
        self.assertIn('今天有点累',json.dumps(drafted_context,ensure_ascii=False))
        self.assertEqual(s.public_prompt_context['text'],self.proposal['text'])
        self.assertEqual(s.public_prompt_context['playback_state'],'completed')
        self.assertEqual(s.reception.conversation.topics,['music']);self.assertEqual(s.reception.conversation.count,1)
        self.assertEqual(s.expression.active['expression'],'auto')
        self.assertIn(self.proposal['text'],s.context_snapshot()[0])
        self.assertFalse(any(e.get('kind')=='cloud_action' for e in self.events))

    async def test_explicit_expression_is_not_overwritten_by_public_social_pose(self):
        s=self.s;command=s.expression.publish(dict(expression='neutral',source='command'),2)
        reader=SimpleNamespace(generate=AsyncMock(return_value=self.audio))
        with patch('vivis.speech.social_topics.draft',new=AsyncMock(return_value=self.proposal)), \
                patch('vivis.speech.trusted_reading.RealtimeReader',return_value=reader):
            s.schedule_public_prompt(self.state,welcome=False);await self.drain()
        self.assertEqual(s.expression.active['id'],command['intent_id'])

    async def test_user_start_cancels_slow_topic_without_audio_or_replay(self):
        started=asyncio.Event();cancelled=asyncio.Event()
        async def slow(*_):
            started.set()
            try:await asyncio.Future()
            finally:cancelled.set()
        with patch('vivis.speech.social_topics.draft',side_effect=slow):
            self.s.schedule_public_prompt(self.state,welcome=False)
            await asyncio.wait_for(started.wait(),1)
            self.s.input_generation+=1;self.s.in_speech=True;self.s.cancel_public_prompt()
            await self.s.public_prompt_task
        self.assertTrue(cancelled.is_set());self.assertEqual(self.s.player.count,0)
        self.assertEqual(self.s.reception.conversation.count,0);self.assertFalse(self.events)

    async def test_observation_or_state_change_during_preparation_cannot_play_stale_topic(self):
        for change in ('style','revision'):
            entered=asyncio.Event();release=asyncio.Event()
            async def slow(*_):entered.set();await release.wait();return self.proposal
            with patch('vivis.speech.social_topics.draft',side_effect=slow):
                self.s.schedule_public_prompt(self.state,welcome=False)
                await entered.wait()
                if change=='style':self.s.social.cue['response_style']='concern'
                else:self.s.context['revision']+=1
                release.set();await self.s.public_prompt_task
            self.assertEqual(self.s.player.count,0);self.assertFalse(self.s.public_prompt_started)

    async def test_skip_remains_quiet_and_network_failure_falls_back_locally(self):
        s=self.s
        with patch('vivis.speech.social_topics.draft',new=AsyncMock(return_value=dict(topic='skip',text='',style='neutral'))):
            s.schedule_public_prompt(self.state,welcome=False);await s.public_prompt_task
        self.assertTrue(s.reception.conversation.paused);self.assertEqual(s.player.count,0)
        self.assertEqual(s.thinking_generation,s.input_generation)
        s.reception.conversation.addressed_response();s.fixed_audio=Mock(return_value=self.audio)
        with patch('vivis.speech.social_topics.draft',new=AsyncMock(side_effect=TimeoutError)):
            s.schedule_public_prompt(self.state,welcome=False);await self.drain()
        self.assertEqual(s.reception.conversation.count,1);self.assertFalse(any(e.get('kind')=='cloud_error' for e in self.events))

    async def test_cached_welcome_is_not_delayed_by_new_topic_generation(self):
        self.s.fixed_audio=Mock(return_value=self.audio)
        with patch('vivis.speech.social_topics.draft',new=AsyncMock()) as draft:
            self.s.schedule_public_prompt(self.state,welcome=True);await self.drain()
        draft.assert_not_called();self.assertEqual(self.s.reception.conversation.count,0)

    async def test_optional_pose_file_failure_does_not_cancel_public_audio(self):
        reader=SimpleNamespace(generate=AsyncMock(return_value=self.audio))
        with patch('vivis.speech.social_topics.draft',new=AsyncMock(return_value=self.proposal)), \
                patch('vivis.speech.trusted_reading.RealtimeReader',return_value=reader), \
                patch.object(self.s.expression,'publish',side_effect=OSError('temporary state file failure')):
            self.s.schedule_public_prompt(self.state,welcome=False);await self.drain()
        self.assertEqual(self.s.public_prompt_context['playback_state'],'completed')

    async def test_disabling_topics_keeps_existing_cached_invitation(self):
        self.s.social.c=dict(self.s.social.c,topics_enabled=False)
        self.s.fixed_audio=Mock(return_value=self.audio)
        with patch('vivis.speech.social_topics.draft',new=AsyncMock()) as draft:
            self.s.schedule_public_prompt(self.state,welcome=False);await self.drain()
        draft.assert_not_called();self.assertEqual(self.s.reception.conversation.count,1)

    async def test_receive_loop_admits_only_current_final_turn_and_clears_it_on_task(self):
        s=self.s;s.audio=Mock(output_stream=None);s.reception.c=dict(s.reception.c,enabled=False)
        inbox=asyncio.Queue()
        class Socket:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await inbox.get())
            async def send(self,_):pass
        async def connect(ws,session,log):return session
        with patch('vivis.speech.cloud_managed.websocket_connection',return_value=Socket()), \
                patch('vivis.speech.cloud_managed.connect_session',side_effect=connect):
            job=asyncio.create_task(s._connection())
            try:
                await self.until(lambda:s.ws is not None)
                for event in (
                    dict(type='input_audio_buffer.speech_started',item_id='current'),
                    dict(type='input_audio_buffer.speech_stopped',item_id='current'),
                    dict(type='conversation.item.input_audio_transcription.completed',item_id='current',transcript='今天工作有点累。'),
                    dict(type='conversation.item.input_audio_transcription.completed',item_id='old',transcript='过时的访客话语。')):
                    inbox.put_nowait(event)
                await self.until(lambda:len(s.social_turns)==1 and inbox.empty())
                self.assertEqual(s.social_turns[0]['text'],'今天工作有点累。')
                self.assertEqual(s.social_turns[0]['session'],s.memory_session)
                s.request(dict(kind='voice_context',revision=3,mode='task',target=next(iter(s.program_ids)),ready=True))
                self.assertFalse(s.social_turns)
            finally:
                job.cancel();await asyncio.gather(job,return_exceptions=True)


if __name__=='__main__':unittest.main()
