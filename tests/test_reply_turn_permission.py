"""A permitted utterance may finish across presence expiry; no new wake rights."""
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vivis.interaction.cloud_control import SpeechPresence
from vivis.speech.realtime import VoiceSession


class ReplyTurnPermissionTests(unittest.TestCase):
    def setUp(self):
        from test_cloud_control import CloudBackendTests
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.s=CloudBackendTests().service(self.tmp.name,[])
        self.addCleanup(self.s.log.close);self.addCleanup(self.s.recorders.close)
        self.addCleanup(self.s.conversation_animation.close)
        self.s.context.update(revision=1,mode='interaction',target='interaction',ready=True)
        self.s.enabled=True;self.s.identified_person=lambda:None
        self.s.identity_fusion=None
        self.s.presence.visible=Mock(return_value=True)
        self.s.input_generation=1
        self.state=VoiceSession(self.s.player,self.s.log);self.s.cloud_state=self.state

    def begin_and_expire(self):
        self.s.begin_user_turn('question');self.s.in_speech=True
        self.s.presence.visible.return_value=False
        self.s.presence.last_face=time.monotonic()-self.s.presence.c['face_loss_grace_s']-1
        self.assertFalse(self.s.can_speak())

    def test_onset_permission_survives_expiry_before_speech_end_and_response(self):
        self.begin_and_expire()
        self.assertTrue(self.s.turn_reply_pending())
        self.assertTrue(self.s.can_answer_turn('question'))
        self.s.in_speech=False;self.s.turn_ended_at=time.monotonic()
        self.assertTrue(self.s.turn_reply_pending())
        self.assertTrue(self.s.can_answer_turn('question'))
        self.assertEqual(self.s.presence.wake_until,0.)
        self.assertFalse(self.s.can_speak())  # No permission for unsolicited/new speech.

    def test_no_permission_at_onset_cannot_be_manufactured_by_engaged_history(self):
        self.s.presence.visible.return_value=False
        self.s.begin_user_turn('question')
        self.s.memory_turns['question']['engaged']=True
        self.assertFalse(self.s.can_answer_turn('question'))
        self.assertFalse(self.s.turn_reply_pending())

    def test_new_turn_invalid_turn_connection_revision_and_session_revoke_old_grant(self):
        self.begin_and_expire()
        for field,value in (('message_connection',object()),('memory_session','new-visit'),
                            ('input_generation',2),('turn_item_id','another')):
            previous=getattr(self.s,field);setattr(self.s,field,value)
            self.assertFalse(self.s.can_answer_turn('question'),field)
            setattr(self.s,field,previous)
        self.s.context['revision']=2
        self.assertFalse(self.s.can_answer_turn('question'))
        self.s.context['revision']=1;self.s.memory_turns['question']['valid']=False
        self.assertFalse(self.s.can_answer_turn('question'))
        self.assertFalse(self.s.can_answer_turn(None))
        self.assertFalse(self.s.turn_reply_authorized({}))

    def test_mode_stop_output_handoff_and_recovery_still_block(self):
        self.begin_and_expire()
        self.s.context['mode']='standby';self.assertFalse(self.s.can_answer_turn('question'))
        self.s.context['mode']='interaction'
        for field,value in (('enabled',False),('control_pending',True),('awaiting_recovery_user',True),
                            ('explicit_wake_pending',{'pending':True}),('relay_identity_hold',True)):
            previous=getattr(self.s,field);setattr(self.s,field,value)
            self.assertFalse(self.s.can_answer_turn('question'),field)
            setattr(self.s,field,previous)

    def test_completed_reply_does_not_keep_conversation_audio_mode_open(self):
        self.begin_and_expire();self.s.in_speech=False;self.s.turn_ended_at=time.monotonic()
        self.state.responses['answer']=dict(status='completed',user_turn=dict(item_id='question'))
        self.assertFalse(self.s.turn_reply_pending())
        self.assertFalse(self.s.can_answer_turn('question'))
        self.s.finish_responses.add('answer')
        self.assertTrue(self.s.turn_reply_pending())
        self.assertFalse(self.s.can_answer_turn('question'))  # A tail is not a new generation grant.
        self.s.finish_responses.clear()
        self.s.input_generation+=1;self.s.begin_user_turn('next')
        self.assertFalse(self.s.can_answer_turn('next'))

    def test_cancelled_response_and_private_relay_cannot_be_resurrected(self):
        self.begin_and_expire()
        self.s.speech_responses.append('cancelled');self.state.generating='cancelled'
        self.state.responses['cancelled']=dict(user_turn=dict(item_id='question'))
        self.state.blocked.add('cancelled')
        self.assertFalse(self.s.can_play_response('cancelled'))
        self.s.speech_responses.append('relay');self.state.generating='relay'
        self.s.relay_recipients['relay']='another-person'
        self.assertFalse(self.s.can_play_response('relay'))


class ReplyTurnConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_receive_loop_keeps_audio_then_silences_next_unwoken_turn(self):
        from test_cloud_control import CloudBackendTests, config
        from test_managed_voice import FakeAudio
        events=[];messages=asyncio.Queue();sent=[]
        class WS:
            async def __aenter__(self):return self
            async def __aexit__(self,*_):return False
            async def send(self,text):
                value=json.loads(text);sent.append(value);item=value.get('item',{})
                if value.get('type')=='conversation.item.create' and item.get('id','').startswith('controller_'):
                    await messages.put(dict(type='conversation.item.created',item=item))
            def __aiter__(self):return self
            async def __anext__(self):return json.dumps(await messages.get())
        async def connected(ws,session,log):return session
        with tempfile.TemporaryDirectory() as tmp, \
                patch('vivis.speech.managed.AudioIO',FakeAudio), \
                patch('vivis.speech.cloud_managed.websocket_connection',return_value=WS()), \
                patch('vivis.speech.cloud_managed.connect_session',side_effect=connected):
            s=CloudBackendTests().service(tmp,events)
            self.addCleanup(s.log.close);self.addCleanup(s.recorders.close)
            self.addCleanup(s.conversation_animation.close)
            s.presence=SpeechPresence(config('configs/perception/speech_presence.json'),tmp)
            s.presence.visible=Mock(return_value=True)
            s.identity_fusion=None;s.identified_person=lambda:None
            s.recognized_person=lambda:None;s.reviewing_identity=lambda:None
            s.reception.c=dict(s.reception.c,enabled=False)
            task=asyncio.create_task(s.run())
            async def capture():
                while not s.shutdown.is_set():
                    stamp=time.monotonic();s.audio.last_capture=stamp
                    if s.audio.output_stream is not None:s.audio.last_output=stamp
                    s.audio.uplink.put((stamp,b'\0'*640,b'\0'*640,b'\0'*640))
                    await asyncio.sleep(.02)
            producer=asyncio.create_task(capture())
            async def until(predicate):
                deadline=time.monotonic()+3
                while not predicate():
                    if task.done():task.result()
                    if time.monotonic()>deadline:raise AssertionError('Receive-loop test timed out')
                    await asyncio.sleep(.01)
            def last_modalities():
                return next((m['session']['modalities'] for m in reversed(sent)
                    if m.get('type')=='session.update' and 'modalities' in m['session']),None)
            try:
                await until(lambda:s.ws is not None)
                s.controls.put(dict(kind='voice_context',revision=1,mode='interaction',target='interaction',ready=True))
                s.controls.put(dict(kind='voice_mode',epoch=1,enabled=True))
                await until(lambda:s.audio.output_stream is not None and s.can_speak())
                await messages.put(dict(type='input_audio_buffer.speech_started',item_id='current'))
                await until(lambda:s.turn_item_id=='current')
                s.presence.visible.return_value=False
                s.presence.last_face=time.monotonic()-s.presence.c['face_loss_grace_s']-1
                await until(lambda:any(m.get('type')=='session.update'
                    and all(flag in m['session'].get('instructions','') for flag in (
                        '"current_turn_reply_permission": true','"wake_permission": false','"face_visible": false'))
                    for m in sent))
                self.assertEqual(last_modalities(),['text','audio'])
                self.assertFalse(s.can_speak());self.assertTrue(s.turn_reply_pending())
                await messages.put(dict(type='input_audio_buffer.speech_stopped',item_id='current'))
                await messages.put(dict(type='conversation.item.input_audio_transcription.completed',item_id='current',transcript='为什么候鸟会迁徙？'))
                for index,(name,args,accepted) in enumerate((('get_robot_status',{},True),
                        ('set_expression',dict(expression='concern',source='social'),True),
                        ('search_person_memory',dict(keywords='偏好'),False))):
                    rid='tool-'+str(index);call='call-'+str(index)
                    before=sum(m.get('type')=='response.create' for m in sent)
                    await messages.put(dict(type='response.created',response=dict(id=rid)))
                    await messages.put(dict(type='response.function_call_arguments.done',response_id=rid,
                        call_id=call,name=name,arguments=json.dumps(args)))
                    await until(lambda:any(m.get('item',{}).get('call_id')==call for m in sent))
                    result=next(json.loads(m['item']['output']) for m in sent if m.get('item',{}).get('call_id')==call)
                    self.assertEqual(result['accepted'],accepted,result)
                    if not accepted:self.assertEqual(result['status'],'identity_confirmation_required')
                    await messages.put(dict(type='response.done',response=dict(id=rid,status='completed')))
                    await until(lambda:sum(m.get('type')=='response.create' for m in sent)>before)
                await messages.put(dict(type='response.created',response=dict(id='answer')))
                pcm=base64.b64encode((np.ones(960,dtype='<i2')*1000).tobytes()).decode()
                await messages.put(dict(type='response.audio.delta',response_id='answer',delta=pcm))
                await messages.put(dict(type='response.audio.done',response_id='answer'))
                await messages.put(dict(type='response.audio_transcript.done',response_id='answer',transcript='候鸟迁徙是为了寻找适合生存和繁殖的环境。'))
                await messages.put(dict(type='response.done',response=dict(id='answer',status='completed')))
                await until(lambda:s.player.count>0)
                self.assertNotIn('answer',s.blocked)
                await until(lambda:any(e.get('role')=='assistant' for e in events))
                self.assertTrue(np.any(s.player.render(s.player.count,time.monotonic())))
                s.cloud_state.poll_playback()
                await until(lambda:not s.finish_responses and not s.turn_reply_pending())
                await until(lambda:last_modalities()==['text'])
                await messages.put(dict(type='input_audio_buffer.speech_started',item_id='unwoken'))
                await messages.put(dict(type='input_audio_buffer.speech_stopped',item_id='unwoken'))
                await messages.put(dict(type='response.created',response=dict(id='blocked')))
                await messages.put(dict(type='response.audio.delta',response_id='blocked',delta=pcm))
                await until(lambda:'blocked' in s.blocked)
                self.assertEqual(s.player.count,0)
            finally:
                s.shutdown.set();producer.cancel();await asyncio.gather(producer,return_exceptions=True)
                await asyncio.wait_for(task,4)


if __name__=='__main__':unittest.main()
