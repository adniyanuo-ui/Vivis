"""Optional brief side acknowledgement, isolated from the main private turn."""
import asyncio
from collections import deque
import json
import time

from vivis.interaction.sound_attention import Mailbox,SideSpeechGate,configuration,recent
from vivis.speech.runtime_policy import bounded_send


class SoundAttentionVoice:
    def init_sound_attention(self):
        self.sound_attention=None;self.sound_gate=None;self.sound_mailbox=None;self.direction_observer=None
        self.side_items=deque(maxlen=128);self.side_responses=deque(maxlen=128)
        self.side_commands=deque(maxlen=128)
        self.side_onsets={}
        self.side_response_sources={}
        self.side_wire_item=None;self.side_task=None;self.side_stable=None
        c=configuration()
        definition=json.loads(self.robot_path.read_text())
        if not c['enabled'] or definition.get('name')!='xiaoshuai':return
        try:
            import webrtcvad
            vad=webrtcvad.Vad(2)
        except ImportError:
            self.log.write('sound_attention_disabled',reason='webrtcvad_not_installed');return
        self.sound_gate=SideSpeechGate(c,lambda pcm:vad.is_speech(pcm,16000))
        self.side_audio=self.fixed_audio('稍等。')  # Validate/load before audio scheduling.
        self.sound_config=c
        hardware=json.loads((self.robot_path.parent/definition['hardware']).read_text())
        from vivis.speech.sound_direction import DirectionObserver,configuration as direction_configuration
        self.direction_observer=DirectionObserver(direction_configuration(),hardware['runtime']['port'])
        self.sound_mailbox=Mailbox(self.folder/'sound_attention_feedback.json',self.folder/'sound_attention.json')
        # Lazy start after microphone input, not during construction/offline check.
        self.sound_started=False;self.side_suppressed=False;self.side_cancelled=False

    def sound_frozen(self):return getattr(self,'sound_attention',None) is not None

    def acoustic_microphone(self,stamp,pcm,metrics):
        """Runs on the existing async drain, never on the PortAudio callback."""
        gate=getattr(self,'sound_gate',None)
        if gate is None:return False
        if not self.sound_started:
            self.sound_started=True;self.sound_mailbox.start();self.direction_observer.start()
            self.log.write('sound_attention_started',budget_s=self.sound_config['budget_s'],
                mount_calibrated=self.sound_config['mount']['calibrated'],checksum_verified=False)
        state=self.cloud_state;now=time.monotonic()
        if (self.context['mode']!='interaction' or not self.context['ready']
                or not self.health_available or not self.enabled):
            gate.reset(keep_history=True,now=now);self.side_suppressed=False
            return False  # Standby wake and task stop retain their original input path.
        available=(state is not None and self.context['mode']=='interaction' and self.context['ready']
            and self.health_available and self.enabled and not self.in_speech and not self.control_pending
            and self.presence.single_visible()
            and not self.sound_frozen() and not self.turn_reply_pending() and self.farewell is None
            and not self.expression_pending and not (self.message_lane and self.message_lane.task)
            and not self.relay_pending)
        # Do not pause confidential/unfinished business output to answer a stranger.
        # Ordinary already-complete conversational speech may be paused briefly.
        if state and self.player.count and not self.in_speech:
            available=available and state.generating is None
        suppressed,request=gate.observe(stamp,pcm,metrics,self.direction_observer.latest,
            self.sound_mailbox.latest,available=available,now=now)
        self.side_suppressed=suppressed
        if request:
            audio=self.side_audio
            if audio is None:
                self.log.write('sound_attention_skipped',reason='verified_short_audio_missing')
                return suppressed
            self.side_stable=self.stable_identity()
            request.update(person=self.side_stable['person_id'],session=self.memory_session,
                revision=self.context['revision'],connection=self.message_connection,
                stable=self.side_stable,audio=audio)
            self.sound_attention=request
            if self.continuity:self.continuity.submit('invalidate_identity_audio')
            self.log.write('sound_attention_begin',attention_id=request['id'],sector=request['sector'],
                bearing_deg=request['bearing_deg'],main_person=request['person'],
                business_identity_retained=True,side_business_authorized=False)
            self.emit(dict(kind='voice_warning',epoch=self.epoch,
                message='声源插话：短暂转向回应，保留主聊天身份；不办理侧方私人业务。'))
            self.side_task=asyncio.create_task(self.side_acknowledgement(request,state))
        return suppressed

    async def side_acknowledgement(self,request,state):
        """Priority cached audio through the SAME DAC/AEC/mouth; no cloud wait."""
        rid='side_'+request['id'];paused=self.player.pause(state.generating)['responses']
        self.player.begin_priority(rid)
        audio=request['audio'];pcm=audio.pcm24_bytes
        state.responses[rid]=dict(source='side_acknowledgement',user_turn=None,archive_no_user=True,
            status='completed',received_audio_s=len(pcm)/48000)
        try:
            if self.player.mouth:self.player.mouth.install(rid,audio.mouth_order,audio.mouth_frames)
            for offset in range(0,len(pcm),4800):
                if self.sound_attention is not request:return
                self.player.add(rid,pcm[offset:offset+4800]);await asyncio.sleep(0)
            self.player.finish(rid)
            announced=False
            while self.sound_attention is request:
                if not announced and rid in self.player.started_responses:
                    announced=True
                    self.emit(dict(kind='cloud_transcript',epoch=self.epoch,role='assistant',
                        text='稍等。',source='side_acknowledgement'))
                await asyncio.sleep(.02)
        except asyncio.CancelledError:raise
        except Exception as exc:
            self.log.write('sound_attention_audio_failed',error_type=type(exc).__name__)
        finally:
            self.player.cancel(rid);self.player.end_priority(rid)
            # Never restore server-cancelled speech, changed visits or new commands.
            same=(request['connection'] is self.message_connection and request['session']==self.memory_session
                and request['revision']==self.context['revision'] and self.health_available and not self.shutdown.is_set())
            if same:self.player.resume(set(paused)-state.blocked-state.server_cancelled)

    def acoustic_tick(self):
        request=getattr(self,'sound_attention',None)
        if request is None:return
        now=time.monotonic();feedback=self.sound_mailbox.latest
        same=(request['connection'] is self.message_connection and request['session']==self.memory_session
            and request['revision']==self.context['revision'] and self.context['mode']=='interaction'
            and self.context['ready'] and self.health_available)
        if not same or now-request['created_monotonic']>12:
            self.cancel_sound_attention('state_changed_or_deadline');return
        fresh_feedback=recent(feedback,now,self.sound_config['lease_s']) and feedback.get('owner')==request['owner']
        if fresh_feedback and feedback.get('attention_id')==request['id']:request['motion_seen']=True
        if not request.get('motion_seen') and now-request['created_monotonic']>min(1.,self.sound_config['budget_s']/2):
            self.cancel_sound_attention('motion_not_admitted');return
        main_present=((self.knowledge.current_identity().get('state')=='recognized'
            and self.recognized_person()==request['person']) if request['person'] else self.presence.single_visible())
        wire={key:request[key] for key in ('id','owner','sector','bearing_deg','heard_monotonic',
            'created_monotonic','action','mount_calibrated')}
        wire.update(updated_monotonic=now,main_present=main_present)
        self.sound_mailbox.submit(wire)
        if (recent(feedback,now,self.sound_config['lease_s']) and feedback.get('owner')==request['owner']
                and feedback.get('completed',{} ) and feedback['completed'].get('id')==request['id']):
            if not request.get('return_logged'):
                request['return_logged']=True
                self.log.write('sound_attention_returned',**feedback['completed'],
                    target_budget_s=self.sound_config['budget_s'],elapsed_is_target_timeline=True)
            if feedback.get('phase')=='idle':
                self.cancel_sound_attention(feedback.get('result') or 'returned')

    def cancel_sound_attention(self,reason):
        request=getattr(self,'sound_attention',None)
        if request is None:return
        self.sound_attention=None
        self.sound_mailbox.submit(dict(id=request['id'],owner=request['owner'],action='cancel',updated_monotonic=time.monotonic()))
        if self.side_task and not self.side_task.done():self.side_task.cancel()
        self.log.write('sound_attention_finished',attention_id=request['id'],reason=reason,
            elapsed_s=time.monotonic()-request['heard_monotonic'])
        if reason in ('found','empty'):
            # The original person left. Only fresh normal recognition may bind
            # the newly found visitor; the side utterance itself grants nothing.
            if self.identity_fusion:self.identity_fusion.reset()
            from vivis.interaction.identity_fusion import unavailable
            self.voice_business=unavailable('main_left_during_acoustic_attention')
            self.voice_pending=None;self.voice_display=None
            if self.continuity:self.continuity.submit('invalidate_identity_audio')
            if self.sound_gate:self.sound_gate.reset()

    async def filter_side_event(self,event,timeline,ws):
        gate=getattr(self,'sound_gate',None)
        if gate is None:return False
        kind=event.get('type','');item=event.get('item_id')
        rid=event.get('response_id') or event.get('response',{}).get('id')
        if kind=='error' and event.get('error',{}).get('event_id') in self.side_commands:
            self.log.write('side_context_cleanup_failed',reason=event['error'].get('code'));return True
        if kind=='input_audio_buffer.speech_started':
            stamp=timeline.at(event.get('audio_start_ms'))
            if isinstance(item,str):
                self.side_onsets[item]=stamp
                if len(self.side_onsets)>128:self.side_onsets.pop(next(iter(self.side_onsets)))
            if gate.contains(stamp) or (stamp is None and getattr(self,'side_suppressed',False)):
                if not isinstance(item,str) or not item:return True
                if item not in self.side_items:self.side_items.append(item)
                self.side_wire_item=item
            else:self.side_wire_item=None
        elif isinstance(item,str) and item in self.side_onsets and gate.contains(self.side_onsets[item]):
            if item not in self.side_items:
                self.side_items.append(item)
                if item==self.turn_item_id:
                    turn=self.memory_turns.get(item)
                    if turn:turn['valid']=False
                    if self.cloud_state:
                        self.cloud_state.event(dict(type='input_audio_buffer.speech_stopped',item_id=item,reason='turn_invalid'))
                    if self.continuity:
                        self.continuity.end(item,timeline.at(event.get('audio_end_ms')),False,{})
                        self.continuity.final(item,'',False)
                    self.in_speech=False;self.turn_engaged=False;self.turn_item_id=None;self.turn_ended_at=None
                    self.side_wire_item=item
        if item in self.side_items:
            if kind=='conversation.item.input_audio_transcription.completed':
                self.log.write('side_speech_isolated',item_id=item,private_actor=None,
                    text=event.get('transcript','')[:500],business_executed=False)
                command='side_delete_'+item;self.side_commands.append(command)
                await bounded_send(ws,json.dumps(dict(type='conversation.item.delete',event_id=command,item_id=item)))
            return True
        if kind=='response.created' and self.side_wire_item:
            self.side_responses.append(rid)
            self.side_response_sources[rid]=self.side_wire_item
            if len(self.side_response_sources)>128:self.side_response_sources.pop(next(iter(self.side_response_sources)))
        if rid in self.side_responses:
            if kind=='response.done' and self.side_response_sources.get(rid)==self.side_wire_item:
                self.side_wire_item=None
            if kind=='response.function_call_arguments.done' and event.get('call_id'):
                await bounded_send(ws,json.dumps(dict(type='conversation.item.create',item=dict(type='function_call_output',
                    call_id=event['call_id'],output='{"accepted":false,"reason":"side_voice_has_no_main_actor_authority"}'))))
            return True
        return False

    def close_sound_attention(self):
        # Joins run off the event loop; cancellation remains on that loop.
        if self.sound_mailbox:self.sound_mailbox.close()
        if self.direction_observer:self.direction_observer.close()
