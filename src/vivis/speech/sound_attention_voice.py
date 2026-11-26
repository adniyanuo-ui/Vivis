"""Optional brief side acknowledgement, isolated from the main private turn."""
import asyncio
from collections import deque
import json
import time

from vivis.interaction.sound_attention import Mailbox,SideSpeechGate,configuration,recent
from vivis.speech.runtime_policy import bounded_send


GATE_REASONS=dict(direction_unavailable='没有新鲜的声源坐标',gaze_feedback_unavailable='跟随控制反馈未就绪',
    gaze_busy='正在进行一次转向',main_face_unavailable='当前没有单人主聊天画面',
    gaze_not_settled='眼颈跟随尚未稳定',mount_unconfirmed='方位安装朝向尚未确认',
    same_direction='声源与主人方向差未达到设定门槛',robot_playback_overlap='声音与音箱回采重叠',
    main_busy='主聊天人的话轮或业务尚未结束',cooldown='距离上次转向过近')


class SoundAttentionVoice:
    def init_sound_attention(self):
        self.sound_attention=None;self.sound_gate=None;self.sound_mailbox=None;self.direction_observer=None
        self.sound_status_publisher=None;self.sound_diagnostic_at=0.;self.sound_observer_key=None
        self.sound_frame_key=None;self.sound_gate_log_at=0.;self.sound_busy={}
        self.sound_gate_warning=None;self.sound_gate_warning_at=0.
        self.side_items=deque(maxlen=128);self.side_responses=deque(maxlen=128)
        self.side_commands=deque(maxlen=128)
        self.side_onsets={}
        self.side_response_sources={}
        self.side_wire_item=None;self.side_task=None;self.side_stable=None
        self.side_dialogue=None;self.side_preroll=deque(maxlen=16)
        self.side_tasks=set()
        c=configuration()
        self.sound_config=c
        definition=json.loads(self.robot_path.read_text())
        self.sound_capabilities=dict(head_yaw_supported='eyes-neck' in definition.get('supported_modes',[]),
            side_attention_configured=c['enabled'] and definition.get('name')=='xiaoshuai',
            manual_turn_tool_available=False,direction_reference_calibrated=c['mount']['calibrated'])
        if not c['enabled'] or definition.get('name')!='xiaoshuai':return
        try:
            import webrtcvad
            vad=webrtcvad.Vad(2)
        except ImportError:
            self.log.write('sound_attention_disabled',reason='webrtcvad_not_installed');return
        self.sound_gate=SideSpeechGate(c,lambda pcm:vad.is_speech(pcm,16000))
        self.side_audio=self.fixed_audio('稍等。')  # Validate/load before audio scheduling.
        hardware=json.loads((self.robot_path.parent/definition['hardware']).read_text())
        from vivis.speech.sound_direction import DirectionObserver,configuration as direction_configuration
        self.direction_observer=DirectionObserver(direction_configuration(),hardware['runtime']['port'])
        self.sound_mailbox=Mailbox(self.folder/'sound_attention_feedback.json',self.folder/'sound_attention.json')
        # Lazy start after microphone input, not during construction/offline check.
        self.sound_started=False;self.side_suppressed=False;self.side_cancelled=False

    def sound_frozen(self):return getattr(self,'sound_attention',None) is not None

    def sound_status(self,*,details=False,now=None):
        now=time.monotonic() if now is None else now
        observer=getattr(self,'direction_observer',None)
        status=observer.status(now) if observer is not None and callable(getattr(observer,'status',None)) else dict(status='disabled')
        result=dict(getattr(self,'sound_capabilities',{}),observer_status=status['status'],
            side_attention_runtime_available=getattr(self,'sound_gate',None) is not None,
            side_glance_active=self.sound_frozen() and not self.sound_attention.get('motion_returned'),
            side_dialogue_active=self.sound_frozen(),current_speaker_location_confirmed=False,
            notice='本体支持限位内眼颈跟随和自动短暂循声；没有手动转头工具，不承诺已执行。固定底座不等于头颈不能转。最近声源不等于本轮说话人位置，不确认身份；未知或未标定时不猜您在正前方。')
        if details:
            gate=getattr(getattr(self,'sound_gate',None),'diagnostic',{})
            result.update(observer=status,gate=dict(gate) if isinstance(gate,dict) else {},
                main_busy=dict(getattr(self,'sound_busy',{})))
        return result

    def poll_sound_diagnostics(self,now,metrics):
        if now<getattr(self,'sound_diagnostic_at',0.):return
        self.sound_diagnostic_at=now+.25
        data=self.sound_status(details=True,now=now);status=data['observer']
        key=(status.get('status'),status.get('error'),status.get('generation'))
        if key!=self.sound_observer_key:
            self.sound_observer_key=key;self.log.write('sound_direction_state',**status)
            error=status.get('error')
            message=('声源串口不可用：'+error+'；普通聊天继续。' if error else
                '声源串口已打开；等待有效坐标。' if status.get('serial_open') and status['status']=='waiting' else None)
            if message:self.emit(dict(kind='voice_warning',epoch=self.epoch,message=message))
        frame_key=(status.get('generation'),status.get('sequence'))
        if status['status']=='fresh' and frame_key!=self.sound_frame_key:
            self.sound_frame_key=frame_key
            self.log.write('sound_direction_observation',**status)
            self.emit(dict(kind='voice_warning',epoch=self.epoch,
                message=f"声源方位：收到坐标{status['sector']:02d}；"+
                    ('安装朝向已校准。' if data.get('direction_reference_calibrated') else '安装朝向尚待校准，不据此猜测说话人位置。')))
        if metrics['clean_rms']>=self.sound_config['minimum_clean_rms'] and now-self.sound_gate_log_at>=1.:
            self.sound_gate_log_at=now
            self.log.write('sound_attention_gate',**data['gate'],main_busy=data['main_busy'],
                observer_status=status['status'],clean_rms=metrics['clean_rms'],reference_rms=metrics['reference_rms'])
            reason=data['gate'].get('reason');text=GATE_REASONS.get(reason)
            if text and (reason!=self.sound_gate_warning or now-self.sound_gate_warning_at>=10.):
                self.sound_gate_warning=reason;self.sound_gate_warning_at=now
                self.emit(dict(kind='voice_warning',epoch=self.epoch,message='循声未触发：'+text+'。'))
        publisher=getattr(self,'sound_status_publisher',None)
        if publisher:publisher.submit(dict(updated_monotonic=now,**data))

    def acoustic_microphone(self,stamp,pcm,metrics,*,far=None):
        """Runs on the existing async drain, never on the PortAudio callback."""
        gate=getattr(self,'sound_gate',None)
        if gate is None:return False
        far=bytes(len(pcm)) if far is None else far
        self.side_preroll.append((stamp,pcm,far))
        if not self.sound_started:
            self.sound_started=True;self.sound_mailbox.start();self.direction_observer.start()
            from vivis.motion.ownership import PreviewPublisher
            self.sound_status_publisher=PreviewPublisher(self.folder/'sound_direction.json')
            self.log.write('sound_attention_started',budget_s=self.sound_config['budget_s'],
                mount_calibrated=self.sound_config['mount']['calibrated'],checksum_verified=False)
        state=self.cloud_state;now=time.monotonic()
        if (self.context['mode']!='interaction' or not self.context['ready']
                or not self.health_available or not self.enabled):
            gate.reset(keep_history=True,now=now);self.side_suppressed=False
            self.poll_sound_diagnostics(now,metrics)
            return False  # Standby wake and task stop retain their original input path.
        single=self.presence.single_visible();reply_busy=self.turn_reply_pending()
        self.sound_busy=dict(user_speaking=self.in_speech,reply_pending=reply_busy,single_face=single,
            control_pending=self.control_pending,expression_pending=self.expression_pending is not None,
            business_pending=bool(self.message_lane and self.message_lane.task) or bool(self.relay_pending))
        available=(state is not None and self.context['mode']=='interaction' and self.context['ready']
            and self.health_available and self.enabled and not self.in_speech and not self.control_pending
            and (single or self.sound_frozen()) and self.farewell is None)
        # Do not pause confidential/unfinished business output to answer a stranger.
        # Ordinary already-complete conversational speech may be paused briefly.
        feedback=self.sound_mailbox.latest
        previous=getattr(self,'sound_attention',None)
        if previous and recent(feedback,now,self.sound_config['lease_s']):
            # The camera may briefly face B; the main-person bearing stays A's.
            feedback=dict(feedback,phase='idle',single_face=True,main_bearing_deg=gate.main_anchor)
        suppressed,request=gate.observe(stamp,pcm,metrics,self.direction_observer.latest,
            feedback,available=available,now=now)
        self.side_suppressed=suppressed
        self.poll_sound_diagnostics(now,metrics)
        if request:
            audio=self.side_audio
            self.side_stable=self.stable_identity()
            self.cancel_public_prompt();self.reception.conversation.user_started()
            previous_guest=previous.get('guest') if previous else None
            if previous:self.cancel_sound_attention('new_side_turn')
            request.update(person=self.side_stable['person_id'],session=self.memory_session,
                revision=self.context['revision'],connection=self.message_connection,
                generation=self.input_generation,stable=self.side_stable,audio=audio)
            request['guest']=previous_guest if previous else (self.sound_mailbox.latest or {}).get('visitor')
            request['motion_returned']=not request.get('glance_permitted',True)
            guest=request['guest'] or {}
            request['guest_key']=guest.get('token') or request['id']
            self.sound_attention=request
            if self.continuity:self.continuity.submit('invalidate_identity_audio')
            self.log.write('sound_attention_begin',attention_id=request['id'],sector=request['sector'],
                bearing_deg=request['bearing_deg'],main_person=request['person'],
                business_identity_retained=True,side_identity_separate=True)
            self.emit(dict(kind='voice_warning',epoch=self.epoch,
                message='声源插话：听取侧方原话并回应，保留主聊天身份；侧方业务单独核验。'))
            from vivis.speech.side_dialogue import SideDialogue
            self.side_dialogue=SideDialogue(self,request)
            self.side_dialogue.paused.update(self.player.pause(state.generating)['responses'])
            for frame in self.side_preroll:
                if frame[0]>=request['heard_monotonic']-.04:self.side_dialogue.feed(*frame)
            self.side_task=asyncio.create_task(asyncio.wait_for(self.side_dialogue.run(),130))
            self.side_tasks.add(self.side_task);self.side_task.add_done_callback(self.side_finished)
        elif self.side_dialogue and not self.side_dialogue.capture_done.is_set():
            if suppressed:self.side_dialogue.feed(stamp,pcm,far)
            elif metrics['clean_rms']<self.sound_config['minimum_clean_rms']:
                self.side_dialogue.end()
            else:
                self.cancel_sound_attention('main_user_started')
        return suppressed

    def side_finished(self,task):
        self.side_tasks.discard(task)
        if not task.cancelled():task.exception()  # Observed again by acoustic_tick when current.

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
        if not same or now-request['created_monotonic']>130:
            self.cancel_sound_attention('state_changed_or_deadline');return
        if self.side_task and self.side_task.done():
            try:self.side_task.result()
            except (asyncio.CancelledError,Exception) as exc:
                self.log.write('side_dialogue_task_finished',error_type=type(exc).__name__)
            if request.get('motion_returned'):self.cancel_sound_attention('returned');return
        if request.get('motion_returned'):return
        fresh_feedback=recent(feedback,now,self.sound_config['lease_s']) and feedback.get('owner')==request['owner']
        if fresh_feedback and feedback.get('attention_id')==request['id']:request['motion_seen']=True
        if not request.get('motion_seen') and now-request['created_monotonic']>self.sound_config['budget_s']:
            request['motion_returned']=True
            self.sound_mailbox.submit(dict(id=request['id'],owner=request['owner'],action='cancel',updated_monotonic=now))
            self.log.write('sound_glance_skipped',reason='motion_not_admitted',semantic_listening_continues=True);return
        main_present=((self.knowledge.current_identity().get('state')=='recognized'
            and self.recognized_person()==request['person']) if request['person'] else self.presence.single_visible())
        if request.get('guest'):
            current=(feedback or {}).get('visitor') or {}
            main_present=(current.get('owner')==request['guest']['owner']
                and current.get('token')==request['guest']['token'])
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
                request['motion_returned']=True
                self.sound_mailbox.submit(dict(id=request['id'],owner=request['owner'],action='cancel',updated_monotonic=now))
                if feedback.get('result') in ('found','empty'):
                    self.cancel_sound_attention(feedback['result'])

    def cancel_sound_attention(self,reason):
        request=getattr(self,'sound_attention',None)
        if request is None:return
        self.sound_attention=None
        self.sound_mailbox.submit(dict(id=request['id'],owner=request['owner'],action='cancel',updated_monotonic=time.monotonic()))
        dialogue=getattr(self,'side_dialogue',None)
        if dialogue:dialogue.stop()
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
        if self.sound_status_publisher:
            self.sound_status_publisher.finish(dict(updated_monotonic=time.monotonic(),observer_status='stopped'))
