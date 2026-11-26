"""Qwen-only voice decisions, continuous cloud listening and leased speech output."""
import asyncio
import base64
from collections import deque
import json
import queue
import sqlite3
import time
import uuid
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo

from vivis.paths import ROOT
from vivis.speech.runtime_policy import policy,VoiceTransportError,bounded_send
from vivis.interaction.cloud_control import SpeechPresence,controller_intent,normalize_program,ReceptionGate,preparation_reply
from vivis.interaction.expression_intent import IntentWriter,configuration as expression_configuration
from vivis.interaction.speech_dialect import SpeechDialect
from vivis.speech.managed import ManagedVoice,session_with_program_context
from vivis.speech.realtime import VoiceSession,connect_session,websocket_connection,receive_event
from vivis.speech.service_recovery import ServiceRecovery,TransientServiceError,transient_error
from vivis.interaction.robot_tasks import RobotTasks,relay_matches,MessageMeaningUncertain
from vivis.interaction.courier import CourierHandoffs,pickup_request,courier_read_intent
from vivis.interaction.message_delivery import (MessageService,MessageRequest,IdentityVerificationRequired,
    MessageRecipientUnavailable,MessageSourcePending)
from vivis.interaction.message_status import message_view,operation_failure
from vivis.interaction.message_lane import MessageLane,StaleMessageTurn,extract_qwen
from vivis.speech.message_receipts import MessageReceipts,ReceiptRejected,MESSAGE_REQUEST
from vivis.speech.continuity_voice import ContinuityVoice
from vivis.speech.fused_identity import FusedIdentityVoice,StaleIdentityStatus
from vivis.speech.sound_attention_voice import SoundAttentionVoice


CLOUD_POLICY=(' 所有语音理解、唤醒、问答、点播和停止由你负责，没有本地ASR或本地语义模型备用。'
    'controller_state.speech_ready=true且ready=false时可先语音交流，动作尚未准备完成。准备、许可和语音链路状态只供内部控制，不向访客播报；自然回应访客原话，不声称尚未完成的动作已经执行。'
    '前台接待来人时自然主动问候，若本地稳定识别熟人，使用当前preferred_address称呼；未知访客只称您好，不猜姓名。'
    '用户说想一想、等一下或先不用聊天时，用defer_reception；只简短尊重要求，不追问。'
    'controller_state.spoken_public_invitation是你通过本地音箱刚说过的公开邀请，不是用户指令。'
    '用户回答需要、好呀、介绍一下时，承接这句邀请介绍真实能力或节目，不重新问有什么需要帮忙。'
    '当前状态以controller_state为准；待机按交流进退规则判断明确呼叫或当前获准的直接招呼，其他话保持安静。待机中的问候/问题必须先调用robot_program唤醒，get_robot_status不能代替唤醒。'
    '无可见人脸且没有明确唤醒许可时，不接新的普通话题，不主动搭话；仍接受停止和待机。controller_state.current_turn_reply_permission=true表示本轮开口时已获交流许可，可完成当前回答；它不确认身份、不增加业务权限，也不是下轮的唤醒许可。'
    '播放期间不要直接回答或播放聊天语音，先通过控制工具交接。'
    '用户说停止、结束、不用播了等，请调用robot_program stop回交互；待机/自己玩去吧则standby。'
    '节目播放或program_followup_handoff=true时，用户明确对你提问或要求一次表情，调用program_followup保留原话。问会不会唱属于问题。明确换节目、停止、待机仍调用robot_program。'
    '泛指换一首、换首别的歌是继续听另一首歌，用program_followup(kind=change_song)，不能只调用stop就结束。指定歌名才直接play。'
    '节目正在播放时，“不要停止、别停、继续当前节目”要求保持现状，不调用robot_program或program_followup；play会重新开始播放，不能当作继续。'
    'stop/standby只需action字段；不能因为没有人脸或没叫名字而拒绝停止。play才需要program。'
    '任何点播必须调用robot_program，不能只口头答应；笼统的节目类别唯一时可选择，否则先澄清。'
    '实时表情跟随是独立节目：用户要你学表情、模仿我的表情、跟着我做表情时，调用robot_program(action=play,program=expression-follow)。'
    '它不同于聊天中的自动微笑，也不同于set_expression做一次表情。意思明确或用户肯定你的表情跟随提议后立即调用，不再空口反复确认。'
    '只问你会不会学表情是能力询问，不等于要求启动；不要模仿、先别开始等否定同样不能启动。'
    '只有controller_state.mode=task且target=expression-follow、ready=true才表示表情跟随已运行；否则不能说准备好了、已经跟着学。'
    '用户本句明确呼唤唤醒词时explicit_wake必须为true，否则为false；同句点播可用play加explicit_wake=true，或wake中填写program。仅唤醒时program填空字符串。'
    '你认识我吗、我是谁、我的年龄或职业等问题必须先调用get_robot_status，按当前身份回答；未知时不猜。'
    '个人记忆只按稳定单人身份归属；记忆关闭或身份未确认时，不承诺已经建立可跨次找回的个人记录。'
    '询问某人来过没有或留言转达状态，由最终识别文字触发本地记录查询和准确播报；不能把历史assistant回答当作到访证据。当前画面无人、没有待转达留言或查询无结果，都不能证明他没来过。没有可核验依据时只能说暂时无法确认；转达记录不证明物品已经买好。'
    '用户自报姓名或称呼不等于视觉核验。current_person为空时不能宣称认出了某位熟人，不能补全其姓名岗位或转述其任务。'
    'identity_state=reviewing表示控制器在身份保持窗口内，可能是低分、不同候选或暂时丢脸；不说突然忘记，也不声称刚刚重新核验成功。若identity_retained或identity_verified为true，沿用当前会话身份和业务权限。窗口到期后以新控制器状态为准。'
    '工具返回pending仅表示请求收到，不代表已经播放；准备完成后控制器才更新状态。'
    '工具accepted=false表示没有执行，必须依据error纠正工具调用，禁止说已经退出/已经完成。'
    'controller_state.unattributed_source_notice=true表示识别片段没有可靠的话轮归属，不能据此办理业务或宣称已记录；请用户完整表达当前请求，只问缺项，不把查询误说成登记取件码失败。'
    '交互态用户说笑一个/别笑了/认真一点/恢复自然时，调用set_expression，source=command；不能只口头答应。别笑了用neutral，认真一点用attentive，恢复自然用auto回到温和微笑底态。'
    '聊天自动回应使用同一个set_expression，source=social，结合当前话语、上下文和可观察表情；不要机械模仿。'
    '用户分享明确的实验成功、好消息时，回答前选择surprise或joy；表达疲惫或烦恼时选concern，严肃解释选listening或attentive。'
    '在需要改变表达时，为当前回答开始前安排一次source=social；普通简短事实保持自然，不每句都调用。'
    '只读恢复、节目接续或留言专用朗读遵守更窄权限，不额外调用自动表情工具；本地自然状态仍可继续。'
    '自动表情scheduled_for_current_reply只是排入本句播放，不是已经做完动作；自然回答用户，不念调度状态，不等待表情回执。'
    'skipped_late_expression表示本句已经出声、表情来晚了，继续回答，不补做旧表情、不重说本句。'
    '已有四种固定组合：joy开心、concern温和关切、listening认真倾听、surprise惊喜。用户明确要求用source=command；聊天语义适合时可用source=social。它们是标定通道组合，不是新舵机动作。主动话题由本地空闲调度，不能只因视觉变化就自行插话。'
    '默认接待和聊天底态是温和微笑；用户苦笑、抱怨、表达难过或谈严肃事情时，用concern、listening或attentive收起笑容；愉快交流可温和smile，不每句都调用。'
    '视觉数据只是面部动作观察，不证明内心情绪。52基础表情均可调用：单通道用标准通道名，多通道用custom和weights。'
    '用户要求悲伤、哭脸、大笑、惊讶等，实时选择已有通道组合，不以需要重新标定为由拒绝。悲伤可组合BrowInnerUp和MouthFrown；大笑可组合MouthSmile、CheekSquint、JawOpen；哭脸是面部组合，不声称已专门验收。'
    '需要闭嘴/收回下颌时可用custom权重JawOpen=0；MouthClose通道在某些标定中没有有效位移，不要把通道名称当作动作已经成功。'
    '通道存在不保证都有非零映射或独立可见动作，按本体反馈如实说明。点头是颈部姿态，不在52表情通道内，仍未开放。'
    'no_calibrated_displacement表示该本体对应通道没有有效标定位移，不能声称已完成可见动作。'
    'blocked_by_ownership表示自动回应暂让位于说话或跟踪，不是没有标定；不能声称该动作已完成。'
    '表情target_accepted仅表示控制循环接收了平滑目标，不是实测皮肤已经到位；pending时不要声称完成。'
    '不要声称能修改摄像头、舵机参数或执行未提供的动作。')

SOUND_POLICY=(' controller_state.sound_attention给出当前眼颈能力与方位状态。底座固定不等于头颈不能转；'
    '支持范围内视觉跟随和自动短暂循声，不得笼统说本体不能转头。没有手动转头工具，不能把收到请求说成已转过去。'
    '询问声源定位、转头能力或我在哪一边时先调用get_robot_status；observer_status非fresh或安装朝向未校准，'
    '如实说明本轮位置尚不能确认，不猜您在正前方或看得清清楚楚。最近声源也不等于本轮说话人，不能据它确认身份或读取私密记录。')

WORK_POLICY=(' 留言由本地固定流程处理：最终识别文字到齐后才提取结构化收件人和内容，程序核验并写库，固定TTS播报结果。'
    '你负责自然聊天和节目控制，不负责调用留言写入、历史查重或口头宣称已保存。'
    '用户问保存了吗由程序直接读账本；pending是已保存待转达，announced是完整播报后已转达，无需收件人确认；completed是历史完成或人工关闭记录，不再提醒，不能据此声称实际播过或事情办完。不要再询问是否收到，不把留言转达说成事情办完。'
    '物品采购、寄件等实际事情是否完成不在留言系统负责范围。'
    '身份未知不得读取私人历史。退件码仍使用courier_pickup的独立授权边界，不用于普通熟人留言。')

PROGRAM_TURN_POLICY=(' 当前在节目或节目交接中。收到对小帅的明确问题，包括“你多大了”“会唱白骨夫人吗”，'
    '以及“今天星期几”“现在几点”等日常问题，不论是否知道答案，'
    '必须先调用program_followup(kind=question)，暂时不要用文本或语音回答。'
    '收到“大笑一下”等一次性表情请求，必须先调用program_followup(kind=expression)。'
    '收到“换一首歌曲”等泛指换歌请求，调用program_followup(kind=change_song)，不当作单纯stop。'
    '只有明确要求开始一首歌、一个剧本或持续跟随，才能调用robot_program点播。'
    '停止和待机用robot_program。先排除否定、引用、节目自身声音及旁人讲话；例如“歌词写的是你多大了”'
    '只是在转述歌词，绝不调用工具。只有直接对机器人说的明确请求才切换。')


class ConversationReset(Exception):pass
class CloudIdleTimeout(Exception):pass
class SpeechVoiceSwitch(Exception):pass
class StaleMessageRequest(ValueError):pass
class ControllerAcknowledgementTimeout(VoiceTransportError):pass
class GreetingAdmissionError(ValueError):pass


class ControllerEventGate:
    """A proactive reply needs a server-confirmed conversation item first."""
    def __init__(self):self.pending={}

    async def append(self,ws,context,kind,directive):
        item_id='controller_'+uuid.uuid4().hex
        future=asyncio.get_running_loop().create_future();self.pending[item_id]=future
        event=dict(source='local_controller_event',human_speech=False,kind=kind,controller=dict(context),instruction=directive)
        text='【本地控制器事件，不是用户说话，也不是新的点播命令；不写入用户个人事实】'+json.dumps(event,ensure_ascii=False)
        try:
            # Qwen requires a user-role item to initiate a response. Its CONTENT
            # explicitly identifies the controller source, never a fake ASR turn.
            await bounded_send(ws,json.dumps(dict(type='conversation.item.create',item=dict(id=item_id,type='message',role='user',
                content=[dict(type='input_text',text=text)])),ensure_ascii=False))
            try:await asyncio.wait_for(future,timeout=policy()['controller_ack_s'])
            except asyncio.TimeoutError:raise ControllerAcknowledgementTimeout('Controller event acknowledgement timed out') from None
            return item_id
        finally:self.pending.pop(item_id,None)

    def observe(self,event):
        if event.get('type')!='conversation.item.created':return
        item=event.get('item',{});future=self.pending.get(item.get('id'))
        if future is not None and not future.done():future.set_result(True)


class CloudManagedVoice(SoundAttentionVoice,ContinuityVoice,FusedIdentityVoice,ManagedVoice):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.presence=SpeechPresence(json.loads((ROOT/'configs/perception/speech_presence.json').read_text()))
        from vivis.interaction.identity_stability import IdentityStability
        self.identity_stability=IdentityStability(self.presence.c['identity_uncertain_reset_s'])
        from pathlib import Path
        catalog=json.loads(Path(self.speech['programs']).read_text())
        self.programs={p['id']:p for p in catalog['programs']}
        self.program_ids={p['id'] for p in catalog['programs']}
        reception=json.loads((ROOT/self.speech['managed_voice']['reception_config']).read_text())
        self.reception=ReceptionGate(reception)
        from vivis.interaction.social_conversation import SocialConversation,configuration as social_configuration
        self.social=SocialConversation(social_configuration(self.speech['managed_voice'].get('social_conversation_config')),
            self.presence.c['max_age_s'])
        self.last_social_cue=None
        self.social_turns=deque(maxlen=6)
        self.public_prompt_task=None;self.public_prompt_started=False;self.thinking_generation=None
        self.public_prompt_context=None
        from vivis.interaction.memory import PersonMemory
        self.memory_config=json.loads((ROOT/self.speech['managed_voice']['memory_config']).read_text())
        self.memory=PersonMemory(self.memory_config) if self.memory_config['enabled'] else None
        self.tasks_config=json.loads((ROOT/self.speech['managed_voice']['tasks_config']).read_text())
        self.tasks=RobotTasks(self.memory,self.tasks_config) if self.memory and self.tasks_config['enabled'] else None
        self.courier=CourierHandoffs(self.memory) if self.memory else None;self.courier_offer=None;self.courier_query_pending=None
        self.task_cache=None;self.task_refresh_at=0.;self.task_cache_version=-1
        self.task_visit_announced=set();self.task_notice_attempts={};self.relay_pending={};self.turn_item_id=None;self.turn_message_id=None;self.turn_ended_at=None
        self.memory_turns={}
        self.program_waiters={}
        self.response_request_pending=False;self.last_response_request_at=None
        self.memory_task=None;self.memory_retry_at=0.;self.anonymous='anonymous:'+uuid.uuid4().hex
        self.turn_person=self.anonymous;self.turn_engaged=False;self.turn_saved=False;self.pending_user_text=None;self.reply_people={};self.pending_memories={}
        self.memory_session=self.folder.parent.name+':'+uuid.uuid4().hex;self.conversation_person=None;self.visit_seen=False;self.absence_since=None;self.identity_missing_since=None
        self.next_identity_verification=0.;self.relay_identity_hold=False;self.relay_recipients={}
        self.identity_display=None
        self.question_onset=None;self.invalid_question=None;self.recovered_query_generation=None
        self.program_followup=None;self.followup_handoff=False;self.followup_response=None
        self.program_responses={};self.program_check_task=None
        self.recovered_expression=None
        robot_definition=json.loads(self.robot_path.read_text())
        self.wake_words=(self.speech.get('wake_words') or robot_definition.get('identity',{}).get('wake_words')
            or [robot_definition.get('identity',{}).get('display_name',robot_definition['name'])])
        from vivis.interaction.presence_entry import configuration as entry_configuration
        self.entry=entry_configuration()
        self.explicit_wake_pending=None;self.explicit_wake_seen=deque(maxlen=128)
        self.farewell=None;self.farewell_task=None;self.controller_reply_id=None
        from vivis.speech.fixed_replies import FixedReplyCache
        directory=self.speech['managed_voice'].get('fixed_replies')
        self.fixed_replies=FixedReplyCache(ROOT/directory) if directory else None
        self.cloud_audio_enabled=False
        self.identity_reset_pending=False;self.finish_responses=set();self.speech_responses=deque(maxlen=128)
        self.input_generation=0;self.response_inputs={};self.message_receipts=MessageReceipts()
        self.message_lane=None;self.message_claimed=None;self.message_speaker=None;self.courier_waiters={};self.unattributed_source_notice=False;self.message_connection=object()
        self.response_waiting_user_end=False
        self.context=dict(revision=0,mode='standby',target='standby',ready=False)
        self.calls=deque(maxlen=128);self.response_revisions={};self.blocked=deque(maxlen=128)
        self.continue_response=False;self.in_speech=False;self.last_snapshot=None;self.last_permission=False
        self.announcement_revision=None;self.announcement_kind='welcome';self.tool_retry_budget=2;self.idle_sleep_revision=None
        self.tool_repair=None;self.control_pending=False;self.expression_pending=None;self.expression_outcome=None
        self.expression=IntentWriter(self.folder/'expression_intent.json',expression_configuration())
        from vivis.interaction.conversation_expression import configuration as conversation_configuration
        from vivis.speech.conversation_animation import ConversationAnimation
        self.conversation_animation=ConversationAnimation(self.folder/'conversation_expression.json',
            conversation_configuration(),self.expression.c)
        self.dialect=SpeechDialect()
        self.active_voice=self.session['voice'];self.pending_voice_switch=None
        self.service_recovery=ServiceRecovery(self.config['service_recovery'])
        self.awaiting_recovery_user=False
        self.transport_fault=None;self.health_available=True;self.audio_ready=True;self.audio_recovery=None
        self.expression.clear();self.resume_history=deque(maxlen=max(12,2*self.config['max_history_turns']))
        self.summary['control_backend']='qwen_realtime';self.summary['local_asr_started']=False
        self.log.write('cloud_control_backend',name='qwen_realtime',local_asr_started=False,reception_enabled=reception['enabled'],memory_enabled=self.memory is not None)
        self.init_continuity()
        self.init_fused_identity()
        self.init_sound_attention()

    def sync_conversation_animation(self):
        animation=getattr(self,'conversation_animation',None)
        if animation is None:return None
        scope=(id(self.message_connection),self.memory_session,self.conversation_person,
            self.context['revision'],self.input_generation)
        enabled=(self.context['mode']=='interaction' and self.context['ready']
            and (self.presence.allowed() or self.turn_reply_pending() or self.finish_responses) and self.health_available)
        animation.context(scope,enabled,self.in_speech)
        return scope

    def health(self,available,reason,*,recovery_kind=None):
        if available==self.health_available:return
        self.health_available=available
        self.sync_conversation_animation()
        if not available:
            self.message_connection=object()
            if self.message_lane:self.message_lane.reset_context()
            if self.cloud_state:self.cloud_state.interrupt(source='voice_unavailable')
        self.emit(dict(kind='voice_health',available=available,reason=reason,recovery_kind=recovery_kind))

    def input_overflow(self):
        self.cloud_queue=None
        self.transport_fault='Cloud input queue full; current turn invalidated'
        self.health(False,'input_overflow')

    async def recover_audio(self,reason):
        """Close the previous owner before reopening; no hardware-controller restart."""
        self.log.write('audio_recovery_started',reason=reason,
            capture_clock_fault=self.audio.stats.get('capture_clock_fault'),
            captured_frames=self.audio.stats.get('captured_frames'),processed_frames=self.audio.stats.get('processed_frames'),
            recent_callback_warnings=self.audio.stats.get('callback_warning_events',[])[-8:])
        interval=(self.audio.stats.get('capture_clock_fault') or {}).get('interval_ms')
        recovery_kind=('capture_gap' if type(interval) in (int,float) and 35<interval<1000*policy()['callback_stall_s']
                       and reason.startswith('Gap in microphone capture clock:') else None)
        self.audio_ready=False;self.health(False,reason,recovery_kind=recovery_kind)
        self.message_connection=object();self.awaiting_recovery_user=True
        self.cloud_queue=None
        if self.cloud_task:
            self.cloud_task.cancel();await asyncio.gather(self.cloud_task,return_exceptions=True)
        self.cloud_task=None
        self.enabled=False
        if self.mouth:self.mouth.deactivate()
        # Never open a second owner when close has not actually completed.
        await asyncio.wait_for(asyncio.to_thread(self.audio.close),policy()['audio_close_s'])
        self.state('off')
        from vivis.speech.realtime_audio import AudioIO
        while not self.shutdown.is_set():
            board=self.speech['managed_voice'].get('board_reference_device')
            audio=AudioIO(self.speech,self.speaker,self.player,aec=board is None,
                frontend=self.config['frontend'],board_reference_device=board)
            try:
                devices=await asyncio.wait_for(asyncio.to_thread(audio.open,with_output=False),policy()['audio_open_s'])
            except asyncio.TimeoutError:
                # The native call may still own devices. Do not start another open.
                raise RuntimeError('Audio open did not finish; automatic reopening stopped') from None
            except (ValueError,TypeError):raise
            except Exception as exc:
                await asyncio.wait_for(asyncio.to_thread(audio.close),policy()['audio_close_s'])
                self.log.write('audio_reopen_failed',error_type=type(exc).__name__)
                await asyncio.sleep(policy()['recovery_cooldown_s']);continue
            self.audio=audio;self.audio_ready=True
            self.opened=time.monotonic();self.last_input=self.opened
            self.summary['microphone_opens']+=1
            self.log.write('audio_reopened',**devices)
            self.cloud_task=asyncio.create_task(self.cloud(0))
            return

    def start_decoder(self):
        # The base class owns only audio/AEC; deliberately no Decoder construction.
        self.cloud_task=asyncio.create_task(self.cloud(0))

    def output_ready(self):
        return (self.enabled and self.context['mode']=='interaction' and self.context.get('speech_ready',self.context['ready'])
                and self.farewell is None
                and not self.program_waiters
                and self.pending_mode is None
                and not self.tool_repair and not self.control_pending and self.expression_pending is None
                and self.pending_voice_switch is None and not self.awaiting_recovery_user
                and not self.relay_identity_hold)

    def can_speak(self):
        """Fresh presence/wake permission, separate from an admitted turn's reply."""
        return self.explicit_wake_pending is None and self.output_ready() and self.presence.allowed()

    def turn_reply_authorized(self,item_id):
        """Onset permission belongs to one attributed turn, never to an identity."""
        if not isinstance(item_id,str) or item_id!=self.turn_item_id:return False
        turn=self.memory_turns.get(item_id,{})
        return bool(turn.get('reply_permission_at_onset') is True and turn.get('valid') is True
            and turn.get('generation')==self.input_generation
            and turn.get('connection') is self.message_connection
            and turn.get('revision')==self.context['revision'] and turn.get('session')==self.memory_session)

    def can_answer_turn(self,item_id):
        state=self.cloud_state
        return self.can_speak() or bool(self.explicit_wake_pending is None and self.output_ready()
            and self.turn_reply_authorized(item_id) and state
            and (self.user_turn_pending(state) or state.generating))

    def turn_reply_pending(self):
        state=self.cloud_state
        return bool(state and self.turn_reply_authorized(self.turn_item_id)
            and (self.user_turn_pending(state) or state.generating or self.player.count or self.finish_responses))

    def finish_farewell(self,ticket,reason):
        if self.farewell is not ticket:return
        self.farewell=None
        if ticket.get('archive_job') and reason=='played' and self.continuity:
            self.continuity.submit('job_notice',ticket['archive_job'])
        self.log.write('transition_reply_finished',reply_id=ticket['id'],reason=reason)
        self.emit(dict(kind='voice_reply_done',epoch=self.epoch,reply_id=ticket['id'],reason=reason))

    def cancel_farewell(self,reason):
        ticket=self.farewell
        if ticket is None:return
        if self.farewell_task and not self.farewell_task.done():self.farewell_task.cancel()
        self.finish_farewell(ticket,reason)

    def fixed_audio(self,text):
        if self.fixed_replies is None:return None
        try:
            return self.fixed_replies.get(text,dict(self.session,model=self.config['model'],voice=self.active_voice),
                self.mouth.order if self.mouth else None)
        except Exception:
            self.log.write('fixed_reply_cache_rejected',reason='asset_validation_failed')
            return None

    def schedule_farewell(self,state):
        """One bounded transition/incident reply; never ordinary task/idle chat."""
        ticket=self.farewell
        if ticket is None:return
        if ticket['connection'] is not self.message_connection or ticket['generation']!=self.input_generation:
            self.cancel_farewell('new_input_or_connection');return
        if self.farewell_task and not self.farewell_task.done():return
        def live():
            return (self.farewell is ticket and ticket['connection'] is self.message_connection
                    and ticket['generation']==self.input_generation and self.cloud_state is state)
        def current():
            return (live() and self.context['target']==ticket.get('target','standby')
                and (self.context.get('standby_fault') if ticket.get('fault') else self.context.get('speech_ready',self.context['ready']))
                and self.controller_reply_id==ticket['id']
                and self.enabled and self.pending_mode is None)
        async def speak():
            from vivis.speech.trusted_reading import RealtimeReader
            from vivis.speech.message_voice import MessageSpeaker
            from vivis.speech.fixed_replies import REPLIES
            text=REPLIES['standby_fault'] if ticket.get('fault') else ticket.get('text',REPLIES['standby'])
            reason='not_played'
            try:
                if not text:
                    # Keep the same reply-ID handoff, with no TTS, PCM or mouth
                    # frames. Silence is one of the user's five choices.
                    while live() and not current():await asyncio.sleep(.02)
                    if not current():raise StaleMessageTurn()
                    reason='silent';return
                # Reuse the verified Qwen reader and the SAME output/AEC/A2F
                # path. No DB operation, personal identity or local TTS model.
                engine=None;audio=self.fixed_audio(text)
                self.log.write('fixed_reply_audio',cached=audio is not None,characters=len(text),voice=self.active_voice)
                if audio is None:
                    # Missing or differently voiced assets keep the existing
                    # bounded reader; never substitute another cached voice.
                    engine=RealtimeReader(self.url,self.secret,dict(self.session,voice=self.active_voice),self.log)
                    audio=await asyncio.wait_for(engine.generate(text),policy()['playback_stall_s'])
                while live() and not current():await asyncio.sleep(.02)
                if not current():raise StaleMessageTurn()
                class PreparedAudio:
                    async def generate(self,_):return audio
                speaker=MessageSpeaker(engine,self.player,state,lambda _:current(),
                    lambda words:self.emit(dict(kind='cloud_transcript',epoch=self.epoch,role='assistant',
                        text=words,source='standby_fault' if ticket.get('fault') else ticket.get('source','transition_farewell'))),self.log.write,
                    tts=PreparedAudio(),source='transition_reply')
                complete=await speaker.say(text,ticket)
                reason='played' if complete else 'interrupted'
            except asyncio.CancelledError:reason='cancelled'
            except StaleMessageTurn:reason='superseded'
            except Exception as exc:
                reason='speech_failed';self.log.write('transition_reply_failed',error_type=type(exc).__name__)
                self.emit(dict(kind='voice_warning',epoch=self.epoch,message=(
                    '待机故障语音未完整播出；故障记录保留，下轮继续检查。' if ticket.get('fault') else
                    '准备提示未完整播出；执行结果请查看控制状态，语音监听继续。')))
            finally:self.finish_farewell(ticket,reason)
        self.farewell_task=asyncio.create_task(speak())

    def can_play_response(self,rid):
        if self.program_followup is not None and self.followup_handoff:return False
        followup=self.followup_response
        if (followup and self.response_inputs.get(rid)==followup['generation']
                and not self.player.has_started([rid]) and not self.followup_current(followup)):return False
        if not self.output_ready() or rid in self.blocked:return False
        if self.cloud_state and rid in self.cloud_state.blocked:return False
        if self.can_speak():return True
        # Audio can arrive between presence expiry and the next context tick.
        # Establish the same permission here before rejecting a valid tail.
        if self.cloud_state:self.preserve_current_speech(self.cloud_state,'no_face_no_wake')
        return rid in self.finish_responses

    def preserve_current_speech(self,state,reason):
        """Only replies already authorized in this connection may finish.

        A response.done event ends generation, not DAC playback. Include queued
        PCM and the last scheduled DAC span before clearing personal context.
        Relay audio is never granted this continuation permission.
        """
        buffered=set(self.player.response_ids())
        active=set()
        public=[rid for rid,response in state.responses.items()
            if response.get('source') in ('public_reception','transition_reply')]
        for rid in set(self.speech_responses)|set(public):
            response=state.responses.get(rid,{})
            if rid in self.blocked or rid in state.blocked or rid in self.relay_recipients:continue
            if (rid==state.generating or rid in buffered
                    or (response.get('status')=='completed' and response.get('received_audio_s')
                        and not self.audio_completed(state,rid))):active.add(rid)
        fresh=active-self.finish_responses
        self.finish_responses=active
        if fresh:self.log.write('speech_finish_before_silence',reason=reason,response_ids=sorted(fresh))
        return bool(active)

    def update_speech_permission(self,state,permission):
        if not permission:self.preserve_current_speech(state,'no_face_no_wake')
        elif not self.identity_reset_pending:self.finish_responses.clear()
        self.last_permission=permission

    def user_turn_pending(self,state):
        if self.in_speech or self.response_waiting_user_end or self.response_request_pending or self.continue_response:return True
        if self.turn_item_id and self.turn_ended_at is not None:
            # Do not reconnect in the gap between speech_stopped and the
            # response carrying its tool call (especially stop/standby).
            return not any(response.get('user_turn',{}).get('item_id')==self.turn_item_id
                and response.get('status') in ('completed','cancelled','failed')
                for response in state.responses.values() if response.get('user_turn'))
        return False

    def speech_preference(self):
        return dict(**self.dialect.snapshot(),active_voice=self.active_voice,
            requested_voice=self.dialect.voice(self.session['voice']),voice_switch_pending=self.pending_voice_switch is not None)

    def voice_ready(self,voice):
        if voice!=self.dialect.voice(self.session['voice']):raise ValueError('Wrong dialect voice acknowledgement')
        previous=self.active_voice;pending=self.pending_voice_switch
        self.active_voice=voice;self.pending_voice_switch=None;self.summary['active_voice']=voice
        if pending:
            if self.context['mode']=='interaction' and self.context['ready']:
                self.announcement_kind='dialect';self.announcement_revision=self.context['revision']
            self.emit(dict(kind='voice_warning',epoch=self.epoch,message='语音音色已确认：'+voice+'；当前'+self.dialect.name+'，实际口音仍需试听。'))
        self.log.write('speech_voice_ready',previous=previous,**self.speech_preference())

    def departure_requires_reset(self,now):
        return (self.visit_seen and self.absence_since is not None
                and now-self.absence_since>=self.reception.c['departure_hold_s']
                and not self.presence.allowed(now))

    def request(self,command):
        if command.get('kind')=='voice_execution_notice':
            reply=command.get('reply_id');text=command.get('text');target=command.get('target')
            if (not isinstance(reply,str) or not 1<=len(reply)<=64 or not isinstance(text,str) or not 1<=len(text)<=200
                    or target not in {'standby','interaction',*self.program_ids}):return
            self.cancel_farewell('execution_notice_replaced')
            self.farewell=dict(id=reply,generation=self.input_generation,connection=self.message_connection,
                target=target,text=text,source='archive_result' if command.get('archive_job') else 'execution_failure',
                archive_job=command.get('archive_job'))
            return
        if command.get('kind')=='voice_fault_notice':
            reply=command.get('reply_id')
            if not isinstance(reply,str) or not 1<=len(reply)<=64:return
            self.cancel_farewell('fault_notice_replaced')
            self.farewell=dict(id=reply,generation=self.input_generation,connection=self.message_connection,fault=True)
            return
        if command.get('kind')=='voice_context':
            revision=command.get('revision');mode=command.get('mode');target=command.get('target');reply=command.get('transition_reply_id')
            if (type(revision) is not int or revision<=self.context['revision'] or mode not in ('standby','interaction','task')
                    or target not in {'standby','interaction',*self.program_ids} or type(command.get('ready')) is not bool):return
            if ('speech_ready' in command and (type(command['speech_ready']) is not bool or not (
                    mode=='interaction' and target=='interaction' or
                    isinstance(reply,str) and 1<=len(reply)<=64 and not command.get('standby_fault')))):return
            if 'standby_fault' in command and (type(command['standby_fault']) is not bool or mode!='standby'):return
            if (mode=='task')!=(target in self.program_ids) or (mode!='task' and mode!=target):return
            previous=self.context
            if mode!='interaction' and (mode!=previous['mode'] or target!=previous['target']):
                self.program_followup=None;self.followup_handoff=False;self.followup_response=None
            self.controller_reply_id=reply if isinstance(reply,str) and 1<=len(reply)<=64 else None
            if self.farewell and (target!=self.farewell.get('target','standby') or self.controller_reply_id!=self.farewell['id']
                    or self.farewell.get('fault') and not command.get('standby_fault')):
                self.cancel_farewell('controller_changed')
            if mode=='standby':self.presence.clear_greeting()
            if mode=='standby' and previous['mode']!='standby' and self.presence.greeting_visible():
                self.reception.dismiss_current_visit()
            if mode=='interaction':
                if reply is not None or self.followup_handoff:self.announcement_kind='quiet'
                elif previous['mode']=='standby':self.announcement_kind='welcome'
                elif previous['mode']=='task':self.announcement_kind='return'
                elif previous['ready'] and not command['ready']:self.announcement_kind='quiet'
            if mode!='standby' and self.memory_task and not self.memory_task.done():self.memory_task.cancel()
            self.context={k:command[k] for k in ('revision','mode','target','ready')}
            self.sync_conversation_animation()
            if 'speech_ready' in command:self.context['speech_ready']=command['speech_ready']
            if 'standby_fault' in command:self.context['standby_fault']=command['standby_fault']
            if self.explicit_wake_pending:
                if mode=='interaction' and previous['mode'] in ('standby','interaction'):
                    self.explicit_wake_pending['revision']=revision
                    if command['ready']:
                        self.explicit_wake_pending.setdefault('controller_ready_monotonic',time.monotonic())
                else:self.explicit_wake_pending=None
            self.tool_repair=None;self.control_pending=False
            if mode!='interaction' or not command['ready']:
                self.expression.clear();self.expression_pending=None;self.expression_outcome=None
                self.social_turns.clear()
            if previous['target']!=target or not self.context.get('speech_ready',self.context['ready']):
                self.continue_response=False
                self.message_claimed=None
                if self.message_lane:self.message_lane.reset_context()
                self.message_receipts=MessageReceipts()
                self.courier_offer=None
                self.courier_query_pending=None
                if self.cloud_state:self.cloud_state.interrupt(source='controller_context_changed')
            if mode=='interaction' and self.context['ready'] and (previous['mode']!='interaction' or not previous['ready']):
                self.announcement_revision=None if (self.announcement_kind=='quiet' or self.explicit_wake_pending
                    or previous.get('speech_ready',False)) else revision
            return
        if command.get('kind')=='voice_mode':
            epoch=command.get('epoch');enabled=command.get('enabled')
            newest=max(self.epoch,self.pending_mode[0] if self.pending_mode else self.epoch)
            if type(epoch) is not int or epoch<=newest or type(enabled) is not bool:return
            if enabled:
                session_with_program_context(self.session,command.get('last_program'))
                self.last_program=command.get('last_program')
            self.pending_mode=(epoch,enabled)
            if self.cloud_state:self.cloud_state.interrupt(source='speaker_handoff')
            self.release_at=time.monotonic()+self.config['fade_ms']/1000+.04
            return
        super().request(command)

    async def apply_mode(self):
        if self.pending_mode is None or time.monotonic()<self.release_at:return
        self.epoch,self.enabled=self.pending_mode;self.pending_mode=None
        if not self.audio_ready:
            self.enabled=False
            if self.audio.output_stream is None:self.state('off')
            return
        self.audio.disable_output()
        if self.enabled:
            try:self.audio.enable_output(self.player)
            except (ValueError,TypeError):raise
            except Exception as exc:
                self.audio.fault='Voice output open: '+str(exc)
                self.health(False,'output_open_failed');return
        self.state('ready' if self.enabled and self.ws is not None else 'off')

    def recognized_person(self):
        person=self.knowledge.current_person()
        if not person or not self.presence.visible():return None
        try:
            observation=json.loads(self.presence.path.read_text())
            if observation.get('faces')!=1:return None
        except (OSError,ValueError,AttributeError):return None
        return person['person_id']

    def identified_person(self):
        return self.stable_identity()['person_id']

    def stable_identity(self):
        attention=getattr(self,'sound_attention',None)
        if attention is not None:return attention['stable']
        if self.identity_fusion:return self.fused_identity()
        with self.identity_stability.lock:
            return self.identity_stability.observe(self.recognized_person(),time.monotonic())

    def identity_labels(self, stable=None):
        stable=self.stable_identity() if stable is None else stable
        if self.identity_fusion:return self.fused_identity_labels(stable)
        def label(pid):
            person=self.knowledge.people.get(pid,{})
            return person.get('preferred_address') or person.get('display_name') or 'other'
        logical=label(stable['person_id']);raw=label(stable['candidate_id'])
        return dict(identity_label=logical+'-'+raw+('/图-other/声' if self.continuity else ''),logical_label=logical,realtime_label=raw)

    def reviewing_identity(self):
        stable=self.stable_identity()
        person=self.knowledge.people.get(stable['person_id'])
        if not stable['held'] or not person:return None
        return dict(state='reviewing',person=dict(person_id=stable['person_id'],**person),
                    identity_retained=True,hold_remaining_s=stable['remaining_s'])

    def verify_identity(self,state):
        if getattr(self,'sound_attention',None) is not None:return
        now=time.monotonic()
        if (self.identity_fusion and not self.user_turn_pending(state) and not state.generating
                and not self.player.count):self.admit_voice_identity()
        present=self.presence.visible()
        person=self.identified_person()
        review=self.reviewing_identity()
        reviewing_person=review['person']['person_id'] if review else None
        labels=self.identity_labels()
        display=(person,bool(review),labels['identity_label'])
        if now>=self.next_identity_verification or display!=self.identity_display:
            self.identity_display=display
            interval=self.presence.c.get('identity_verification_interval_s',3.)
            self.next_identity_verification=now+interval
            self.log.write('identity_verified',person_id=person,face_visible=present,
                previous_person_id=self.conversation_person,interval_s=interval,
                identity_state='reviewing' if review else 'recognized' if person else 'unknown',
                reviewing_person_id=reviewing_person,identity_hold_s=None if self.identity_fusion else self.identity_stability.hold_s,
                identity_policy='evidence_fusion' if self.identity_fusion else 'timed_hold',**labels)
            try:
                path=self.folder/'identity.json';temporary=path.with_suffix('.tmp')
                label=self.knowledge.people.get(person,{}).get('preferred_address') or person
                temporary.write_text(json.dumps(dict(observed_monotonic=now,person_id=person,label=label,
                    state='reviewing' if review else 'recognized' if person else 'unknown',
                    hold_s=None if self.identity_fusion else self.identity_stability.hold_s,**labels),ensure_ascii=False))
                temporary.replace(path)
            except OSError:pass  # Optional preview must not block dialogue or authority checks.
        reason=None
        if present:
            self.visit_seen=True;self.absence_since=None
        elif self.absence_since is None:
            self.absence_since=now
        if person:
            previous=self.conversation_person
            self.conversation_person=person;self.identity_missing_since=None;self.identity_reset_pending=False
            if previous!=person:
                # An initial anonymous welcome is public, not a conversation
                # with a different speaker. Enrich the live context without
                # clearing its queued audio or welcoming the same visitor twice.
                # Prior anonymous turns keep their original attribution and
                # cannot gain private authority. Known-person changes and
                # recipient-bound output still require a fresh conversation.
                has_user_history=any(item.get('role')=='user' for item in self.resume_history)
                bound=self.memory_turns.get(self.turn_item_id,{})
                newly_bound=(self.identity_fusion and previous is None and self.turn_person==person
                    and bound.get('person')==person and bound.get('connection') is self.message_connection
                    and bound.get('session')==self.memory_session)
                if (previous is not None or self.relay_recipients
                        or (self.turn_item_id or has_user_history) and not newly_bound and not self.public_turn_can_enrich_identity(now)):
                    public_inflight=any(r.get('source') in ('public_reception','transition_reply')
                        and rid not in state.blocked and r.get('received_audio_s') and not self.audio_completed(state,rid)
                        for rid,r in state.responses.items())
                    if previous is None and not self.relay_recipients and self.anonymous_identity_turn_pending(now):
                        # The current input began anonymously. Preserve its
                        # audio/source until final ASR can classify it; private
                        # context remains withheld during this narrow interval.
                        self.conversation_person=None
                        notice=(person,self.input_generation,self.turn_item_id)
                        if getattr(self,'identity_enrichment_notice',None)!=notice:
                            self.identity_enrichment_notice=notice
                            self.log.write('identity_enrichment_waiting_final',person_id=person,item_id=self.turn_item_id,
                                user_audio_preserved=True,private_context_withheld=True)
                    elif previous is None and not self.relay_recipients and public_inflight:
                        # Complete this public sentence before isolating older,
                        # unattributed history. Re-evaluate on the next tick.
                        self.conversation_person=None
                    else:reason='confirmed_person_changed'
                else:
                    self.log.write('identity_context_enriched',person_id=person,
                        audio_preserved=True,prior_turns_reassigned=False)
        elif reviewing_person and reviewing_person==self.conversation_person:
            # Same tracked face, bounded visual review retains this conversation.
            self.identity_missing_since=None
            self.identity_reset_pending=False
        elif self.conversation_person:
            if self.identity_missing_since is None:
                released=self.identity_fusion.released_since if self.identity_fusion else self.identity_stability.released_since
                self.identity_missing_since=(now-self.presence.c['identity_uncertain_reset_s'] if self.identity_fusion
                    else released if released is not None else now)
            if now-self.identity_missing_since>=self.presence.c['identity_uncertain_reset_s']:
                self.identity_reset_pending=True
                finishing=self.preserve_current_speech(state,'identity_uncertain')
                # Keep control listening alive through the current user turn
                # and any tool response. Fresh identity remains mandatory in
                # every personal-data/task tool throughout this deferral.
                if not finishing and not self.user_turn_pending(state) and state.generating is None:
                    self.conversation_person=None;reason='identity_uncertain'
        if not present and self.departure_requires_reset(now):
            if not self.preserve_current_speech(state,'visitor_departed') and not self.user_turn_pending(state) and state.generating is None:
                self.visit_seen=False;self.conversation_person=None;reason='visitor_departed'
        if reason:
            if self.identity_fusion and reason in ('visitor_departed','identity_uncertain'):self.identity_fusion.reset(now)
            self.anonymous='anonymous:'+uuid.uuid4().hex;self.turn_person=self.anonymous
            self.memory_session=self.folder.parent.name+':'+uuid.uuid4().hex
            self.continue_response=False
            if reason=='confirmed_person_changed':
                self.reception.armed=True;self.reception.present_since=None
            state.interrupt(source=reason)
            raise ConversationReset(reason)

    def public_turn_can_enrich_identity(self,now):
        source=self.question_onset;text=source.get('final_text') if source else None
        return bool(source and source['person']==self.anonymous and source['generation']==self.input_generation
            and source['item']==self.turn_item_id and source['connection'] is self.message_connection
            and self.presence.single_visible() and self.turn_ended_at is not None
            and 0<=now-self.turn_ended_at<=policy()['message_extract_s']
            and isinstance(text,str) and text.strip() and not MESSAGE_REQUEST.search(text)
            and '取件码' not in text and courier_read_intent(text) is None)

    def anonymous_identity_source_current(self):
        source=self.question_onset
        return bool(source and isinstance(self.turn_item_id,str) and self.turn_item_id
            and source.get('person')==self.anonymous and source.get('generation')==self.input_generation
            and source.get('item')==self.turn_item_id and source.get('connection') is self.message_connection)

    def anonymous_identity_turn_pending(self,now=None):
        """Wait only for this anonymous input's final text, never old history."""
        now=time.monotonic() if now is None else now
        return bool(self.anonymous_identity_source_current() and not self.question_onset.get('final_text')
            and (self.in_speech or self.turn_ended_at is not None
                and 0<=now-self.turn_ended_at<=policy()['message_extract_s']))

    def identity_enrichment_guarded(self):
        # A final event can arrive before the verification tick. Keep private
        # data closed until that tick explicitly accepts the public enrichment.
        return self.conversation_person is None and self.anonymous_identity_source_current()

    def reception_turn_in_progress(self,state):
        source=self.question_onset
        return bool(self.context['mode']=='interaction' and self.context['ready'] and source
            and source['item']==self.turn_item_id and source['generation']==self.input_generation
            and source['connection'] is self.message_connection and self.user_turn_pending(state))

    def task_overview(self):
        if not self.tasks or not self.memory:return dict(enabled=False)
        if self.identity_enrichment_guarded():return dict(enabled=False,identity_pending=True)
        now=time.monotonic();person=self.identified_person()
        if self.task_cache is None or now>=self.task_refresh_at or self.task_cache_version!=self.memory.version or self.task_cache.get('current_person')!=person:
            fields=('task_id','kind','creator','recipient','status','not_before','due_at')
            fields+=('saved','delivery_status','delivery_label')
            def brief(task):
                view=message_view(task)
                return {k:view.get(k) for k in fields}
            self.task_cache=dict(enabled=True,current_person=person,
                identity_verified=person is not None,
                counts=self.tasks.counts(person) if person else {},
                active=[brief(t) for t in self.tasks.list(visible_to=person)] if person else [],
                due_task_inbox=[brief(t) for t in self.tasks.due(person)] if person else [])
            self.task_cache_version=self.memory.version;self.task_refresh_at=now+self.tasks.c['refresh_s']
        return self.task_cache

    def known_people(self):
        return [dict(person_id=pid,display_name=p['display_name'],preferred_address=p.get('preferred_address','')) for pid,p in self.knowledge.people.items()]

    def message_operation(self):
        lane=self.message_lane
        out=dict(phase='processing' if lane and lane.task and not lane.task.done() else 'idle',
            final_source_available=bool(self.pending_user_text))
        if lane and lane.result and getattr(lane,'result_actor',None)==self.identified_person():
            out.update({k:lane.result[k] for k in ('operation_status','saved') if k in lane.result})
        return out

    def message_request(self,*,automatic=False):
        person=self.identified_person();actor=person if automatic else self.turn_person
        generation=self.input_generation;revision=self.context['revision'];connection=self.message_connection
        def notice_guard():
            if (self.message_connection is not connection or self.input_generation!=generation or self.context['revision']!=revision
                    or not self.can_speak()):raise StaleMessageTurn()
            if self.voice_conflicts_with_actor(actor):raise StaleMessageTurn()
            current=self.identified_person()
            if current and actor and not actor.startswith('anonymous:') and current!=actor:raise StaleMessageTurn()
        def guard():
            notice_guard()
            if self.identified_person()!=person:raise StaleMessageTurn()
        recent=tuple(t['text'] for item,t in self.memory_turns.items()
            if item!=self.turn_item_id and t['person']==actor and t['session']==self.memory_session
            and t.get('connection') is connection and t.get('revision')==revision
            and t.get('valid') and t.get('engaged') and isinstance(t.get('text'),str) and 0<len(t['text'])<=500)[-4:]
        return MessageRequest(actor,self.memory_session,None if automatic else self.turn_item_id,
            None if automatic else self.turn_message_id,None if automatic else self.pending_user_text,person,guard=guard,notice_guard=notice_guard,read_only=self.recovered_query_generation==self.input_generation,
            recent_user_turns=() if automatic else recent)

    def message_valid(self,request):
        try:request.guard();return True
        except StaleMessageTurn:return False

    def messages(self):
        if self.message_lane is None:
            self.message_lane=MessageLane(self.tasks,self.knowledge.people,self.extract_message,self.speak_message,
                self.message_valid,self.log.write,courier=self.courier,notify=self.speak_message_notice,on_intent=self.message_intent_decided)
        return self.message_lane

    async def speak_message_notice(self,text,request):
        from dataclasses import replace
        if request.notice_guard is None:return False
        request.notice_guard()
        public=replace(request,actor=self.anonymous,verified_person=None,guard=request.notice_guard)
        return await self.speak_message(text,public)

    async def extract_message(self,text,people,hint):
        return await extract_qwen(self.url,self.secret,self.session,self.log,text,people,hint)

    async def speak_message(self,text,request):
        from vivis.speech.trusted_reading import RealtimeReader
        from vivis.speech.message_voice import MessageSpeaker
        engine=RealtimeReader(self.url,self.secret,self.session,self.log)
        started=False
        def spoken(text):
            nonlocal started
            started=True
            self.emit(dict(kind='cloud_transcript',epoch=self.epoch,role='assistant',text=text,source='trusted_message_tts'))
            if self.memory:self.memory.append(request.actor,request.session,'assistant',text,'audio_started_completion_unconfirmed')
        speaker=MessageSpeaker(engine,self.player,self.cloud_state,self.message_valid,
            spoken,self.log.write,tts=engine)
        try:return await speaker.say(text,request)
        except StaleMessageTurn:raise
        except Exception:
            self.emit(dict(kind='voice_warning',epoch=self.epoch,
                message=('本次留言语音未完整播出' if started or speaker.started else '本次留言朗读生成失败，尚未开始播报')+
                    '；数据库结果保留，不会因此重新建单或标记已转告。'))
            raise

    def ordinary_message_turn(self,event):
        import re
        return (event.get('type')=='conversation.item.input_audio_transcription.completed'
            and event.get('item_id')==self.turn_item_id and event.get('transcript','').strip() and self.tasks and self.memory and self.can_speak()
            and not re.search(r'取件码|快递员|(?:我来|上门).{0,6}取.{0,8}(?:快递|退件)',event.get('transcript',''))
            and courier_read_intent(event.get('transcript','')) is None
            and (MESSAGE_REQUEST.search(event.get('transcript','')) or
                (self.message_lane is not None and self.message_lane.accepts_followup(
                    self.identified_person() or self.anonymous,self.memory_session,event.get('transcript','')))))

    def final_message(self,event,state):
        if (event.get('type')!='conversation.item.input_audio_transcription.completed'
                or event.get('item_id')!=self.turn_item_id or not self.pending_user_text
                or not self.tasks or not self.memory or not self.can_speak()):return
        if self.final_courier_read(self.pending_user_text,state):return
        if not self.ordinary_message_turn(event):return
        self.message_receipts.expect_intent(self.turn_item_id)
        self.messages().start(self.message_request())

    def message_intent_decided(self,request,business):
        request.guard()
        self.message_receipts.decide_intent(request.source_item,business)
        if not business:return
        state=self.cloud_state
        self.message_claimed=self.input_generation;self.continue_response=False
        self.message_receipts.pending=None;self.announcement_revision=None
        state.interrupt(source='fixed_message_flow')
        for rid,generation in self.response_inputs.items():
            if generation==self.input_generation and rid not in self.blocked:self.blocked.append(rid)
        if state.generating:asyncio.create_task(self.cancel_inference(self.ws,state.generating))

    def final_courier_read(self,text,state):
        """Public pickup reads and owner visit queries have one final-ASR executor."""
        intent=courier_read_intent(text);previous=self.courier_query_pending
        if intent is None and previous and previous.get('generation',-2)+1==self.input_generation:
            if (previous.get('connection') is self.message_connection and previous.get('actor')==self.turn_person
                    and previous.get('revision')==self.context['revision']):
                from vivis.interaction.courier import phone_suffixes
                aliases=[a for p in self.knowledge.people.values() for a in (p['display_name'],p.get('preferred_address','')) if a]
                if phone_suffixes(text) or any(a in text for a in aliases):
                    intent='lookup';text=previous['source']+'。'+text
        if intent is None or not self.courier:return False
        if self.message_claimed==self.input_generation:return True
        request=self.message_request();lane=self.messages();lane.cancel()
        self.message_claimed=self.input_generation;self.continue_response=False
        self.message_receipts.pending=None;self.courier_query_pending=None
        state.interrupt(source='fixed_courier_read')
        for rid,generation in self.response_inputs.items():
            if generation==self.input_generation and rid not in self.blocked:self.blocked.append(rid)
        if state.generating:asyncio.create_task(self.cancel_inference(self.ws,state.generating))
        generation=self.input_generation
        async def read_and_speak():
            try:
                request.guard()
                if intent=='visit':
                    if not request.verified_person or request.verified_person!=request.actor:
                        await self.speak_message_notice('查询您委托的退件转告记录需要确认本人。来取码的快递员不需要认脸，只需提供委托人称呼或手机尾号。',request);return
                    result=self.courier.visit_query(request.actor)
                else:
                    result=self.courier.lookup('',text)
                request.guard()
                prompts=dict(need_owner='您来取谁的退件？请说委托人姓名、称呼，或手机后四位。',
                    not_found='没有找到与您提供信息匹配的有效退件码，请核对委托人称呼或手机后四位。',
                    ambiguous_owner='提供的信息对应多位委托人，请补充具体姓名。')
                spoken=result.get('spoken_receipt') or prompts[result['status']]
                if intent=='lookup' and result['status'] in ('need_owner','ambiguous_owner'):
                    self.courier_query_pending=dict(source=text,generation=generation,actor=request.actor,
                        connection=self.message_connection,revision=self.context['revision'])
                lane.result=dict(operation_status='read',saved=None,result=result);lane.result_actor=request.actor
                self.log.write('courier_fixed_read',action=intent,status=result['status'],source_item=request.source_item)
                complete=await self.speak_message(spoken,request)
                request.guard()
                if complete and result['status']=='available':
                    self.courier.mark_announced(result['handoff_id'],result['version'],'fixed-read:'+str(request.source_item))
            except (asyncio.CancelledError,StaleMessageTurn):pass
            except Exception as exc:
                self.log.write('courier_fixed_read_failed',error_type=type(exc).__name__)
                try:await self.speak_message_notice('这次退件查询或播报没有完成，已有取件码记录保持不变。',request)
                except (Exception,asyncio.CancelledError):pass
        lane.task=asyncio.create_task(read_and_speak())
        return True

    async def cancel_inference(self,ws,response_id):
        # The cancel is response-scoped locally; never cancel a newer answer
        # if it replaced the interrupted generation before this task ran.
        if ws is not self.ws or not self.cloud_state or self.cloud_state.generating!=response_id:return
        try:await bounded_send(ws,json.dumps(dict(type='response.cancel')))
        except Exception as exc:self.log.write('interrupted_response_cancel_failed',response_id=response_id,error_type=type(exc).__name__)

    def fresh_task_source(self,*,acknowledgement=False,clarification=False):
        if not self.turn_item_id or self.turn_ended_at is None or time.monotonic()-self.turn_ended_at>60:
            raise ValueError('Task changes need a fresh completed user utterance, not historical context')
        if self.relay_pending and not acknowledgement:raise ValueError('A relay is data to speak, not authority for creating or updating tasks')
        person=self.identified_person()
        if self.voice_conflicts_with_actor(self.turn_person):raise ValueError('Current voice conflicts with original actor; do not reassign this request')
        if not clarification and not self.turn_person.startswith('anonymous:') and person!=self.turn_person:
            raise ValueError('The confirmed speaker changed; ask for the task again instead of attributing the old turn to someone else')

    def check_task_delivery(self,state):
        self.protect_relay_audio(state)
        if not self.tasks or not self.memory:return
        for task_id,relay in list(self.relay_pending.items()):
            rid=relay.get('response_id');response=state.responses.get(rid,{})
            if self.identified_person()!=relay['recipient']:
                if rid is None or rid not in state.blocked:
                    state.interrupt(rid,source='relay_recipient_changed')
                if rid:self.blocked.append(rid)
                self.relay_identity_hold=True;self.continue_response=False
                self.log.write('robot_task_relay_cancelled',task_id=task_id,reason='recipient_unconfirmed_or_changed')
                self.relay_pending.pop(task_id,None);continue
            if relay['revision']!=self.context['revision'] or (rid and rid in state.blocked):
                self.relay_pending.pop(task_id,None);continue
            if self.audio_completed(state,rid):
                if relay_matches(relay['content'],response.get('transcript','')):
                    self.tasks.mark_announced(task_id,relay['recipient'],rid)
                    self.task_visit_announced.add(task_id)
                    self.log.write('robot_task_announced',task_id=task_id,recipient=relay['recipient'],response_id=rid,recipient_acknowledged=False)
                else:self.log.write('robot_task_relay_unverified',task_id=task_id,response_id=rid,reason='spoken_text_does_not_include_complete_message')
                self.relay_pending.pop(task_id,None)

    def audio_completed(self,state,rid):
        response=state.responses.get(rid,{})
        count=round(response.get('received_audio_s',0)*48000)
        return (count>0 and response.get('status')=='completed'
                and self.player.consumed.get(rid,0)>=count
                and response.get('empty_dac',float('inf'))<=time.monotonic())

    def protect_relay_audio(self,state):
        if not self.relay_recipients:return
        person=self.identified_person()
        for rid,recipient in list(self.relay_recipients.items()):
            response=state.responses.get(rid,{})
            no_audio=(response.get('status') in ('completed','cancelled','failed')
                      and not response.get('received_audio_s'))
            if rid in state.blocked or no_audio or self.audio_completed(state,rid):
                self.relay_recipients.pop(rid,None)
            elif person!=recipient:
                state.interrupt(rid,source='relay_recipient_changed')
                self.blocked.append(rid);self.relay_recipients.pop(rid,None)
                self.relay_identity_hold=True;self.continue_response=False
                self.log.write('relay_audio_blocked',response_id=rid,expected_person_id=recipient,person_id=person)

    def schedule_task_notice(self,state):
        if (not self.tasks or not self.memory or not self.tasks.c['auto_announce'] or not self.can_speak()
                or self.in_speech or state.generating or self.player.count or self.continue_response or self.response_request_pending
                or self.response_waiting_user_end or self.announcement_revision is not None
                or len(self.task_visit_announced)>=self.tasks.c['max_auto_per_visit']
                or self.identity_enrichment_guarded()):return
        person=self.identified_person()
        if not person:return
        lane=self.messages()
        if lane.task and not lane.task.done():return
        for task in self.tasks.due(person):
            if task['task_id'] in self.task_visit_announced:continue
            request=self.message_request(automatic=True)
            async def deliver(task_id=task['task_id'],request=request):
                await lane.relay(task_id,request)
                if self.tasks.get(task_id)['status']=='announced':self.task_visit_announced.add(task_id)
            lane.task=asyncio.create_task(deliver());return

    def response_busy(self,event,state):
        error=event.get('error',{})
        if (event.get('type')=='error' and error.get('code')=='invalid_value'
                and error.get('param')=='response.create'
                and error.get('message')=='Cannot create response while user is speaking.'
                and self.last_response_request_at is not None and 0<=time.monotonic()-self.last_response_request_at<5):
            self.response_request_pending=False;self.continue_response=False
            self.announcement_revision=None;self.response_waiting_user_end=True
            self.log.write('response_create_waiting_user_end',event_id=event.get('event_id'),commands_replayed=False)
            return True
        if (event.get('type')=='error' and error.get('code')=='invalid_value'
                and error.get('param')=='response.create'
                and error.get('message')=='Cannot create response while another response is in progress.'
                and self.last_response_request_at is not None and time.monotonic()-self.last_response_request_at<5):
            self.response_request_pending=state.generating is None;self.continue_response=False
            self.log.write('response_create_coalesced',event_id=event.get('event_id'),response_id=state.generating)
            return True
        return False

    def question_started(self,item_id,state):
        now=time.monotonic()
        latest_dac=max((r.get('empty_dac',-float('inf')) for r in state.responses.values()),default=-float('inf'))
        quiet=not self.player.response_ids(state.generating) and now-latest_dac>=1.
        outputs=set(self.player.response_ids(state.generating))
        outputs.update(rid for rid,r in state.responses.items() if now-r.get('empty_dac',-float('inf'))<1.)
        self.invalid_question=None
        self.recovered_expression=None
        self.question_onset=dict(item=item_id,generation=self.input_generation,revision=self.context['revision'],
            connection=self.message_connection,person=self.turn_person,quiet=quiet,outputs=outputs,
            visible_person=self.presence.greeting_visible(),
            conversation_presence=self.context['mode']=='interaction' and self.context['ready'] and self.presence.allowed(),
            preparation_presence=self.context['mode']=='interaction' and not self.context['ready'] and self.presence.allowed(),
            eligible=self.context['mode']=='interaction' and self.context['ready'] and self.presence.visible() and self.can_speak())

    def greeting_wake_allowed(self,require_ready=True):
        return (self.entry['enabled'] and self.entry['visible_greeting_wake']
                and self.context['mode'] in ('standby','interaction') and (self.context['ready'] or not require_ready)
                and (self.presence.greeting_visible() or
                     self.context['mode']=='interaction' and not self.context['ready'] and self.presence.allowed()))

    def continuing_greeting(self,source):
        # A current participant does not disappear on one missed camera frame.
        # Reuse the existing grace at BOTH onset and completion; never create
        # an empty-frame standby wake or refresh the grace without observation.
        return (source is not None and source.get('conversation_presence') is True
                and self.entry['enabled'] and self.entry['visible_greeting_wake']
                and self.context['mode']=='interaction' and self.presence.allowed())

    def verify_wake_source(self,greeting=True):
        """Model proposals need the same final-text/source gate as wake recovery."""
        from vivis.interaction.presence_entry import whole_greeting
        from vivis.speech.explicit_wake import addressed_wake
        source=self.question_onset
        if (not source or source['generation']!=self.input_generation
                or source['revision']!=self.context['revision'] or source['connection'] is not self.message_connection
                or (not greeting and source['person']!=(self.identified_person() or self.anonymous))
                or (greeting and source.get('visible_person') is not True and not self.continuing_greeting(source))):
            raise GreetingAdmissionError('greeting_source_not_current')
        turn=self.memory_turns.get(source['item'])
        text=turn.get('text') if turn else None
        if not text and source['item']==self.turn_item_id:text=self.pending_user_text
        if not text:raise GreetingAdmissionError('greeting_final_text_pending')
        ended=self.turn_ended_at
        if not ended and self.invalid_question and self.invalid_question['item']==source['item']:
            ended=self.invalid_question['ended']
        if self.in_speech or ended is None or time.monotonic()-ended>policy()['message_extract_s']:
            raise GreetingAdmissionError('wake_final_turn_pending_or_expired')
        if greeting and not whole_greeting(text,self.entry):raise GreetingAdmissionError('not_a_direct_greeting')
        if not greeting and not addressed_wake(text,self.wake_words):raise GreetingAdmissionError('not_a_direct_name_call')
        if not source['quiet']:
            from vivis.speech.invalid_question import distinct_from_playback
            state=self.cloud_state
            transcripts=([state.responses.get(r,{}).get('transcript') for r in source['outputs']] if state else [])
            if not distinct_from_playback(text,transcripts):raise GreetingAdmissionError('greeting_echo_not_excluded')

    def admit_explicit_wake(self,event,state):
        from vivis.speech.explicit_wake import addressed_wake
        from vivis.interaction.presence_entry import whole_greeting
        from vivis.speech.invalid_question import distinct_from_playback
        if self.context.get('standby_fault'):return False
        item=event.get('item_id');text=event.get('transcript');source=self.question_onset
        named=addressed_wake(text,self.wake_words)
        greeting=(whole_greeting(text,self.entry) and (
            self.greeting_wake_allowed(require_ready=self.context['mode']!='interaction') and source is not None
            and (source.get('visible_person') is True or source.get('preparation_presence') is True)
            or self.continuing_greeting(source)))
        if (not (named or greeting) or item in self.explicit_wake_seen
                or not source or item!=source['item'] or self.context['mode'] not in ('standby','interaction')
                or source['generation']!=self.input_generation or source['revision']!=self.context['revision']
                or source['connection'] is not self.message_connection
                or (not greeting and source['person']!=(self.identified_person() or self.anonymous)) or self.in_speech):return False
        turn=self.memory_turns.get(item);ended=self.turn_ended_at
        if not ended and self.invalid_question and self.invalid_question['item']==item:ended=self.invalid_question['ended']
        if not turn or ended is None or time.monotonic()-ended>policy()['message_extract_s']:return False
        outputs=source.get('outputs',set())
        if not source['quiet'] and not distinct_from_playback(text,[state.responses.get(r,{}).get('transcript') for r in outputs]):return False
        if set(self.player.response_ids(state.generating))-outputs and any(
                self.response_inputs.get(r)!=self.input_generation for r in self.player.response_ids(state.generating)):return False
        self.explicit_wake_seen.append(item)
        already_audible=self.can_speak() and self.cloud_audio_enabled and turn['valid']
        if named:self.presence.wake()
        elif greeting:
            self.presence.greet();self.reception.dismiss_current_visit()
        self.turn_engaged=True;turn['engaged']=True;turn['valid']=True
        self.turn_item_id=item;self.turn_person=source['person'];self.turn_ended_at=ended;self.pending_user_text=text
        self.invalid_question=None
        self.log.write('explicit_wake_permission',item_id=item,mode=self.context['mode'],
            audio_recovery_required=not already_audible,face_visible=self.presence.visible(),
            wake_reason='name' if named else 'greeting')
        if already_audible:return False
        self.explicit_wake_pending=dict(source,text=text,ended=ended,wake_reason='name' if named else 'greeting',
                                       admitted_monotonic=time.monotonic())
        self.announcement_revision=None;self.continue_response=False
        for rid,generation in self.response_inputs.items():
            if generation==self.input_generation:
                state.interrupt(rid,source='explicit_wake_replace_silent_reply')
                if rid not in self.blocked:self.blocked.append(rid)
        needs_audio_handoff=(self.context['mode']=='standby' or
            (self.context['mode']=='interaction' and not self.context.get('speech_ready',self.context['ready'])))
        if needs_audio_handoff and not self.control_pending:
            self.control_pending=True
            self.emit(dict(kind='cloud_action',revision=self.context['revision'],
                arguments=dict(action='wake',program='',explicit_wake=named,
                               **({'wake_reason':'greeting'} if not named else {})),text='直接招呼：'+text))
        return True

    def explicit_wake_current(self,source,require_presence=True):
        return (self.explicit_wake_pending is source and source['generation']==self.input_generation
            and source['connection'] is self.message_connection and source['revision']==self.context['revision']
            and (source.get('wake_reason')=='greeting' or source['person']==(self.identified_person() or self.anonymous))
            and self.turn_item_id==source['item']
            and (not require_presence or source.get('wake_reason')!='greeting'
                 or not self.context.get('speech_ready',self.context['ready'])
                 or self.greeting_wake_allowed(require_ready=False) or self.continuing_greeting(source)))

    async def complete_explicit_wake(self,ws,state,controller_events,audio_gate,instructions):
        source=self.explicit_wake_pending
        if source is None:return False
        if not self.explicit_wake_current(source,require_presence=False):
            self.explicit_wake_pending=None;return False
        if not self.output_ready() or self.in_speech:return False
        if not self.explicit_wake_current(source):
            if not self.context['ready']:
                # The old camera may have closed before the new worker's first
                # body result. Keep the admitted turn during bounded warmup;
                # do not turn temporary missing evidence into either a grant
                # or a cancelled greeting. Full-ready still uses its old gate.
                return False
            now=time.monotonic()
            if self.presence.greeting_detection_pending():
                if 'presence_wait_started' not in source:
                    self.log.write('explicit_wake_waiting_body',item_id=source['item'],budget_s=self.presence.c['max_age_s'])
                started=source.setdefault('presence_wait_started',now)
                if now-started<self.presence.c['max_age_s']:return False
            self.log.write('explicit_wake_cancelled',item_id=source['item'],reason='greeting_presence_not_confirmed_after_prepare')
            self.explicit_wake_pending=None;return False
        if source.get('wake_reason')!='greeting':self.presence.wake()
        else:self.presence.greet()
        if state.generating:await self.cancel_inference(ws,state.generating)
        await audio_gate.enable(ws,instructions)
        if not self.explicit_wake_current(source) or self.in_speech:return False
        self.cloud_audio_enabled=True
        # Handoff intentionally resets business execution state. Restore only
        # this verified FINAL user text before creating its continuation, so a
        # normal hello cannot be stranded in await_source and treated as a DB
        # receipt. Do not re-deliver/log ASR or restore old business results.
        final_event=dict(type='conversation.item.input_audio_transcription.completed',item_id=source['item'],transcript=source['text'])
        self.message_receipts.source_turns[source['item']]=self.input_generation
        restored=self.message_receipts.filter(final_event)
        self.message_receipts.released.extend(e for e in restored if e.get('type')!='conversation.item.input_audio_transcription.completed')
        if (state.current_user_turn or {}).get('item_id')!=source['item']:
            # Semantic VAD can reject a genuine greeting. Its validated FINAL
            # wake must bind the reply to this item, never the previous turn.
            state.current_user_turn=dict(item_id=source['item'],confirmed_end_monotonic=source['ended'],
                endpoint_definition='validated_final_wake_after_semantic_rejection_not_physical_voice_end')
        directive=('请自然回应访客这句原话：'+source['text']+
            '。只是招呼时简短答应，例如“您好，我在呢”；有问题就直接回答。不要复述任何内部控制状态，'
            '不要重复自我介绍，也不要要求再次唤醒。这不代表熟人身份核验。')
        if not self.context['ready']:
            directive+='动作尚未就绪，不要声称已执行；无需向访客解释准备过程。'
        await controller_events.append(ws,self.context,'explicit_wake',directive)
        if not self.explicit_wake_current(source) or self.in_speech:return False
        self.explicit_wake_pending=None;self.announcement_revision=None
        self.final_message(final_event,state)
        self.continue_response=self.message_claimed!=self.input_generation
        now=time.monotonic();admitted=source['admitted_monotonic'];ready=source.get('controller_ready_monotonic')
        self.log.write('explicit_wake_audio_ready',item_id=source['item'],original_question_preserved=True,
            before_motion_ready=not self.context['ready'],
            final_source_restored=True,wake_to_audio_permission_s=now-admitted,
            wake_to_controller_ready_s=None if ready is None else ready-admitted,
            controller_ready_to_audio_permission_s=None if ready is None else now-ready)
        self.emit(dict(kind='voice_warning',epoch=self.epoch,message='已收到明确唤醒，说话许可和云端语音输出已恢复。'))
        return True

    def question_invalidated(self,event):
        source=self.question_onset
        if source and source['item']==event.get('item_id'):
            self.invalid_question=dict(source,ended=time.monotonic())

    def recover_invalid_question(self,event,state):
        from vivis.speech.invalid_question import complete_question,distinct_from_playback,explicit_expression_request
        source=self.invalid_question
        if not source or source['item']!=event.get('item_id'):return False
        self.invalid_question=None  # One decision per final item; duplicates never replay it.
        expression=explicit_expression_request(event.get('transcript'))
        self.log.write('invalid_final_transcript',item_id=source['item'],
            expression_candidate=expression,question_candidate=complete_question(event.get('transcript')))
        person=self.identified_person() or self.anonymous
        active=set(self.player.response_ids(state.generating))
        outputs=source.get('outputs',set())
        barge_in=(not source['quiet'] and not active-outputs and
            distinct_from_playback(event.get('transcript',''),[state.responses.get(rid,{}).get('transcript') for rid in outputs]))
        if (not source['eligible'] or not (source['quiet'] or barge_in)
                or not (complete_question(event.get('transcript')) or expression)
                or (expression and not source['quiet'])
                or source['generation']!=self.input_generation or source['revision']!=self.context['revision']
                or source['connection'] is not self.message_connection or source['person']!=person
                or time.monotonic()-source['ended']>policy()['message_extract_s']
                or self.in_speech or (active and not barge_in)
                or not self.presence.visible() or not self.can_speak()):return False
        turn=self.memory_turns.get(source['item'])
        if not turn or turn['person']!=source['person']:return False
        if barge_in:
            for rid in outputs:
                state.interrupt(rid,source='final_question_barge_in')
                if rid not in self.blocked:self.blocked.append(rid)
        turn['valid']=True
        self.turn_item_id=source['item'];self.turn_person=source['person'];self.turn_ended_at=source['ended']
        self.recovered_query_generation=self.input_generation
        if expression:self.recovered_expression=dict(source,consumed=False)
        self.pending_user_text=event['transcript']
        state.current_user_turn=dict(item_id=source['item'],confirmed_end_monotonic=source['ended'])
        state.speech_stopped_at=source['ended']
        self.announcement_revision=None;self.continue_response=False
        self.log.write('invalid_expression_recovered' if expression else 'invalid_question_recovered',
            item_id=source['item'],read_only=not expression,allowed_action='set_expression_once' if expression else None,
            reason='explicit_expression_while_robot_quiet' if expression else
                'final_question_barge_in' if barge_in else 'final_question_while_robot_quiet',person_id=person)
        return True

    async def ask_recovered_question(self,ws,controller_events,state,request):
        try:
            request.notice_guard()
            if self.message_claimed==self.input_generation:return
            if state.generating in state.blocked:await self.cancel_inference(ws,state.generating)
            directive=('云端最终识别到当前访客的完整问句，但语义轮次检测误判为无效。原话：'+request.source_text+
                '。这是原话轮的只读恢复，不是新授权。请回答或调用只读查询；不保存、修改、取消留言，不启动节目或动作。')
            expression=self.recovered_expression
            if expression:
                directive=('云端最终识别到当前访客的明确表情请求，但语义轮次检测误判为无效。原话：'+request.source_text+
                    '。请理解该请求，只通过set_expression执行一次对应表情，再依据执行结果简短回答。'
                    '仍使用现有标定表情及强度、时长限制；不要办理留言、切换节目或调用其他动作。纯笑声不是动作指令。')
            await controller_events.append(ws,self.context,'recovered_expression' if expression else 'recovered_question',directive)
            request.notice_guard()
            if not self.in_speech and (state.generating is None or state.generating in state.blocked) and self.message_claimed!=self.input_generation:
                self.continue_response=True
        except (asyncio.CancelledError,StaleMessageTurn):pass
        except Exception as exc:
            self.log.write('invalid_question_recovery_failed',error_type=type(exc).__name__)
            self.emit(dict(kind='voice_warning',epoch=self.epoch,message='这句请求未能恢复，请再说一次。'))

    def begin_user_turn(self,item_id):
        if self.identity_fusion:self.admit_voice_identity(previous_turn=True)
        self.explicit_wake_pending=None
        self.program_followup=None;self.followup_response=None
        if self.program_check_task and not self.program_check_task.done():self.program_check_task.cancel()
        if self.farewell and self.farewell.get('source')=='program_interrupt':self.cancel_farewell('new_user_turn')
        self.reception.conversation.user_started()
        if self.message_lane:
            actor=self.identified_person() or self.anonymous
            self.message_lane.next_turn(actor,self.memory_session)
            if self.message_lane.has_continuation(actor,self.memory_session):
                self.message_receipts.expect_details(self.input_generation-1,'message_followup')
        self.turn_item_id=item_id;self.turn_message_id=None;self.turn_ended_at=None
        self.turn_person=self.identified_person() or self.anonymous
        self.turn_saved=False;self.pending_user_text=None;self.turn_engaged=self.can_speak()
        if isinstance(item_id,str) and item_id:
            self.memory_turns[item_id]=dict(person=self.turn_person,session=self.memory_session,
                engaged=self.turn_engaged,reply_permission_at_onset=self.turn_engaged,generation=self.input_generation,
                valid=True,text=None,message_id=None,connection=self.message_connection,revision=self.context['revision'])
            while len(self.memory_turns)>128:self.memory_turns.pop(next(iter(self.memory_turns)))
        self.finish_identity_turn_start()

    def remember(self,event):
        if not self.memory:return
        kind=event.get('type');rid=event.get('response_id')
        current=self.memory_turns.get(self.turn_item_id)
        if current:current['engaged']|=self.turn_engaged
        candidates={self.turn_item_id} if current else set()
        if kind=='conversation.item.input_audio_transcription.completed':
            item_id=event.get('item_id');turn=self.memory_turns.get(item_id)
            if turn:
                turn['text']=event.get('transcript','');candidates.add(item_id)
                if item_id==self.turn_item_id:self.pending_user_text=turn['text'];self.unattributed_source_notice=False
            else:
                self.log.write('memory_transcript_unattributed',item_id=item_id,reason='unknown_user_turn',
                    event_fields=sorted(event),previous_item_id=event.get('previous_item_id'),
                    current_item_id=self.turn_item_id,not_assigned_to_current=True)
                # An orphaned short acknowledgement is not a business request.
                # Keep it unattributed without asking for an unrelated pickup code.
                from vivis.speech.message_receipts import normalized
                if normalized(event.get('transcript','')) in ('是','嗯','啊','对','好','好的','哦','噢','呃','这','这这'):return
                if not MESSAGE_REQUEST.search(event.get('transcript','')) and not self.message_receipts.awaiting_details:
                    return  # Unattributed ordinary speech is not a failed message operation.
                self.unattributed_source_notice=True
                self.emit(dict(kind='voice_warning',epoch=self.epoch,
                    message='收到语音片段，但无法核验它属于哪个话轮；未据此办理业务。请将取件码或留言连成一句再说。'))
        for item_id in candidates:
            turn=self.memory_turns[item_id];text=turn['text']
            if turn['valid'] and turn['engaged'] and isinstance(text,str) and text.strip() and turn['message_id'] is None:
                if self.continuity and self.continuity.admissions.get((turn['session'],item_id)) is not True:
                    continue  # No unbounded legacy copy while the new journal is paused.
                mid=self.memory.append(turn['person'],turn['session'],'user',text);turn['message_id']=mid
                if self.tasks:self.tasks.bind_source(turn['session'],item_id,mid)
                if item_id==self.turn_item_id:self.turn_message_id=mid;self.turn_saved=True
        if kind=='response.audio_transcript.done' and rid and rid not in self.blocked:
            self.pending_memories[rid]=(self.reply_people.get(rid,self.anonymous),event.get('transcript',''))
        for key,(person,text) in list(self.pending_memories.items()):
            if key in self.player.started_responses:
                response=self.cloud_state.responses.get(key,{}) if self.cloud_state else {}
                source=None if response.get('archive_no_user') else (response.get('user_turn') or {}).get('item_id')
                turn=self.memory_turns.get(source,{})
                admitted=not self.continuity or self.continuity.admissions.get((turn.get('session'),source)) is True
                if text.strip() and admitted:self.memory.append(person,turn.get('session',self.memory_session),'assistant',text,'audio_started_completion_unconfirmed')
                del self.pending_memories[key]
        while len(self.pending_memories)>64:self.pending_memories.pop(next(iter(self.pending_memories)))

    async def compact_memory(self):
        try:
            person=await asyncio.to_thread(self.memory.pending)
            if not person:return
            batch=await asyncio.to_thread(self.memory.batch,person)
            if not batch:return
            from vivis.speech.memory_compaction import summarize
            result=await summarize(self.url,self.secret,self.session,batch,self.memory_config['compression_timeout_s'])
            if self.context['mode']=='standby':
                self.memory.compact(batch,result)
                self.log.write('memory_compacted',person_id=person,message_count=len(batch['messages']),raw_archive_retained=True)
        except asyncio.CancelledError:raise
        except Exception as exc:self.log.write('memory_compaction_deferred',error=type(exc).__name__)
        finally:self.memory_retry_at=time.monotonic()+60

    def context_snapshot(self):
        confirmed=self.identified_person()
        review=self.reviewing_identity()
        identity_pending=self.identity_enrichment_guarded()
        if identity_pending:confirmed=None;review=None
        identity=self.knowledge.person_context(allow_identification=confirmed is not None,allow_review=review is not None,
            retained_person_id=confirmed if review else None,stable_identity=self.stable_identity())
        permission=self.presence.allowed()
        facts=dict(**self.context,unattributed_source_notice=self.unattributed_source_notice,face_visible=self.presence.last_visible,
            wake_permission=permission and not self.presence.last_visible)
        facts['current_turn_reply_permission']=self.turn_reply_pending()
        facts['sound_attention']=self.sound_status()
        introduction=getattr(self,'introduced_guest',None)
        if introduction and introduction['session']==self.memory_session:
            visitor=(getattr(self,'sound_mailbox',None).latest or {}).get('visitor') if getattr(self,'sound_mailbox',None) else None
            if visitor and visitor.get('token')==introduction['guest']['token'] and visitor.get('owner')==introduction['guest']['owner']:
                facts['introduced_visitor']=dict(information=introduction['information'],
                    notice='熟人刚介绍的当前访客，可用介绍中的称呼交流；不是新的图像或声纹确认，不额外授予私人业务权限。')
        facts['body_visible']=self.presence.person_count()>0
        if self.identity_fusion:
            from vivis.interaction.identity_fusion import recognition_report
            # Make current channel evidence available even when the model skips
            # the status tool. A retained address is not a fresh voice match.
            status=self.speaker_status(confirmed)
            facts['identity_recognition']=recognition_report(confirmed,self.knowledge.people,
                self.fused_face_evidence(),status.get('current_result',{}),
                (self.fusion_last or {}).get('reason'))
        if self.continuity:
            from vivis.interaction.continuity import actor
            facts['recent_memory_paused']=actor(confirmed) in self.continuity.blocked_people
            facts['conversation_recording']=self.continuity.finalized.get((self.memory_session,self.turn_item_id),dict(saved=False,status='pending_or_unavailable'))
            if facts['recent_memory_paused']:
                facts['memory_notice']='近期记忆已满且旧记录尚未整理。本轮不保存新聊天记忆，不能声称记住了；请简短说明，但留言仍按独立账本执行。'
        facts['program_followup_handoff']=self.followup_handoff
        facts['identity_state']='reviewing' if review else 'recognized' if confirmed else 'unknown'
        facts['identity_authorized']=confirmed is not None
        facts['identity_enrichment_pending']=identity_pending
        facts['message_operation']=self.message_operation()
        invitation=self.public_prompt_context
        if (invitation and invitation['connection'] is self.message_connection
                and invitation['session']==self.memory_session
                and self.input_generation<=invitation['generation']+1):
            facts['spoken_public_invitation']={key:invitation[key] for key in ('text','playback_state')}
        local_zone=self.tasks.c['timezone'] if self.tasks and self.memory else 'Asia/Shanghai'
        local_time=datetime.now(ZoneInfo(local_zone))
        facts['current_local_time']=local_time.isoformat(timespec='minutes')
        facts['current_weekday']='星期'+'一二三四五六日'[local_time.weekday()]
        facts['timezone']=local_zone
        if self.tasks and self.memory:
            facts['robot_tasks']=self.task_overview()
            facts['known_people']=self.known_people() if confirmed else []
        observation=None
        try:
            observation=json.loads(self.presence.path.read_text())
            expression=observation.get('expression') if isinstance(observation,dict) else None
            if (isinstance(observation,dict) and self.presence.visible() and observation.get('faces')==1 and isinstance(expression,dict)
                    and expression.get('observation_only') is True
                    and type(expression.get('bilateral_smile')) in (float,int)
                    and 0<=expression['bilateral_smile']<=1):
                facts['visual_expression_observation']=dict(bilateral_smile=expression['bilateral_smile'],observation_only=True)
        except (OSError,ValueError):pass
        social_active=self.context['mode']=='interaction' and self.context['ready']
        cue=self.social.observe(observation,time.monotonic(),(self.memory_session,confirmed or self.anonymous),active=social_active)
        if social_active and (self.social.c['tone_enabled'] or self.social.c['topics_enabled']):
            if self.social.c['tone_enabled']:facts['visual_social_cue']=cue
            if cue!=self.last_social_cue:
                self.last_social_cue=dict(cue)
                self.log.write('social_speech_cue_changed',**cue,tone_enabled=self.social.c['tone_enabled'])
        if self.expression_outcome:facts['expression_outcome']=self.expression_outcome
        facts['greeting_wake_allowed']=self.greeting_wake_allowed()
        instructions=session_with_program_context(self.session,self.last_program)['instructions']
        from vivis.interaction.presence_entry import instructions as entry_instructions
        instructions+=CLOUD_POLICY+SOUND_POLICY+entry_instructions(self.entry)+WORK_POLICY+identity['instructions']+' controller_state='+json.dumps(facts,ensure_ascii=False)
        person=confirmed
        if self.memory and person:instructions+=self.memory.context(person)
        if getattr(self,'continuity',None):
            instructions+=self.archive_context_cache.get(person,'')
            instructions+=' 管理员明确要求整理档案时调用organize_archives。整理只在待机执行，必须由程序确认提交后才能说档案更新完毕。'
            instructions+=' 聊天记录由程序后台保存。只有controller_state.conversation_recording.saved=true才可声称本轮聊天已保存；否则自然回应内容，不承诺永久记住。留言仍以独立账本回执为准。'
        instructions+=self.dialect.instructions()
        if social_active:instructions+=self.social.instructions()
        if self.program_turn_only():instructions+=PROGRAM_TURN_POLICY
        return instructions,identity,permission

    def program_turn_only(self):
        return self.context['mode']=='task' or (self.followup_handoff and not self.context['ready'])

    def available_tools(self):
        if self.program_turn_only():
            return [tool for tool in self.session['tools'] if tool['name'] in ('robot_program','program_followup')]
        return [tool for tool in self.session['tools'] if tool['name']!='program_followup']

    def response_instructions(self,directive):
        # response.create.instructions REPLACES session instructions. Always
        # retain the current canonical persona, knowledge and local date.
        return self.context_snapshot()[0]+'\n本次回复的具体要求：'+directive

    def observe_program_response(self,event):
        kind=event.get('type');rid=event.get('response_id') or event.get('response',{}).get('id')
        if kind=='response.created' and self.context['mode']=='task':
            self.program_responses[rid]=dict(text=False,tool=False,completed=False,checked=False)
            while len(self.program_responses)>128:self.program_responses.pop(next(iter(self.program_responses)))
        record=self.program_responses.get(rid)
        if record is None:return
        if kind in ('response.text.done','response.audio_transcript.done'):
            record['text']=bool(event.get('text') or event.get('transcript'))
        if kind=='response.function_call_arguments.done':record['tool']=True
        if kind=='response.done':record['completed']=event['response'].get('status')=='completed'

    def schedule_program_check(self,state):
        """One bounded semantic check for an unhandled direct answer, never a keyword action."""
        if self.program_check_task and not self.program_check_task.done():return
        source=self.question_onset
        if (self.context['mode']!='task' or self.control_pending or self.followup_handoff or self.program_followup
                or not source or source.get('program_intent_checked') or not source.get('final_text')
                or self.turn_ended_at is None or self.in_speech):return
        for rid,record in reversed(self.program_responses.items()):
            response=state.responses.get(rid,{})
            if (record['checked'] or not record['completed'] or self.response_inputs.get(rid)!=self.input_generation
                    or (response.get('user_turn') or {}).get('item_id')!=self.turn_item_id):continue
            record['checked']=True
            if record['tool'] or not record['text']:continue
            request=dict(source,session=self.memory_session,ended=self.turn_ended_at)
            if not self.followup_current(request):return
            source['program_intent_checked']=True
            async def check(request=request,rid=rid):
                from vivis.speech.program_followup import classify
                try:
                    result=await classify(self.url,self.secret,self.session,request['final_text'],self.log,
                        max(.1,policy()['message_extract_s']-(time.monotonic()-request['ended'])))
                    if (not self.followup_current(request) or self.context['revision']!=request['revision']
                            or self.context['mode']!='task' or self.control_pending or self.program_followup):return
                    self.log.write('program_followup_tool_omission_checked',item_id=request['item'],decision=result)
                    if result!='ignore':self.stage_program_followup(dict(kind=result),rid)
                except asyncio.CancelledError:raise
                except Exception as exc:self.log.write('program_followup_check_failed',error_type=type(exc).__name__)
            self.program_check_task=asyncio.create_task(check())
            return

    def continuing_program_turn(self):
        return self.followup_handoff or any(request and self.followup_current(request)
            for request in (self.program_followup,self.followup_response))

    def followup_current(self,request):
        now=time.monotonic()
        if (request['generation']!=self.input_generation or request['item']!=self.turn_item_id
                or request['connection'] is not self.message_connection or request['session']!=self.memory_session
                or self.in_speech or not 0<=now-request['ended']<=policy()['message_extract_s']):return False
        if not (self.presence.single_visible() or (not self.presence.greeting_visible() and now<self.presence.wake_until)):
            return False
        raw_person=self.recognized_person()
        # A fresh different identity overrides the old greeting's eight-second
        # display hold for an action waiting to cross the program boundary.
        if request['person']==self.anonymous:
            # Recognition can enrich a still-current public anonymous turn.
            # Never relabel its stored speaker or grant it private authority.
            return raw_person is None or raw_person==self.identified_person()
        return (raw_person is None or raw_person==request['person']) and request['person']==self.identified_person()

    def stage_program_followup(self,args,rid,*,native_stop_review=False):
        source=self.question_onset
        response=self.cloud_state.responses.get(rid,{}) if self.cloud_state else {}
        if (not isinstance(args,dict) or set(args)!={'kind'} or args['kind'] not in ('question','expression','change_song')
                or self.context['mode']!='task' and not self.followup_handoff
                or (self.control_pending and not self.followup_handoff and not native_stop_review) or self.program_followup is not None or not source
                or self.response_inputs.get(rid)!=self.input_generation
                or (response.get('user_turn') or {}).get('item_id')!=self.turn_item_id
                or source['revision']!=self.context['revision'] or self.turn_ended_at is None):
            raise ValueError('没有可接续的当前节目用户话轮。')
        request=dict(source,session=self.memory_session,ended=self.turn_ended_at,kind=args['kind'],accepted=False,
            old_program=self.context['target'])
        if not self.followup_current(request):raise ValueError('来访者或话轮已变化。')
        if native_stop_review:
            request.update(kind='control_review',stop_already_requested=True)
            self.followup_handoff=True
        if self.followup_handoff and self.control_pending and self.context['mode']=='task':request['stop_already_requested']=True
        self.program_followup=request
        self.accept_program_followup(source.get('final_text'))
        if self.program_followup is None:raise ValueError('本轮最终识别未提供有效的接续请求。')
        return dict(accepted=True,status='pending_program_followup')

    def review_stopped_request(self,request):
        """Stopping stays immediate. Review its original goal concurrently with handoff."""
        if request.get('review_started'):return
        request['review_started']=True
        if self.program_check_task and not self.program_check_task.done():self.program_check_task.cancel()
        async def check():
            from vivis.speech.program_followup import classify
            try:
                result=await classify(self.url,self.secret,self.session,request['text'],self.log,
                    max(.1,policy()['message_extract_s']-(time.monotonic()-request['ended'])))
                if self.program_followup is not request or not self.followup_current(request):return
                self.log.write('stopped_request_reviewed',item_id=request['item'],decision=result)
                if result=='ignore':self.program_followup=None;self.followup_handoff=False
                else:request['kind']=result
            except asyncio.CancelledError:raise
            except Exception as exc:
                if self.program_followup is request:self.program_followup=None;self.followup_handoff=False
                self.log.write('stopped_request_review_failed',error_type=type(exc).__name__)
        self.program_check_task=asyncio.create_task(check())

    def accept_program_followup(self,text):
        request=self.program_followup
        if request is None or request['accepted'] or text is None:return
        if not isinstance(text,str) or not 1<=len(text.strip())<=500 or not self.followup_current(request):
            self.program_followup=None
            return
        from vivis.speech.message_receipts import normalized
        from vivis.interaction.program_source import veto
        compact=normalized(text)
        quoted=compact.startswith(('歌词写的是','歌词里说','歌词唱的是','剧本里说','剧本台词是','台词说的是'))
        incidental=compact in ('嗯','啊','哈','对','好','是','继续','不要停','不用停','不要停止','不用停止','不退出','不要退出')
        action='play' if request['kind']=='change_song' else 'stop'
        if not compact or quoted or incidental or veto(text,action) or all(c in '哈呵嘿啊哦呃嗯' for c in compact):
            self.program_followup=None
            return
        request.update(text=text,accepted=True)
        self.announcement_revision=None
        self.log.write('program_followup_accepted',item_id=request['item'],kind_of_request=request['kind'])
        if self.context['mode']=='task' and not request.get('stop_already_requested'):
            from vivis.speech.fixed_replies import interruption_reply
            self.followup_handoff=True;self.control_pending=True
            self.cloud_state.interrupt(source='program_followup_requested')
            self.cancel_farewell('replaced')
            self.farewell=dict(id=uuid.uuid4().hex,generation=self.input_generation,connection=self.message_connection,
                target='interaction',text=interruption_reply(),source='program_interrupt')
            self.emit(dict(kind='cloud_action',revision=self.context['revision'],arguments=dict(action='stop'),
                text='节目中接续用户请求',transition_reply_id=self.farewell['id']))

    async def resume_program_followup(self,ws,controller_events,state):
        request=self.program_followup
        if request is None:return
        if not self.followup_current(request):
            self.program_followup=None
            self.log.write('program_followup_skipped',reason='stale_source',item_id=request['item'])
            return
        if request['accepted'] and request['kind']=='control_review':
            self.review_stopped_request(request);return
        if (not request['accepted'] or self.context['mode']!='interaction' or not self.context['ready']
                or not self.can_speak() or state.generating is not None or self.response_request_pending):return
        self.announcement_revision=None;self.continue_response=False
        if request['kind']=='change_song':
            from vivis.interaction.cloud_control import next_song
            program=next_song(self.programs,request['old_program'])
            if program:
                self.program_followup=None;self.followup_handoff=False;self.control_pending=True
                self.emit(dict(kind='cloud_action',revision=self.context['revision'],
                    arguments=dict(action='play',program=program),text='接续换歌请求'))
                self.log.write('program_followup_song_changed',item_id=request['item'],program=program)
                return
            request['kind']='question'
        directive=('旧节目已退出。接续这一条用户原话，只回答问题，可以查机器人状态，不得点播、启动动作或读写私人业务。'
            if request['kind']=='question' else
            '旧节目已退出。接续这一条用户原话，只能用set_expression(source=command)做一次明确要求的表情，随后按控制结果回应。')
        directive+=' 本轮原话是 '+json.dumps(request['text'],ensure_ascii=False)+self.dialect.instructions()
        # This final ASR turn already exists on the server. A synthetic user
        # item/ack is unnecessary and used to lose the request when removing
        # the short-reply ID advanced the controller revision during the wait.
        revision=self.context['revision']
        self.program_followup=None;self.followup_handoff=False
        # Context changes clear receipt provenance. Restore only the already
        # verified original turn. This never re-enters the business dispatcher.
        self.message_receipts.filter(dict(type='conversation.item.input_audio_transcription.completed',
            item_id=request['item'],transcript=request['text']))
        if MESSAGE_REQUEST.search(request['text']):self.message_receipts.decide_intent(request['item'],False)
        self.followup_response=dict(request,revision=revision,expression_used=False)
        self.response_request_pending=True;self.last_response_request_at=time.monotonic()
        await bounded_send(ws,json.dumps(dict(type='response.create',response=dict(modalities=['text','audio'],
            instructions=self.response_instructions(directive))),ensure_ascii=False))
        self.log.write('program_followup_resumed',item_id=request['item'],kind_of_request=request['kind'])

    def public_prompt_available(self,state):
        latest_dac=max((r.get('empty_dac',0) for r in state.responses.values()),default=0)
        return (not self.sound_frozen() and self.context['ready'] and self.presence.single_visible() and self.can_speak()
                and not self.awaiting_recovery_user and not self.user_turn_pending(state) and state.generating is None
                and not self.player.response_ids() and time.monotonic()>=latest_dac and not self.finish_responses
                and not self.program_followup and not (self.message_lane and self.message_lane.task and not self.message_lane.task.done())
                and not self.relay_pending and not self.courier_offer)

    def cancel_public_prompt(self):
        if self.public_prompt_task and not self.public_prompt_task.done():self.public_prompt_task.cancel()

    def schedule_public_prompt(self,state,*,welcome):
        from vivis.interaction.reception_conversation import public_invitation
        from vivis.speech.message_voice import MessageSpeaker
        from vivis.speech.trusted_reading import RealtimeReader
        conversation=self.reception.conversation
        if not welcome:conversation.last_attempt=time.monotonic()
        self.public_prompt_started=False
        connection=self.message_connection;generation=self.input_generation;revision=self.context['revision']
        session=self.memory_session;person=self.recognized_person()
        address=self.knowledge.people.get(person,{}).get('preferred_address') if person else None
        observed_style=None
        def current():
            return (not self.sound_frozen() and self.cloud_state is state and connection is self.message_connection and session==self.memory_session
                    and (self.public_prompt_started or (generation==self.input_generation and not self.in_speech))
                    and revision==self.context['revision']
                    and not conversation.declined and not conversation.paused and self.context['ready'] and self.can_speak()
                    and (self.public_prompt_started or self.presence.single_visible()
                         and (person is None or person==self.recognized_person()))
                    and (self.public_prompt_started or observed_style is None
                         or self.social.cue['response_style'] in ('neutral',observed_style))
                    and not (self.message_lane and self.message_lane.task and not self.message_lane.task.done()))
        async def speak():
            nonlocal observed_style
            pose_id=None
            try:
                turns=()
                verified_person=person if person==self.identified_person() else None
                if verified_person and self.memory and (welcome or self.social.c['topics_enabled']):
                    try:turns=(await asyncio.to_thread(self.memory.search,person,[],session))['matches']
                    except sqlite3.Error:self.log.write('public_prompt_history_unavailable')
                if not current():return
                text=public_invitation(address,turns,welcome=welcome,index=conversation.count)
                previous=self.public_prompt_context
                if (not welcome and previous and previous['session']==session and previous['text'].endswith(text)):
                    text=public_invitation(index=conversation.count+1)
                topic=None;speech_style=None
                if not welcome and self.social.c['topics_enabled']:
                    from vivis.interaction.social_conversation import public_context
                    from vivis.speech.social_topics import draft
                    observed_style=self.social.cue['response_style']
                    current_turns=[{k:t[k] for k in ('role','text')} for t in self.social_turns if t['session']==session]
                    context=public_context(self.knowledge,verified_person,turns,current_turns,conversation.topics)
                    # Send facial ACTIONS, never the robot's suggested style as
                    # if it proved the visitor's mood or overrode their words.
                    context['visual_cue']={k:v for k,v in self.social.cue.items() if k!='response_style'}
                    context['previous_prompt']=previous['text'] if previous and previous['session']==session else None
                    try:
                        proposal=await draft(self.url,self.secret,dict(self.session,voice=self.active_voice),context,self.log,
                            self.social.c['topic_max_chars'],policy()['message_extract_s'])
                        if not current():return
                        if proposal['topic']=='skip':
                            conversation.defer('thinking');self.thinking_generation=self.input_generation
                            self.log.write('social_topic_deferred');return
                        if previous and previous['session']==session and previous['text'].endswith(proposal['text']):
                            raise ValueError('Repeated public invitation')
                        text=proposal['text'];topic=proposal
                        if self.social.c['tone_enabled']:speech_style=proposal['style']
                    except asyncio.CancelledError:raise
                    except Exception as exc:
                        self.log.write('social_topic_fallback',reason=type(exc).__name__)
                        observed_style=None  # Existing neutral cached invitation remains valid.
                if not current():return
                reader=None;audio=None if speech_style is not None else self.fixed_audio(text)
                if audio is None:
                    options={'style':speech_style} if speech_style is not None else {}
                    reader=RealtimeReader(self.url,self.secret,dict(self.session,voice=self.active_voice),self.log,**options)
                    audio=await asyncio.wait_for(reader.generate(text),policy()['playback_stall_s'])
                # Check again immediately before queuing speech. The same guard
                # remains active through playback and cancellation.
                if not current() or not self.public_prompt_available(state):return
                class PreparedAudio:
                    async def generate(self,_):return audio
                def started(words):
                    nonlocal pose_id
                    self.public_prompt_started=True
                    self.public_prompt_context=dict(text=words,playback_state='started',connection=connection,
                        session=session,generation=generation)
                    if not welcome:
                        conversation.played(time.monotonic())
                        if topic:
                            conversation.remember_topic(topic['topic'])
                            if self.social.c['topic_expression_enabled'] and topic['style']!='neutral':
                                try:
                                    outcome=self.expression.publish(dict(expression=topic['style'],source='social'),revision)
                                    if outcome['status']=='pending_expression_target':pose_id=outcome['intent_id']
                                    self.log.write('social_topic_expression',expression=topic['style'],**outcome)
                                except (OSError,ValueError) as exc:
                                    self.log.write('social_topic_expression_failed',reason=type(exc).__name__)
                    self.emit(dict(kind='cloud_transcript',epoch=self.epoch,role='assistant',text=words,
                        source='public_welcome' if welcome else 'public_idle_prompt'))
                    self.log.write('public_prompt_played',welcome=welcome,count=conversation.count,
                        topic=topic['topic'] if topic else None,speech_style=speech_style)
                speaker=MessageSpeaker(reader,self.player,state,lambda _:current(),started,self.log.write,
                    tts=PreparedAudio(),source='public_reception')
                complete=False
                try:complete=await speaker.say(text,None)
                finally:
                    invitation=self.public_prompt_context
                    if invitation and invitation['session']==session and invitation['generation']==generation:
                        invitation['playback_state']='completed' if complete else 'interrupted'
                    if not welcome and self.public_prompt_started and session==self.memory_session:
                        # Measure the gap from the end so DAC scheduling
                        # cannot shorten the minimum gap between invitations.
                        conversation.last_prompt=time.monotonic()
            except (asyncio.CancelledError,StaleMessageTurn):pass
            except Exception as exc:self.log.write('public_prompt_failed',error_type=type(exc).__name__)
            finally:
                if pose_id and self.expression.active and self.expression.active['id']==pose_id:
                    try:self.expression.publish(dict(expression='auto',source='social'),self.context['revision'])
                    except (OSError,ValueError) as exc:self.log.write('social_topic_expression_release_failed',reason=type(exc).__name__)
        self.public_prompt_task=asyncio.create_task(speak())

    def courier_request(self,args,event):
        if not self.courier or not self.memory:raise ValueError('退件取件码功能未启用。')
        if not isinstance(args,dict) or not set(args)<={'action','owner_name','code','handoff_id','version'}:
            raise ValueError('Invalid courier request')
        if (not self.turn_item_id or self.turn_ended_at is None or time.monotonic()-self.turn_ended_at>60
                or self.in_speech or not self.can_speak()):raise ValueError('需要本轮明确的取件或登记请求。')
        if not self.pending_user_text:
            raise MessageSourcePending('等待本轮识别文字；没有读取或修改退件记录。')
        self.turn_engaged=True;self.remember({})
        action=args.get('action')
        if action in ('owner_query','owner_visit'):
            owner=self.identified_person()
            if not owner or self.turn_person!=owner:raise IdentityVerificationRequired('查询本人退件留言需要确认委托人身份。')
            self.fresh_task_source()
            result=(self.courier.visit_query(owner) if action=='owner_visit' else self.courier.owner_query(owner));text=result['spoken_receipt']
            self.courier_offer=None
        elif action=='lookup':
            source=self.pending_user_text;previous=self.courier_query_pending
            if (previous and previous['generation']+1==self.input_generation
                    and previous.get('actor')==self.turn_person and previous.get('connection') is self.message_connection
                    and previous.get('revision')==self.context['revision']):
                source=previous['source']+'。'+source
            result=self.courier.lookup(args.get('owner_name',''),source)
            self.courier_offer=None
            prompts=dict(need_owner='您来取谁的退件？请说委托人姓名、称呼，或手机后四位。',
                not_found='目前没有这位委托人授权转告的退件取件码，请让本人先登记。',
                ambiguous_owner='这个称呼对应不止一位委托人，请说具体姓名。')
            text=result.get('spoken_receipt') or prompts[result['status']]
            if result['status']=='available':
                self.courier_offer=dict(result,response_id=None)
            self.courier_query_pending=(dict(source=source,generation=self.input_generation,actor=self.turn_person,
                connection=self.message_connection,revision=self.context['revision'])
                if result['status'] in ('need_owner','ambiguous_owner') and pickup_request(source) else None)
        elif action=='register':
            owner=self.identified_person()
            if not owner or self.turn_person!=owner:raise IdentityVerificationRequired('只有委托人登记或修改时需要核验本人；来取码的快递员不需要。')
            self.fresh_task_source()
            profile=self.knowledge.people[owner];label=profile.get('preferred_address') or profile['display_name']
            aliases=list(dict.fromkeys([label,profile['display_name']]))
            cutoff=(datetime.now(timezone.utc)-timedelta(minutes=5)).isoformat()
            with self.memory.db() as db:
                ids=[r['id'] for r in db.execute("SELECT id FROM messages WHERE person=? AND role='user' AND created>=? ORDER BY id DESC LIMIT 12",(owner,cutoff))]
            result=self.courier.register(owner,label,aliases,args.get('code',''),ids,
                source_reason='已核验本人本轮明确授权，取件码核对本人最近五分钟原话。')
            self.courier_offer=None
            text='您的退件取件码'+ '、'.join(result['code'])+'已保存，快递员来取您的退件时，我会转告。'
            result={k:result[k] for k in ('id','owner_label','status','version')}
        elif action=='acknowledge':
            offer=self.courier_offer
            if (not offer or args.get('handoff_id',offer['handoff_id'])!=offer['handoff_id']
                    or args.get('version',offer['version'])!=offer['version']):
                raise ValueError('请先说明取谁的退件并查询取件码，再确认收到；不从历史猜测。')
            self.remember({})
            result=self.courier.acknowledge(offer['handoff_id'],offer['version'],self.turn_message_id)
            text='已记录您收到取件码。';self.courier_offer=None
        elif action=='cancel':
            owner=self.identified_person()
            if not owner or self.turn_person!=owner:raise IdentityVerificationRequired('取消退件转告需要委托人本人确认。')
            self.fresh_task_source()
            hid=args.get('handoff_id')
            if not hid:
                self.courier.ensure_schema()
                with self.memory.db() as db:
                    row=db.execute('SELECT id FROM courier_handoffs WHERE owner=?',(owner,)).fetchone()
                if not row:raise ValueError('本人没有退件取件码记录。')
                hid=row['id']
            record=self.courier.cancel(hid,owner)
            result=dict(handoff_id=hid,status=record['status']);text='这条退件取件码转告已取消。';self.courier_offer=None
        else:raise ValueError('Unsupported courier action')
        self.tool_repair=None;self.continue_response=self.can_speak()
        spoken=text
        return dict(accepted=True,result=result,face_identity_required=action in ('register','cancel','owner_query'),
            scope='owner_authorized_courier_code_only',physical_collection_verified=False,
            spoken_receipt=spoken)

    def check_courier_delivery(self,state):
        offer=self.courier_offer
        if not offer or not offer.get('response_id'):return
        if self.audio_completed(state,offer['response_id']):
            self.courier.mark_announced(offer['handoff_id'],offer['version'],offer['response_id'])
            offer['response_id']=None

    def report_receipt_failure(self,record):
        """An unassociated voice reply is not evidence of a failed DB action."""
        if record.get('turn')!=self.input_generation:return
        if record.get('kind') in ('await_source','ordinary'):
            self.emit(dict(kind='voice_warning',epoch=self.epoch,
                message='这句语音与回答未能可靠关联，已停止该段回答；请再说一次。'))
            return
        self.receipt_failure_notice()

    def receipt_failure_notice(self):
        """One truthful fixed notice, not another model-generated write receipt."""
        lane=self.messages()
        if lane.task and not lane.task.done():return  # Never cancel a real DB operation.
        request=self.message_request()
        text=('刚才的语音片段无法可靠关联，这次请求尚未完成；已有记录没有改变，请完整说一下要查询或办理什么。'
              if self.unattributed_source_notice else
              '刚才的操作回执未能核验，我不能确认这次操作成功。可以查询账本核对；已有记录不会因此删除。')
        self.message_receipts.pending=None;self.continue_response=False
        self.message_claimed=self.input_generation
        lane.result=dict(operation_status='notice',saved=None);lane.result_actor=request.actor
        async def notify():
            try:await self.speak_message_notice(text,request)
            except (asyncio.CancelledError,StaleMessageTurn):pass
            except Exception as exc:self.log.write('receipt_failure_notice_failed',error_type=type(exc).__name__)
        lane.task=asyncio.create_task(notify())
        self.emit(dict(kind='voice_warning',epoch=self.epoch,message=text))

    async def resume_courier_calls(self,ws):
        """Continue only the original call, speaker, final item and connection."""
        for call,waiting in list(self.courier_waiters.items()):
            error=None
            try:waiting['request'].guard()
            except StaleMessageTurn:
                self.courier_waiters.pop(call,None)
                output=dict(accepted=False,status='stale_user_turn',saved=False,
                    error='话轮或身份已变化，未执行旧的退件请求。')
                await bounded_send(ws,json.dumps(dict(type='conversation.item.create',item=dict(type='function_call_output',
                    call_id=call,output=json.dumps(output,ensure_ascii=False))),ensure_ascii=False))
                self.log.write('courier_wait_cancelled',call_id=call,reason='request_guard_changed')
                continue
            ready=(self.turn_item_id==waiting['item'] and bool(self.pending_user_text))
            if not ready and time.monotonic()>=waiting['deadline']:
                error='没有收到可归属的完整取件码，本次没有登记；请说“退件取件码”加数字。'
            if not ready and error is None:continue
            self.courier_waiters.pop(call,None)
            await self.tool(ws,waiting['event'],_resuming=True,_source_error=error)

    async def resume_program_calls(self,ws):
        for call,waiting in list(self.program_waiters.items()):
            live=(waiting['connection'] is self.message_connection and waiting['generation']==self.input_generation
                and waiting['revision']==self.context['revision'] and waiting['item']==self.turn_item_id)
            final=self.question_onset.get('final_text') if self.question_onset else None
            if live and not final and time.monotonic()<waiting['deadline']:continue
            self.program_waiters.pop(call,None)
            if live and final:
                await self.tool(ws,waiting['event'],_resuming=True)
            else:
                await bounded_send(ws,json.dumps(dict(type='conversation.item.create',item=dict(type='function_call_output',
                    call_id=call,output=json.dumps(dict(accepted=False,status='program_source_expired',
                        instruction='本轮完整原话未核验或话轮已变化，未执行节目切换；等待新的明确请求。'),ensure_ascii=False))),ensure_ascii=False))
                self.log.write('program_control_discarded',call_id=call,reason='stale_or_missing_source')

    def program_source(self,event,args):
        """Final ASR precedes every mutating program proposal, even a fast stop."""
        action=args.get('action')
        # wake may carry a programme, or interrupt a currently running task.
        # Its spelling must not bypass the same source/preservation boundary.
        if action=='wake':
            action='play' if args.get('program') else 'stop' if self.context['mode']=='task' else 'wake'
        if action not in ('stop','standby','play'):return None
        source=self.question_onset;rid=event.get('response_id')
        response=self.cloud_state.responses.get(rid,{}) if self.cloud_state else {}
        if (not source or source['item']!=self.turn_item_id or source['generation']!=self.input_generation
                or source['connection'] is not self.message_connection or self.in_speech or self.turn_ended_at is None
                or self.response_inputs.get(rid)!=self.input_generation
                or (response.get('user_turn') or {}).get('item_id')!=self.turn_item_id):
            return 'unbound_program_source'
        if time.monotonic()-self.turn_ended_at>policy()['message_extract_s']:return 'expired_program_source'
        text=source.get('final_text')
        if not text:
            if self.program_waiters and event['call_id'] not in self.program_waiters:return 'program_source_already_pending'
            self.program_waiters[event['call_id']]=dict(event=dict(event),item=self.turn_item_id,
                generation=self.input_generation,revision=self.context['revision'],connection=self.message_connection,
                deadline=self.turn_ended_at+policy()['message_extract_s'])
            self.log.write('program_control_waiting_final',call_id=event['call_id'],item_id=self.turn_item_id)
            return 'waiting'
        from vivis.interaction.program_source import veto
        return veto(text,action)

    async def tool(self,ws,event,*,_resuming=False,_source_error=None):
        call=event.get('call_id');name=event.get('name');raw=event.get('arguments','')
        if not isinstance(call,str) or not 1<=len(call)<=128 or (call in self.calls and not _resuming):return
        if not _resuming:self.calls.append(call)
        output=dict(accepted=False);args=None;switch_request=None;dialect_rollback=None;expression_recovery=False
        response=self.cloud_state.responses.get(event.get('response_id'),{}) if self.cloud_state else {}
        reply_item=((response.get('user_turn') or {}).get('item_id')
            if self.response_inputs.get(event.get('response_id'))==self.input_generation else None)
        try:
            if not isinstance(raw,str) or len(raw)>4096:raise ValueError('Invalid tool arguments')
            args=json.loads(raw)
            revision=self.response_revisions.get(event.get('response_id'))
            generation=self.response_inputs.get(event.get('response_id'),self.input_generation)
            if name=='get_robot_status' and (revision!=self.context['revision'] or generation!=self.input_generation):
                raise StaleIdentityStatus('身份查询已过期，不改变当前话轮的回复状态。')
            if revision!=self.context['revision']:raise ValueError('Stale tool result after state change')
            if name in ('robot_program','set_expression','set_speech_dialect','program_followup','defer_reception') and generation!=self.input_generation:
                raise ValueError('Stale tool result after a new user turn')
            if self.program_turn_only() and name not in ('robot_program','program_followup'):
                raise ValueError('Program handoff must finish before interaction tools can run')
            followup=self.followup_response
            if followup:
                if self.response_inputs.get(event.get('response_id'))!=followup['generation']:
                    raise ValueError('Stale tool result from a different user turn')
                if not self.followup_current(followup) or followup['revision']!=revision:
                    raise ValueError('The continued user turn expired')
                if name!='get_robot_status' and not (name=='set_expression' and followup['kind']=='expression'
                        and not followup['expression_used'] and isinstance(args,dict) and args.get('source')=='command'):
                    raise ValueError('Only the original question or one explicit expression may be continued')
            if self.recovered_query_generation==self.input_generation:
                source=self.recovered_expression
                expression_recovery=(name=='set_expression' and source is not None and not source['consumed']
                    and source['generation']==self.input_generation and source['revision']==self.context['revision']
                    and source['connection'] is self.message_connection
                    and source['person']==(self.identified_person() or self.anonymous)
                    and source['item']==self.turn_item_id and not self.in_speech and self.presence.visible()
                    and self.response_inputs.get(event.get('response_id'))==self.input_generation
                    and time.monotonic()-source['ended']<=policy()['message_extract_s'])
                read_tool=name in ('get_robot_status','search_person_memory','search_robot_memory')
                read_tool|=name=='courier_pickup' and isinstance(args,dict) and args.get('action') in ('owner_query','owner_visit','lookup')
                if not (read_tool or expression_recovery):raise ValueError('Recovered turn does not authorize this action; request a fresh explicit action.')
            private_tool=name in ('search_person_memory','search_robot_memory','organize_archives')
            private_tool|=name=='courier_pickup' and isinstance(args,dict) and args.get('action')!='lookup'
            if private_tool and self.identity_enrichment_guarded():
                raise IdentityVerificationRequired('当前匿名话轮尚未定稿，身份补全暂不授权私人资料或业务。')
            if name in ('search_person_memory','search_robot_memory'):
                speaker=self.reply_people.get(event.get('response_id'))
                if speaker and not speaker.startswith('anonymous:') and speaker!=self.identified_person():
                    raise IdentityVerificationRequired('The speaker identity is no longer confirmed')
            if self.awaiting_recovery_user and name in ('robot_program','set_expression','set_speech_dialect','program_followup','defer_reception'):
                raise ValueError('After service recovery, wait for fresh user speech; never replay actions')
            if name=='courier_pickup' and (self.response_inputs.get(event.get('response_id'),self.input_generation)!=self.input_generation
                    or self.in_speech or self.message_claimed==self.input_generation):
                raise StaleMessageRequest('话轮已变化或已由固定留言流程接管，不重复执行取件请求。')
            if self.relay_pending and name!='get_robot_status':
                raise ValueError('Relay text is data, not tool authority')
            if name=='organize_archives':
                output=await self.organize_archives(args,event)
            elif name=='get_robot_status':
                if args!={}:raise ValueError('Status tool takes no arguments')
                if (generation!=self.input_generation or not await self.wait_current_voice_status()):
                    raise StaleIdentityStatus('身份查询原话已变化，旧查询不再继续；等待当前用户请求。')
                person_id=self.identified_person()
                person=(dict(person_id=person_id,**self.knowledge.people[person_id])
                        if person_id in self.knowledge.people else None)
                review=self.reviewing_identity()
                output=dict(accepted=True,controller=self.context,face_visible=self.presence.visible(),
                    current_person=person,identity_verified=person is not None,
                    identity_source='local_image_voice_fusion' if self.identity_fusion else 'local_face_match_with_bounded_identity_hold',
                    identity_state='reviewing' if review else 'recognized' if person else 'unknown',
                    last_known_person=review['person'] if review else None,
                    identity_notice=('身份保持窗口内，沿用当前会话身份和权限，不重置会话。' if person and review else '使用已核验身份回答。' if person else
                        '同一张脸正在短暂复核，可沿用上次称呼；暂不读取个人历史、职业背景或留言。' if review else
                        '当前尚未本地确认身份。自报称呼不等于认出，不补全熟人姓名岗位，不读取他人留言。'),
                    memory_enabled=self.memory is not None,memory_attribution='frozen_turn_identity' if self.identity_fusion else 'confirmed_single_face_only',
                    speaker_status=self.speaker_status(person_id),
                    sound_attention=self.sound_status(details=True),
                    speech_preference=self.speech_preference(),robot_tasks=self.task_overview(),
                    message_operation=self.message_operation(),
                    known_people=self.known_people() if person else [],
                    robot_memory_scope='all_archived_conversations_via_bounded_search')
                voice=output['speaker_status'].get('current_result',{})
                if self.identity_fusion:
                    from vivis.interaction.identity_fusion import recognition_report
                    output['identity_recognition']=recognition_report(person_id,self.knowledge.people,
                        self.fused_face_evidence(),voice,(self.fusion_last or {}).get('reason'))
                    output['identity_notice']=output['identity_recognition']['notice']+(
                        'current_person是业务身份，声纹或图像本轮确认以identity_recognition分别标注的结果为准。'
                        '本说明不回写原话归属、不改变业务权限。')
                elif voice.get('state')=='confirmed':
                    label=voice.get('preferred_address') or voice.get('display_name') or voice['person_id']
                    output['identity_notice']=('本轮声音已确认：'+label+'。回答听声认人时据此称呼；'
                        'current_person是业务会话身份，可能尚未接纳本轮结果，不因此否认已通过的声纹。'
                        '图像只有current_face_result.state=confirmed时才可声称看出来；图声确认冲突时分别说明，以图像为主。'
                        '本结果不回写已接纳原话，不额外授权私人记忆或留言。')
                self.tool_repair=None
                self.continue_response=self.can_answer_turn(reply_item)
            elif name=='program_followup':
                if not self.program_turn_only():
                    self.tool_repair=None;self.continue_response=self.can_answer_turn(reply_item)
                    output=dict(accepted=False,status='already_in_interaction',
                        instruction='当前已在交互态，不需要节目交接；直接回答本轮原话，不再调用program_followup。')
                else:output=self.stage_program_followup(args,event.get('response_id'))
            elif name=='defer_reception':
                source=(self.cloud_state.responses.get(event.get('response_id'),{}).get('user_turn') or {}) if self.cloud_state else {}
                if (not isinstance(args,dict) or set(args)!={'kind'} or self.context['mode']!='interaction'
                        or not self.context['ready'] or self.in_speech or not self.presence.single_visible()
                        or self.response_inputs.get(event.get('response_id'))!=self.input_generation
                        or not self.turn_item_id or source.get('item_id')!=self.turn_item_id):
                    raise ValueError('Need a current visitor turn to pause proactive reception')
                self.reception.conversation.defer(args['kind'])
                self.thinking_generation=self.input_generation
                self.announcement_revision=None;self.cancel_public_prompt()
                self.tool_repair=None;self.continue_response=self.can_answer_turn(reply_item)
                output=dict(accepted=True,status='public_prompts_deferred',instruction='简短回应即可，不再问问题。')
            elif name=='robot_program':
                args=normalize_program(args,self.program_ids)
                source_error=self.program_source(event,args)
                if source_error=='waiting':return
                if source_error:
                    output=dict(accepted=False,status=source_error,
                        instruction='原话否定、引用了该动作，或来源未核验；保持当前节目，不执行该工具，不宣称已停止。')
                    self.tool_repair=None;self.continue_response=False
                    self.log.write('program_control_vetoed',call_id=call,action=args['action'],reason=source_error)
                    await bounded_send(ws,json.dumps(dict(type='conversation.item.create',item=dict(type='function_call_output',
                        call_id=call,output=json.dumps(output,ensure_ascii=False))),ensure_ascii=False))
                    return
                intent=controller_intent(args,self.program_ids,self.context['mode'])
                greeting=args.get('wake_reason')=='greeting'
                if greeting and (not self.greeting_wake_allowed() or self.in_speech
                        or self.response_inputs.get(event.get('response_id'),self.input_generation)!=self.input_generation):
                    raise ValueError('Greeting wake needs a fresh visible face/body and current finished user turn')
                if greeting or args['action']=='wake' or args.get('explicit_wake') is True:
                    self.verify_wake_source(greeting)
                if args['action']!='wake':
                    self.program_followup=None;self.followup_handoff=False
                if greeting:
                    self.presence.greet();self.reception.dismiss_current_visit()
                if not greeting and (intent['kind']=='wake' or args.get('explicit_wake') is True):self.presence.wake();self.turn_engaged=True
                if args['action']=='wake' and self.context['mode']=='interaction':
                    self.announcement_kind='wake';self.announcement_revision=self.context['revision']
                elif intent['kind']=='select' and not self.presence.allowed():raise ValueError('No face; call wake only if the user explicitly called the wake word')
                already_requested=greeting and self.control_pending
                self.turn_engaged=True;self.remember({})
                if self.cloud_state and not already_requested:self.cloud_state.interrupt(source='cloud_program_request')
                already_ready=(self.context['ready'] and ((self.context['mode']=='interaction' and args['action'] in ('stop','wake'))
                               or (self.context['mode']=='standby' and args['action'] in ('stop','standby'))))
                self.tool_repair=None;self.control_pending=not already_ready
                if already_ready:
                    output=dict(accepted=True,status='already_in_requested_state',controller=self.context)
                    self.continue_response=self.can_answer_turn(reply_item)
                else:
                    if not already_requested:
                        reply=None
                        notice=preparation_reply(intent,self.programs,interrupting=self.context['mode']=='task',choose_interruption=True)
                        if notice and self.question_onset and self.question_onset['generation']==self.input_generation:
                            self.cancel_farewell('replaced')
                            self.farewell=dict(id=uuid.uuid4().hex,generation=self.input_generation,connection=self.message_connection,**notice)
                            reply=self.farewell['id']
                        self.emit(dict(kind='cloud_action',revision=revision,arguments=args,text='Qwen: '+args['action'],
                            **({'transition_reply_id':reply} if reply else {})))
                        if args['action']=='stop' and self.context['mode']=='task':
                            try:self.stage_program_followup(dict(kind='question'),event.get('response_id'),native_stop_review=True)
                            except ValueError:self.log.write('stopped_request_not_retained',reason='no_current_public_source')
                    output=dict(accepted=True,status='pending_controller_execution')
            elif name=='set_expression':
                animation=getattr(self,'conversation_animation',None)
                social=animation is not None and animation.c['enabled'] and args.get('source')=='social'
                permission=self.presence.allowed() or social and self.turn_reply_authorized(reply_item)
                if self.context['mode']!='interaction' or not self.context['ready'] or not permission:
                    raise ValueError('Expression is only available in ready interaction; stop the current programme first if user requested it')
                if social:
                    active=self.expression.active
                    if (active and active['source']=='command' and active['expression']!='auto'
                            and time.monotonic()<active['expires_monotonic']):
                        result=dict(status='suppressed_by_explicit_command',intent_id=active['id'])
                    else:
                        result=animation.stage(args,self.sync_conversation_animation())
                else:result=self.expression.publish(args,revision)
                if followup:followup['expression_used']=True
                if expression_recovery:self.recovered_expression['consumed']=True
                output=dict(accepted=True,**result)
                self.tool_repair=None
                if result['status']=='pending_expression_target':
                    self.expression_pending=dict(intent_id=result['intent_id'],created=time.monotonic())
                else:self.continue_response=self.can_answer_turn(reply_item)
            elif name=='set_speech_dialect':
                if self.context['mode']!='interaction' or not self.context['ready'] or not self.presence.allowed():
                    raise ValueError('Dialect switching needs ready interaction and a face or explicit wake permission; stop the programme first')
                requested=self.dialect.requested(args);previous=self.dialect.name
                if self.pending_voice_switch is not None:raise ValueError('Voice switch pending; wait for server confirmation')
                self.dialect.name=requested
                desired=self.dialect.voice(self.session['voice'])
                if desired!=self.active_voice:
                    dialect_rollback=previous
                    switch_request=dict(voice=desired,dialect=requested,call_id=call,ready=False)
                    self.pending_voice_switch=switch_request
                    if self.cloud_state:self.cloud_state.interrupt(source='speech_voice_switch')
                else:
                    try:
                        # Same voice: ordered before tool continuation, without reconnecting.
                        instructions,_,_=self.context_snapshot()
                        await asyncio.wait_for(ws.send(json.dumps(dict(type='session.update',session=dict(instructions=instructions)),ensure_ascii=False)),timeout=policy()['send_timeout_s'])
                    except BaseException:
                        self.dialect.name=previous;raise
                self.tool_repair=None;self.continue_response=self.can_answer_turn(reply_item)
                output=dict(accepted=True,status='pending_voice_reconnect' if switch_request else 'speech_instruction_updated',
                    **self.speech_preference(),pronunciation_verified=False)
                self.summary['speech_dialect']=requested
                self.log.write('speech_dialect_changed',previous=previous,**self.dialect.snapshot(),pronunciation_verified=False)
            elif name=='search_person_memory':
                if not self.identified_person():raise IdentityVerificationRequired('身份正在复核或尚未确认，暂不读取个人历史。')
                if not isinstance(args,dict) or set(args)!={'keywords'}:raise ValueError('Memory search needs keywords only')
                output=dict(accepted=True,result=self.memory.search(self.identified_person(),args['keywords'],self.memory_session) if self.memory else dict(enabled=False))
                self.tool_repair=None;self.continue_response=self.can_answer_turn(reply_item)
            elif name=='search_robot_memory':
                if not self.identified_person():raise IdentityVerificationRequired('当前身份尚未核验，请正对摄像头确认身份后再查询历史；口头自报不能替代核验。')
                if not isinstance(args,dict) or not {'keywords'}<=args.keys()<={'keywords','person_id','before_id'}:raise ValueError('Invalid robot memory query')
                output=dict(accepted=True,result=self.memory.search_all(args['keywords'],args.get('person_id') or None,args.get('before_id')) if self.memory else dict(enabled=False))
                self.tool_repair=None;self.continue_response=self.can_answer_turn(reply_item)
            elif name=='courier_pickup':
                if _source_error:raise ValueError(_source_error)
                output=self.courier_request(args,event)
            elif name=='manage_robot_task':
                output=dict(accepted=False,status='final_text_pipeline_only',saved=False,
                    error='普通留言仅由最终识别文字启动固定流程，不接受聊天模型直接写入。')
                self.continue_response=False;self.tool_repair=None
            else:raise ValueError('Unknown tool')
        except StaleIdentityStatus as exc:
            # This old read must not overwrite a newer turn's response flags.
            output.update(status='stale_user_turn',error=str(exc))
        except StaleMessageRequest as exc:
            output.update(status='stale_user_turn',error=str(exc));self.tool_repair=None;self.continue_response=False
        except MessageSourcePending as exc:
            if name=='courier_pickup' and not _resuming and len(self.courier_waiters)<4:
                self.courier_waiters[call]=dict(event=dict(event),request=self.message_request(),
                    item=self.turn_item_id,deadline=time.monotonic()+policy()['message_extract_s'])
                self.message_receipts.responses.pop(event.get('response_id'),None)
                self.tool_repair=None;self.continue_response=False
                self.log.write('courier_waiting_final',call_id=call,item_id=self.turn_item_id)
                return  # No provisional tool result for the model to mistake for completion.
            output.update(status='final_text_required',saved=False,error=str(exc))
            self.tool_repair=None;self.continue_response=False
        except MessageMeaningUncertain as exc:
            output.update(status='meaning_clarification_required',error=str(exc),saved=False,
                spoken_receipt=self.message_receipts.set('meaning',self.input_generation))
            self.tool_repair=None;self.continue_response=self.can_answer_turn(reply_item)
        except MessageRecipientUnavailable as exc:
            output.update(status='recipient_unavailable',error=str(exc),saved=False,
                spoken_receipt=self.message_receipts.set('unsupported_recipient',self.input_generation))
            self.tool_repair=None;self.continue_response=self.can_answer_turn(reply_item)
        except IdentityVerificationRequired as exc:
            output.update(status='identity_confirmation_required',error=str(exc))
            self.tool_repair=None;self.continue_response=self.can_answer_turn(reply_item)
        except GreetingAdmissionError as exc:
            output.update(status=str(exc),error='唤醒尚未通过本轮原话与在场条件核验；未执行，等待最终识别或下一次明确呼叫。')
            self.tool_repair=None;self.continue_response=False
        except (ValueError,TypeError) as exc:
            output['error']=str(exc);self.continue_response=self.tool_retry_budget>0
            self.tool_repair=dict(tool=name,error=str(exc),controller=self.context)
            self.tool_retry_budget=max(0,self.tool_retry_budget-1)
        except PermissionError as exc:
            if name=='organize_archives':output.update(status='administrator_required',error=str(exc))
            elif name in ('manage_robot_task','courier_pickup'):
                output.update(status='database_operation_failed',error='PermissionError：留言状态未能确认。')
            else:raise
            self.tool_repair=None;self.continue_response=self.can_answer_turn(reply_item)
        except (sqlite3.Error,OSError,RuntimeError) as exc:
            if name not in ('manage_robot_task','courier_pickup','organize_archives'):raise
            output.update(status='database_operation_failed',error=type(exc).__name__+
                ('：档案整理任务未能确认，不能声称整理成功。' if name=='organize_archives' else '：留言状态未能确认。'))
            self.tool_repair=None;self.continue_response=self.can_answer_turn(reply_item)
        if output.get('spoken_receipt'):
            rid=event.get('response_id')
            self.message_receipts.responses.pop(rid,None)
            if rid not in self.blocked:
                self.blocked.append(rid)
                if self.cloud_state:self.cloud_state.interrupt(rid,source='message_receipt_required')
        if name=='courier_pickup' and not output['accepted'] and output.get('status') not in ('stale_user_turn','source_transcript_pending','final_text_required'):
            text=('查询本人退件留言、登记或修改取件码需要核验委托人身份。请正对摄像头；取码的快递员不用人脸鉴定，请说明来取谁的退件。'
                  if output.get('status')=='identity_confirmation_required' else '这次退件取件码操作失败了，请再试一次。')
            output['spoken_receipt']=text
            self.tool_repair=None;self.continue_response=self.can_answer_turn(reply_item)
            rid=event.get('response_id');self.message_receipts.responses.pop(rid,None);self.blocked.append(rid)
            if self.cloud_state:self.cloud_state.interrupt(rid,source='courier_operation_failed')
        if name in ('manage_robot_task','courier_pickup') and not output['accepted']:
            output.update(operation_failure(output.get('status'),output.get('error','本次操作未执行。')))
        if name=='courier_pickup' and output.get('spoken_receipt'):
            text=output['spoken_receipt'];self.message_receipts.pending=None;self.continue_response=False
            request=self.message_request();lane=self.messages();lane.cancel()
            lane.result=dict(operation_status='read',saved=output.get('accepted'));lane.result_actor=request.actor
            offer=(self.courier_offer if output.get('accepted') and
                output.get('result',{}).get('status')=='available' else None)
            async def speak_code():
                try:
                    complete=await (self.speak_message_notice(text,request)
                        if output.get('status')=='identity_confirmation_required' else self.speak_message(text,request))
                    if complete and offer:self.courier.mark_announced(offer['handoff_id'],offer['version'],'trusted-tts')
                except (asyncio.CancelledError,StaleMessageTurn):pass
                except Exception as exc:self.log.write('courier_voice_failed',error_type=type(exc).__name__)
            lane.task=asyncio.create_task(speak_code())
        self.log.write('cloud_tool',name=name,call_id=call,accepted=output['accepted'],revision=self.context['revision'],
            error=output.get('error'),arguments=args if name in ('robot_program','set_expression','set_speech_dialect') and isinstance(args,dict) else None,
            status=output.get('status'))
        try:
            await bounded_send(ws,json.dumps(dict(type='conversation.item.create',item=dict(type='function_call_output',
                call_id=call,output=json.dumps(output,ensure_ascii=False))),ensure_ascii=False))
        except BaseException:
            if switch_request is not None:
                self.dialect.name=dialect_rollback;self.pending_voice_switch=None
            raise
        if switch_request is not None:switch_request['ready']=True

    async def cloud(self,_epoch):
        try:
            await self.conversation_cloud_loop(_epoch)
        finally:
            animation=getattr(self,'conversation_animation',None)
            if animation:await asyncio.to_thread(animation.close)

    async def conversation_cloud_loop(self,_epoch):
        while not self.shutdown.is_set():
            try:
                await self._connection();return
            except TransientServiceError as exc:
                self.health(False,exc.code)
                self.relay_pending.clear()
                try:retry=self.service_recovery.next()
                except RuntimeError as exhausted:
                    self.log.write('cloud_service_recovery_exhausted',code=exc.code,event_id=exc.event_id)
                    retry=dict(attempt=0,maximum=self.config['service_recovery']['max_reconnects'],
                        delay_s=policy()['recovery_cooldown_s'],window_s=self.config['service_recovery']['window_s'])
                detail=str(exc)
                if self.secret:detail=detail.replace(self.secret,'[redacted]')
                detail=' '.join(detail.split())[:600]
                self.log.write('cloud_service_recovery',code=exc.code,event_id=exc.event_id,error=detail,**retry,
                    microphone_reopened=False,hardware_restarted=False,commands_replayed=False)
                progress=(f'快速重试已达上限，等待{retry["delay_s"]:g}秒后再尝试'
                    if retry['attempt']==0 else f'将在{retry["delay_s"]:g}秒后重连 {retry["attempt"]}/{retry["maximum"]}')
                self.emit(dict(kind='voice_warning',epoch=self.epoch,
                    message=f'Qwen语音连接暂不可用：{detail}；{progress}；不重放节目指令。'))
                self.response_revisions.clear();self.reply_people.clear();self.pending_memories.clear()
                self.continue_response=False;self.tool_repair=None;self.in_speech=False
                self.turn_item_id=None;self.turn_message_id=None;self.pending_user_text=None;self.turn_ended_at=None
                self.response_request_pending=False;self.response_waiting_user_end=False;self.control_pending=False
                self.input_generation+=1
                self.announcement_revision=None;self.awaiting_recovery_user=True
                await asyncio.sleep(retry['delay_s'])
            except SpeechVoiceSwitch:
                self.relay_pending.clear()
                self.log.write('speech_voice_reconnect',**self.speech_preference(),microphone_reopened=False,hardware_restarted=False)
                self.response_revisions.clear();self.reply_people.clear();self.pending_memories.clear()
                self.continue_response=False;self.tool_repair=None
                await asyncio.sleep(.05)
            except ConversationReset as exc:
                self.relay_pending.clear();self.turn_item_id=None;self.turn_message_id=None;self.turn_ended_at=None
                self.relay_identity_hold=False
                self.identity_reset_pending=False;self.finish_responses.clear()
                self.in_speech=False;self.turn_engaged=False;self.pending_user_text=None;self.turn_saved=False
                if str(exc)!='identity_uncertain':self.task_visit_announced.clear();self.task_notice_attempts.clear()
                self.log.write('conversation_reset',reason=str(exc),person_id=self.conversation_person)
                # Unknown identity clears personal context, but does not prove
                # a different visitor requested another public speech style.
                if str(exc)!='identity_uncertain':self.dialect.reset()
                self.pending_voice_switch=None
                self.awaiting_recovery_user=False
                self.summary['speech_dialect']=self.dialect.name
                self.response_revisions.clear();self.reply_people.clear();self.pending_memories.clear()
                self.resume_history.clear();self.expression.clear();self.expression_pending=None
                self.expression_outcome=None;self.tool_repair=None;self.control_pending=False
                await asyncio.sleep(.05)
            except CloudIdleTimeout:
                self.health(False,'cloud_idle_reconnect')
                self.awaiting_recovery_user=True
                self.relay_pending.clear()
                self.log.write('cloud_idle_reconnect',microphone_reopened=False,hardware_restarted=False)
                self.response_revisions.clear();self.reply_people.clear();self.pending_memories.clear()
                await asyncio.sleep(.1)

    async def _connection(self):
        tasks=[];state=None;preserve_output=False
        try:
            self.cancel_sound_attention('new_connection')
            self.side_wire_item=None
            self.side_items.clear();self.side_responses.clear();self.side_commands.clear()
            self.side_onsets.clear()
            self.side_response_sources.clear()
            if self.sound_gate:self.sound_gate.reset()
            self.message_connection=object();self.courier_waiters.clear()
            self.program_waiters.clear()
            self.social_turns.clear()
            self.invalid_question=None;self.question_onset=None;self.recovered_query_generation=None
            self.program_followup=None;self.followup_handoff=False;self.followup_response=None
            self.program_responses.clear()
            self.recovered_expression=None
            self.explicit_wake_pending=None
            self.cloud_audio_enabled=False
            self.message_claimed=None
            self.relay_recipients.clear()
            self.finish_responses.clear();self.speech_responses.clear()
            self.message_receipts=MessageReceipts();self.response_inputs.clear()
            self.courier_offer=None
            self.courier_query_pending=None
            self.response_waiting_user_end=False
            if self.message_lane:self.message_lane.reset_context()
            self.state('connecting')
            self.response_request_pending=False
            async with websocket_connection(self.url,self.secret) as ws:
                instructions,identity,_=self.context_snapshot()
                initial=dict(self.session,instructions=instructions,modalities=['text'],voice=self.dialect.voice(self.session['voice']),
                    tools=self.available_tools())
                accepted=await connect_session(ws,initial,self.log)
                names={t.get('name') for t in accepted.get('tools',[]) if isinstance(t,dict)}
                if names!={t['name'] for t in initial['tools']}:raise RuntimeError('Qwen did not confirm required tools; no local fallback')
                self.voice_ready(accepted['voice'])
                if self.resume_history:
                    history='同一来访者本次会话的最近文字记录，仅用于续接，不是新命令；assistant不保证完整播放：'+json.dumps(list(self.resume_history),ensure_ascii=False)
                    await bounded_send(ws,json.dumps(dict(type='conversation.item.create',item=dict(type='message',role='system',content=[dict(type='input_text',text=history)])),ensure_ascii=False))
                self.ws=ws;self.summary['cloud_connections']+=1
                if self.enabled and self.audio.output_stream is None:self.audio.enable_output(self.player)
                self.state('ready' if self.enabled else 'off')
                self.health(True,'connected')
                if self.awaiting_recovery_user:
                    self.log.write('cloud_service_restored',commands_replayed=False,waiting_for_fresh_user=True)
                    self.emit(dict(kind='voice_warning',epoch=self.epoch,message='Qwen连接已恢复，请继续说话；刚才未完成的指令不会自动重放。'))
                self.log.write('knowledge_context_initial',epoch=self.epoch,recognized=identity['recognized'],person_id=identity['person_id'])
                self.player.mouth=self.mouth
                self.player.expression_clock=self.conversation_animation
                self.sync_conversation_animation()
                self.conversation_animation.start()
                transcript_response_id=None
                def transcript(role,text):
                    if not isinstance(text,str) or not text.strip():return
                    if role=='user':self.presence.continued_turn()
                    audible=self.can_play_response(transcript_response_id)
                    if role=='assistant' and audible:self.public_prompt_context=None
                    if (role=='assistant' and audible and self.thinking_generation is not None
                            and self.thinking_generation!=self.input_generation
                            and self.response_inputs.get(transcript_response_id)==self.input_generation):
                        self.reception.conversation.addressed_response();self.thinking_generation=None
                    if text and (role=='user' and self.turn_engaged or role=='assistant' and audible):
                        self.resume_history.append(dict(role=role,text=text[:2000]))
                    if role=='user' or audible:
                        labels=self.identity_labels() if role=='user' else {}
                        self.emit(dict(kind='cloud_transcript',epoch=self.epoch,role=role,text=text,**labels))
                state=VoiceSession(self.player,self.log,on_transcript=transcript,interruption_hold_ms=self.config['interruption_hold_ms']);self.cloud_state=state
                suppressed=deque(maxlen=128)
                controller_events=ControllerEventGate()
                from vivis.speech.welcome import WelcomeTurnGate
                welcome_gate=WelcomeTurnGate()
                from vivis.speech.explicit_wake import AudioPermissionGate
                audio_gate=AudioPermissionGate()
                uplink=asyncio.Queue(maxsize=max(1,round(policy()['uplink_max_age_s']/.02)));self.cloud_queue=uplink
                from vivis.speech.round_recording import SentTimeline
                journal_timeline=SentTimeline()
                self.transport_fault=None
                async def send():
                    while True:
                        stamp,pcm=await uplink.get()
                        if time.monotonic()-stamp>policy()['uplink_max_age_s']:raise VoiceTransportError('Cloud input audio expired; incomplete turn discarded')
                        await asyncio.wait_for(ws.send(json.dumps(dict(type='input_audio_buffer.append',audio=base64.b64encode(pcm).decode()))),timeout=policy()['send_timeout_s'])
                        journal_timeline.sent(stamp,pcm)
                async def reject_receipt(rid,reason):
                    if rid in self.blocked or rid in state.blocked:
                        self.message_receipts.responses.pop(rid,None);return
                    record=self.message_receipts.responses.get(rid,{})
                    entry_guard=record.get('kind')=='needs_tool' and record.get('turn')==self.input_generation
                    self.blocked.append(rid);state.interrupt(rid,source='unverified_message_receipt')
                    if state.generating==rid:
                        try:await asyncio.wait_for(ws.send(json.dumps(dict(type='response.cancel'))),timeout=policy()['send_timeout_s'])
                        except Exception:self.log.write('message_receipt_cancel_failed',response_id=rid)
                        state.generating=None
                    retry=self.message_receipts.retry(rid,self.input_generation)
                    if not retry and entry_guard:self.message_receipts.pending=None
                    self.log.write('message_receipt_rejected',response_id=rid,reason=reason,retry=retry)
                    if record.get('turn')!=self.input_generation:return
                    self.continue_response=retry and self.can_answer_turn(self.turn_item_id)
                    if not retry:self.report_receipt_failure(record)
                async def deliver(event):
                    nonlocal transcript_response_id
                    rid=event.get('response_id');kind=event.get('type');transcript_response_id=rid
                    if kind=='response.audio_transcript.done' and self.courier_offer:
                        from vivis.speech.message_receipts import code_normalized
                        if code_normalized(event.get('transcript',''))==code_normalized(self.courier_offer['spoken_receipt']):
                            self.courier_offer['response_id']=rid
                    if kind=='response.audio.delta':self.protect_relay_audio(state)
                    if kind=='response.audio.delta' and (rid in self.blocked or not self.can_play_response(rid)):
                        if rid not in suppressed:
                            self.log.write('cloud_audio_suppressed',response_id=rid,mode=self.context['mode'],ready=self.context['ready'],
                                face_visible=self.presence.last_visible,previously_blocked=rid in self.blocked,
                                expression_pending=self.expression_pending is not None,control_pending=self.control_pending)
                            suppressed.append(rid)
                        if rid not in self.blocked:self.blocked.append(rid)
                        return
                    self.remember(event)
                    if kind=='conversation.item.input_audio_transcription.completed':
                        self.final_message(event,state)
                        await self.resume_courier_calls(ws)
                    warning=await receive_event(ws,state,event)
                    if kind=='response.created':
                        response=state.responses[event['response']['id']]
                        item=(response.get('user_turn') or {}).get('item_id')
                        response['archive_session']=self.memory_turns.get(item,{}).get('session',self.memory_session)
                        if event['response']['id'] in welcome_gate.responses:
                            response.update(archive_no_user=True,archive_session=self.memory_session)
                    if warning:self.emit(dict(kind='voice_warning',epoch=self.epoch,**warning))
                async def cancel_welcome(rids,reason):
                    self.continue_response=False;self.announcement_revision=None;self.tool_repair=None
                    for response_id in rids:
                        if response_id not in self.blocked:self.blocked.append(response_id)
                        state.interrupt(response_id,source='welcome_yields_to_user')
                    # Cancel generation only for a known welcome response, never
                    # globally cancel a new user answer or a fixed DB receipt.
                    if rids and (state.generating in rids or self.response_request_pending):
                        await bounded_send(ws,json.dumps(dict(type='response.cancel')))
                    self.log.write('welcome_yielded_before_playback',reason=reason,response_ids=list(rids),
                        user_turn_preserved=True)

                async def receive():
                    nonlocal transcript_response_id
                    async for message in ws:
                        event=json.loads(message);kind=event.get('type');rid=event.get('response_id')
                        if await self.filter_side_event(event,journal_timeline,ws):continue
                        self.observe_program_response(event)
                        if kind=='response.audio.delta':event['_network_received_monotonic']=time.monotonic()
                        transcript_response_id=rid
                        controller_events.observe(event)
                        audio_gate.observe(event)
                        if welcome_gate.user_event(event,self.player):
                            await cancel_welcome(welcome_gate.responses,event['type'])
                        if (not self.public_prompt_started and kind in ('input_audio_buffer.speech_started','conversation.item.input_audio_transcription.delta',
                                    'conversation.item.input_audio_transcription.completed') and event.get('item_id')):
                            self.cancel_public_prompt()
                        if kind=='response.created' and welcome_gate.created(event['response']['id']):
                            # The user may speak before response.created reaches
                            # us; quarantine this late welcome and its output.
                            await cancel_welcome([event['response']['id']],'late_welcome_response')
                        if kind=='response.function_call_arguments.done' and rid in self.blocked and rid in welcome_gate.responses:
                            self.log.write('welcome_tool_discarded',response_id=rid,name=event.get('name'))
                            if isinstance(event.get('call_id'),str):
                                await bounded_send(ws,json.dumps(dict(type='conversation.item.create',item=dict(
                                    type='function_call_output',call_id=event['call_id'],output=json.dumps(dict(
                                        accepted=False,status='welcome_cancelled',instruction='欢迎已让给真实用户；不续播欢迎，不执行该工具。'),ensure_ascii=False))),ensure_ascii=False))
                            continue
                        if kind=='session.updated' and 'modalities' in event.get('session',{}):
                            self.cloud_audio_enabled='audio' in event['session']['modalities']
                        if self.response_busy(event,state):continue
                        if kind=='error' and event.get('error',{}).get('code')=='response_idle_timeout':
                            self.log.write('cloud_idle_timeout',code='response_idle_timeout')
                            raise CloudIdleTimeout()
                        retryable=transient_error(event)
                        if retryable:
                            self.log.write('cloud_transient_service_error',code=retryable.code,event_id=retryable.event_id,
                                response_id=rid or state.generating,message=str(retryable))
                            raise retryable
                        if kind=='response.created':
                            rid=event['response']['id'];self.response_revisions[rid]=self.context['revision']
                            self.response_inputs[rid]=self.input_generation
                            if len(self.response_inputs)>128:self.response_inputs.pop(next(iter(self.response_inputs)))
                            source_item=(state.current_user_turn or {}).get('item_id')
                            self.message_receipts.bind(rid,self.input_generation,source_item)
                            self.reply_people[rid]=self.turn_person
                            scope=self.sync_conversation_animation()
                            reply_allowed=self.can_answer_turn(source_item)
                            if source_item and source_item==self.turn_item_id and reply_allowed:
                                self.conversation_animation.register(rid,scope)
                            if len(self.reply_people)>128:self.reply_people.pop(next(iter(self.reply_people)))
                            if len(self.response_revisions)>128:self.response_revisions.pop(next(iter(self.response_revisions)))
                            if reply_allowed and self.message_claimed!=self.input_generation:
                                self.speech_responses.append(rid)
                                if not self.can_speak():self.log.write('reply_permission_retained',item_id=source_item,
                                    response_id=rid,reason='authorized_turn_before_presence_expiry',next_turn_authorized=False)
                            else:self.blocked.append(rid)
                            for relay in self.relay_pending.values():
                                relay['response_id']=rid
                                self.relay_recipients[rid]=relay['recipient']
                            while len(self.relay_recipients)>128:self.relay_recipients.pop(next(iter(self.relay_recipients)))
                            # Consume the pending request only after admission:
                            # its permission must survive this response boundary.
                            self.response_waiting_user_end=False
                            self.response_request_pending=False;self.continue_response=False
                        if kind=='input_audio_buffer.speech_started':
                            self.input_generation+=1;self.message_receipts.pending=None;self.message_claimed=None
                            self.relay_pending.clear()
                            self.relay_identity_hold=False
                            self.awaiting_recovery_user=False
                            self.in_speech=True;self.continue_response=False;self.tool_retry_budget=2
                            self.tool_repair=None
                            self.presence.continued_turn()
                            if self.can_speak():self.announcement_revision=None
                            self.begin_user_turn(event.get('item_id'))
                            self.sync_conversation_animation()
                            self.continuity_start(event,journal_timeline)
                            self.question_started(event.get('item_id'),state)
                            self.message_receipts.source_turns[event.get('item_id')]=self.input_generation
                        elif kind=='input_audio_buffer.speech_stopped' and event.get('item_id')==self.turn_item_id:
                            self.in_speech=False
                            if event.get('reason')=='turn_invalid':
                                self.question_invalidated(event)
                                turn=self.memory_turns.get(self.turn_item_id)
                                if turn:turn['valid']=False
                                self.turn_item_id=None;self.turn_ended_at=None
                            else:self.turn_ended_at=time.monotonic()
                        if kind=='input_audio_buffer.speech_stopped':
                            self.sync_conversation_animation()
                            self.response_waiting_user_end=False
                            if self.continuity:self.continuity.end(event.get('item_id'),journal_timeline.at(event.get('audio_end_ms')),
                                event.get('reason')!='turn_invalid',self.continuity_evidence(self.turn_person))
                        if kind=='response.function_call_arguments.done':
                            await self.dispatch_cloud_tool(ws,event,tasks);continue
                        if (kind=='conversation.item.input_audio_transcription.completed' and self.question_onset
                                and event.get('item_id')==self.turn_item_id==self.question_onset['item']
                                and self.question_onset['generation']==self.input_generation and self.turn_ended_at is not None):
                            self.question_onset['final_text']=event.get('transcript','')
                            if self.context['mode']=='interaction' and self.context['ready'] and self.turn_engaged:
                                self.social_turns.append(dict(role='user',text=self.question_onset['final_text'],session=self.memory_session))
                            self.accept_program_followup(self.question_onset['final_text'])
                            await self.resume_program_calls(ws)
                        waking=(kind=='conversation.item.input_audio_transcription.completed' and self.admit_explicit_wake(event,state))
                        recovered=(not waking and kind=='conversation.item.input_audio_transcription.completed' and self.recover_invalid_question(event,state))
                        if kind=='conversation.item.input_audio_transcription.completed' and self.continuity:
                            turn=self.memory_turns.get(event.get('item_id'),{})
                            current=event.get('item_id')==self.turn_item_id
                            engaged=(turn.get('engaged') or current and (self.turn_engaged or
                                self.context['mode'] in ('interaction','task') and self.presence.allowed()))
                            self.continuity.final(event.get('item_id'),event.get('transcript',''),bool(waking or recovered or turn.get('valid') and engaged))
                        if self.ordinary_message_turn(event):self.message_receipts.expect_intent(event['item_id'])
                        try:events=self.message_receipts.filter(event)
                        except ReceiptRejected as exc:
                            await reject_receipt(exc.response_id,str(exc));continue
                        for released in events:await deliver(released)
                        if recovered:
                            tasks[:]=[task for task in tasks if not task.done()]
                            tasks.append(asyncio.create_task(self.ask_recovered_question(ws,controller_events,state,self.message_request())))
                    raise VoiceTransportError('Qwen cloud voice disconnected')
                async def context_loop():
                    previous=initial['instructions'];previous_modalities=['text']
                    previous_tools=initial['tools']
                    while True:
                        await asyncio.sleep(self.speech['managed_voice'].get('control_poll_interval_s',.05))
                        self.acoustic_tick()
                        if self.sound_frozen():continue
                        for released in self.message_receipts.drain_released():
                            rid=released.get('response_id') or released.get('response',{}).get('id')
                            if released.get('type')=='response.done' or (self.response_inputs.get(rid)==self.input_generation and rid not in self.blocked):
                                await deliver(released)
                                await asyncio.sleep(0)
                        for rid in self.message_receipts.expired():await reject_receipt(rid,'Receipt verification timed out')
                        rejected=self.message_receipts.rejected;self.message_receipts.rejected=[]
                        for rid,reason in rejected:await reject_receipt(rid,reason)
                        if self.pending_voice_switch and self.pending_voice_switch['ready']:raise SpeechVoiceSwitch()
                        self.remember({})
                        await self.resume_courier_calls(ws)
                        await self.resume_program_calls(ws)
                        self.check_task_delivery(state)
                        self.check_courier_delivery(state)
                        if self.message_lane and self.message_lane.task and self.message_lane.task.done():
                            if not self.message_lane.task.cancelled():
                                try:self.message_lane.task.result()
                                except Exception as exc:self.log.write('message_job_failed',error_type=type(exc).__name__)
                            self.message_lane.task=None
                        if self.expression_pending:
                            pending=self.expression_pending
                            ack=self.expression.acknowledgement(pending['intent_id'])
                            if ack or time.monotonic()-pending['created']>self.expression.c['ack_timeout_s']:
                                self.expression_outcome=ack or dict(intent_id=pending['intent_id'],status='failed',error='Control loop did not acknowledge expression target')
                                self.log.write('expression_result',**self.expression_outcome)
                                self.expression_pending=None;self.continue_response=True
                        if self.context['ready']:self.verify_identity(state)
                        if self.turn_engaged and self.presence.wake_until>0 and (self.in_speech or state.generating or self.player.count):
                            self.presence.wake()
                        instructions,identity,permission=self.context_snapshot()
                        self.sync_conversation_animation()
                        animation=self.conversation_animation
                        if animation.error!=getattr(self,'conversation_animation_error',None):
                            self.conversation_animation_error=animation.error
                            self.log.write('conversation_expression_publisher',error=animation.error,
                                audio_blocked=False)
                        for _ in range(16):
                            try:pose=animation.events.get_nowait()
                            except queue.Empty:break
                            self.log.write('conversation_expression_state',**pose)
                        arrived=self.context['ready'] and self.reception.observe(self.presence.last_visible,self.context['mode'],
                            person_id=self.identified_person(),occupied=self.presence.person_count()>0)
                        if arrived and (self.continuing_program_turn() or self.reception_turn_in_progress(state)):
                            # A mode handoff is still this conversation. Consume
                            # the arrival without erasing its retained source.
                            event=('reception_arrival_during_program_followup' if self.continuing_program_turn()
                                   else 'reception_arrival_during_user_turn')
                            self.log.write(event,session_retained=True)
                        elif arrived:
                            self.log.write('reception_arrival',mode=self.context['mode'],
                                observed_hold_s=time.monotonic()-self.reception.present_since,
                                configured_hold_s=self.reception.c['arrival_hold_s'])
                            self.anonymous='anonymous:'+uuid.uuid4().hex
                            self.memory_session=self.folder.parent.name+':'+uuid.uuid4().hex
                            self.reception.conversation.reset();self.thinking_generation=None
                            if self.reception.conversation.enabled:
                                # A real new arrival can greet again. A nearby
                                # user turn that has already begun takes priority.
                                recent=self.turn_ended_at is not None and time.monotonic()-self.turn_ended_at<self.reception.conversation.delay
                                welcome_gate.user_seen=self.in_speech or recent
                            if self.context['mode']=='standby':
                                self.control_pending=True
                                self.emit(dict(kind='cloud_action',revision=self.context['revision'],arguments=dict(action='wake',program=''),text='前台检测到访客'))
                            else:
                                self.announcement_kind='welcome';self.announcement_revision=self.context['revision']
                        if (self.reception.c['enabled'] and self.context['mode']=='interaction' and self.context['ready'] and not permission
                                and self.absence_since is not None and time.monotonic()-self.absence_since>=self.reception.c['return_to_standby_after_s']
                                and not state.generating and not self.player.count and not self.finish_responses
                                and not self.user_turn_pending(state)
                                and self.idle_sleep_revision!=self.context['revision']):
                            self.idle_sleep_revision=self.context['revision']
                            self.emit(dict(kind='cloud_action',revision=self.context['revision'],arguments=dict(action='standby',program='',explicit_wake=False),text='前台持续无人，回接待待机'))
                        await self.continuity_tick(state)
                        if (self.memory and not self.continuity and self.context['mode']=='standby' and self.context['ready']
                                and time.monotonic()>=self.memory_retry_at and (self.memory_task is None or self.memory_task.done())):
                            self.memory_retry_at=time.monotonic()+60
                            self.memory_task=asyncio.create_task(self.compact_memory())
                        self.update_speech_permission(state,permission)
                        self.schedule_farewell(state)
                        if self.explicit_wake_pending:
                            if await self.complete_explicit_wake(ws,state,controller_events,audio_gate,instructions):
                                previous=instructions;previous_modalities=['text','audio']
                            if self.explicit_wake_pending:continue
                        modalities=['text','audio'] if (self.can_speak() or self.finish_responses
                            or self.turn_reply_pending() and self.can_answer_turn(self.turn_item_id)) else ['text']
                        current_tools=self.available_tools()
                        if instructions!=previous or modalities!=previous_modalities or current_tools!=previous_tools:
                            update=dict(instructions=instructions,modalities=modalities)
                            if current_tools!=previous_tools:update['tools']=current_tools
                            await bounded_send(ws,json.dumps(dict(type='session.update',session=update),ensure_ascii=False))
                            previous=instructions;previous_modalities=modalities;previous_tools=current_tools
                            self.log.write('cloud_context_updated',revision=self.context['revision'],face_visible=self.presence.last_visible,
                                recognized=identity['recognized'],person_id=identity['person_id'])
                        self.schedule_task_notice(state)
                        self.schedule_program_check(state)
                        await self.resume_program_followup(ws,controller_events,state)
                        if (self.followup_handoff and self.program_followup is None and self.context['mode']=='interaction'
                                and self.context['ready'] and not self.in_speech and not self.user_turn_pending(state)):
                            self.followup_handoff=False
                        conversation=self.reception.conversation
                        if conversation.enabled and self.reception.c['enabled']:
                            available=(self.public_prompt_task is None or self.public_prompt_task.done()) and self.public_prompt_available(state)
                            welcome=self.announcement_kind=='welcome' and self.announcement_revision==self.context['revision']
                            if welcome:
                                if welcome_gate.user_seen:
                                    self.announcement_revision=None
                                elif available:
                                    self.announcement_revision=None
                                    self.schedule_public_prompt(state,welcome=True)
                            elif conversation.due(time.monotonic(),available=available and self.announcement_revision is None):
                                self.schedule_public_prompt(state,welcome=False)
                        announce=self.announcement_revision==self.context['revision'] and self.can_speak()
                        if conversation.enabled and self.reception.c['enabled'] and self.announcement_kind=='welcome':announce=False
                        if self.program_followup is not None or (self.public_prompt_task and not self.public_prompt_task.done()):continue
                        if self.message_claimed!=self.input_generation and not self.awaiting_recovery_user and not self.response_waiting_user_end and not self.response_request_pending and not self.in_speech and state.generating is None and not (self.message_lane and self.message_lane.task and not self.message_lane.task.done()) and (self.continue_response or (announce and self.can_speak())):
                            request_revision=self.context['revision'];request_turn=self.turn_item_id
                            self.continue_response=False
                            if announce:self.announcement_revision=None
                            if announce:self.turn_person=self.identified_person() or self.anonymous
                            request=dict(type='response.create')
                            if self.message_receipts.instruction():
                                instruction=self.message_receipts.instruction()
                                request['response']=dict(instructions=self.response_instructions(instruction))
                            elif announce:
                                if self.announcement_kind!='welcome':welcome_gate.active=False
                                if self.announcement_kind=='welcome' and not welcome_gate.arm():
                                    self.log.write('controller_announcement_skipped',reason='user_turn_already_seen',announcement='welcome')
                                    continue
                                state.mark_proactive()
                                if self.announcement_kind=='welcome':
                                    directive='控制器已完成准备，当前为前台交互态。先查询get_robot_status确认当前身份，有确认称呼就用该称呼自然欢迎/问候；身份未知只说您好。留言正文只由本地固定流程播报；欢迎语不读留言、不承诺转告、不询问是否收到。不重复长篇介绍。若唤醒时有问题，简短问候后回答。'
                                elif self.announcement_kind=='dialect':directive='控制器已确认目标音色和方言设置。直接使用当前方言简短回应用户上一句，继续原来的聊天；不要再次调用切换工具，不夸口音正宗，不重新介绍自己或主动点播。'
                                else:directive='控制器已完成准备，已回交互态或收到用户再次呼唤。自然简短回应，若有明确问题就回答；不重复欢迎、不重复自我介绍，不自行再次点播。'
                                directive+=self.dialect.instructions()
                                try:item_id=await controller_events.append(ws,self.context,self.announcement_kind,directive)
                                except ControllerAcknowledgementTimeout:
                                    self.log.write('controller_announcement_skipped',reason='ack_timeout',announcement=self.announcement_kind)
                                    continue  # Missing proactive greeting must not tear down an otherwise live conversation.
                                self.log.write('controller_event_confirmed',item_id=item_id,announcement=self.announcement_kind,
                                    revision=request_revision,human_speech=False)
                                request['response']=dict(instructions=self.response_instructions(directive))
                            elif self.tool_repair:
                                request['response']=dict(instructions=self.response_instructions('上一次工具调用没有执行。根据以下错误纠正工具调用，不能只口头宣称完成；不要执行无关操作：'+json.dumps(self.tool_repair,ensure_ascii=False)+self.dialect.instructions()))
                            if (self.in_speech or state.generating is not None or self.context['revision']!=request_revision or self.turn_item_id!=request_turn
                                    or (announce and not self.can_speak())
                                    or (announce and self.announcement_kind=='welcome' and welcome_gate.cancelled)):continue
                            welcome_gate.sent()
                            self.response_request_pending=True;self.last_response_request_at=time.monotonic()
                            await bounded_send(ws,json.dumps(request,ensure_ascii=False))
                async def fault_watch():
                    while True:
                        if self.transport_fault:raise VoiceTransportError(self.transport_fault)
                        await asyncio.sleep(.02)
                tasks=[asyncio.create_task(send()),asyncio.create_task(receive()),asyncio.create_task(context_loop()),asyncio.create_task(fault_watch())]
                try:
                    done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
                    for task in done:task.result()
                except Exception as exc:
                    from vivis.speech.service_recovery import transport_error
                    if isinstance(exc,TransientServiceError) or transport_error(exc):
                        self.health(False,type(exc).__name__)  # Before bounded websocket cleanup waits.
                    raise
                finally:
                    # Detach before WebSocket close waits; audio capture never queues behind close.
                    self.cloud_queue=None
                    for task in tasks:task.cancel()
                    await asyncio.gather(*tasks,return_exceptions=True)
        except (SpeechVoiceSwitch,TransientServiceError):
            preserve_output=True;raise
        except (asyncio.CancelledError,ConversationReset,CloudIdleTimeout):raise
        except Exception as exc:
            from vivis.speech.service_recovery import transport_error
            if transport_error(exc):
                self.health(False,type(exc).__name__)
                raise TransientServiceError('TransportUnavailable',str(exc)) from exc
            self.emit(dict(kind='cloud_error',epoch=self.epoch,error=str(exc)))
            self.enabled=False;self.shutdown.set()
        finally:
            if self.program_check_task:
                self.program_check_task.cancel();await asyncio.gather(self.program_check_task,return_exceptions=True)
            self.cancel_public_prompt()
            animation=getattr(self,'conversation_animation',None)
            if animation:
                animation.context(None,False,False)
            if self.public_prompt_task:await asyncio.gather(self.public_prompt_task,return_exceptions=True)
            self.cancel_farewell('connection_closed')
            if self.farewell_task:
                self.farewell_task.cancel();await asyncio.gather(self.farewell_task,return_exceptions=True)
            for task in tasks:task.cancel()
            if tasks:await asyncio.gather(*tasks,return_exceptions=True)
            if self.memory_task:
                self.memory_task.cancel();await asyncio.gather(self.memory_task,return_exceptions=True)
            self.message_connection=object();self.courier_waiters.clear()
            self.program_waiters.clear()
            self.invalid_question=None;self.question_onset=None;self.recovered_query_generation=None
            self.recovered_expression=None
            self.explicit_wake_pending=None
            self.cloud_audio_enabled=False
            self.cloud_queue=None;self.ws=None;self.message_lane.cancel() if self.message_lane else None
            if state:
                state.interrupt(source='cloud_closed');state.poll_playback()
                if self.continuity:
                    self.continuity.poll(state,self.memory_session);self.continuity.submit('disconnect')
                self.summary['provisional_pauses']+=state.provisional_pauses
                self.summary['invalid_turn_resumes']+=state.invalid_turn_resumes
                self.summary['local_cancel_operations']+=state.cancel_operations
                self.summary['playback_overflows']=self.summary.get('playback_overflows',0)+state.playback_overflows
                self.log.write('cloud_session_closed',epoch=self.epoch,responses=state.responses,usage=state.usage,
                    invalid_turns=state.invalid_turns,invalid_turn_resumes=state.invalid_turn_resumes)
            if not preserve_output:self.audio.disable_output()
            self.cloud_state=None
