"""Source-scoped side conversation using the existing Qwen voice model.

No second microphone/output stream, no main-person history, no side-to-main
identity mutation, and no reference enrollment from an introducer's recording.
"""
import asyncio
import base64
from datetime import datetime
import json
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from vivis.interaction.introductions import Introductions
from vivis.speech.realtime import connect_session,websocket_connection
from vivis.speech.runtime_policy import bounded_send


INTRO_TOOL=dict(type='function',name='introduce_visitor',
    description='仅当侧方发言人明确介绍镜头前这位新访客时调用。姓名display_name或称呼preferred_address至少一项，必须摘自介绍原话，不补全姓名。background保存介绍的职业背景，可为空。缺姓名和称呼则调用空字段，由程序要求补充。程序单独核验介绍人本轮声纹、访客绑定及首次注册；不能把介绍人当主访客。',
    parameters=dict(type='object',properties={k:dict(type='string',description=d) for k,d in (
        ('display_name','介绍原话中的姓名；没有则空字符串'),('preferred_address','介绍原话中的交流称呼，如杨教授；没有则空字符串'),
        ('background','仅概括原话介绍的职业背景，不猜测补充；没有则空字符串'))},
        required=['display_name','preferred_address','background'],additionalProperties=False))
MESSAGE_TOOL=dict(type='function',name='side_message',
    description='仅当本轮侧方原话明确要求留言、转告、带话或查询自己的留言时调用。无需复制内容或猜收件人参数，程序从本轮定稿原话提取，并只按侧方说话人的权限执行。普通聊天不调用。',
    parameters=dict(type='object',properties={},required=[],additionalProperties=False))


class SideDialogue:
    def __init__(self,voice,request):
        self.v=voice;self.request=request;self.frames=[];self.capture_done=asyncio.Event()
        self.cancelled=False;self.actor=None;self.transcript='';self.rids=set();self.spoken=[]
        self.result=None;self.paused=set();self.message_lane=None
        self.terminal_tool=False

    def valid(self):
        v=self.v;r=self.request
        return (not self.cancelled and not v.shutdown.is_set() and v.health_available and v.enabled
            and r['connection'] is v.message_connection and r['session']==v.memory_session
            and r.get('generation',v.input_generation)==v.input_generation
            and r['revision']==v.context['revision'] and v.context['mode']=='interaction' and v.context['ready'])

    def feed(self,stamp,pcm,far):
        if self.capture_done.is_set() or not self.valid():return
        if self.frames and not .005<=stamp-self.frames[-1][0]<=.035:
            self.cancelled=True;self.capture_done.set();return
        if len(self.frames)>=4500:
            self.cancelled=True;self.capture_done.set();return  # 90 s, bounded.
        self.frames.append((stamp,pcm,far))

    def end(self):self.capture_done.set()

    def stop(self):
        self.cancelled=True;self.capture_done.set()
        if self.message_lane:self.message_lane.cancel()
        for rid in self.rids:self.v.player.cancel(rid)

    async def identify(self,pcm,far):
        speaker=getattr(getattr(self.v,'continuity',None),'speaker',None)
        if not speaker:return
        source=dict(turn_id='side_'+self.request['id'],session=self.request['session'],
            voice_context='side:'+self.request['guest_key']+':'+str(self.request.get('sector')),audio_epoch=speaker.audio_epoch,
            audio_started=self.frames[0][0],audio_ended=self.frames[-1][0]+.02)
        self.result=await speaker.recognize_external(pcm,far,source,self.valid)
        evidence=(self.result or {}).get('identity_evidence',{})
        if evidence.get('state')=='confirmed' and evidence.get('person_id') in self.v.knowledge.people:
            self.actor=evidence['person_id']
        self.v.log.write('side_speaker_result',side_turn=source['turn_id'],main_person=self.request['person'],
            actor=self.actor,evidence=evidence,enrollment_allowed=False)

    def instructions(self):
        v=self.v
        actor=dict(person_id=self.actor,**v.knowledge.people[self.actor]) if self.actor else None
        draft=self.draft()
        return (v.knowledge.persona+'\n'+v.knowledge.business+
            '\n你正在回应侧方插话者，镜头前另有主聊天人，两者不能混淆。只简短回应本句，一般一句话，随后继续接待主访客。'
            '普通插话尽量用8到15个字回答，不展开长篇解释；复杂问题可简短提议稍后细聊。介绍后的欢迎是面向主访客，可用一两句自然接话。'
            '没有发送图像，不猜测画面、位置或身份。不要说固定底座所以脖子不能转。程序管理眼颈短暂侧看和返回，不要承诺已转向。'
            '明确介绍新访客时调用introduce_visitor，等工具回执后使用访客称呼自然欢迎，可顺背景问一个简单问题。'
            '介绍人、被介绍访客、主聊天身份分别独立；本轮speaker仅代表插话者，不代表被介绍人。'
            '姓名或称呼缺失时友好问“这位老师怎么称呼？”；未注册完不能声称下次一定认识或已经采集完毕。'
            '工具失败如实说明简短原因，不编造成功；查询天气等无实时数据时不猜。'
            '有效动作需求必须调工具，问会不会不等于要求执行。留言用side_message。'
            '已介绍的职业背景仅来自介绍人，不等于已核实。工具返回待识别时可以改口称呼，不能冒充图声识别成功。'
            '\ncontroller_public='+json.dumps(dict(speaker=actor,guest=self.request.get('guest'),
                unfinished_introduction=draft,
                programs=v.programs,current_local_time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='minutes')),
                ensure_ascii=False))

    def draft(self):
        draft=getattr(self.v,'introduction_draft',None)
        return draft if (draft and self.actor and draft['introducer']==self.actor
            and draft['guest_key']==self.request['guest_key'] and draft['session']==self.request['session']
            and 0<=time.monotonic()-draft['stamp']<=120) else None

    async def tool(self,name,args):
        if not self.valid():raise ValueError('这句话已经过期，不执行旧请求')
        if not isinstance(args,dict):raise ValueError('工具参数格式不完整，请重新说明本次请求')
        v=self.v;r=self.request
        def guard():
            if not self.valid():raise ValueError('这句话已过期，未提交的操作不再执行')
        if name=='introduce_visitor':
            # Use the exact selected visitor captured BEFORE the head glances;
            # never the face currently seen in the introducer's direction.
            if not self.actor:raise ValueError('还没确认介绍人的声音，请熟人把姓名、称呼和介绍连成一句再说')
            if r['person']:raise ValueError('主访客已有业务身份，不能按陌生人重复登记')
            guest=r.get('guest')
            if not guest:raise ValueError('未锁定镜头前的单人访客，请让被介绍人正对摄像头')
            # The immutable token was fresh at the utterance onset. Allow the
            # introduction to finish; vision validates current token at commit.
            guest=dict(guest,stamp=time.monotonic())
            draft=self.draft();original=(draft['text']+'。' if draft else '')+self.transcript
            output=await asyncio.to_thread(Introductions(v.knowledge.root).request,args,guest=guest,introducer=self.actor,
                source=dict(turn_id='side_'+r['id'],text=original),session=r['session'])
            if output.get('accepted'):
                v.introduction_draft=None
                v.introduced_guest=dict(guest=guest,information=args,person_id=output['person_id'],session=r['session'])
            elif output.get('status')=='needs_address':
                v.introduction_draft=dict(introducer=self.actor,guest_key=r['guest_key'],session=r['session'],
                    text=original[-3000:],information=args,stamp=time.monotonic())
            v.log.write('visitor_introduction',side_turn=r['id'],introducer=self.actor,**output)
            return output
        if name=='get_robot_status':
            return dict(accepted=True,side_speaker=self.actor,main_person_retained=r['person'],
                current_voice_result=(self.result or {}).get('identity_evidence'),sound_attention=v.sound_status(),
                notice='本结果只说明本轮侧方声音，不改变主聊天人的身份。')
        if name=='set_expression':
            if args.get('source')=='social':
                return dict(accepted=True,status='keep_existing_social_animation',
                    instruction='继续自然语音回应；侧方不另发无回答绑定的自动表情，不宣称动作已完成。')
            return dict(accepted=True,**v.expression.publish(args,r['revision']))
        if name=='robot_program':
            from vivis.interaction.cloud_control import controller_intent,preparation_reply
            intent=controller_intent(args,v.programs,'interaction')  # Same validated catalogue/actions.
            notice=preparation_reply(intent,v.programs)
            if notice:await self.say(notice['text'])
            if not self.valid():raise ValueError('新请求已取代旧的节目要求')
            v.emit(dict(kind='cloud_action',revision=r['revision'],arguments=args,text='侧方有效请求',
                source='side_dialogue',side_turn=r['id']))
            return dict(accepted=True,status='pending_controller_execution')
        if name=='side_message':
            if args!={}:raise ValueError('留言工具不接受额外参数')
            if not self.actor:raise ValueError('本轮侧方说话人身份尚未确认，不能借用主聊天人的权限查询或保存私人留言')
            if not v.tasks:raise ValueError('留言账本未启用')
            from vivis.interaction.message_lane import MessageLane
            from vivis.interaction.message_delivery import MessageRequest
            receipts=[]
            async def say(words,request):
                complete=await self.say(words)
                if complete:receipts.append(words)
                return complete
            self.message_lane=MessageLane(v.tasks,v.knowledge.people,v.extract_message,say,
                lambda _:self.valid(),v.log.write,courier=v.courier)
            turn='side_'+r['id']
            mid=await asyncio.to_thread(v.memory.append,self.actor,r['session'],'user',self.transcript)
            request=MessageRequest(self.actor,r['session'],turn,mid,self.transcript,self.actor,
                guard=guard)
            await self.message_lane.run(request)
            self.terminal_tool=(self.message_lane.result or {}).get('operation_status')!='chat'
            if self.terminal_tool and not receipts:
                v.emit(dict(kind='voice_warning',epoch=v.epoch,
                    message='侧方留言语音回执未完整播出，请核对账本结果；不据此宣称留言失败或已经转达。'))
            return dict(self.message_lane.result or {},spoken_receipts=receipts,
                instruction=('本次仅由程序核验并播报业务回执；不要再生成成功或完成声明。'
                    if self.terminal_tool else '本句不是留言事务，继续回答原话，不承诺保存或转告。'))
        if name=='courier_pickup':
            if not v.courier:raise ValueError('快递账本未启用')
            action=args.get('action')
            if action=='lookup':result=await asyncio.to_thread(v.courier.lookup,args.get('owner_name',''),self.transcript)
            else:
                if not self.actor:raise ValueError('本轮侧方身份未确认，不能办理委托人的私人业务')
                if action=='owner_query':result=await asyncio.to_thread(v.courier.owner_query,self.actor)
                elif action=='owner_visit':result=await asyncio.to_thread(v.courier.visit_query,self.actor)
                elif action=='register':
                    if not v.memory:raise ValueError('取件码账本未启用')
                    mid=await asyncio.to_thread(v.memory.append,self.actor,r['session'],'user',self.transcript)
                    if not self.valid():raise ValueError('原取件码登记已失效')
                    p=v.knowledge.people[self.actor];label=p.get('preferred_address') or p['display_name']
                    result=await asyncio.to_thread(v.courier.register,self.actor,label,list(dict.fromkeys([label,p['display_name']])),
                        args.get('code',''),[mid],source_reason='本轮侧方声纹确认，取件码核对该发言人定稿原话。',guard=guard)
                    result=dict(accepted=True,spoken_receipt='您的退件取件码'+ '、'.join(result['code'])+'已保存，快递员来取时我会转告。')
                elif action=='cancel':
                    result=dict(await asyncio.to_thread(v.courier.cancel,args.get('handoff_id',''),self.actor,guard=guard),spoken_receipt='这条退件取件码转告已取消。')
                else:raise ValueError('请明确要登记、查询还是取消取件码转告')
            receipt=result.get('spoken_receipt')
            if receipt:
                self.terminal_tool=True
                complete=await self.say(receipt)
                if complete and self.valid() and action=='lookup' and result.get('status')=='available':
                    await asyncio.to_thread(v.courier.mark_announced,result['handoff_id'],result['version'],'side:'+r['id'])
                return dict(accepted=True,spoken=complete,status=result.get('status'),
                    instruction='回执已由程序播报，不重复朗读；只有完整播完的取件码才记录转告。')
            return result
        raise ValueError('此插话工具尚不可用，请回到主交互继续')

    async def say(self,text):
        """Program receipts use existing verified TTS, DAC and A2F."""
        from vivis.speech.trusted_reading import RealtimeReader
        from vivis.speech.message_voice import MessageSpeaker
        previous=self.v.player.priority_response
        if previous in self.rids:
            await self.drain(previous);self.v.player.end_priority(previous)
        engine=RealtimeReader(self.v.url,self.v.secret,dict(self.v.session,voice=self.v.active_voice),self.v.log)
        async def generate(words):
            cached=self.v.fixed_audio(words)
            return cached if cached is not None else await engine.generate(words)
        speaker=MessageSpeaker(engine,self.v.player,self.v.cloud_state,lambda _:self.valid(),
            lambda words:self.v.emit(dict(kind='cloud_transcript',epoch=self.v.epoch,role='assistant',text=words,source='side_dialogue')),
            self.v.log.write,tts=SimpleNamespace(generate=generate),source='side_dialogue',priority=True)
        return await speaker.say(text,SimpleNamespace(source_item='side_'+self.request['id'],session=self.request['session']))

    async def run(self):
        v=self.v
        identity_task=None
        try:
            if not self.valid():return
            session=dict(v.session,instructions=self.instructions(),enable_search=False,turn_detection=None,
                max_history_turns=1,voice=v.active_voice,tools=[INTRO_TOOL,MESSAGE_TOOL,*[
                    t for t in v.session['tools'] if t['name'] in ('robot_program','set_expression','get_robot_status','courier_pickup')]])
            async with websocket_connection(v.url,v.secret) as ws:
                await connect_session(ws,session,v.log)
                cursor=0;deadline=time.monotonic()+91
                # Connect/stream WHILE the introducer speaks, rather than add a
                # new connection and re-upload the whole clip after they finish.
                while not self.capture_done.is_set() or cursor<len(self.frames):
                    if not self.valid():return
                    if time.monotonic()>deadline:raise TimeoutError('Side capture deadline')
                    batch=self.frames[cursor:cursor+10]
                    if not batch:await asyncio.sleep(.02);continue
                    part=b''.join(f[1] for f in batch);cursor+=len(batch)
                    await bounded_send(ws,json.dumps(dict(type='input_audio_buffer.append',audio=base64.b64encode(part).decode())))
                if not self.valid() or not self.frames:return
                pcm=b''.join(f[1] for f in self.frames);far=b''.join(f[2] for f in self.frames)
                identity_task=asyncio.create_task(self.identify(pcm,far))
                await bounded_send(ws,json.dumps(dict(type='input_audio_buffer.commit')))
                async for raw in ws:
                    if not self.valid():return
                    event=json.loads(raw);kind=event.get('type')
                    if kind=='error':raise RuntimeError('Side voice provider error')
                    if kind=='conversation.item.input_audio_transcription.failed':raise RuntimeError('侧方语音转写未完成')
                    if kind=='conversation.item.input_audio_transcription.completed':
                        self.transcript=event.get('transcript','').strip()
                        if not self.transcript:return
                        await identity_task
                        if not self.valid():return
                        await bounded_send(ws,json.dumps(dict(type='session.update',session=dict(instructions=self.instructions()))))
                        profile=v.knowledge.people.get(self.actor,{})
                        label=profile.get('preferred_address') or profile.get('display_name') or 'other'
                        v.emit(dict(kind='cloud_transcript',epoch=v.epoch,role='user',text=self.transcript,
                            source='side_dialogue',side_actor=self.actor,main_person=self.request['person'],
                            identity_label='侧方'+label+'（主聊天身份保持）'))
                        v.log.write('side_utterance',side_turn=self.request['id'],text=self.transcript,
                            actor=self.actor,main_person=self.request['person'],enrollment_allowed=False)
                        if v.continuity:
                            v.continuity.external_turn('side_'+self.request['id'],self.request['session'],self.actor,pcm,self.transcript)
                        await bounded_send(ws,json.dumps(dict(type='response.create')));break
                else:raise RuntimeError('Side speech disconnected before final transcript')
                self.paused.update(v.player.pause(None)['responses'])
                calls=set();pending=False;rounds=0;total=0;buffered=[];words=None
                async for raw in ws:
                    if not self.valid():return
                    event=json.loads(raw);kind=event.get('type');rid=event.get('response_id') or event.get('response',{}).get('id')
                    if kind=='error':raise RuntimeError('Side reply provider error')
                    if kind=='response.created':
                        previous=v.player.priority_response
                        if previous in self.rids:v.player.end_priority(previous)
                        self.rids.add(rid);v.player.begin_priority(rid)
                        buffered=[];words=None
                        v.cloud_state.responses[rid]=dict(source='side_dialogue',user_turn=None,
                            archive_item='side_'+self.request['id'],archive_session=self.request['session'],
                            status='in_progress',received_audio_s=0.)
                    elif kind=='response.function_call_arguments.done':
                        call=event.get('call_id')
                        if call in calls:continue
                        calls.add(call)
                        if len(calls)>6:raise ValueError('Too many side tool calls')
                        try:
                            args=json.loads(event.get('arguments','{}'))
                            output=await self.tool(event.get('name'),args)
                        except (ValueError,OSError) as exc:output=dict(accepted=False,notice=str(exc)[:240])
                        await bounded_send(ws,json.dumps(dict(type='conversation.item.create',item=dict(type='function_call_output',
                            call_id=call,output=json.dumps(output,ensure_ascii=False))),ensure_ascii=False));pending=True
                    elif kind=='response.audio.delta':
                        if rid not in self.rids:raise ValueError('Unbound side audio')
                        part=base64.b64decode(event['delta'],validate=True);total+=len(part)
                        if len(part)%2 or total>48000*30:raise ValueError('Side reply exceeds thirty seconds')
                        # Do not play a model's premature “saved/done” before
                        # discovering a business tool later in this response.
                        buffered.append(part)
                    elif kind=='response.audio_transcript.done':
                        words=event.get('transcript','')
                    elif kind=='response.done':
                        if event.get('response',{}).get('status')!='completed':
                            raise RuntimeError('Side response did not complete')
                        if not self.terminal_tool:
                            for part in buffered:
                                # Bound each scheduler turn even if a provider
                                # supplies a large audio delta in one frame.
                                for offset in range(0,len(part),4800):
                                    packet=part[offset:offset+4800]
                                    while v.player.count+len(packet)>v.player.maximum:
                                        if not self.valid():return
                                        await asyncio.sleep(.02)
                                    if not self.valid():return
                                    v.player.add(rid,packet)
                                    v.cloud_state.responses[rid]['received_audio_s']+=len(packet)/48000
                                    await asyncio.sleep(0)
                            if words:
                                self.spoken.append((rid,words));v.cloud_state.responses[rid]['transcript']=words
                        v.player.finish(rid)
                        if rid in v.cloud_state.responses:v.cloud_state.responses[rid]['status']='completed'
                        if pending:
                            if self.terminal_tool:break
                            rounds+=1
                            if rounds>4:raise ValueError('Side tool continuation limit')
                            await self.drain(rid)
                            v.player.end_priority(rid)
                            pending=False;await bounded_send(ws,json.dumps(dict(type='response.create')))
                        else:break
            deadline=time.monotonic()+35;announced=set()
            while self.valid() and time.monotonic()<deadline:
                v.cloud_state.poll_playback()
                for rid,words in self.spoken:
                    if rid in v.player.started_responses and rid not in announced:
                        announced.add(rid);v.emit(dict(kind='cloud_transcript',epoch=v.epoch,role='assistant',text=words,source='side_dialogue'))
                if all(self.drained(r) for r in self.rids):break
                await asyncio.sleep(.02)
        except asyncio.CancelledError:raise
        except Exception as exc:
            v.log.write('side_dialogue_failed',side_turn=self.request['id'],error_type=type(exc).__name__,error=str(exc)[:240])
            if self.valid():
                v.emit(dict(kind='voice_warning',epoch=v.epoch,message='侧方这句话暂未完整处理，请再说一次；主聊天身份保留。'))
        finally:
            if identity_task and not identity_task.done():
                identity_task.cancel();await asyncio.gather(identity_task,return_exceptions=True)
            for rid in self.rids:v.player.cancel(rid);v.player.end_priority(rid)
            if self.valid() and v.cloud_state:
                v.player.resume(self.paused-set(v.cloud_state.blocked)-set(v.cloud_state.server_cancelled))
            v.log.write('side_dialogue_finished',side_turn=self.request['id'],actor=self.actor,
                main_person=self.request['person'],cancelled=self.cancelled)

    async def drain(self,rid):
        deadline=time.monotonic()+35
        while self.valid() and time.monotonic()<deadline:
            self.v.cloud_state.poll_playback()
            if self.drained(rid):return
            await asyncio.sleep(.02)
        if self.valid():raise RuntimeError('Side audio playback did not progress')

    def drained(self,rid):
        response=self.v.cloud_state.responses.get(rid,{})
        expected=round(response.get('received_audio_s',0)*48000)
        return not expected or (self.v.player.consumed.get(rid,0)>=expected
            and response.get('empty_dac',float('inf'))<=time.monotonic())
