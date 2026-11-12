"""Two fixed message flows: final text -> commit; verified recipient -> delivery."""
import asyncio
from dataclasses import replace
import json

from vivis.interaction.message_delivery import MessageService,MessageRequest
from vivis.interaction.message_status import RECEIPTS,message_view
from vivis.interaction.robot_tasks import MessageMeaningUncertain


class StaleMessageTurn(ValueError):pass


class MessageLane:
    def __init__(self,store,people,extract,speak,valid,log,*,courier=None,notify=None):
        self.store=store;self.service=MessageService(store);self.people=people
        self.extract=extract;self.speak=speak;self.valid=valid;self.log=log
        self.last={};self.offered={};self.seen=set();self.task=None;self.active_relay=None
        self.result=None;self.extracting=None;self.continuation=None
        self.courier=courier;self.notify=notify

    async def notice(self,text,request):
        if self.notify:
            try:await self.notify(text,request)
            except (asyncio.CancelledError,StaleMessageTurn):raise
            except Exception as exc:self.log('message_notice_failed',error_type=type(exc).__name__)
        elif self.valid(request):await self.say(text,request)

    def check(self,request):
        if not self.valid(request):raise StaleMessageTurn('话轮、说话人或模式已变化；未提交的旧请求不执行。')

    def cancel(self):
        if self.task and not self.task.done():self.task.cancel()

    def start(self,request):
        key=(request.actor,request.session,request.source_item)
        if key in self.seen:return False
        self.seen.add(key)
        if len(self.seen)>128:self.seen={key}
        self.cancel();self.task=asyncio.create_task(self.run(request));return True

    def next_turn(self,actor,session):
        # A new speech turn may supplement an UNCOMMITTED extraction. It must
        # never silently edit an already-saved message or cross a person/session.
        pending=self.extracting
        if pending and pending.actor==actor and pending.session==session:
            self.continuation=dict(actor=actor,session=session,source=pending.source_text)
        elif self.continuation and (self.continuation['actor']!=actor or self.continuation['session']!=session):
            self.continuation=None

    def has_continuation(self,actor,session):
        return bool(self.continuation and self.continuation['actor']==actor and self.continuation['session']==session)

    def target(self,request,instruction):
        target=instruction.get('task_id') or self.last.get(request.actor,{}).get('task_id')
        if not target:raise ValueError('请说明是哪一条留言。')
        record=self.store.get(target)
        if record['creator']!=request.actor:raise ValueError('这条留言不属于当前人员。')
        return target

    async def say(self,text,request):
        self.check(request)
        try:return await self.speak(text,request)
        except (asyncio.CancelledError,StaleMessageTurn):raise
        except Exception as exc:
            # Voice is an output of the operation, never its database result.
            if self.result is not None:self.result['speech_status']='failed'
            self.log('message_voice_failed',error_type=type(exc).__name__)
            return False

    async def run(self,request):
        """Extraction never runs on partial ASR; it cannot read/modify the DB."""
        self.result=None;self.result_actor=request.actor
        try:
            self.check(request)
            if not request.source_item or not request.source_text:raise ValueError('需要本轮最终识别文字。')
            if request.verified_person!=request.actor:
                self.result=dict(operation_status='needs_clarification',saved=False)
                await self.notice('请正对摄像头确认身份后再交代或查询私人留言。',request);return
            hint=dict(self.last.get(request.actor,{}))
            if self.has_continuation(request.actor,request.session):
                hint['draft_source']=self.continuation['source'];self.continuation=None
            if request.actor in self.offered:
                offered=self.store.get(self.offered[request.actor])
                hint=dict(task_id=offered['task_id'],content=offered['content'],delivery_complete=offered['status'] in ('announced','completed'))
            hint=dict(hint,current_person_id=request.actor,recent_user_turns=list(request.recent_user_turns))
            self.extracting=request
            try:instruction=await self.extract(request.source_text,self.people,hint)
            finally:
                if self.extracting is request:self.extracting=None
            self.check(request);action=instruction['action']
            if request.read_only and action not in ('query','visit_query','courier_query','clarify','none'):
                raise ValueError('本次只恢复查询；如需保存、修改或转达，请重新明确提出请求。')
            self.log('message_fields',actor=request.actor,**{k:instruction[k] for k in ('action','recipient_id','status','missing') if k in instruction})
            recipient=instruction.get('recipient_id')
            if action in ('query','visit_query','relay') and recipient and recipient not in self.people:
                instruction=dict(action='clarify',missing='recipient');action='clarify'
            # Reading someone else's message is a sender query, never delivery
            # to the current actor. Do not silently ignore the requested target.
            if action=='relay' and recipient and recipient!=request.actor:action='query'

            if action=='none':
                if hint.get('draft_source'):self.last.pop(request.actor,None)
                return
            if action=='visit_query':
                if recipient not in self.people:raise ValueError('请说明要查询谁的到访记录。')
                from vivis.interaction.visit_query import visit_evidence,visit_receipt
                result=await asyncio.to_thread(visit_evidence,self.store,request.actor,recipient)
                self.check(request);self.result=dict(operation_status='read',saved=None,result=result)
                self.log('visit_query_result',actor=request.actor,person_id=recipient,evidence=result)
                await self.say(visit_receipt(result,self.people[recipient]),request);return
            if action=='courier_query':
                if not self.courier:raise ValueError('退件留言查询尚未启用。')
                result=await asyncio.to_thread(self.courier.owner_query,request.actor)
                self.check(request);self.result=dict(operation_status='read',saved=None,result=result)
                self.log('courier_owner_query',actor=request.actor,status=result['status'])
                await self.say(result['spoken_receipt'],request);return
            if action=='clarify':
                field=instruction.get('missing','content')
                question={'recipient':'这条留言是给谁的？','content':'请补充要转告的具体内容。',
                    'target':'请说明是哪一条留言。','meaning':'留言里的物品或要求没有听清，请补充这一部分。'}[field]
                self.result=dict(operation_status='needs_clarification',saved=None)
                self.last[request.actor]=dict(draft_source=request.source_text,missing=field)
                await self.say(question+'请确认后我再处理；现有留言记录没有改变。',request);return
            guard=lambda:self.check(request)
            guarded=replace(request,guard=guard,clarified=hint.get('missing')=='meaning')
            if action=='query':
                args=dict(action='list',status=instruction.get('status','all'))
                if instruction.get('recipient_id'):args['recipient_id']=instruction['recipient_id']
                if instruction.get('content_keyword'):args['content_keyword']=instruction['content_keyword']
                out=await asyncio.to_thread(self.service.execute,args,guarded,self.people)
                self.check(request);self.result=out.payload()
                rows=out.result['tasks'];words=[]
                import re
                recent=self.last.get(request.actor,{}).get('task_id')
                if recent and not args.get('content_keyword') and re.search(r'刚才|那条|这条',request.source_text):
                    rows=[message_view(await asyncio.to_thread(self.store.get,recent))]
                self.log('message_query_result',actor=request.actor,recipient_id=args.get('recipient_id'),status=args['status'],count=len(rows),content_keyword=args.get('content_keyword'))
                for row in rows:
                    words.append(row.get('delivery_label','历史已取消')+'。'+row['content'])
                recipient_name=(self.people[args['recipient_id']].get('preferred_address') or self.people[args['recipient_id']].get('display_name') or '该收件人') if args.get('recipient_id') else None
                empty=('没有查到您有权查看的、给'+recipient_name+'的对应状态留言。') if recipient_name else '没有查到您有权查看的对应状态留言。'
                text=empty if not words else '查询到'+str(len(words))+'条留言。'+'。'.join(words)
                if out.result['has_more']:text+='还有更多留言，可以按状态分别查询。'
                await self.say(text,request);return
            if action=='relay':
                target=instruction.get('task_id')
                if target and self.store.get(target)['recipient']!=request.actor:
                    raise ValueError('这条留言不是给当前人员的，不能转告。')
                if not target:
                    waiting=self.store.list('active',recipient=request.actor,limit=2)
                    if not waiting:
                        self.result=dict(operation_status='read',saved=None)
                        await self.say('目前没有待转达的留言。',request);return
                    target=waiting[0]['task_id']
                await self.relay(target,request)
                self.result=dict(operation_status='read',saved=True,result=message_view(self.store.get(target)))
                return
            if action=='retry':
                prior=self.last.get(request.actor)
                if not prior:raise ValueError('没有明确定位到刚才的留言，请先查询留言内容。')
                if prior.get('task_id'):
                    record=message_view(await asyncio.to_thread(self.store.get,prior['task_id']))
                    self.result=dict(operation_status='read',saved=True,result=record)
                    await self.say(record['delivery_label']+'。'+record['content'],request);return
                # The user explicitly retries the same saved draft; keep its
                # original request ID, while rechecking this NEW authorization.
                args=prior['args'];guarded=replace(prior['request'],guard=guard)
            elif action=='create':
                recipient=instruction.get('recipient_id');content=instruction.get('content','').strip()
                if recipient not in self.people:
                    self.last[request.actor]=dict(draft_source=request.source_text,missing='recipient')
                    raise ValueError('这条留言是给谁的？这条留言尚未保存。')
                if not content:
                    self.last[request.actor]=dict(draft_source=request.source_text,missing='content')
                    raise ValueError('请补充要转告的具体内容。这条留言尚未保存。')
                args=dict(action='create',kind='message',title=content[:80],content=content,recipient_id=recipient)
                self.last[request.actor]=dict(args=args,request=request)
            elif action=='supplement':
                target=self.target(request,instruction)
                args=dict(action='supplement',task_id=target,content=instruction.get('content',''))
            else:raise ValueError('本轮仅支持保存、补充、查询和转达留言。')
            out=await asyncio.to_thread(self.service.execute,args,guarded,self.people)
            self.check(request);self.result=out.payload()
            if action in ('create','retry','supplement'):
                self.last[request.actor]=dict(task_id=out.result['task_id'],content=out.result['content'])
            text=RECEIPTS[out.receipt_effect]
            if out.receipt_effect=='created':
                person=self.people[out.result['recipient']]
                label=person.get('preferred_address') or person.get('display_name') or '收件人'
                text='已保存，等见到'+label+'后转告。'
            await self.say(text,request)
        except asyncio.CancelledError:
            if self.result is None:self.result=dict(operation_status='rejected',saved=None)
            self.log('message_operation_cancelled');return
        except StaleMessageTurn:
            if self.result is None:self.result=dict(operation_status='rejected',saved=None)
            self.log('message_operation_cancelled',reason='request_guard_changed')
            try:await self.notice('身份正在复核，本次留言处理已暂停。已有记录不会因此删除，请正对摄像头确认后再查询。',request)
            except (asyncio.CancelledError,StaleMessageTurn):pass
            return
        except MessageMeaningUncertain:
            self.last[request.actor]=dict(draft_source=request.source_text,missing='meaning')
            self.result=dict(operation_status='needs_clarification',saved=False)
            if self.valid(request):await self.say('留言中的物品或要求没有听清，请只补充这一部分。这条留言尚未保存。',request)
        except ValueError as exc:
            self.result=dict(operation_status='needs_clarification',saved=False)
            if self.valid(request):await self.say(str(exc),request)
        except Exception as exc:
            # A timeout can occur after commit: do not say success OR failure.
            if self.result and self.result.get('saved') is True:
                self.log('message_voice_failed_after_commit',error_type=type(exc).__name__);return
            self.result=dict(operation_status='unconfirmed',saved=None)
            self.log('message_operation_unconfirmed',error_type=type(exc).__name__)
            if self.valid(request):
                try:await self.say(RECEIPTS['unconfirmed'],request)
                except Exception:self.log('message_failure_voice_unavailable')

    async def relay(self,task_id,request):
        self.active_relay=task_id;self.offered.pop(request.actor,None)
        try:
            self.check(request)
            row=await asyncio.to_thread(self.store.prepare_relay,task_id,request.actor)
            self.check(request)
            owner=self.people.get(row['creator'],{})
            label=owner.get('preferred_address') or owner.get('display_name') or '委托人'
            from datetime import datetime
            from zoneinfo import ZoneInfo
            day=datetime.fromisoformat(row['created']).astimezone(ZoneInfo(self.store.c['timezone'])).strftime('%m月%d日')
            complete=await self.say(label+'在'+day+'给您留言：'+row['content'],request)
            self.check(request)
            if complete:
                await asyncio.to_thread(self.store.mark_announced,task_id,request.actor,'trusted-tts',guard=lambda:self.check(request))
                self.offered[request.actor]=task_id
                self.log('message_delivered',task_id=task_id,recipient=request.actor,acknowledgement_required=False)
        except (asyncio.CancelledError,StaleMessageTurn):self.log('message_relay_interrupted',task_id=task_id)
        except Exception as exc:self.log('message_relay_failed',task_id=task_id,error_type=type(exc).__name__)
        finally:self.active_relay=None


def query_recipient_fields(out,text,people,hint):
    """Validate an extracted addressee against literal source evidence, not similarity."""
    if out.get('action') not in ('query','visit_query','relay'):return out
    import re
    named=re.search(r'给([^，。？！?]{1,20}?)的留言',text)
    reference=named.group(1).strip() if named else out.get('recipient_reference','').strip()
    recipient=out.get('recipient_id')
    clarify=dict(action='clarify',missing='recipient')
    if not reference:
        return clarify if recipient else out  # Unscoped list stays within current actor's permissions.
    if reference not in text:return clarify
    if reference in ('我','自己','本人'):
        resolved=hint.get('current_person_id')
    elif reference in ('他','她','那位','那个人') and out['action'] in ('query','visit_query'):
        resolved=None
        # Use only the nearest explicit topic in bounded USER speech supplied
        # by the caller. Assistant guesses and global history are not evidence.
        for previous in reversed(hint.get('recent_user_turns',[])):
            matches={pid for pid,p in people.items() if any(a and a in previous for a in
                (p.get('display_name'),p.get('preferred_address'),*p.get('aliases',[])))}
            if matches:
                resolved=next(iter(matches)) if len(matches)==1 else None
                break
    else:
        matches=[]
        for pid,person in people.items():
            aliases=[person.get('display_name'),person.get('preferred_address'),*person.get('aliases',[])]
            if reference in aliases:matches.append(pid)
        resolved=matches[0] if len(matches)==1 else None
    if resolved is None or recipient and recipient!=resolved:return clarify
    return dict(out,recipient_id=resolved)


def extraction_tool(people=None):
    tool=dict(type='function',name='message_fields',description='只把最终识别原话提取为一条指令，不操作数据库、不判断保存成功、不回复用户。写入create必须有明确recipient_id和content；缺收件人或内容用clarify及missing，不猜。supplement必须明确补充旧留言且能定位task_id。查询query只提取收件人/范围/关键词，不生成新内容；relay只用于向当前收件人转达。',
        parameters=dict(type='object',additionalProperties=False,required=['action','recipient_reference'],
            properties=dict(recipient_reference=dict(type='string',description='本轮原话中的目标称呼或代词，逐字摘录，例如小王、他、我；不把他改写成姓名。'),action=dict(type='string',enum=['create','supplement','retry','query','visit_query','courier_query','relay','clarify','none']),
                recipient_id=dict(type='string',description='从给定people中对应原话称呼；create必需，不能从当前说话人猜收件人。'),
                content=dict(type='string',maxLength=500,description='create/supplement的留言正文，保留原话事实、数量和时间；query不填新留言。'),
                task_id=dict(type='string',description='仅使用提供的旧留言ID；明确补充或重试旧留言时填写，不编造。'),
                content_keyword=dict(type='string',maxLength=40,description='查询物品名或事项名，不含买/准备/让等动作及数量。例如买冰淇淋填冰淇淋，准备两份测试资料填测试资料。必须逐字存在于本轮原话。查询全部留言时留空。'),
                status=dict(type='string',enum=['active','all','pending','announced','completed','cancelled']),
                missing=dict(type='string',enum=['','recipient','content','target','meaning']))))
    if people is not None:tool['parameters']['properties']['recipient_id']['enum']=['',*people]
    return tool


async def extract_qwen(url,secret,session,log,text,people,hint):
    """Same configured Qwen model, isolated text-only extraction, one result."""
    from vivis.speech.realtime import websocket_connection,connect_session
    public=[dict(id=k,name=v.get('display_name',''),address=v.get('preferred_address','')) for k,v in people.items()]
    # Only one explicitly referenced recent message, never a history search.
    recent={k:v for k,v in hint.items() if k in ('task_id','content','draft_source','missing','delivery_complete','current_person_id','recent_user_turns')}
    settings=dict(session,enable_search=False,modalities=['text'],tools=[extraction_tool(people)],instructions=(
        '你是结构化提取器。只调用一次message_fields，不说话、不查历史、不判断保存成功。'
        '提取用户当前最终文字中的收件人和留言内容。自然交代让同事买/带/准备东西视为create。'
        '明确说补充才supplement；明确重新保存刚才同一请求才retry；问保存了吗或状态是query。完整播报即已转达，不需要确认，单纯收到、知道了、明白了为none。'
        '只有明确要听给当前说话人自己的留言时才用relay，recipient_id填recent.current_person_id。刚转告的内容没听清也可relay。'
        '给别人的留言保存了吗、把内容完整说一下、帮我查给小王的留言都用query；完整读出不等于转告收件人，不用relay。'
        '本人询问自己给快递员的留言有没有、退件留言保存了吗、我的取件码是什么，使用courier_query，只查询当前核验委托人自己的独立退件账本；不查普通同事留言、不替别人查、不把快递员猜成某位熟人。'
        'query和relay必须用recipient_reference逐字摘录原话中的收件人称呼，没有明确称呼时填空。不要把第三方名字替换成当前说话人。'
        '查询某人来过没有、见过没有，用visit_query，recipient_id为被查询的人；不回答来过或没来过，由程序查实际记录。'
        'query和visit_query允许根据recent.recent_user_turns中最近明确提到的唯一人物理解本轮他/她；recipient_reference仍填本轮代词，recipient_id填那个明确人物。前文提到多人有歧义才问recipient。这些原话只用于只读查询指代，不作为新保存、修改或转告的授权。'
        '人名只按people中的姓名和称呼对应ID，不能拼写一个新ID，也不能把听错或不明的人名猜成名单中的别人。比如小儿不是小王，收件人不明用clarify和missing=recipient，不查询一个猜测对象后说没有留言。'
        '普通聊天none。缺谁问recipient，缺内容问content，物品不明问meaning，不猜。'
        '查询默认status=all；用户明确问待转达或未完成用active，已转达或已完成留言用announced（含历史关闭记录），明确只问历史确认记录才用completed。announced就是已转达，不是待确认或未完成。'
        'query若提及具体收件人，填写recipient_id；只提刚才那条时，可用recent中的task_id定位。'
        '问特定事项例如买冰淇淋的事转告了吗，status=all且content_keyword=冰淇淋，只查询该事项，不读全部历史。关键词必须逐字存在于本轮原话，不能从历史猜测。'
        'content_keyword使用物品名/事项名，禁止带动作或数量：买冰淇淋→冰淇淋（不是买冰淇淋），买USB hub→USB hub，准备两份测试资料→测试资料；否则会漏掉账本里买一个冰淇淋等不同数量表达。'
        'recent.delivery_complete=true表示刚完整转达过这条留言；只有明确没听清、要求重说时才用relay和该task_id重复播报。'
        'recent含未完成的draft_source时，本句是对缺项的补充回答，结合该单条草稿提取完整内容；不得扩展到别的历史。'
        '不自动遍历、合并历史，不输出成功回执；task_id仅能使用给出的最近留言ID或用户明确提供的ID。'
        '保留数量、日期、地点、用途，不把转告当实际执行。相对日期保留原词。数据：'
        +json.dumps(dict(people=public,recent=recent),ensure_ascii=False)))
    async def run():
        async with websocket_connection(url,secret) as ws:
            await connect_session(ws,settings,log)
            await ws.send(json.dumps(dict(type='conversation.item.create',item=dict(type='message',role='user',content=[dict(type='input_text',text=text)])),ensure_ascii=False))
            await ws.send(json.dumps(dict(type='response.create')))
            async for raw in ws:
                event=json.loads(raw)
                if event.get('type')=='error':raise RuntimeError('Message extraction provider error')
                if event.get('type')=='response.function_call_arguments.done':
                    if event.get('name')!='message_fields':raise ValueError('Unexpected extraction tool')
                    out=json.loads(event['arguments']);schema=extraction_tool()['parameters']
                    if not isinstance(out,dict) or not {'action'}<=out.keys()<=schema['properties'].keys():raise ValueError('Invalid extracted fields')
                    if out['action'] not in schema['properties']['action']['enum']:raise ValueError('Invalid extracted action')
                    if any(not isinstance(v,str) for v in out.values()) or len(out.get('content',''))>500:raise ValueError('Invalid extracted values')
                    if out['action']=='query' and out.get('status','all') not in schema['properties']['status']['enum']:
                        raise ValueError('Invalid extracted query status')
                    if out['action']=='clarify' and out.get('missing') not in ('recipient','content','target','meaning'):
                        out['missing']='meaning'
                    return query_recipient_fields(out,text,people,hint)
                if event.get('type')=='response.done':raise ValueError('未能提取明确留言指令，请只补充缺少的对象或内容。')
        raise RuntimeError('Message extraction disconnected')
    from vivis.speech.runtime_policy import policy
    return await asyncio.wait_for(run(),timeout=policy()['message_extract_s'])
