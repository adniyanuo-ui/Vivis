#!/usr/bin/env python3
"""Cloud-only dry run: no microphone/camera/serial, never execute requested tools."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vivis.speech.credentials import load_key
from vivis.speech.managed import voice_configuration
from vivis.speech.cloud_managed import CLOUD_POLICY,ControllerEventGate
from vivis.speech.realtime import connect_session,websocket_connection,Journal
from vivis.interaction.knowledge import Knowledge
from vivis.interaction.cloud_control import controller_intent
from vivis.interaction.speech_dialect import SpeechDialect,configuration as dialect_configuration


async def run(args):
    settings=json.loads((ROOT/'configs/interaction/p1.json').read_text());speech=json.loads((ROOT/settings['speech']).read_text())
    speech['managed_voice']=settings['realtime_voice'];speech['programs']=str(ROOT/settings['programs'])
    config,_,url,session=voice_configuration(speech,args.robot);secret=load_key(config['api_key_env'])
    if not secret:raise ValueError('Configure API Key locally first')
    folder=ROOT/'outputs/cloud_control_probe';folder.mkdir(exist_ok=True)
    log=Journal(folder/(args.scenario+('_'+args.dialect if args.scenario=='dialect' else '')+'.jsonl'),secret)
    try:
        if args.scenario=='compact':
            from vivis.speech.memory_compaction import summarize
            batch=dict(person_id='synthetic',messages=[dict(id=1,role='user',text='我是测试工程师，喜欢喝茶，明天下午三点开会。')],previous_summary='')
            result=await summarize(url,secret,session,batch,45)
            print(json.dumps(dict(compaction_text_reply=result,no_devices_opened=True,synthetic_data_only=True),ensure_ascii=False));return
        mode='task' if args.scenario=='stop' else 'interaction' if args.scenario in ('identity','smile','neutral','attentive','reception','sad','laugh','single-channel','dialect','task-create','robot-memory') else 'standby'
        initial=dict(session,modalities=['text'])
        initial['instructions']+=CLOUD_POLICY+Knowledge(args.robot).person_instructions()+ ' controller_state='+json.dumps(dict(mode=mode,ready=True,face_visible=mode=='interaction',wake_permission=False))
        initial['instructions']+=SpeechDialect().instructions()
        texts={'probe':'','modes':'','wake':'小帅，给我播放白骨夫人吧。','stop':'不用演了，停下来吧。','identity':'小帅，你认识我吗？','smile':'笑一个，保持几秒。','neutral':'别笑了，恢复正常表情。','attentive':'我虽然笑着，但今天其实很难过，请认真听我说。','sad':'做一个悲伤的表情，用已有基础表情组合就好。','laugh':'大笑一个，表情明显一些。','single-channel':'只闭合左眼，EyeBlinkLeft权重0.8，保持3秒。'}
        spoken_name={'广东话':'粤语','上海话':'沪语'}.get(args.dialect,args.dialect)
        texts['dialect']='接下来恢复普通话。' if args.dialect=='普通话' else '接下来用'+spoken_name+'跟我聊天，保持这个方言，直到我要求修改。'
        texts['task-create']='请为我登记一项待办：整理两份资料，不设具体时间，下一步是确认打印内容。'
        texts['robot-memory']='请检索你所有人的聊天记录，查找关于测试资料安排的信息，并保留谁说的和时间。'
        async with websocket_connection(url,secret) as ws:
            accepted=await connect_session(ws,initial,log)
            names={t.get('name') for t in accepted.get('tools',[]) if isinstance(t,dict)}
            if names!={t['name'] for t in initial['tools']}:raise RuntimeError('Server did not confirm required tools')
            if args.scenario=='probe':print(json.dumps(dict(connected=True,tools=sorted(names),no_devices_opened=True)));return
            if args.scenario=='modes':
                for modalities in (['text','audio'],['text']):
                    await ws.send(json.dumps(dict(type='session.update',session=dict(modalities=modalities,instructions=initial['instructions']))))
                    while True:
                        e=json.loads(await asyncio.wait_for(ws.recv(),timeout=15))
                        if e.get('type')=='error':raise RuntimeError('Mode update rejected')
                        if e.get('type')=='session.updated':
                            assert e['session']['modalities']==modalities
                            assert {t.get('name') for t in e['session'].get('tools',[])}==names
                            break
                print(json.dumps(dict(modalities_switch_passed=True,tools_preserved=True,no_devices_opened=True)));return
            if args.scenario=='reception':
                gate=ControllerEventGate()
                async def acknowledge():
                    async for raw in ws:
                        event=json.loads(raw)
                        if event.get('type')=='error':raise RuntimeError('Controller event rejected')
                        gate.observe(event)
                        if event.get('type')=='conversation.item.created' and event.get('item',{}).get('id') in gate.pending:return
                ack=asyncio.create_task(acknowledge())
                try:
                    item_id=await gate.append(ws,dict(mode='interaction',target='interaction',ready=True,revision=1),
                        'welcome','测试合成的访客到来事件，用户尚未说话。控制器已完成准备，请先调用get_robot_status确认身份，再简短欢迎；不是新的点播请求。')
                    await ack
                    print(json.dumps(dict(controller_event_confirmed=True,item_id=item_id,human_speech=False,no_devices_opened=True)))
                finally:
                    ack.cancel();await asyncio.gather(ack,return_exceptions=True)
            else:
                await ws.send(json.dumps(dict(type='conversation.item.create',item=dict(type='message',role='user',content=[dict(type='input_text',text=texts[args.scenario])])),ensure_ascii=False))
            await ws.send(json.dumps(dict(type='response.create',response=dict(modalities=['text']))))
            async def receive():
                async for raw in ws:
                    event=json.loads(raw)
                    if event.get('type')=='error':raise RuntimeError('Qwen returned error: '+str(event.get('error',{})))
                    if event.get('type')=='response.function_call_arguments.done':
                        arguments=json.loads(event['arguments'])
                        print(json.dumps(dict(observed_tool=event['name'],arguments=arguments),ensure_ascii=False))
                        if event['name']=='set_expression':
                            from vivis.interaction.expression_intent import arguments as expression_arguments,configuration
                            expression_arguments(arguments,configuration())
                        if event['name']=='robot_program':controller_intent(arguments,{p['id'] for p in json.loads((ROOT/settings['programs']).read_text())['programs']},mode)
                        if args.scenario=='dialect':
                            if event['name']!='set_speech_dialect' or SpeechDialect().requested(arguments)!=args.dialect:
                                raise RuntimeError('Model did not request the expected dialect')
                        if args.scenario=='task-create' and (event['name']!='manage_robot_task' or arguments.get('action')!='create'):
                            raise RuntimeError('Model did not request task creation')
                        if args.scenario=='robot-memory' and event['name']!='search_robot_memory':raise RuntimeError('Model did not request global memory search')
                        print(json.dumps(dict(model_tool_request=event['name'],arguments=arguments,tool_not_executed=True,no_devices_opened=True),ensure_ascii=False));return
                    if event.get('type')=='response.done':raise RuntimeError('Model answered without the expected tool; inspect synthetic request')
            await asyncio.wait_for(receive(),timeout=30)
    except Exception as exc:raise RuntimeError(str(exc).replace(secret,'[redacted]')) from None
    finally:log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenario',choices=('probe','wake','stop','identity','compact','modes','smile','neutral','attentive','reception','sad','laugh','single-channel','dialect','task-create','robot-memory'),default='probe')
    p.add_argument('--dialect',choices=dialect_configuration()['dialects'],default='四川话')
    p.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json');args=p.parse_args()
    asyncio.run(run(args))


if __name__=='__main__':main()
