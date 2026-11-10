#!/usr/bin/env python3
"""Synthetic cloud identity/memory roundtrip; no audio, camera or motor devices."""
import asyncio,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.paths import ROOT
from vivis.speech.managed import voice_configuration
from vivis.speech.realtime import websocket_connection,connect_session,Journal
from vivis.speech.credentials import load_key
from vivis.interaction.memory import PersonMemory

async def main():
 p=json.loads((ROOT/'configs/interaction/p1.json').read_text());s=json.loads((ROOT/p['speech']).read_text());s.update(managed_voice=p['realtime_voice'],programs=str(ROOT/p['programs']))
 config,_,url,session=voice_configuration(s,ROOT/'configs/robots/xiaoshuai/robot.json');key=load_key(config['api_key_env'])
 session.update(modalities=['text'],instructions='你是合成数据测试机器人。当前controller_state为interaction且ready=true、face_visible=true，current_visual_identity为recognized、person_id=synthetic。不要唤醒或播放节目。回答年龄需查询get_robot_status；回答历史偏好需查询search_person_memory，用原话关键词，无命中时换词或空数组查询，不猜。')
 person=dict(person_id='synthetic',display_name='测试工',preferred_address='测试工',age_years=44)
 with tempfile.TemporaryDirectory() as tmp:
  m=PersonMemory(json.loads((ROOT/'configs/interaction/memory.json').read_text()),tmp);m.append('synthetic','previous','user','我喜欢喝绿茶。')
  log=Journal(Path(tmp)/'events.jsonl',key);called=[];texts=[];continuing=False
  async with websocket_connection(url,key) as ws:
   await connect_session(ws,session,log)
   await ws.send(json.dumps(dict(type='conversation.item.create',item=dict(type='message',role='user',content=[dict(type='input_text',text='我今年几岁了？你还记得我喜欢喝什么吗？')])),ensure_ascii=False))
   await ws.send(json.dumps(dict(type='response.create',response=dict(modalities=['text']))))
   async def receive():
    nonlocal continuing
    async for raw in ws:
     e=json.loads(raw);kind=e.get('type')
     if kind=='error':raise RuntimeError('Roundtrip cloud error')
     if kind=='response.function_call_arguments.done':
      name=e['name'];a=json.loads(e['arguments']);called.append(name);print('SYNTHETIC_TOOL',name,json.dumps(a,ensure_ascii=False),flush=True)
      if len(called)>6:raise RuntimeError('Too many tool rounds')
      if name=='get_robot_status':result=dict(current_person=person,face_visible=True,memory_enabled=True)
      elif name=='search_person_memory':
       try:result=m.search('synthetic',a['keywords'],'current')
       except (ValueError,TypeError) as exc:result=dict(accepted=False,error=str(exc))
      else:raise RuntimeError('Unexpected programme action')
      await ws.send(json.dumps(dict(type='conversation.item.create',item=dict(type='function_call_output',call_id=e['call_id'],output=json.dumps(result,ensure_ascii=False))),ensure_ascii=False));continuing=True
     if kind in ('response.text.done','response.audio_transcript.done'):texts.append(e.get('text',e.get('transcript','')))
     if kind=='response.done':
      if continuing:
       continuing=False;await ws.send(json.dumps(dict(type='response.create',response=dict(modalities=['text']))))
      else:return
   await asyncio.wait_for(receive(),timeout=45)
  log.close();text=' '.join(texts)
  report=dict(synthetic_data_only=True,age44_correct='44' in text or '四十四' in text,green_tea_recalled='绿茶' in text,tools=called,reply=text,no_devices_opened=True)
  folder=ROOT/'outputs/cloud_control_probe';folder.mkdir(exist_ok=True);(folder/'memory_roundtrip.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps(report,ensure_ascii=False))
  assert report['age44_correct'] and report['green_tea_recalled']
if __name__=='__main__':asyncio.run(main())
