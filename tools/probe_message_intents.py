#!/usr/bin/env python3
"""Synthetic cloud extraction acceptance; no real DB, microphone or motors."""
import asyncio
import json
import time
from datetime import datetime
from vivis.paths import ROOT
from vivis.interaction.message_lane import extract_qwen
from vivis.speech.credentials import load_key
from vivis.speech.managed import voice_configuration
from vivis.speech.realtime import Journal

async def main():
    secret=load_key()
    if not secret:raise SystemExit('请先配置本机百炼密钥；测试不会打印密钥。')
    speech=json.loads((ROOT/'configs/perception/speech.json').read_text())
    speech['managed_voice']=json.loads((ROOT/'configs/interaction/p1.json').read_text())['realtime_voice']
    speech['programs']=str(ROOT/'configs/interaction/programs.json')
    _,_,url,session=voice_configuration(speech,ROOT/'configs/robots/xiaoshuai/robot.json')
    folder=ROOT/'outputs/message_intents'/datetime.now().strftime('%Y%m%d_%H%M%S');folder.mkdir(parents=True)
    log=Journal(folder/'events.jsonl',secret)
    people={'alpha':dict(display_name='测试甲',preferred_address='甲工'),'beta':dict(display_name='测试乙',preferred_address='小乙')}
    cases=[
        ('遇到甲工，让帮我问一下他准备哪份资料。','beta',dict(action='create',recipient_id='alpha')),
        ('一会儿见到小乙，让他帮我买包烟。','alpha',dict(action='create',recipient_id='beta')),
        ('这条留言是给小乙的，测试乙。','alpha',dict(action='clarify',missing='content')),
        ('我给小乙的留言，让他帮我买买包烟，你转告了吗？','alpha',dict(action='query',recipient_id='beta',creator_id='alpha',content_keyword='烟')),
        ('小乙有给我的留言吗？','alpha',dict(action='query',recipient_id='alpha',creator_id='beta',status='active')),
        ('有我的留言吗？','alpha',dict(action='query',recipient_id='alpha',status='active')),
        ('还有别的留言吗？','beta',dict(action='query',recipient_id='beta',status='active')),
        ('还有别的跟我有关的事吗？','beta',dict(action='query',recipient_id=None,creator_id=None,status='active')),
        ('哦，这是谁给我的留言？','beta',dict(action='query',recipient_id='beta',query_detail='sender')),
        ('今天天气真好。','alpha',dict(action='none')),
        ('小帅，停止播放。','alpha',dict(action='none')),
        ('请给我讲个笑话。','alpha',dict(action='none')),
    ]
    cases=[(text,actor,expected,{}) for text,actor,expected in cases]
    cases += [
        ('打火机买好了。','beta',dict(action='create',recipient_id='alpha'),
            dict(draft_source='见到甲工，帮我给他带句话。')),
        ('明天早上9点到。','beta',dict(action='create',recipient_id='alpha'),
            dict(draft_source='帮我给甲工留个言。',missing='content')),
        ('小乙。','alpha',dict(action='create',recipient_id='beta'),
            dict(draft_source='帮我带句话，明天准备两份资料。',missing='recipient')),
        ('打火机买好了。','beta',dict(action='create',recipient_id='alpha'),
            dict(draft_source='帮我带句话。\n给甲工。',missing='content')),
        ('先不留言了，给我讲个笑话。','beta',dict(action='none'),dict(draft_source='见到甲工帮我带句话。')),
        ('算了，不留言了。','beta',dict(action='none'),dict(draft_source='见到甲工帮我带句话。')),
        ('打火机买好了。','beta',dict(action='none'),{}),
    ]
    reports=[]
    try:
        for text,actor,expected,hint in cases:
            start=time.monotonic()
            try:
                result=await extract_qwen(url,secret,session,log,text,people,dict(hint,current_person_id=actor))
                passed=all(result.get(k)==v for k,v in expected.items())
                report=dict(text=text,result=result,expected=expected,passed=passed,elapsed_s=round(time.monotonic()-start,3))
            except Exception as exc:report=dict(text=text,passed=False,error_type=type(exc).__name__)
            reports.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
        (folder/'summary.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in reports),reports=reports,physical_devices_opened=False),ensure_ascii=False,indent=2)+'\n')
        print('Evidence:',folder,flush=True)
        return 0 if all(r['passed'] for r in reports) else 1
    finally:log.close()

if __name__=='__main__':raise SystemExit(asyncio.run(main()))
