#!/usr/bin/env python3
"""Cloud Qwen extraction + temp SQLite + TTS + virtual DAC; no physical devices."""
import asyncio
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time

from vivis.paths import ROOT
from vivis.interaction.memory import PersonMemory
from vivis.interaction.robot_tasks import RobotTasks
from vivis.interaction.message_delivery import MessageRequest
from vivis.interaction.message_lane import MessageLane,extract_qwen
from vivis.speech.cloud_tts import CloudTTS
from vivis.speech.credentials import load_key
from vivis.speech.managed import voice_configuration
from vivis.speech.message_voice import MessageSpeaker
from vivis.speech.realtime import Journal,VoiceSession
from vivis.speech.realtime_audio import VoiceBuffer


async def main():
    secret=load_key();speech=json.loads((ROOT/'configs/perception/speech.json').read_text())
    speech['managed_voice']=json.loads((ROOT/'configs/interaction/p1.json').read_text())['realtime_voice']
    speech['programs']=str(ROOT/'configs/interaction/programs.json')
    _,_,url,session=voice_configuration(speech,ROOT/'configs/robots/xiaoshuai/robot.json')
    folder=ROOT/'outputs/message_lane'/datetime.now().strftime('%Y%m%d_%H%M%S');folder.mkdir(parents=True)
    journal=Journal(folder/'events.jsonl',secret);player=VoiceBuffer(max_seconds=120);state=VoiceSession(player,journal)
    engine=CloudTTS(dict(speech['tts'],timeout_s=20),secret=secret,cache_directory=False)
    spoken=[];current=None;reports=[]
    def valid(request):return request==current
    speaker=MessageSpeaker(engine,player,state,valid,spoken.append,journal.write)
    async def virtual_dac():
        while True:
            if player.count:player.render(min(player.count,48000),time.monotonic()-2)
            await asyncio.sleep(.005)
    pump=asyncio.create_task(virtual_dac())
    try:
        with tempfile.TemporaryDirectory() as tmp:
            memory=PersonMemory(json.loads((ROOT/'configs/interaction/memory.json').read_text()),tmp)
            policy=json.loads((ROOT/'configs/interaction/tasks.json').read_text());store=RobotTasks(memory,policy)
            people={'alpha':dict(display_name='测试甲',preferred_address='甲工'),'beta':dict(display_name='测试乙',preferred_address='小乙')}
            async def extract(text,people,hint):return await extract_qwen(url,secret,session,journal,text,people,hint)
            lane=MessageLane(store,people,extract,speaker.say,valid,journal.write)
            cases=[('alpha','给小乙留条测试留言：明天准备两份资料。'),
                ('alpha','刚才给小乙的留言保存了吗？'),('alpha','重新保存刚才那条。'),
                ('beta','把给我的留言说一下。'),('beta','没听清，暂时没收到。')]
            for index,(actor,text) in enumerate(cases):
                mid=memory.append(actor,'synthetic','user',text)
                current=MessageRequest(actor,'synthetic',str(index),mid,text,actor)
                await lane.run(current);assert len(store.list('all'))==1
                reports.append(dict(case=index,result=lane.result,last_spoken=spoken[-1]));print('PASS',index,spoken[-1],flush=True)
            tid=store.list('all')[0]['task_id']
            assert store.get(tid)['status']=='announced'
            mid=memory.append('beta','synthetic','user','收到了，谢谢。')
            current=MessageRequest('beta','synthetic','ack',mid,'收到了，谢谢。','beta')
            await lane.run(current);assert store.get(tid)['status']=='completed';assert not store.due('beta')
            fresh=RobotTasks(PersonMemory(memory.c,tmp),policy);assert fresh.get(tid)['status']=='completed'
            reports.append(dict(case='delivery_ack_reload',status='completed',last_spoken=spoken[-1]))
            print('PASS delivery/ack/reload',spoken[-1],flush=True)
        (folder/'summary.json').write_text(json.dumps(dict(passed=True,synthetic_only=True,physical_devices_opened=False,
            virtual_dac=True,reports=reports),ensure_ascii=False,indent=2)+'\n')
        print('Evidence:',folder)
    finally:
        pump.cancel();await asyncio.gather(pump,return_exceptions=True);journal.close()


if __name__=='__main__':asyncio.run(main())
