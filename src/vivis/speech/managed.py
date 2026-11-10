"""One persistent microphone: local fast ASR and mode-scoped cloud voice."""
import argparse
import asyncio
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import wave
from types import SimpleNamespace

import numpy as np

from vivis.paths import ROOT
from vivis.speech.listen import Decoder
from vivis.speech.realtime import configuration,VoiceSession,Journal,connect_session,websocket_connection,receive_event
from vivis.speech.realtime_audio import AudioIO,VoiceBuffer
from vivis.interaction.knowledge import Knowledge


def voice_configuration(speech,robot):
    options=speech['managed_voice']
    if options.get('region','beijing') not in ('beijing','singapore'):raise ValueError('Invalid realtime voice region')
    args=SimpleNamespace(config=ROOT/options['config'],region=options.get('region','beijing'),robot=Path(robot),
        workspace_id=None,voice=None,model=None,seconds=None,speaker=None,mic=None,
        aec_backend='aec3',aec_profile=options.get('profile','voice-preserve'),managed=True)
    config,_,speaker,url,session=configuration(args)
    catalog=json.loads(Path(speech.get('programs',ROOT/'configs/interaction/programs.json')).read_text())
    titles='、'.join(p['title'] for p in catalog['programs'])
    abilities='、'.join(catalog.get('capabilities',['唱歌','演剧本','学表情']))
    planned='、'.join(catalog.get('planned_capabilities',[]))
    prepared='、'.join(p['title'] for p in catalog['programs'] if p['kind']=='prerecorded')
    session['instructions']+=' 你代表真实机器人小帅，本体有眼睛和摄像头，但云端没有收到视频帧；视觉观察与跟随由本地控制器负责。你当前处于交互态。可用本地节目为'+titles+'；其中'+prepared+'是已经准备好的静态作品。用户点播时由本地控制器切换，你不要自行编造、演唱或讲述替代内容，不声称已开始播放或已进入待机。用户只笼统要求表演节目时，请询问选择哪个。节目名称听不清或不在列表时，请确认完整名称，不要口头答应开始播放。'
    session['instructions']+=' 介绍你会什么时，只介绍能力类别：'+abilities+'；不要默认报具体作品名。用户询问有哪些歌曲或剧本、或者明确选作品时，才介绍当前节目库。未来规划包括'+planned+'，必须明确它们尚在规划中，不能说已经能执行。实际执行结果没有返回给你时，不要声称已经安排、已经播放或已经停止；只可简短确认收到请求。'
    library=[dict(id=p['id'],title=p['title'],category=p.get('category',p['kind'])) for p in catalog['programs']]
    session['instructions']+=' 当前本地节目库是以下完整事实，不是举例：'+json.dumps(library,ensure_ascii=False)+'。song是歌曲、script是剧本、expression是表情跟随。询问歌曲或剧本列表时，只按对应类别列出这里的名称；不要说没有清单，不要虚构其他流行歌曲，也不要把剧本当歌曲。用户点播库外作品时说明暂未收录。'
    if options.get('control_backend')=='qwen_realtime':
        interval=options.get('control_poll_interval_s',.05)
        if type(interval) not in (int,float) or not math.isfinite(interval) or not .02<=interval<=.2:
            raise ValueError('Invalid cloud control polling interval')
        from vivis.interaction.cloud_control import tools
        from vivis.interaction.cloud_control import validate_presence_config,ReceptionGate
        from vivis.interaction.memory import validate_config
        validate_presence_config(json.loads((ROOT/'configs/perception/speech_presence.json').read_text()))
        ReceptionGate(json.loads((ROOT/options['reception_config']).read_text()))
        memory_config=json.loads((ROOT/options['memory_config']).read_text());validate_config(memory_config)
        from vivis.interaction.robot_tasks import validate_config as validate_tasks
        task_config=json.loads((ROOT/options['tasks_config']).read_text());validate_tasks(task_config)
        if task_config['enabled'] and not memory_config['enabled']:raise ValueError('Robot tasks require persistent memory storage')
        from vivis.interaction.expression_intent import configuration as expression_configuration
        expression_configuration()
        session['tools']=tools([p['id'] for p in catalog['programs']])
        session['tool_choice']='auto'
        robot_definition=json.loads(Path(robot).read_text())
        wake_words=speech.get('wake_words') or robot_definition.get('identity',{}).get('wake_words') or [robot_definition.get('identity',{}).get('display_name',robot_definition['name'])]
        session['instructions']+=' 配置的唤醒词是'+json.dumps(wake_words,ensure_ascii=False)+'。用户本句呼唤这些称呼时，robot_program.explicit_wake必须为true。'
    return config,speaker,url,session


def session_with_program_context(session,context):
    result=dict(session)
    if context is None:return result
    if (not isinstance(context,dict) or set(context)!= {'id','title','outcome'}
            or context.get('outcome') not in ('completed','interrupted')
            or any(not isinstance(context.get(k),str) or not 1<=len(context[k])<=100 for k in ('id','title'))):
        raise ValueError('Invalid controller program context')
    result['instructions']+=' 控制器已确认的最近一次节目记录（不是播放请求）：'+json.dumps(context,ensure_ascii=False)+'。completed表示已经实际播完，interrupted表示中途取消；回答刚才播放了什么时以此记录为准，不要否认已发生的表演。该记录不代表下一次已经开始，也不包含作者或表演者身份。'
    return result


async def refresh_knowledge(ws,knowledge,base_session,context,is_active,log,epoch):
    """Update social context only; never create a response or acquire devices."""
    while True:
        await asyncio.sleep(.5)
        if not is_active():return
        snapshot=knowledge.person_context() if knowledge.config else dict(instructions='',recognized=False,person_id=None)
        current=snapshot['instructions']
        if current==context:continue
        await asyncio.wait_for(ws.send(json.dumps(dict(type='session.update',session=dict(
            instructions=base_session['instructions']+current)),ensure_ascii=False)),timeout=.5)
        context=current
        log.write('knowledge_context_updated',epoch=epoch,recognized=snapshot['recognized'],person_id=snapshot['person_id'])


class ManagedVoice:
    """Capture fans out without waiting for the network or sentence review."""
    def __init__(self,speech,robot,folder,emit):
        self.speech=speech;self.emit=emit;self.folder=folder;self.robot_path=robot
        self.config,self.speaker,self.url,self.session=voice_configuration(speech,robot)
        self.knowledge=Knowledge(robot)
        self.secret=os.environ.get(self.config['api_key_env'],'')
        if not self.secret or self.secret!=self.secret.strip():raise ValueError('Realtime voice API key missing/invalid')
        self.log=Journal(folder/'cloud_events.jsonl',self.secret)
        self.player=VoiceBuffer(self.config['output_gain'],self.config['fade_ms'],self.config['max_playback_buffer_s'])
        self.audio=AudioIO(speech,self.speaker,self.player,True,frontend=self.config['frontend'])
        self.controls=queue.Queue(maxsize=32);self.local=queue.Queue(maxsize=speech['capture']['queue_frames'])
        self.shutdown=threading.Event();self.asr_fault=None;self.cloud_queue=None
        self.cloud_task=None;self.cloud_state=None;self.ws=None;self.epoch=0;self.enabled=False
        self.pending_mode=None;self.release_at=None;self.opened=None;self.last_input=None;self.last_program=None
        self.decoder=None;self.decoder_thread=None
        self.summary=dict(cloud_connections=0,microphone_opens=0,provisional_pauses=0,invalid_turn_resumes=0,
                          local_cancel_operations=0,frontend=self.config['frontend'],voice=self.config['voice'],
                          output_gain=self.config['output_gain'],realtime_lip_sync=False)
        self.recorders=ExitStack();self.writers={};self.frame_log=None;self.recorded_samples=0
        self.mouth=None;self.reported_mouth_error=None

    def state(self,status):
        self.emit(dict(kind='voice_state',epoch=self.epoch,status=status,speaker_owned=self.audio.output_stream is not None))

    def request(self,command):
        if command.get('kind')=='voice_mode':
            epoch=command.get('epoch');enabled=command.get('enabled')
            newest=max(self.epoch,self.pending_mode[0] if self.pending_mode else self.epoch)
            if type(epoch) is not int or type(enabled) is not bool or epoch<=newest:return
            if enabled:
                context=command.get('last_program')
                session_with_program_context(self.session,context)
                self.last_program=context
            self.pending_mode=(epoch,enabled)
            if self.mouth:self.mouth.deactivate()
            if self.cloud_state:self.cloud_state.interrupt(source='mode_change')
            self.cloud_queue=None
            if self.cloud_task:self.cloud_task.cancel()
            self.release_at=time.monotonic()+self.config['fade_ms']/1000+.04
        elif command.get('kind')=='voice_interrupt' and command.get('epoch')==self.epoch:
            if self.cloud_state:self.cloud_state.interrupt(source='local_control')
            if self.ws and self.cloud_state and self.cloud_state.generating:
                # Network cancellation is best effort; local audio has already stopped.
                asyncio.create_task(self.cancel_inference(self.ws))

    async def cancel_inference(self,ws):
        try:await asyncio.wait_for(ws.send(json.dumps(dict(type='response.cancel'))),timeout=.5)
        except Exception as exc:self.log.write('cancel_network_error',error=str(exc))

    def start_decoder(self):
        self.decoder=Decoder(self.speech,self.emit)
        def decode():
            try:
                while not self.shutdown.is_set():
                    try:stamp,pcm,metrics,reference=self.local.get(timeout=.1)
                    except queue.Empty:continue
                    if time.monotonic()-stamp>.5:raise RuntimeError('Local ASR backlog exceeded500ms')
                    self.decoder.feed(np.frombuffer(pcm,'<i2').astype(np.float32)/32768,stamp,metrics,
                        reference=np.frombuffer(reference,'<i2').astype(np.float32)/32768)
            except Exception as exc:self.asr_fault=str(exc)
        self.decoder_thread=threading.Thread(target=decode,daemon=True);self.decoder_thread.start()

    async def cloud(self,epoch):
        state=None
        try:
            self.state('connecting')
            async with websocket_connection(self.url,self.secret) as ws:
                base_session=session_with_program_context(self.session,self.last_program)
                initial=dict(base_session)
                snapshot=self.knowledge.person_context() if self.knowledge.config else dict(instructions='',recognized=False,person_id=None)
                context=snapshot['instructions']
                initial['instructions']+=context
                await connect_session(ws,initial,self.log)
                self.log.write('knowledge_context_initial',epoch=epoch,recognized=snapshot['recognized'],person_id=snapshot['person_id'])
                if epoch!=self.epoch or not self.enabled:return
                player=VoiceBuffer(self.config['output_gain'],self.config['fade_ms'],self.config['max_playback_buffer_s'])
                player.mouth=self.mouth
                self.audio.enable_output(player);self.player=player
                state=VoiceSession(player,self.log,interruption_hold_ms=self.config['interruption_hold_ms'],on_transcript=lambda role,text:self.emit(
                    dict(kind='cloud_transcript',epoch=epoch,role=role,text=text)))
                self.cloud_state=state;self.ws=ws;uplink=asyncio.Queue(maxsize=25);self.cloud_queue=uplink
                self.summary['cloud_connections']+=1;first_sent=asyncio.Event()
                async def send():
                    while True:
                        stamp,pcm=await uplink.get()
                        if time.monotonic()-stamp>.5:raise RuntimeError('Cloud uplink backlog exceeded500ms')
                        import base64
                        await asyncio.wait_for(ws.send(json.dumps(dict(type='input_audio_buffer.append',
                            audio=base64.b64encode(pcm).decode()))),timeout=.5)
                        first_sent.set()
                async def receive():
                    async for message in ws:
                        warning=await receive_event(ws,state,json.loads(message))
                        if warning:self.emit(dict(kind='voice_warning',epoch=self.epoch,**warning))
                    raise RuntimeError('Realtime service disconnected')
                tasks=[asyncio.create_task(send()),asyncio.create_task(receive()),
                    asyncio.create_task(refresh_knowledge(ws,self.knowledge,base_session,context,
                        lambda:epoch==self.epoch and self.enabled and self.pending_mode is None,self.log,epoch))]
                try:
                    await asyncio.wait_for(first_sent.wait(),timeout=1.)
                    deadline=time.monotonic()+1.
                    while self.audio.last_output is None:
                        if time.monotonic()>deadline:raise RuntimeError('Voice output did not start')
                        await asyncio.sleep(.005)
                    if epoch==self.epoch and self.enabled and self.pending_mode is None:self.state('ready')
                    done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
                    for task in done:task.result()
                finally:
                    # Detach immediately, before websocket context cleanup can
                    # wait. Otherwise an idle-timeout is masked by queue-full.
                    if self.cloud_state is state:self.cloud_queue=None
                    for task in tasks:task.cancel()
                    await asyncio.gather(*tasks,return_exceptions=True)
        except asyncio.CancelledError:raise
        except Exception as exc:
            if epoch==self.epoch and self.enabled:
                self.emit(dict(kind='cloud_error',epoch=epoch,error=str(exc)))
                self.enabled=False
        finally:
            if self.mouth:self.mouth.deactivate()
            if state:
                state.interrupt(source='cloud_closed');state.poll_playback()
                self.summary['provisional_pauses']+=state.provisional_pauses
                self.summary['invalid_turn_resumes']+=state.invalid_turn_resumes
                self.summary['local_cancel_operations']+=state.cancel_operations
                self.log.write('cloud_session_closed',epoch=epoch,responses=state.responses,
                    invalid_turns=state.invalid_turns,invalid_turn_resumes=state.invalid_turn_resumes,
                    server_cancelled_response_count=len(state.server_cancelled),usage=state.usage)
            # Release before announcing failure; mode changes also release in the
            # main loop so a slow WebSocket close cannot hold up the speaker.
            self.audio.disable_output();self.cloud_queue=None;self.cloud_state=None;self.ws=None
            if epoch==self.epoch and self.pending_mode is None:self.state('unavailable' if not self.enabled else 'off')

    async def apply_mode(self):
        if self.pending_mode is None or time.monotonic()<self.release_at:return
        self.audio.disable_output()
        if self.cloud_task is not None and not self.cloud_task.done():
            # A mode switch may proceed physically once output is released; a
            # new connection waits for the previous task's bounded close.
            epoch,enabled=self.pending_mode
            if not enabled:
                self.epoch=epoch;self.enabled=False;self.pending_mode=None;self.state('off')
            return
        if self.cloud_task is not None:
            await asyncio.gather(self.cloud_task,return_exceptions=True);self.cloud_task=None
        self.epoch,self.enabled=self.pending_mode;self.pending_mode=None
        if self.enabled:self.cloud_task=asyncio.create_task(self.cloud(self.epoch))
        else:self.state('off')

    async def run(self):
        try:
            lip=self.speech['managed_voice'].get('lip_sync',{})
            if lip.get('enabled',False):
                import csv
                from vivis.runtime.robot import Robot
                from vivis.speech.mouth import SpeechMouth
                robot=Robot(self.robot_path)
                with robot.resolve('channels').open(encoding='utf-8-sig',newline='') as f:order=next(csv.reader(f))[2:]
                self.mouth=SpeechMouth(ROOT/lip['binary'],ROOT/lip['model'],self.folder/'mouth.json',order,
                    secret_env=(self.config['api_key_env'],),buffer_seconds=self.config['max_playback_buffer_s'])
                self.summary['realtime_lip_sync']=True
            if self.speech['managed_voice'].get('record_audio',False):
                for name in ('microphone','reference','uplink'):
                    w=self.recorders.enter_context(wave.open(str(self.folder/(name+'_16k.wav')),'wb'))
                    w.setparams((1,2,16000,0,'NONE','not compressed'));self.writers[name]=w
                self.frame_log=self.recorders.enter_context((self.folder/'frames.jsonl').open('w'))
            self.start_decoder();devices=self.audio.open(with_output=False)
            self.summary['microphone_opens']+=1;self.opened=time.monotonic();self.last_input=self.opened
            ready=False
            reported_mouth_warning=None
            while not self.shutdown.is_set():
                now=time.monotonic()
                if self.mouth and self.mouth.error and self.mouth.error!=self.reported_mouth_error:
                    self.reported_mouth_error=self.mouth.error
                    self.emit(dict(kind='mouth_error',error=self.mouth.error))
                warning=getattr(self.mouth,'warning',None)
                if warning and warning!=reported_mouth_warning:
                    reported_mouth_warning=warning;self.emit(dict(kind='mouth_error',error=warning))
                if self.audio.fault:raise RuntimeError(self.audio.fault)
                if self.asr_fault:raise RuntimeError(self.asr_fault)
                if self.cloud_task is not None and self.cloud_task.done():
                    try:self.cloud_task.result()
                    except asyncio.CancelledError:pass
                if now-(self.audio.last_capture or self.opened)>1:raise RuntimeError('Microphone callback watchdog')
                if now-self.last_input>1:raise RuntimeError('Processed microphone watchdog')
                if self.audio.output_stream is not None and now-(self.audio.last_output or self.audio.output_opened_at)>1:
                    raise RuntimeError('Voice output callback watchdog')
                for _ in range(32):
                    try:command=self.controls.get_nowait()
                    except queue.Empty:break
                    self.request(command)
                await self.apply_mode()
                for _ in range(32):
                    try:stamp,pcm,raw,far=self.audio.uplink.get_nowait()
                    except queue.Empty:break
                    self.last_input=now
                    for name,data in [('microphone',raw),('reference',far),('uplink',pcm)]:
                        if name in self.writers:self.writers[name].writeframesraw(data)
                    if self.frame_log:
                        self.frame_log.write(json.dumps(dict(sample=self.recorded_samples,adc_monotonic=stamp))+'\n')
                        self.recorded_samples+=320
                    level=lambda data:float(np.sqrt(np.mean((np.frombuffer(data,'<i2')/32768.)**2)))
                    metrics=dict(raw_rms=level(raw),clean_rms=level(pcm),reference_rms=level(far))
                    if self.decoder is not None:
                        try:self.local.put_nowait((stamp,pcm,metrics,far))
                        except queue.Full:raise RuntimeError('Local ASR queue exceeded500ms')
                    if self.cloud_queue is not None:
                        try:self.cloud_queue.put_nowait((stamp,pcm))
                        except asyncio.QueueFull:
                            self.emit(dict(kind='cloud_error',epoch=self.epoch,error=('Cloud queue full; no local speech fallback' if self.speech['managed_voice'].get('control_backend')=='qwen_realtime' else 'Cloud queue full; local control continues')))
                            self.enabled=False;self.cloud_task.cancel()
                    if not ready:
                        ready=True;self.emit(dict(kind='ready',**devices,aec=True,frontend=self.config['frontend'],
                            asr_backend='qwen_realtime' if self.decoder is None else self.speech['asr'].get('backend'),managed_voice=True))
                if self.cloud_state:self.cloud_state.poll_playback()
                while not self.audio.meters.empty():self.log.write(**self.audio.meters.get_nowait())
                await asyncio.sleep(.005)
        finally:
            self.shutdown.set()
            if self.cloud_task:
                self.cloud_task.cancel();await asyncio.gather(self.cloud_task,return_exceptions=True)
            self.audio.close()
            if self.mouth:
                self.mouth.close()
                self.summary['mouth']=dict(generated_frames=self.mouth.frames,published=self.mouth.published,
                    prebuffer_wait_callbacks=self.mouth.gated,prebuffer_timeouts=getattr(self.mouth,'prebuffer_timeouts',0),error=self.mouth.error)
            if self.decoder_thread:self.decoder_thread.join(timeout=2)
            if self.decoder:self.decoder.close()
            self.recorders.close()
            self.summary.update(audio=self.audio.stats,recorded_samples=self.recorded_samples)
            (self.folder/'summary.json').write_text(json.dumps(self.summary,ensure_ascii=False,indent=2)+'\n')
            self.log.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--robot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check',action='store_true');args=p.parse_args(argv)
    speech=json.loads(args.config.read_text())
    if args.check:
        config,*_=voice_configuration(speech,args.robot)
        print(json.dumps(dict(valid=True,frontend=config['frontend'],hardware_opened=False,network_called=False)));return 0
    args.output.mkdir(parents=True,exist_ok=True)
    secret_name=json.loads((ROOT/speech['managed_voice']['config']).read_text())['api_key_env']
    secret=os.environ.get(secret_name,'');lock=threading.Lock()
    def emit(event):
        text=json.dumps(event,ensure_ascii=False,allow_nan=False)
        if secret:text=text.replace(secret,'[redacted]')
        with lock:print(text,flush=True)
    backend=ManagedVoice
    if speech['managed_voice'].get('control_backend')=='qwen_realtime':
        from vivis.speech.cloud_managed import CloudManagedVoice
        backend=CloudManagedVoice
    service=backend(speech,args.robot,args.output,emit)
    def commands():
        for line in sys.stdin:
            try:command=json.loads(line)
            except ValueError:continue
            try:service.controls.put(command,timeout=.5)
            except queue.Full:service.shutdown.set();return
        service.shutdown.set()  # Supervisor lost; release its speaker/capture.
    threading.Thread(target=commands,daemon=True).start()
    handlers={s:signal.signal(s,lambda *_:service.shutdown.set()) for s in (signal.SIGINT,signal.SIGTERM)}
    try:asyncio.run(service.run())
    except Exception as exc:emit(dict(kind='asr_error',error=str(exc)));return 1
    finally:
        for s,h in handlers.items():signal.signal(s,h)
    return 0


if __name__=='__main__':raise SystemExit(main())
