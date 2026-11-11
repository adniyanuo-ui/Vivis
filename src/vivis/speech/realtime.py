"""Standalone Qwen speech-to-speech test. No robot, camera, A2F or task tools."""
import argparse
import asyncio
import base64
from contextlib import ExitStack
from datetime import datetime
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
from urllib.parse import urlencode,urlsplit,urlunsplit
import wave

from vivis.paths import ROOT
from vivis.speech.realtime_audio import AudioIO,VoiceBuffer,PlaybackBufferFull
from vivis.speech.service_recovery import ServiceRecovery,transient_error
from vivis.speech.voice_frontend import select_frontend,AEC_PROFILES


def configuration(args):
    config=json.loads(args.config.read_text())
    if config.get('schema_version')!='vivis-realtime-voice-test-1.0':raise ValueError('Unsupported voice test config')
    config['frontend']=select_frontend(config.get('frontend',{'backend':'speex'}),
        getattr(args,'aec_backend',None),getattr(args,'aec_profile',None))
    for name in ('voice','model','seconds','output_gain'):
        value=getattr(args,name,None)
        if value is not None:config[name]=value
    if args.region:
        host='dashscope.aliyuncs.com' if args.region=='beijing' else 'dashscope-intl.aliyuncs.com'
        config['endpoint']='wss://'+host+'/api-ws/v1/realtime'
    if not 10<=config['seconds']<=900:raise ValueError('Voice test duration must be10..900 seconds')
    if not .5<=config['speech_rate']<=2.:raise ValueError('Invalid speech rate')
    ServiceRecovery(config['service_recovery'])
    for key,default,lo,hi in [('vad_threshold',.5,-1,1),('interruption_hold_ms',0,0,500)]:
        value=config.get(key,default)
        if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:raise ValueError('Invalid '+key)
        config[key]=value
    for key,lo,hi in [('output_gain',.1,2.),('fade_ms',0,100),('max_playback_buffer_s',1,120)]:
        value=config[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:raise ValueError('Invalid '+key)
    if args.workspace_id:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.workspace_id):raise ValueError('Invalid workspace ID')
        region='ap-southeast-1' if 'intl' in config['endpoint'] or '.ap-southeast-1.' in config['endpoint'] else 'cn-beijing'
        config['endpoint']='wss://'+args.workspace_id+'.'+region+'.maas.aliyuncs.com/api-ws/v1/realtime'
    parts=urlsplit(config['endpoint']);host=parts.hostname or ''
    if (parts.scheme!='wss' or parts.username or parts.password or parts.query or parts.fragment
            or host not in ('dashscope.aliyuncs.com','dashscope-intl.aliyuncs.com') and not host.endswith('.maas.aliyuncs.com')):
        raise ValueError('Use the official regional Qwen WSS endpoint')
    url=urlunsplit((parts.scheme,parts.netloc,parts.path,urlencode(dict(model=config['model'])),''))
    robot_path=args.robot or ROOT/config['robot'];robot=json.loads(robot_path.read_text())
    identity=robot.get('identity',dict(display_name=robot['name']))
    hardware=json.loads((robot_path.parent/robot['hardware']).read_text())
    speaker=args.speaker or hardware['runtime']['audio_device']
    speech=json.loads((ROOT/config['speech']).read_text())
    if args.mic:speech['capture']['device']=args.mic
    session=dict(modalities=['text','audio'],voice=config['voice'],speech_rate=config['speech_rate'],
                 enable_speech_emotion=config['enable_speech_emotion'],
                 instructions='本体身份：'+json.dumps(identity,ensure_ascii=False)+'。'+config['instructions'],
                 input_audio_format='pcm',output_audio_format='pcm',input_audio_transcription=dict(language='zh'),
                 output_audio=dict(language='zh'),turn_detection=dict(type=config['turn_detection'],threshold=config['vad_threshold']),
                 max_history_turns=config['max_history_turns'],enable_search=False,tools=[])
    from vivis.interaction.knowledge import Knowledge
    session['instructions']+=Knowledge(robot_path).base_instructions(include_people=getattr(args,'include_people',True))
    if not getattr(args,'managed',False):
        session['instructions']+=' 你正在做纯语音测试，没有相机、机器人动作或外部工具，不声称执行了动作。'
    return config,speech,speaker,url,session


def handshake_diagnostic(exc,secret):
    response=getattr(exc,'response',None)
    if response is None:return None
    detail=dict(http_status=response.status_code)
    try:
        body=json.loads(bytes(response.body[:8192]).decode('utf-8'))
        if isinstance(body.get('error'),dict):body=body['error']
        for field in ('code','message','request_id'):
            if isinstance(body.get(field),str):detail[field]=body[field][:1000].replace(secret,'[redacted]')
    except (ValueError,TypeError,AttributeError):pass
    headers=getattr(response,'headers',{})
    request_id=headers.get('x-request-id') or headers.get('x-dashscope-request-id')
    if request_id:detail['request_id']=str(request_id).replace(secret,'[redacted]')
    return detail


def key_status(secret):
    return dict(present=bool(secret),length=len(secret),starts_with_sk=secret.startswith('sk-'),
                subscription_key=secret.startswith('sk-sp-'),contains_whitespace=bool(re.search(r'\s',secret)))


class Journal:
    def __init__(self,path,secret=''):
        self.file=path.open('w');self.secret=secret;self.lock=threading.Lock()

    def write(self,kind,**values):
        line=json.dumps(dict(monotonic=time.monotonic(),kind=kind,**values),ensure_ascii=False,allow_nan=False)
        if self.secret:line=line.replace(self.secret,'[redacted]')
        with self.lock:self.file.write(line+'\n');self.file.flush()

    def close(self):self.file.close()


class VoiceSession:
    def __init__(self,player,log,on_transcript=None,interruption_hold_ms=0):
        self.player=player;self.log=log;self.generating=None;self.blocked=set();self.responses={}
        self.speech_stopped_at=None;self.cancel_operations=0;self.late_packets=0;self.usage=[]
        self.server_starts=0;self.server_cancelled=set();self.invalid_turns=0
        self.pending_interruption=None;self.provisional_pauses=0;self.invalid_turn_resumes=0
        self.on_transcript=on_transcript
        self.current_user_turn=None;self.latency_reported=set()
        self.interruption_hold_s=interruption_hold_ms/1000;self.playback_overflows=0

    def mark_proactive(self):
        self.current_user_turn=None

    def interrupt(self,response_id=None,source='keyboard'):
        result=self.player.cancel(response_id)
        ids=set(result['responses'])
        if response_id:ids.add(response_id)
        if response_id is None and self.generating:ids.add(self.generating)
        self.blocked.update(ids)
        if response_id is None:self.pending_interruption=None
        if ids:self.cancel_operations+=1
        self.log.write('local_interrupt',source=source,**result,blocked_responses=sorted(ids))

    def confirm_interruption(self,source):
        pending=self.pending_interruption
        if pending is None:return
        self.pending_interruption=None
        for rid in pending['responses']:
            if rid not in self.blocked:self.interrupt(rid,source=source)
        self.log.write('interruption_confirmed',item_id=pending['item_id'],source=source,
                       response_ids=pending['responses'])

    def event(self,event):
        kind=event.get('type','unknown');rid=event.get('response_id');now=time.monotonic()
        if kind=='response.audio.delta':
            now=event.get('_network_received_monotonic',now)
            if not rid:raise RuntimeError('Audio packet missing response_id; cannot safely reject stale audio')
            if rid in self.blocked:
                self.late_packets+=1;return
            pcm=base64.b64decode(event['delta'],validate=True)
            response=self.responses.setdefault(rid,{})
            if 'first_audio_received' not in response:
                response['first_audio_received']=now
                self.log.write('first_audio_received',response_id=rid,bytes=len(pcm))
            previous=response.get('last_audio_received',now)
            response['max_packet_gap_ms']=max(response.get('max_packet_gap_ms',0.),(now-previous)*1000)
            response['last_audio_received']=now
            response['audio_packets']=response.get('audio_packets',0)+1
            response['received_audio_s']=response.get('received_audio_s',0.)+len(pcm)/48000
            try:self.player.add(rid,pcm)
            except PlaybackBufferFull:
                buffered=self.player.count/48000;self.playback_overflows+=1
                self.interrupt(rid,source='response_buffer_overflow')
                result=dict(response_id=rid,buffered_s=buffered,limit_s=self.player.maximum/48000,
                    message='本句语音超出播放缓冲上限，已取消本句；麦克风及交互继续。')
                self.log.write('playback_overflow_recovered',**result)
                return result
            self.log.write('audio_packet',response_id=rid,bytes=len(pcm),buffered_s=self.player.count/48000,
                           gap_ms=(now-previous)*1000,network_received_monotonic=now,
                           gate_hold_ms=max(0.,(time.monotonic()-now)*1000))
            return
        fields={k:event[k] for k in ('event_id','item_id','response_id','audio_start_ms','audio_end_ms','reason','transcript','text') if k in event}
        if kind=='response.created':
            # A new answer supersedes retained old speech even if a preceding
            # committed/stopped event is missing. Never revive it on a late invalid.
            self.confirm_interruption('new_response')
            rid=event['response']['id'];self.generating=rid
            self.responses[rid]=dict(created=now,speech_stopped_received=self.speech_stopped_at,
                user_turn=dict(self.current_user_turn) if self.current_user_turn else None)
            fields['response_id']=rid
        elif kind=='input_audio_buffer.speech_started':
            self.server_starts+=1
            # smart_turn_v2 may later retract a short acknowledgement. Pause now
            # for prompt barge-in; only confirmed turns may destroy the PCM.
            if self.interruption_hold_s:
                result=dict(responses=self.player.response_ids(self.generating),retained_samples=0)
            else:result=self.player.pause(self.generating)
            if self.pending_interruption:
                result['responses']=sorted(set(result['responses'])|set(self.pending_interruption['responses']))
            self.pending_interruption=dict(item_id=event.get('item_id'),responses=result['responses'],
                pause_at=now+self.interruption_hold_s,paused=not self.interruption_hold_s)
            if not self.interruption_hold_s:self.provisional_pauses+=1
            self.log.write('interruption_candidate' if self.interruption_hold_s else 'provisional_interrupt',
                item_id=event.get('item_id'),hold_ms=self.interruption_hold_s*1000,**result)
        elif kind=='input_audio_buffer.speech_stopped':
            matches=self.pending_interruption is not None and self.pending_interruption['item_id']==event.get('item_id')
            if event.get('reason')=='turn_invalid':
                self.invalid_turns+=1
                if matches:
                    pending=self.pending_interruption;self.pending_interruption=None
                    result=self.player.resume(set(pending['responses'])-self.blocked-self.server_cancelled)
                    if result['responses']:self.invalid_turn_resumes+=1
                    self.log.write('invalid_turn_resume',item_id=event.get('item_id'),**result)
            else:
                self.speech_stopped_at=now
                self.current_user_turn=dict(item_id=event['item_id'],confirmed_end_monotonic=now) if isinstance(event.get('item_id'),str) else None
                if matches:self.confirm_interruption('valid_speech_stopped')
        elif kind=='input_audio_buffer.committed':
            if self.pending_interruption is not None and self.pending_interruption['item_id']==event.get('item_id'):
                self.confirm_interruption('input_committed')
        elif kind=='response.done':
            response=event['response'];rid=response['id'];fields.update(response_id=rid,status=response.get('status'))
            self.responses.setdefault(rid,{})['status']=response.get('status')
            if response.get('status')=='cancelled':
                self.server_cancelled.add(rid);self.interrupt(rid,source='server_cancelled')
            if rid==self.generating:self.generating=None
            self.player.finish(rid)
            if response.get('usage'):
                self.usage.append(response['usage']);fields['usage']=response['usage']
        elif kind=='error':
            error=event.get('error',{});self.log.write('server_error',detail=error)
            if error.get('param')=='response.cancel' and error.get('type')=='invalid_request_error':return
            raise RuntimeError('Realtime service error: '+str(error.get('code'))+' '+str(error.get('message')))
        elif kind in ('conversation.item.input_audio_transcription.completed','response.audio_transcript.done'):
            role='user' if kind.startswith('conversation.') else 'assistant'
            text=event.get('transcript','')
            if role=='assistant' and rid:self.responses.setdefault(rid,{})['transcript']=text
            if self.on_transcript:self.on_transcript(role,text)
            else:print(('你：' if role=='user' else '机器人：')+text,flush=True)
        self.log.write('server_event',type=kind,**fields)

    def poll_playback(self):
        pending=self.pending_interruption
        if pending and not pending['paused'] and time.monotonic()>=pending['pause_at']:
            result=self.player.pause()
            pending['responses']=sorted(set(pending['responses'])|set(result['responses']))
            pending['paused']=True;self.provisional_pauses+=1
            self.log.write('provisional_interrupt',item_id=pending['item_id'],**result)
        while not self.player.notices.empty():
            event=self.player.notices.get_nowait();rid=event['response_id']
            if event['kind']=='playback_buffer_empty':
                self.responses.setdefault(rid,{})['empty_dac']=event['dac_monotonic']
            if event['kind']=='playback_started':
                response=self.responses.setdefault(rid,{})
                response.setdefault('first_dac_monotonic',event['dac_monotonic'])
                response['playback_starts']=response.get('playback_starts',0)+1
                turn=response.get('user_turn')
                if turn and turn['item_id'] not in self.latency_reported:
                    if len(self.latency_reported)>=256:self.latency_reported.clear()
                    self.latency_reported.add(turn['item_id'])
                    packet=response.get('first_audio_received')
                    self.log.write('reply_latency',response_id=rid,item_id=turn['item_id'],
                        confirmed_end_to_first_dac_s=event['dac_monotonic']-turn['confirmed_end_monotonic'],
                        confirmed_end_to_first_packet_s=None if packet is None else packet-turn['confirmed_end_monotonic'],
                        first_packet_to_dac_s=None if packet is None else event['dac_monotonic']-packet,
                        endpoint_definition='server_confirmed_speech_end_not_physical_voice_end')
            self.log.write(**event)


async def receive_event(ws,state,event):
    """An oversized reply is a response fault, not loss of microphone/cloud control."""
    warning=state.event(event)
    if warning and state.generating==warning['response_id']:
        try:await asyncio.wait_for(ws.send(json.dumps(dict(type='response.cancel'))),timeout=.5)
        except Exception as exc:state.log.write('overflow_cancel_network_error',error=type(exc).__name__)
    return warning


async def connect_session(ws,session,log):
    await ws.send(json.dumps(dict(type='session.update',session=session),ensure_ascii=False))
    for _ in range(20):
        event=json.loads(await asyncio.wait_for(ws.recv(),timeout=15.))
        if event.get('type')=='error':
            retryable=transient_error(event)
            if retryable:raise retryable
            raise RuntimeError('Qwen session configuration failed: '+str(event.get('error',{})))
        if event.get('type')=='session.updated':
            accepted=event['session']
            if (accepted.get('voice')!=session['voice']
                    or accepted.get('turn_detection',{}).get('type')!=session['turn_detection']['type']
                    or ('threshold' in session['turn_detection'] and
                        accepted.get('turn_detection',{}).get('threshold')!=session['turn_detection']['threshold'])):
                raise RuntimeError('Server did not confirm requested voice/turn detection; do not silently fall back')
            log.write('session_ready',model=accepted.get('model'),voice=accepted.get('voice'),turn_detection=accepted.get('turn_detection'))
            return accepted
    raise RuntimeError('Qwen did not acknowledge session configuration')


def check_audio_callbacks(audio,opened_at,now):
    """Opening devices is preparation, not a gap in an already running stream."""
    for name,observed in [('Microphone',audio.last_capture),('Output',audio.last_output)]:
        baseline=opened_at if observed is None else observed
        if now-baseline>1.:
            raise RuntimeError(f'{name} callback watchdog: no callback for {now-baseline:.3f}s after device open')


def websocket_connection(url,key):
    from websockets.asyncio.client import connect
    return connect(url,additional_headers={'Authorization':'Bearer '+key},compression=None,
                   open_timeout=15,close_timeout=3,ping_interval=20,ping_timeout=20,max_size=4*1024*1024,max_queue=16)


async def conversation(args,config,speech,speaker,url,session,key,folder,log,summary):
    async with websocket_connection(url,key) as ws:
        await connect_session(ws,session,log)
        if args.probe:
            print('连接、模型权限、音色及语义轮次配置已确认；未打开麦克风或音箱。',flush=True)
            return dict(probe=True,audio_opened=False)
        player=VoiceBuffer(config['output_gain'],config['fade_ms'],config['max_playback_buffer_s'])
        state=VoiceSession(player,log,interruption_hold_ms=config['interruption_hold_ms'])
        audio=AudioIO(speech,speaker,player,args.aec=='on',frontend=config.get('frontend'));keys=queue.SimpleQueue()
        def read_keys():
            for line in sys.stdin:keys.put(line.strip())
        threading.Thread(target=read_keys,daemon=True).start()
        preparation_started=time.monotonic();started=None;tasks=[];stats={};input_packets_sent=0;ready=False
        try:
            with ExitStack() as stack:
                writers={}
                if args.record_audio:
                    for name in ('uplink','microphone','reference'):
                        wav=stack.enter_context(wave.open(str(folder/(name+'_16k.wav')),'wb'))
                        wav.setparams((1,2,16000,0,'NONE','not compressed'));writers[name]=wav
                devices=audio.open();started=time.monotonic()
                log.write('audio_opened',devices=devices,aec=args.aec,preparation_s=started-preparation_started)
                print('音频设备已打开，等待首帧收发……',flush=True)
                async def send_audio():
                    nonlocal input_packets_sent
                    last=time.monotonic()
                    while True:
                        if audio.fault:raise RuntimeError(audio.fault)
                        try:item=audio.uplink.get_nowait()
                        except queue.Empty:
                            if time.monotonic()-last>1.:raise RuntimeError('Microphone watchdog: no processed capture frames')
                            await asyncio.sleep(.005);continue
                        stamp,pcm,raw,far=item
                        if time.monotonic()-stamp>.5:raise RuntimeError('Network fell behind microphone capture by500ms')
                        await asyncio.wait_for(ws.send(json.dumps(dict(type='input_audio_buffer.append',
                                                 audio=base64.b64encode(pcm).decode()))),timeout=.5)
                        last=time.monotonic()
                        input_packets_sent+=1
                        for name,data in [('uplink',pcm),('microphone',raw),('reference',far)]:
                            if name in writers:writers[name].writeframesraw(data)
                async def receive():
                    async for message in ws:
                        warning=await receive_event(ws,state,json.loads(message))
                        if warning:print('语音提示：'+warning['message'],flush=True)
                    raise RuntimeError('Realtime service disconnected')
                tasks=[asyncio.create_task(send_audio()),asyncio.create_task(receive())]
                while time.monotonic()-started<config['seconds']:
                    for task in tasks:
                        if task.done():task.result()
                    state.poll_playback()
                    if audio.fault:raise RuntimeError(audio.fault)
                    check_audio_callbacks(audio,started,time.monotonic())
                    if not ready and input_packets_sent and audio.last_capture is not None and audio.last_output is not None:
                        ready=True
                        log.write('audio_ready',devices=devices,aec=args.aec,input_packets_sent=input_packets_sent,
                                  startup_s=time.monotonic()-started)
                        print('纯语音聊天已就绪，直接说话，无需唤醒。机器头／相机／A2F均未启用。',flush=True)
                        print('Enter本地停声；/mark echo_only等可标记阶段；/quit或Ctrl+C结束。',flush=True)
                    while not audio.meters.empty():log.write(**audio.meters.get_nowait())
                    if not keys.empty():
                        command=keys.get_nowait()
                        if command=='/quit':break
                        if command.startswith('/mark '):log.write('onsite_marker',label=command[6:][:100])
                        elif command=='':
                            state.interrupt()
                            if state.generating:await ws.send(json.dumps(dict(type='response.cancel')))
                    await asyncio.sleep(.01)
        finally:
            for task in tasks:task.cancel()
            if tasks:await asyncio.gather(*tasks,return_exceptions=True)
            audio.close();state.poll_playback()
            closed_at=time.monotonic()
            stats=dict(audio.stats);stats.update(elapsed_s=closed_at-(started if started is not None else preparation_started),
                audio_preparation_s=(started if started is not None else closed_at)-preparation_started,
                audio_ready=ready,input_packets_sent=input_packets_sent,responses=state.responses,
                local_cancel_operations=state.cancel_operations,cancelled_response_count=len(state.blocked),
                server_cancelled_response_count=len(state.server_cancelled),server_speech_started_events=state.server_starts,
                invalid_turns=state.invalid_turns,rejected_late_packets=state.late_packets,usage=state.usage,clipped_samples=player.clipped)
            stats.update(provisional_pauses=state.provisional_pauses,invalid_turn_resumes=state.invalid_turn_resumes)
            summary.update(probe=False,audio_opened=bool(audio.stats['captured_frames']),**stats)
            log.write('audio_closed',stats=stats)
        return dict(probe=False,audio_opened=True,**stats)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/perception/realtime_voice.json')
    parser.add_argument('--robot',type=Path);parser.add_argument('--voice');parser.add_argument('--mic');parser.add_argument('--speaker')
    parser.add_argument('--workspace-id');parser.add_argument('--model',choices=('qwen-audio-3.1-realtime-plus','qwen-audio-3.1-realtime-flash'))
    parser.add_argument('--region',choices=('beijing','singapore'),help='必须与API Key创建地域一致；不自动跨地域重试')
    parser.add_argument('--seconds',type=int);parser.add_argument('--aec',choices=('on','off'),default='on')
    parser.add_argument('--output-gain',type=float,help='仅本次播放数字幅度；0.5约为-6dB，不改系统音量')
    parser.add_argument('--aec-backend',choices=('speex','aec3'),help='仅本次纯语音测试的前端，不修改P1配置')
    parser.add_argument('--aec-profile',choices=AEC_PROFILES,help='voice-preserve显式选择已试听的人声保护候选；baseline为原版')
    parser.add_argument('--check',action='store_true');parser.add_argument('--probe',action='store_true')
    parser.add_argument('--ask-key',action='store_true',help='在终端安全输入本次密钥，不依赖环境变量，也不保存')
    parser.add_argument('--list-devices',action='store_true');parser.add_argument('--record-audio',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv);config,speech,speaker,url,session=configuration(args)
    if args.check:
        print(json.dumps(dict(config_valid=True,model=config['model'],voice=config['voice'],speaker=speaker,
            frontend=config['frontend'],output_gain=config['output_gain'],
            endpoint=config['endpoint'],
            api_key_status=key_status(os.environ.get(config['api_key_env'],'')),
            microphone=speech['capture']['device'],websockets_installed=importlib.util.find_spec('websockets') is not None,
            hardware_opened=False,network_called=False),ensure_ascii=False));return 0
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices());return 0
    key=os.environ.get(config['api_key_env'])
    if args.ask_key:
        import getpass
        if not sys.stdin.isatty():parser.error('--ask-key需要交互终端')
        key=getpass.getpass('粘贴完整百炼API Key后回车（不回显、不保存）: ')
    if not key:parser.error('请在同一终端配置 '+config['api_key_env']+'；不用把密钥发到聊天或写入仓库')
    if key!=key.strip() or re.search(r'\s',key):parser.error('API Key含空格或换行，请重新复制输入；不会打印密钥')
    if importlib.util.find_spec('websockets') is None:parser.error('请先执行 ./tools/setup_realtime_voice.sh')
    folder=(args.output or ROOT/'outputs/realtime_voice'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    folder.mkdir(parents=True,exist_ok=False);log=Journal(folder/'events.jsonl',key)
    summary=dict(actuators_opened=False,camera_opened=False,record_audio=args.record_audio,error=None)
    print('日志：'+str(folder),flush=True)
    try:
        lock_path=ROOT/'outputs/runtime/p1.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
        with lock_path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('P1或另一个语音测试仍在运行，请先Ctrl+C退出旧实例')
            (folder/'effective.json').write_text(json.dumps(dict(config=config,session=session,aec=args.aec,
                microphone=speech['capture'],speaker=speaker),ensure_ascii=False,indent=2)+'\n')
            summary.update(asyncio.run(conversation(args,config,speech,speaker,url,session,key,folder,log,summary)))
    except KeyboardInterrupt:summary['operator_stop']=True
    except Exception as exc:
        summary['error']=str(exc).replace(key,'[redacted]');print('语音测试已停止：'+summary['error'],file=sys.stderr)
        diagnostic=handshake_diagnostic(exc,key)
        if diagnostic:
            summary['handshake']=diagnostic;log.write('handshake_error',detail=diagnostic)
            print('认证诊断：'+json.dumps(diagnostic,ensure_ascii=False),file=sys.stderr)
    finally:
        (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        log.close()
    return 1 if summary['error'] else 0


if __name__=='__main__':raise SystemExit(main())
