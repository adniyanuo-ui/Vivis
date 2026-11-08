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
from vivis.speech.realtime_audio import AudioIO,VoiceBuffer


def configuration(args):
    config=json.loads(args.config.read_text())
    if config.get('schema_version')!='vivis-realtime-voice-test-1.0':raise ValueError('Unsupported voice test config')
    for name in ('voice','model','seconds'):
        value=getattr(args,name,None)
        if value is not None:config[name]=value
    if args.region:
        host='dashscope.aliyuncs.com' if args.region=='beijing' else 'dashscope-intl.aliyuncs.com'
        config['endpoint']='wss://'+host+'/api-ws/v1/realtime'
    if not 10<=config['seconds']<=900:raise ValueError('Voice test duration must be10..900 seconds')
    if not .5<=config['speech_rate']<=2.:raise ValueError('Invalid speech rate')
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
                 output_audio=dict(language='zh'),turn_detection=dict(type=config['turn_detection']),
                 max_history_turns=config['max_history_turns'],enable_search=False,tools=[])
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
    def __init__(self,player,log):
        self.player=player;self.log=log;self.generating=None;self.blocked=set();self.responses={}
        self.speech_stopped_at=None;self.cancel_operations=0;self.late_packets=0;self.usage=[]
        self.server_starts=0;self.server_cancelled=set();self.invalid_turns=0

    def interrupt(self,response_id=None,source='keyboard'):
        result=self.player.cancel(response_id)
        ids=set(result['responses'])
        if response_id:ids.add(response_id)
        if response_id is None and self.generating:ids.add(self.generating)
        self.blocked.update(ids)
        if ids:self.cancel_operations+=1
        self.log.write('local_interrupt',source=source,**result,blocked_responses=sorted(ids))

    def event(self,event):
        kind=event.get('type','unknown');rid=event.get('response_id');now=time.monotonic()
        if kind=='response.audio.delta':
            if not rid:raise RuntimeError('Audio packet missing response_id; cannot safely reject stale audio')
            if rid in self.blocked:
                self.late_packets+=1;return
            pcm=base64.b64decode(event['delta'],validate=True)
            response=self.responses.setdefault(rid,{})
            if 'first_audio_received' not in response:
                response['first_audio_received']=now
                self.log.write('first_audio_received',response_id=rid,bytes=len(pcm))
            self.player.add(rid,pcm);return
        fields={k:event[k] for k in ('event_id','item_id','response_id','audio_start_ms','audio_end_ms','reason','transcript','text') if k in event}
        if kind=='response.created':
            rid=event['response']['id'];self.generating=rid
            self.responses[rid]=dict(created=now,speech_stopped_received=self.speech_stopped_at)
            fields['response_id']=rid
        elif kind=='input_audio_buffer.speech_started':
            self.server_starts+=1
            # Server VAD already cancels inference. Sending another cancel here could kill the NEW response.
            self.interrupt(source='server_speech_started')
        elif kind=='input_audio_buffer.speech_stopped':
            if event.get('reason')!='turn_invalid':self.speech_stopped_at=now
            else:self.invalid_turns+=1
        elif kind=='response.done':
            response=event['response'];rid=response['id'];fields.update(response_id=rid,status=response.get('status'))
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
        elif kind=='conversation.item.input_audio_transcription.completed':print('你：'+event.get('transcript',''),flush=True)
        elif kind=='response.audio_transcript.done':print('机器人：'+event.get('transcript',''),flush=True)
        self.log.write('server_event',type=kind,**fields)

    def poll_playback(self):
        while not self.player.notices.empty():
            event=self.player.notices.get_nowait();rid=event['response_id']
            if event['kind']=='playback_started':self.responses.setdefault(rid,{}).setdefault('first_dac_monotonic',event['dac_monotonic'])
            self.log.write(**event)


async def connect_session(ws,session,log):
    await ws.send(json.dumps(dict(type='session.update',session=session),ensure_ascii=False))
    for _ in range(20):
        event=json.loads(await asyncio.wait_for(ws.recv(),timeout=15.))
        if event.get('type')=='error':raise RuntimeError('Qwen session configuration failed: '+str(event.get('error',{})))
        if event.get('type')=='session.updated':
            accepted=event['session']
            if (accepted.get('voice')!=session['voice']
                    or accepted.get('turn_detection',{}).get('type')!=session['turn_detection']['type']):
                raise RuntimeError('Server did not confirm requested voice/turn detection; do not silently fall back')
            log.write('session_ready',model=accepted.get('model'),voice=accepted.get('voice'),turn_detection=accepted.get('turn_detection'))
            return
    raise RuntimeError('Qwen did not acknowledge session configuration')


async def conversation(args,config,speech,speaker,url,session,key,folder,log,summary):
    from websockets.asyncio.client import connect
    async with connect(url,additional_headers={'Authorization':'Bearer '+key},compression=None,
                       open_timeout=15,close_timeout=3,ping_interval=20,ping_timeout=20,max_size=4*1024*1024,max_queue=16) as ws:
        await connect_session(ws,session,log)
        if args.probe:
            print('连接、模型权限、音色及语义轮次配置已确认；未打开麦克风或音箱。',flush=True)
            return dict(probe=True,audio_opened=False)
        player=VoiceBuffer(config['output_gain'],config['fade_ms'],config['max_playback_buffer_s'])
        state=VoiceSession(player,log);audio=AudioIO(speech,speaker,player,args.aec=='on');keys=queue.SimpleQueue()
        def read_keys():
            for line in sys.stdin:keys.put(line.strip())
        threading.Thread(target=read_keys,daemon=True).start()
        started=time.monotonic();tasks=[];stats={}
        try:
            with ExitStack() as stack:
                writers={}
                if args.record_audio:
                    for name in ('uplink','microphone','reference'):
                        wav=stack.enter_context(wave.open(str(folder/(name+'_16k.wav')),'wb'))
                        wav.setparams((1,2,16000,0,'NONE','not compressed'));writers[name]=wav
                devices=audio.open();log.write('audio_ready',devices=devices,aec=args.aec)
                print('纯语音聊天已就绪，直接说话，无需唤醒。机器头／相机／A2F均未启用。',flush=True)
                print('Enter本地停声；/mark echo_only等可标记阶段；/quit或Ctrl+C结束。',flush=True)
                async def send_audio():
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
                        for name,data in [('uplink',pcm),('microphone',raw),('reference',far)]:
                            if name in writers:writers[name].writeframesraw(data)
                async def receive():
                    async for message in ws:state.event(json.loads(message))
                    raise RuntimeError('Realtime service disconnected')
                tasks=[asyncio.create_task(send_audio()),asyncio.create_task(receive())]
                while time.monotonic()-started<config['seconds']:
                    for task in tasks:
                        if task.done():task.result()
                    state.poll_playback()
                    if audio.fault:raise RuntimeError(audio.fault)
                    if time.monotonic()-started>1.:
                        if audio.last_capture is None or time.monotonic()-audio.last_capture>1.:
                            raise RuntimeError('Microphone callback watchdog')
                        if audio.last_output is None or time.monotonic()-audio.last_output>1.:
                            raise RuntimeError('Output callback watchdog')
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
            stats=dict(audio.stats);stats.update(elapsed_s=time.monotonic()-started,responses=state.responses,
                local_cancel_operations=state.cancel_operations,cancelled_response_count=len(state.blocked),
                server_cancelled_response_count=len(state.server_cancelled),server_speech_started_events=state.server_starts,
                invalid_turns=state.invalid_turns,rejected_late_packets=state.late_packets,usage=state.usage,clipped_samples=player.clipped)
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
    parser.add_argument('--check',action='store_true');parser.add_argument('--probe',action='store_true')
    parser.add_argument('--ask-key',action='store_true',help='在终端安全输入本次密钥，不依赖环境变量，也不保存')
    parser.add_argument('--list-devices',action='store_true');parser.add_argument('--record-audio',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv);config,speech,speaker,url,session=configuration(args)
    if args.check:
        print(json.dumps(dict(config_valid=True,model=config['model'],voice=config['voice'],speaker=speaker,
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
