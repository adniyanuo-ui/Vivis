"""P1 local voice selection, cancellation and socially responsive standby."""
import argparse
import copy
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.developer import check_competitors,preview_state
from vivis.interaction.intents import IntentRouter,normalize
from vivis.interaction.session import SessionPolicy,STANDBY,INTERACTION
from vivis.interaction.voice_bridge import VoiceBridge
from vivis.motion.mapping import read_json
from vivis.hardware.feetech.sdk import check_serial_access
from vivis.hardware.feetech.protocol import ProtocolError


def dependencies(config,listener_only=False):
    speech=read_json(ROOT/config['speech']);missing=[]
    required=[ROOT/speech['python']]
    qwen_only=not listener_only and config.get('realtime_voice',{}).get('enabled') and config['realtime_voice'].get('control_backend')=='qwen_realtime'
    keys=() if qwen_only else (('encoder','decoder','tokens') if speech['asr'].get('backend')=='paraformer' else ('encoder','decoder','joiner','tokens'))
    for k in keys:required.append(ROOT/speech['asr']['directory']/speech['asr'][k])
    control=speech.get('control_keywords',{})
    if not qwen_only and control.get('enabled',False):
        required.extend(ROOT/control['directory']/control[k] for k in ('encoder','decoder','joiner','tokens'))
        required.append(ROOT/control['keywords_file'])
    if not listener_only:
        lip=config.get('realtime_voice',{}).get('lip_sync',{})
        if config.get('realtime_voice',{}).get('enabled') and lip.get('enabled'):
            required.extend(ROOT/lip[k] for k in ('binary','model'))
        catalog=read_json(ROOT/config['programs'])
        if any(p['kind']=='runtime_a2f' for p in catalog['programs']):
            required.extend([ROOT/'outputs/a2f/build/a2f_chunk_probe',ROOT/'outputs/a2f/model_bundle/model.json'])
        from vivis.speech.prompts import RUNTIME_PROMPTS
        if not qwen_only:
            for name in RUNTIME_PROMPTS:required.append(ROOT/speech['prompts']/(name+'.wav'))
        perception=read_json(ROOT/'configs/perception/face_landmarker.json')
        required.extend(ROOT/perception[k] for k in ('python','model'))
    for path in required:
        if not path.is_file():missing.append(str(path))
    if not listener_only and not qwen_only:
        try:
            from vivis.speech.prompts import prompt_cache_ready
            valid=prompt_cache_ready(speech)
        except (OSError,ValueError):valid=False
        if not valid:missing.append('Prompt cache voice/level differs; run python3 tools/setup_p1.py')
    return speech,missing


def program_command(program,robot,folder,speech,settings):
    stop=folder.with_suffix('.cancel');base=[sys.executable,'-m']
    common=['--robot',str(robot.path),'--output',str(folder),'--session-stop-file',str(stop)]
    if settings.get('_home_speed_counts_s') is not None and (program is None or program['kind']!='runtime_a2f'):
        common+=['--home-speed',str(settings['_home_speed_counts_s'])]
    if program is None or program['kind']=='interaction':
        mode=INTERACTION if program else STANDBY
        idle_config=ROOT/settings[mode]
        voice_options=settings.get('realtime_voice',{})
        if mode==STANDBY and voice_options.get('enabled') and voice_options.get('control_backend')=='qwen_realtime':
            reception=read_json(ROOT/voice_options['reception_config'])
            if reception['enabled']:
                effective=read_json(idle_config);effective['presence_only']=True
                idle_config=folder.with_suffix('.standby.json')
                idle_config.write_text(json.dumps(effective,ensure_ascii=False,indent=2)+'\n')
        continuous=['--continuous'] if voice_options.get('enabled') and voice_options.get('control_backend')=='qwen_realtime' else []
        return base+['vivis.interaction.standby',*common,'--seconds',str(settings[mode+'_seconds']),*continuous,
                     *(['--speech-animation',settings['_speech_animation_path']] if mode==INTERACTION and settings.get('_speech_animation_path') else []),
                     *(['--expression-intent',settings['_expression_intent_path']] if mode==INTERACTION and settings.get('_expression_intent_path') else []),
                     '--config',str(idle_config)]+(['--overlap-warmup'] if settings.get('overlap_warmup',False) else [])
    if not settings.get('diagnostics',False):common+=['--compact-logs']
    if settings.get('overlap_warmup',False):common+=['--overlap-warmup']
    voice_options=settings.get('realtime_voice',{})
    board=voice_options.get('board_reference_device') if voice_options.get('enabled') else None
    if board and program['kind'] in ('prerecorded','runtime_a2f'):
        common+=['--board-reference-device',board]
    if program['kind']=='prerecorded':
        if program.get('realtime_response',False):raise ValueError('Prepared programs cannot enable a social expression overlay')
        return base+['vivis.interaction.prerecorded',*common,'--work',program['work'],'--repeat','1',
            '--audio-reference-port',str(speech['aec']['reference_port']),'--session-id',folder.name,
            '--soft-audio-cancel-ms',str(settings['soft_audio_cancel_ms'])]
    if program['kind']=='runtime_a2f':
        response=program.get('realtime_response',False)
        if type(response) is not bool:raise ValueError('Program realtime_response must be boolean')
        return base+['vivis.interaction.singing',*common,'--work',program['work'],'--repeat','1',
            '--smile','on' if response else 'off',
            '--audio-reference-port',str(speech['aec']['reference_port']),'--session-id',folder.name,
            '--soft-audio-cancel-ms',str(settings['soft_audio_cancel_ms'])]
    if program['kind']=='expression_follow':
        return base+['vivis.interaction.follow',*common,'--tracking','on' if program.get('tracking',True) else 'off',
                     '--seconds',str(settings['expression_seconds'])]
    raise ValueError('Unregistered program kind: '+str(program['kind']))


class Worker:
    """Only this child may own the serial lock. Cancellation is instance-local."""
    def __init__(self,command,folder,events):
        self.folder=folder;self.cancelled_at=None;self.ready=False;self.preparation=dict(state='pending')
        folder.parent.mkdir(parents=True,exist_ok=True)
        self.log=folder.with_suffix('.log').open('w')
        try:
            self.process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONNOUSERSITE':'1','VIVIS_MANAGED_SESSION':'1'},
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        except BaseException:self.log.close();raise
        def reader():
            try:
                for line in self.process.stdout:
                    self.log.write(line)
                    if line.startswith(('SELF-CHECK ','HOME ','WARMUP ','TUNING ','PREPARE COMPLETE','PREPARE FAILED','PREPARE CANCELLED')):
                        state={'START':'running','PASS':'passed','FAIL':'failed','CANCELLED':'cancelled'}.get(line.split()[1]) if line.startswith('SELF-CHECK ') else None
                        if state is not None:self.preparation['state']=state
                        self.preparation['last_message']=line.rstrip();print(line.rstrip(),flush=True)
                        try:events.put_nowait(dict(kind='worker_preparation',pid=self.process.pid,state=state,message=line.rstrip()))
                        except queue.Full:pass
                    if any(text in line for text in ('EXPRESSION FOLLOW START','ROUND 1/1 START','STANDBY START','INTERACTION START')):
                        self.ready=True
                        try:events.put_nowait(dict(kind='worker_ready',pid=self.process.pid))
                        except queue.Full:pass
            finally:self.log.flush()
        self.thread=threading.Thread(target=reader,daemon=True);self.thread.start()

    def cancel(self):
        if self.cancelled_at is None:
            self.folder.with_suffix('.cancel').touch();self.cancelled_at=time.monotonic()

    def finish(self):
        self.thread.join(timeout=2)
        if self.process.stdout:self.process.stdout.close()
        self.log.close()

    def kill_owned_group(self):
        try:os.killpg(self.process.pid,signal.SIGKILL)
        except ProcessLookupError:pass


class PromptPlayer:
    def __init__(self,robot,speech,board_reference_device=None):
        self.device=robot.motion_profile()['audio_device'];self.speech=speech
        self.board_reference_device=board_reference_device
        self.clock=None;self.history=[]

    def stop(self):
        if self.clock is not None:
            try:
                if (ROOT/'outputs/STOP').exists():self.clock.close()
                else:self.clock.close_gracefully(40)
            finally:
                self.clock=None;self.history[-1][1]=time.monotonic()+.4

    def play(self,name):
        self.stop()
        from vivis.runtime.playback import AudioClock
        path=ROOT/self.speech['prompts']/(name+'.wav')
        self.clock=AudioClock(path,self.device,latency=.1,reference=dict(host='127.0.0.1',
            port=self.speech['aec']['reference_port'],session_id='prompt-'+str(uuid.uuid4())),
            board_reference_device=self.board_reference_device)
        self.history.append([time.monotonic(),math.inf]);self.history=self.history[-8:]
        self.clock.start()

    def poll(self):
        if self.clock is not None:
            t=self.clock.time()
            if self.clock.fault:raise RuntimeError('Voice prompt audio fault: '+str(self.clock.fault))
            if t is not None and t>=self.clock.duration:self.stop()

    def finish(self,cancelled=lambda:False,timeout_s=12.):
        """Bounded fault notification; no new program/intent is processed here."""
        deadline=time.monotonic()+timeout_s
        while self.clock is not None and not cancelled():
            self.poll()
            if self.clock is None:return
            if time.monotonic()>=deadline:raise RuntimeError('Fault prompt timed out')
            time.sleep(.02)
        if self.clock is not None:self.stop()

    def contains(self,stamp):return any(start<=stamp<=end for start,end in self.history)


class EventGate:
    def __init__(self,max_age):self.max_age=max_age;self.last=0;self.last_control=0
    def accept(self,event,now):
        control=event.get('kind')=='asr_control'
        if event.get('kind') not in ('asr_final','asr_control'):return False
        sequence=event.get('control_sequence') if control else event.get('utterance_id');stamp=event.get('captured_end_monotonic')
        previous=self.last_control if control else self.last
        if (type(sequence) is not int or sequence<=previous or type(stamp) not in (int,float)
                or not math.isfinite(stamp) or not 0<=now-stamp<=self.max_age):return False
        if control:self.last_control=sequence
        else:self.last=sequence
        return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/interaction/p1.json')
    parser.add_argument('--robot',type=Path);parser.add_argument('--check',action='store_true')
    parser.add_argument('--listen-only',action='store_true',help='Only microphone+ASR+intent, no speaker/camera/servo')
    parser.add_argument('--text-input',action='store_true',help='Type commands instead of opening microphone')
    parser.add_argument('--simulate-text',action='append',help='Offline routing only; no devices or models')
    parser.add_argument('--seconds',type=float);parser.add_argument('--output',type=Path)
    parser.add_argument('--diagnostics',choices=('on','off'),help='按需保留节目完整控制记录与串口追踪')
    parser.add_argument('--semantic',choices=('on','off'),help='交互态的可选LLM语义旁路；本地停止/退出始终优先')
    parser.add_argument('--cloud-chat',choices=('on','off'),help='交互态在线闲聊；无任务或硬件控制权')
    parser.add_argument('--realtime-voice',choices=('on','off'),help='交互态Qwen实时语音；off恢复原本地语音流程')
    parser.add_argument('--ask-key',action='store_true',help='安全输入本次实时语音API Key，不保存')
    parser.add_argument('--record-audio',action='store_true',help='显式保存常驻语音服务的原始／参考／处理后本地录音')
    args=parser.parse_args(argv);settings=read_json(args.config)
    warm_speed=settings.get('warm_home_speed_counts_s',60)
    if type(warm_speed) is not int or not 1<=warm_speed<=32767:parser.error('Invalid warm homing speed')
    if args.diagnostics is not None:settings['diagnostics']=args.diagnostics=='on'
    if settings.get('schema_version')!='vivis-dialog-programs-1.0':parser.error('Unsupported P1 config')
    if type(settings.get('overlap_warmup',False)) is not bool:parser.error('Invalid overlapping warmup switch')
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds<=0):parser.error('Invalid duration')
    if args.listen_only and args.text_input:parser.error('Choose microphone-only or typed-input mode')
    catalog=read_json(ROOT/settings['programs'])
    semantic_config=read_json(ROOT/settings['semantic']) if settings.get('semantic') else None
    conversation_config=read_json(ROOT/settings['conversation']) if settings.get('conversation') else None
    if args.semantic is not None:
        if semantic_config is None:parser.error('No semantic provider configuration')
        semantic_config['enabled']=args.semantic=='on'
    if args.cloud_chat is not None:
        if conversation_config is None:parser.error('No conversation provider configuration')
        conversation_config['enabled']=args.cloud_chat=='on'
    voice_options=copy.deepcopy(settings.get('realtime_voice',dict(enabled=False)))
    if type(voice_options.get('enabled',False)) is not bool:parser.error('Invalid realtime voice enabled switch')
    if args.realtime_voice is not None:voice_options['enabled']=args.realtime_voice=='on'
    voice_options['record_audio']=args.record_audio
    voice_enabled=voice_options.get('enabled',False) and not(args.listen_only or args.text_input or args.simulate_text is not None)
    board_reference_device=voice_options.get('board_reference_device') if voice_enabled else None
    settings['realtime_voice']={**voice_options,'enabled':voice_enabled}
    qwen_only=voice_enabled and voice_options.get('control_backend')=='qwen_realtime'
    reception_enabled=bool(qwen_only and read_json(ROOT/voice_options['reception_config'])['enabled'])
    if args.record_audio and not voice_enabled and not args.check:
        parser.error('--record-audio需要启用实时语音')
    if voice_enabled and (semantic_config and semantic_config.get('enabled') or conversation_config and conversation_config.get('enabled')):
        parser.error('实时语音与旧文本聊天／语义旁路不能同时启用；用--realtime-voice off选择旧流程')
    robot=Robot(args.robot or ROOT/settings['robot']);speech,missing=dependencies(settings,listener_only=args.listen_only)
    wake_words=settings.get('wake_words') or robot.definition.get('identity',{}).get('wake_words')
    if wake_words is None:wake_words=[robot.definition.get('identity',{}).get('display_name',robot.definition['name'])]
    router=IntentRouter(catalog,wake_words=wake_words)
    session=SessionPolicy(p['id'] for p in catalog['programs'])
    last_program=None
    if voice_enabled:
        from vivis.speech.managed import voice_configuration
        voice_configuration(dict(speech,managed_voice=voice_options,programs=str(ROOT/settings['programs'])),robot.path)
    if args.simulate_text is not None:
        for text in args.simulate_text:
            before=session.mode
            intent=router.parse(text,session.mode=='task',current_program=session.desired if session.mode=='task' else None,state=session.mode)
            session.request(intent)
            print(json.dumps(dict(text=text,intent=intent,mode_before=before,mode_after=session.mode,hardware_opened=False),ensure_ascii=False))
        return 0
    if args.check:
        # Plans are pure and validate model-specific ownership without devices.
        from vivis.interaction.follow import plan as follow_plan
        from vivis.interaction.standby import plan as standby_plan
        f=read_json(ROOT/'configs/interaction/expression_follow.json');f['tracking_enabled']=True
        follow_plan(robot,f,1.)
        standby_plan(robot,read_json(ROOT/settings['standby']),1.)
        standby_plan(robot,read_json(ROOT/settings['interaction']),1.)
        for p in catalog['programs']:
            if p['kind'] in ('runtime_a2f','prerecorded'):robot.work(p['work'])
        from vivis.interaction.knowledge import Knowledge
        knowledge=Knowledge(robot.path)
        print(json.dumps(dict(offline_valid=True,knowledge_enabled=knowledge.config is not None,
            familiar_profiles=len(knowledge.people),familiar_faces_configured=knowledge.recognition is not None,
            programs=[p['title'] for p in catalog['programs']],
            modes=[STANDBY,INTERACTION,'task'],wake_words=wake_words,initial_mode=STANDBY,
            realtime_voice=voice_enabled,voice_control_backend=voice_options.get('control_backend','local'),reception_enabled=reception_enabled,realtime_lip_sync=bool(voice_enabled and voice_options.get('lip_sync',{}).get('enabled')),
            board_reference_device=board_reference_device,host_aec_enabled=board_reference_device is None,
            missing_dependencies=missing,hardware_opened=False,requires_onsite=['wake ASR at intended distance','speaker double-talk/AEC','smooth mode transitions']),ensure_ascii=False));return 0
    if missing:parser.error('Missing optional P1 assets; run python3 tools/setup_p1.py and setup_p0.py: '+str(missing))
    if not args.listen_only:
        check_competitors(True)
        if preview_state() is not None:
            raise RuntimeError('已有独立视觉服务；请先关闭，再启动P1以保证真正待机时释放相机')
        try:check_serial_access(robot.motion_profile()['port'])
        except ProtocolError as exc:parser.error(str(exc))
    voice_secret=None;voice_key_env=None
    if voice_enabled:
        trial=read_json(ROOT/voice_options['config']);voice_key_env=trial['api_key_env']
        if args.ask_key:
            if not sys.stdin.isatty():parser.error('--ask-key需要交互终端')
            import getpass
            voice_secret=getpass.getpass('百炼API Key（不回显、不保存）: ')
        else:
            from vivis.speech.credentials import load_key
            try:voice_secret=load_key(voice_key_env)
            except ValueError as exc:parser.error(str(exc))
        if not voice_secret or any(c.isspace() for c in voice_secret):
            parser.error('请运行.venv/bin/python tools/configure_api_key.py保存本机Key，或加--ask-key临时输入；环境变量为'+voice_key_env+'；用--realtime-voice off可运行原本地流程')
    output=(args.output or ROOT/'outputs/p1'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve();output.mkdir(parents=True,exist_ok=False)
    speech_effective=copy.deepcopy(speech)
    speech_effective.update(wake_words=wake_words,programs=str((ROOT/settings['programs']).resolve()))
    if voice_enabled:speech_effective['managed_voice']=voice_options
    if voice_enabled and voice_options.get('lip_sync',{}).get('enabled',False):
        settings['_speech_animation_path']=str(output/'voice/mouth.json')
    if qwen_only:
        settings['_expression_intent_path']=str(output/'voice/expression_intent.json')
    speech_path=output/'speech_effective.json';speech_path.write_text(json.dumps(speech_effective,ensure_ascii=False,indent=2)+'\n')
    (output/'effective.json').write_text(json.dumps(dict(config=settings,robot=str(robot.path),speech=speech),ensure_ascii=False,indent=2)+'\n')
    lock_path=ROOT/'outputs/runtime/p1.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path.open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise RuntimeError('P1语音管理器已经运行，请使用当前实例或先退出它')
    if not args.listen_only:
        control=ROOT/'outputs/developer/control.lock';control.parent.mkdir(parents=True,exist_ok=True)
        try:
            with control.open('a') as probe:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close();raise RuntimeError('已有节目占用控制权，请先停止它，再启动P1语音管理器')
    events=queue.Queue(maxsize=64);worker=asr=None;asr_log=asr_trace=asr_thread=None;active_target=None;active_program=None
    quitting=False;fault=None;phase='starting';turn=0;started=time.monotonic();prompt=None;preparation={};log=(output/'events.jsonl').open('w')
    ready=False;warm_ready=False;reported_fault=None;semantic=cloud=voice=None
    voice_available=True
    def request_shutdown(*_):
        nonlocal quitting
        quitting=True
    handlers={s:signal.signal(s,request_shutdown) for s in (signal.SIGINT,signal.SIGTERM)}
    def record(kind,**fields):
        log.write(json.dumps(dict(monotonic=time.monotonic(),kind=kind,**fields),ensure_ascii=False)+'\n');log.flush()
    last_published=0.
    def publish(force=False):
        nonlocal last_published
        now=time.monotonic()
        if not force and now-last_published<.2:return
        last_published=now
        state=dict(schema_version='vivis-program-state-1.0',updated_monotonic=time.monotonic(),
            voice_available=voice_available,phase=phase,mode=session.mode_of(active_target) if active_target is not None else session.mode,
            desired_mode=session.mode,active_program=active_program,desired_program=session.desired if session.mode=='task' else None,
            microphone_active=ready and asr is not None and asr.poll() is None,
            microphone_pid=getattr(asr,'pid',None),
            wake_words=wake_words,error=fault,output=str(output),self_check=preparation)
        if voice:state['realtime_voice']=dict(status=voice.status,speaker_owned=voice.owned,epoch=voice.epoch)
        path=ROOT/'outputs/runtime/program_status.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,ensure_ascii=False)+'\n');temp.replace(path)
    def start_worker(target):
        nonlocal turn,active_target,active_program,phase,preparation
        turn+=1;folder=output/f'{turn:04d}_{target}'
        program=(dict(id=INTERACTION,title='交互态',kind='interaction') if target==INTERACTION else None
                 if target==STANDBY else next(p for p in catalog['programs'] if p['id']==target))
        worker_settings=dict(settings)
        if warm_ready:
            worker_settings['_home_speed_counts_s']=min(warm_speed,*(m['max_speed_counts_s'] for m in robot.motion_profile()['motors']))
        command=program_command(program,robot,folder,speech,worker_settings)
        active_target=target;active_program=target if session.mode_of(target)=='task' else None
        phase='preparing';preparation=dict(state='pending')
        record('start',program=active_program,mode=session.mode_of(target),session=folder.name);print('准备：'+(program['title'] if program else '真正待机'),flush=True)
        return Worker(command,folder,events)
    gate=EventGate(settings['max_event_age_s'])
    try:
        def optional_event(event):
            try:events.put_nowait(event)
            except queue.Full:pass  # Optional result, never displace a control/capture event.
        identity=robot.definition.get('identity',dict(display_name=robot.definition['name']))
        if semantic_config and semantic_config.get('enabled',False):
            from vivis.interaction.semantic import SemanticRouter,SemanticWorker
            semantic=SemanticWorker(SemanticRouter(semantic_config,catalog,identity),optional_event)
            record('semantic_config',config=semantic_config,identity=identity,spoken_chat=False)
        if conversation_config and conversation_config.get('enabled',False):
            from vivis.interaction.conversation import ConversationRouter
            from vivis.interaction.semantic import SemanticWorker
            from vivis.interaction.knowledge import Knowledge
            cloud=SemanticWorker(ConversationRouter(conversation_config,catalog,identity,Knowledge(robot.path)),optional_event,event_kind='cloud_result')
            record('conversation_config',config=conversation_config,identity=identity,spoken_chat=False)
        if not args.text_input:
            asr_log=(output/'asr.log').open('w')
            asr_trace=(output/'asr_events.jsonl').open('w')
            command=[str(ROOT/speech['python']),'-m','vivis.speech.managed' if voice_enabled else 'vivis.speech.listen',
                     '--config',str(speech_path)]
            child_env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONNOUSERSITE':'1'}
            if voice_enabled:
                command+=['--robot',str(robot.path),'--output',str(output/'voice')]
                child_env[voice_key_env]=voice_secret
            asr=subprocess.Popen(command,
                cwd=ROOT,env=child_env,stdout=subprocess.PIPE,stdin=subprocess.PIPE if voice_enabled else None,
                stderr=asr_log,text=True,start_new_session=True)
            if voice_enabled:voice=VoiceBridge(asr.stdin)
            def read_asr():
                for line in asr.stdout:
                    try:event=json.loads(line)
                    except ValueError:continue
                    asr_trace.write(json.dumps(event,ensure_ascii=False)+'\n');asr_trace.flush()
                    # Partials/review are diagnostics. They cannot evict a stop, capture fault or ready event.
                    if event.get('kind')=='cloud_transcript' and event.get('role')!='user':
                        try:events.put_nowait(event)
                        except queue.Full:pass
                        continue
                    if event.get('kind') not in ('asr_final','asr_control','asr_error','ready','voice_state','voice_warning','cloud_error','cloud_transcript','cloud_action','mouth_error','voice_health'):continue
                    while not quitting:
                        try:events.put(event,timeout=.05);break
                        except queue.Full:
                            if asr.poll() is not None:break
            asr_thread=threading.Thread(target=read_asr,daemon=True);asr_thread.start()
        else:
            def read_text():
                for number,line in enumerate(sys.stdin,1):
                    events.put(dict(kind='asr_final',utterance_id=number,text=line.strip(),captured_end_monotonic=time.monotonic()))
                events.put(dict(kind='shutdown'))
            threading.Thread(target=read_text,daemon=True).start()
        prompt=None if args.listen_only or qwen_only else PromptPlayer(robot,speech,board_reference_device)
        ready=args.text_input
        print('P1三级状态：待机→唤醒交互→任务。停止／退出回交互；待机／自己玩去吧回待机。Ctrl+C退出管理器。',flush=True)
        if board_reference_device:
            print('音频回采：'+board_reference_device+'；主机AEC已关闭。',flush=True)
        print('唤醒词：'+'、'.join(wake_words)+'；管理器运行期间持续监听。',flush=True)
        while True:
            now=time.monotonic()
            if args.seconds is not None and now-started>=args.seconds:quitting=True
            if (ROOT/'outputs/STOP').exists():quitting=True;fault='Emergency stop file present'
            if asr is not None and asr.poll() is not None and not quitting:quitting=True;fault='Speech process ended; see asr.log'
            for _ in range(32):
                try:event=events.get_nowait()
                except queue.Empty:break
                if event.get('kind')=='shutdown':quitting=True;continue
                if event.get('kind')=='voice_health' and qwen_only:
                    voice_available=bool(event['available']);record('voice_health',detail=event)
                    if not voice_available:
                        phase='voice_recovering';session.desired=STANDBY;session.pending_prompt=None
                        if worker:worker.cancel()
                        if voice:voice.enable(False)
                        print('语音链路恢复中；停止当前任务并保持扭矩，管理器继续运行。原因：'+event['reason'],flush=True)
                    else:
                        print('语音链路已恢复；等待新的唤醒或接待，不续播旧节目、不重放旧指令。',flush=True)
                    continue
                if event.get('kind')=='mouth_error':
                    record('mouth_error',detail=event)
                    print('语音口型暂不可用，语音及本地控制继续：'+event['error'],flush=True);continue
                if event.get('kind')=='asr_error':
                    fault='Speech input unavailable: '+str(event);quitting=True
                    if worker:worker.cancel()
                    record('speech_fault',detail=event);continue
                if event.get('kind') in ('asr_partial','asr_refinement','asr_refinement_error','asr_refinement_skipped','asr_overload'):
                    record('speech_diagnostic',detail=event);continue
                if event.get('kind')=='ready':
                    ready=True;record('speech_ready',detail=event)
                    print('麦克风已就绪，请说唤醒词。'+('本次只识别，不驱动机器人。' if args.listen_only else ('当前为前台接待待机，相机观察来人。' if reception_enabled else '当前为真正待机，摄像头关闭。')),flush=True)
                    continue
                if event.get('kind')=='worker_ready' and worker and worker.cancelled_at is None and event['pid']==worker.process.pid:
                    warm_ready=True
                    phase=session.mode_of(active_target);record('worker_ready',program=active_program,mode=phase)
                    print('状态：'+{STANDBY:('①接待待机（相机观察来人）' if reception_enabled else '①待机（摄像头关闭，等待唤醒）'),INTERACTION:'②交互态','task':'③任务执行'}[phase],flush=True)
                    continue
                if event.get('kind')=='worker_preparation' and worker and event['pid']==worker.process.pid:
                    if event['state'] is not None:preparation['state']=event['state']
                    preparation['last_message']=event['message']
                    record('preparation',program=active_program,**preparation);continue
                if event.get('kind')=='cloud_action':
                    if not qwen_only or not voice or not voice_available or event.get('revision')!=voice.context_revision or quitting or fault:
                        record('cloud_action_stale',detail=event);continue
                    from vivis.interaction.cloud_control import controller_intent
                    try:requested=controller_intent(event.get('arguments'),session.programs,session.mode)
                    except ValueError as exc:
                        record('cloud_action_rejected',reason=str(exc));continue
                    event=dict(kind='asr_final',text=event.get('text','Qwen command'),source='qwen_tool',
                        controller_intent=requested,captured_end_monotonic=time.monotonic())
                if event.get('kind') in ('voice_state','voice_warning','cloud_error','cloud_transcript'):
                    if not voice or not voice.accept(event):
                        record('voice_stale',detail=event);continue
                    record(event['kind'],detail=event)
                    if event['kind']=='voice_warning':
                        print('语音提示：'+event['message'],flush=True);continue
                    if event['kind']=='voice_state' and event['status']=='ready':
                        print('实时语音已就绪：'+('唤醒、聊天和节目控制均由Qwen理解；无脸默认静默。' if qwen_only else '可直接聊天、插话；停止／待机／节目切换仍由本地识别执行。'),flush=True)
                    elif event['kind']=='cloud_error':
                        if qwen_only:
                            fault='Qwen语音不可用：'+event['error'];quitting=True
                            if worker:worker.cancel()
                            print(fault+'；停止任务，不启用本地模型替代。',flush=True)
                        else:print('实时聊天暂不可用，本地控制继续。可待机后重新唤醒：'+event['error'],flush=True)
                    elif event['kind']=='cloud_transcript' and event.get('role')=='assistant':
                        print(identity['display_name']+'：'+event.get('text',''),flush=True)
                    elif event['kind']=='cloud_transcript' and event.get('role')=='user' and event.get('text','').strip():
                        print('用户（识别）：'+event['text'],flush=True)
                    if qwen_only:continue  # Cloud tools alone decide actions; transcripts are diagnostics.
                    # A completed USER transcription is a second recognizer,
                    # not an LLM action/tool. Only the same explicit control
                    # whitelist may pass; assistant replies never enter it.
                    candidate=router.parse(event.get('text',''),session.mode=='task',
                        current_program=session.desired if session.mode=='task' else None,state=session.mode,allow_title_correction=True,
                        previous_program=last_program['id'] if last_program else None)
                    if (event['kind']=='cloud_transcript' and event.get('role')=='user'
                            and session.mode==INTERACTION and voice.target
                            and candidate['kind'] in ('standby','interaction','select')):
                        stamp=time.monotonic()
                        event=dict(kind='asr_final',text=event['text'],captured_end_monotonic=stamp,
                            speech_end_monotonic=stamp,source='cloud_user_transcription')
                    else:continue
                if event.get('kind')=='cloud_result':
                    if not cloud or not cloud.accepts(event,session.mode) or quitting or fault:
                        record('cloud_stale',detail=event);continue
                    record('cloud_result',detail=event)
                    if event.get('error'):print('在线聊天暂不可用；本地控制继续。'+event['error'],flush=True)
                    else:
                        cloud.router.commit(event['text'],event['reply'])
                        print(identity['display_name']+'：'+event['reply'],flush=True)
                    continue  # Text only: never apply cloud output to SessionPolicy.
                if event.get('kind')=='semantic_result':
                    if not semantic or not semantic.accepts(event,session.mode) or quitting or fault:
                        record('semantic_stale',detail=event);continue
                    record('semantic_result',detail=event)
                    if event.get('error'):
                        print('语义辅助暂不可用；本地控制继续。'+event['error'],flush=True)
                        if not args.listen_only:session.request(dict(kind='prompt',name='clarify'))
                        continue
                    if event['intent']['kind']=='ignore' and cloud:
                        offered=cloud.offer(event,session.mode)
                        record('cloud_offered',utterance_id=event['utterance_id'],accepted=offered)
                        continue
                    if event['intent']['kind']=='select' and not semantic_config.get('allow_program_selection',False):
                        record('semantic_selection_blocked',detail=event)
                        title=next(p['title'] for p in catalog['programs'] if p['id']==event['intent']['program'])
                        event={**event,'intent':dict(kind='prompt',name='clarify'),
                               'reply':'本地模型仅给出候选；要播放可直接说“播放'+title+'”。'}
                    if event.get('reply'):
                        print(robot.definition.get('identity',{}).get('display_name',robot.definition['name'])+'：'+event['reply'],flush=True)
                    restart=session.request(event['intent'])
                    if args.listen_only:session.pending_prompt=None;phase=session.mode;continue
                    if event['intent']['kind'] in ('select','prompt'):
                        if prompt:prompt.stop()
                        if worker and (session.desired!=active_target or restart):
                            worker.cancel();phase='stopping'
                            record('cancel_requested',program=active_program,reason=event['text'],source='semantic')
                    continue
                if qwen_only and event.get('source')!='qwen_tool':continue
                if event.get('source') not in ('cloud_user_transcription','qwen_tool') and not gate.accept(event,time.monotonic()):continue
                if event.get('kind')=='asr_control' and (session.mode!='task' or event.get('text') not in ('停止','退出','结束','待机')):
                    record('control_keyword_ignored',detail=event);continue
                intent=event['controller_intent'] if qwen_only else router.parse(event.get('text',''),session.mode=='task',
                    current_program=session.desired if session.mode=='task' else None,state=session.mode,allow_title_correction=True,
                    previous_program=last_program['id'] if last_program else None)
                stamp=event.get('speech_end_monotonic') or event['captured_end_monotonic']
                prompt_controls=('standby','interaction','select','wake') if voice else ('standby','interaction')
                if prompt and prompt.contains(stamp) and intent['kind'] not in prompt_controls:
                    record('prompt_turn_ignored',text=event['text']);continue
                if intent['kind']=='ignore' and normalize(event.get('text','')) in ('','嗯','呃','啊','哦','噢','对','是'):
                    record('filler_ignored',text=event.get('text',''));continue
                if voice and not qwen_only and session.mode==INTERACTION:
                    if intent['kind']=='interaction':voice.interrupt()
                    if intent['kind']=='prompt' and intent.get('name')=='welcome' and voice.failed:
                        voice.enable(False);voice.enable(True)
                        record('voice_retry',text=event['text']);continue
                    if intent['kind'] in ('ignore','prompt') and not voice.failed:
                        record('voice_chat_input',text=event['text'],cloud_status=voice.status)
                        continue  # Same cleaned audio already reaches Qwen; no cached TTS duplicate.
                if semantic:semantic.invalidate()
                if cloud:
                    cloud.invalidate()
                    if intent['kind']=='standby':cloud.router.clear()
                record('intent',text=event['text'],intent=intent,metrics=event.get('metrics',{}),
                    source=event.get('source','local_asr'))
                use_semantic=semantic and session.mode==INTERACTION and intent['kind']=='ignore'
                use_cloud=cloud and not semantic and session.mode==INTERACTION and intent['kind']=='ignore'
                label={'standby':'真正待机','interaction':'回到交互','wake':'唤醒交互','ignore':'未匹配','prompt':'交互问答'}.get(intent['kind'],intent['kind'])
                if use_semantic:label='语义理解中'
                elif use_cloud:label='在线聊天中'
                if intent['kind']=='select':label='选择'+next(p['title'] for p in catalog['programs'] if p['id']==intent['program'])
                if fault:label+='；未执行，管理器处于故障状态，请Ctrl+C后检查日志并重启'
                print('识别：'+event['text']+' → '+label,flush=True)
                if fault:continue
                if use_cloud:
                    offered=cloud.offer(event,session.mode)
                    record('cloud_offered',utterance_id=event['utterance_id'],accepted=offered)
                    if not offered:print('在线聊天忙，已跳过本句；本地控制继续。',flush=True)
                    continue
                if use_semantic:
                    offered=semantic.offer(event,session.mode)
                    record('semantic_offered',utterance_id=event['utterance_id'],accepted=offered)
                    if offered:continue
                    print('语义辅助忙，已跳过本句；停止／退出仍随时有效。',flush=True)
                    continue
                restart=session.request(intent)
                if args.listen_only:
                    session.pending_prompt=None;phase=session.mode;continue
                if intent['kind'] in ('wake','select','standby','interaction','prompt'):
                    if voice and session.desired!=INTERACTION:voice.enable(False)
                    if prompt:prompt.stop()
                    if worker and (session.desired!=active_target or restart):
                        worker.cancel();phase='stopping'
                        record('cancel_requested',program=active_program,reason=event['text'])
            if fault and fault.startswith(('Speech process ended','Speech input unavailable')) and fault!=reported_fault:
                print('语音或设备故障，正在停止任务并保持扭矩：'+fault+'；日志：'+str(output/'asr.log'),flush=True)
                reported_fault=fault
            if prompt:prompt.poll()
            if worker is not None:
                code=worker.process.poll()
                if code is not None:
                    cancelled=worker.cancelled_at is not None
                    worker.kill_owned_group();worker.finish();preparation.update(worker.preparation)
                    record('worker_exit',program=active_program,code=code,cancelled=cancelled,self_check=preparation)
                    if active_program is not None and worker.ready and code in (0,130):
                        program=next(p for p in catalog['programs'] if p['id']==active_program)
                        last_program=dict(id=active_program,title=program['title'],outcome='interrupted' if cancelled or code==130 else 'completed')
                        if voice:voice.last_program=dict(last_program)
                        record('program_result',**last_program)
                    failed_log=worker.folder.with_suffix('.log') if code not in (0,130) else None
                    session.completed(active_target,cancelled)
                    worker=None;active_target=None;active_program=None
                    if failed_log is not None:
                        fault='Program failed; see '+str(failed_log);phase='fault';session.pending_prompt=None
                        print('节目故障，已停止；管理器将退出，处理后请重新启动。日志：'+str(failed_log),flush=True)
                        publish(True)
                        if prompt and not quitting and (voice is None or voice.released):
                            try:
                                # play() first cancels the unfinished welcome/menu.
                                prompt.play('fault');prompt.finish(lambda:quitting)
                            except Exception as exc:record('fault_prompt_error',error=str(exc))
                        quitting=True
                elif worker.cancelled_at is not None and now-worker.cancelled_at>settings['cancel_timeout_s']:
                    worker.kill_owned_group();worker.process.wait(timeout=2);worker.finish();worker=None
                    fault='Worker cancellation timed out; hold position, no automatic recovery';phase='fault';quitting=True
            if voice:
                if qwen_only:
                    session.pending_prompt=None
                    voice.context(session.mode,session.desired,bool(worker and worker.ready and worker.cancelled_at is None and active_target==session.desired and not quitting and fault is None and voice_available))
                desired_voice=(voice_available and not quitting and not fault and session.desired==INTERACTION and active_target==INTERACTION
                    and worker is not None and worker.ready and worker.cancelled_at is None
                    and session.pending_prompt is None and (prompt is None or prompt.clock is None)
                    and (prompt is None or not prompt.contains(time.monotonic())))
                voice.enable(desired_voice)
                if (not voice.target and not voice.released and voice.release_requested_at is not None
                        and now-voice.release_requested_at>settings['cancel_timeout_s']):
                    fault='Speech speaker release timed out; no new audio owner started';quitting=True
            if quitting:
                if prompt:prompt.stop()
                if worker:worker.cancel()
                else:break
            elif not args.listen_only and ready and voice_available and worker is None and fault is None and (voice is None or voice.released):worker=start_worker(session.desired)
            if not quitting and worker is not None and prompt and (voice is None or voice.released):
                spoken=session.take_prompt(active_target,getattr(worker,'ready',False) and worker.cancelled_at is None)
                if spoken:prompt.play(spoken)
            publish();time.sleep(.02)
    except BaseException as exc:
        fault=str(exc);raise
    finally:
        cleanup_errors=[]
        def cleanup(action):
            try:action()
            except Exception as exc:cleanup_errors.append(str(exc))
        if semantic:cleanup(semantic.close)
        if cloud:cleanup(cloud.close)
        if prompt:cleanup(prompt.stop)
        if worker is not None:
            cleanup(worker.cancel)
            def wait_worker():
                try:worker.process.wait(timeout=settings['cancel_timeout_s'])
                except subprocess.TimeoutExpired:worker.kill_owned_group();worker.process.wait(timeout=2)
            cleanup(wait_worker)
            cleanup(worker.kill_owned_group)
            cleanup(worker.finish)
        if asr is not None:
            def stop_asr():
                asr.terminate()
                try:asr.wait(timeout=settings['cancel_timeout_s'] if voice else 3)
                except subprocess.TimeoutExpired:asr.kill();asr.wait(timeout=2)
            cleanup(stop_asr)
            if asr_thread:cleanup(lambda:asr_thread.join(timeout=2))
            if asr.stdout:cleanup(asr.stdout.close)
            if asr.stdin:cleanup(asr.stdin.close)
        if asr_log:cleanup(asr_log.close)
        if asr_trace:cleanup(asr_trace.close)
        if cleanup_errors:fault=fault or 'Resource cleanup failed: '+'; '.join(cleanup_errors)
        phase='fault' if fault else 'stopped';cleanup(lambda:publish(True));cleanup(lambda:record('closed',error=fault));cleanup(log.close)
        cleanup(lambda:(output/'summary.json').write_text(json.dumps(dict(phase=phase,error=fault,program_sessions=turn,
            last_mode=session.mode_of(active_target) if active_target is not None else session.mode,
            desired_mode=session.mode,wake_words=wake_words,microphone_process_starts=int(asr is not None),
            cleanup_errors=cleanup_errors,listen_only=args.listen_only,text_input=args.text_input,
            elapsed_seconds=time.monotonic()-started),ensure_ascii=False,indent=2)+'\n'))
        for s,h in handlers.items():signal.signal(s,h)
        lock.close()
    return 1 if fault else 0


if __name__=='__main__':raise SystemExit(main())
