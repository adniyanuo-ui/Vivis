"""P1 local voice program selection, cancellation and camera-free standby."""
import argparse
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
from vivis.runtime.developer import check_competitors
from vivis.interaction.intents import IntentRouter
from vivis.motion.mapping import read_json


def dependencies(config,listener_only=False):
    speech=read_json(ROOT/config['speech']);missing=[]
    required=[ROOT/speech['python']]
    for k in ('encoder','decoder','joiner','tokens'):required.append(ROOT/speech['asr']['directory']/speech['asr'][k])
    if 'asr_final' in speech:
        for k in ('model','tokens'):required.append(ROOT/speech['asr_final']['directory']/speech['asr_final'][k])
    if not listener_only:
        required.extend([ROOT/'outputs/a2f/build/a2f_chunk_probe',ROOT/'outputs/a2f/model_bundle/model.json'])
        for name in ('welcome','menu','clarify','standby','identity','fault'):required.append(ROOT/speech['prompts']/(name+'.wav'))
        perception=read_json(ROOT/'configs/perception/face_landmarker.json')
        required.extend(ROOT/perception[k] for k in ('python','model'))
    for path in required:
        if not path.is_file():missing.append(str(path))
    return speech,missing


def program_command(program,robot,folder,speech,settings):
    stop=folder.with_suffix('.cancel');base=[sys.executable,'-m']
    common=['--robot',str(robot.path),'--output',str(folder),'--session-stop-file',str(stop)]
    if program is None:
        return base+['vivis.interaction.standby',*common,'--seconds',str(settings['standby_seconds']),
                     '--config',str(ROOT/settings['standby'])]
    if not settings.get('diagnostics',False):common+=['--compact-logs']
    if program['kind']=='runtime_a2f':
        return base+['vivis.interaction.singing',*common,'--work',program['work'],'--repeat','1',
            '--audio-reference-port',str(speech['aec']['reference_port']),'--session-id',folder.name,
            '--soft-audio-cancel-ms',str(settings['soft_audio_cancel_ms'])]
    if program['kind']=='expression_follow':
        return base+['vivis.interaction.follow',*common,'--tracking','on' if program.get('tracking',True) else 'off',
                     '--seconds',str(settings['expression_seconds'])]
    raise ValueError('Unregistered program kind: '+str(program['kind']))


class Worker:
    """Only this child may own the serial lock. Cancellation is instance-local."""
    def __init__(self,command,folder,events):
        self.folder=folder;self.cancelled_at=None;self.ready=False
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
                    if any(text in line for text in ('EXPRESSION FOLLOW START','ROUND 1/1 START','STANDBY START')):
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
    def __init__(self,robot,speech):
        self.device=robot.motion_profile()['audio_device'];self.speech=speech
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
            port=self.speech['aec']['reference_port'],session_id='prompt-'+str(uuid.uuid4())))
        self.history.append([time.monotonic(),math.inf]);self.history=self.history[-8:]
        self.clock.start()

    def poll(self):
        if self.clock is not None:
            t=self.clock.time()
            if self.clock.fault:raise RuntimeError('Voice prompt audio fault: '+str(self.clock.fault))
            if t is not None and t>=self.clock.duration:self.stop()

    def contains(self,stamp):return any(start<=stamp<=end for start,end in self.history)


class EventGate:
    def __init__(self,max_age):self.max_age=max_age;self.last=0
    def accept(self,event,now):
        if event.get('kind')!='asr_final':return False
        sequence=event.get('utterance_id');stamp=event.get('captured_end_monotonic')
        if (type(sequence) is not int or sequence<=self.last or type(stamp) not in (int,float)
                or not math.isfinite(stamp) or not 0<=now-stamp<=self.max_age):return False
        self.last=sequence;return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/interaction/p1.json')
    parser.add_argument('--robot',type=Path);parser.add_argument('--check',action='store_true')
    parser.add_argument('--listen-only',action='store_true',help='Only microphone+ASR+intent, no speaker/camera/servo')
    parser.add_argument('--text-input',action='store_true',help='Type commands instead of opening microphone')
    parser.add_argument('--simulate-text',action='append',help='Offline routing only; no devices or models')
    parser.add_argument('--seconds',type=float);parser.add_argument('--output',type=Path)
    parser.add_argument('--diagnostics',choices=('on','off'),help='按需保留节目完整控制记录与串口追踪')
    args=parser.parse_args(argv);settings=read_json(args.config)
    if args.diagnostics is not None:settings['diagnostics']=args.diagnostics=='on'
    if settings.get('schema_version')!='vivis-dialog-programs-1.0':parser.error('Unsupported P1 config')
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds<=0):parser.error('Invalid duration')
    if args.listen_only and args.text_input:parser.error('Choose microphone-only or typed-input mode')
    catalog=read_json(ROOT/settings['programs']);router=IntentRouter(catalog)
    robot=Robot(args.robot or ROOT/settings['robot']);speech,missing=dependencies(settings,listener_only=args.listen_only)
    if args.simulate_text is not None:
        active=False
        for text in args.simulate_text:
            intent=router.parse(text,active)
            if intent['kind']=='select':active=True
            elif intent['kind']=='standby':active=False
            print(json.dumps(dict(text=text,intent=intent,hardware_opened=False),ensure_ascii=False))
        return 0
    if args.check:
        # Plans are pure and validate model-specific ownership without devices.
        from vivis.interaction.follow import plan as follow_plan
        from vivis.interaction.standby import plan as standby_plan
        f=read_json(ROOT/'configs/interaction/expression_follow.json');f['tracking_enabled']=True
        follow_plan(robot,f,1.);standby_plan(robot,read_json(ROOT/settings['standby']),1.)
        for p in catalog['programs']:
            if p['kind']=='runtime_a2f':robot.work(p['work'])
        print(json.dumps(dict(offline_valid=True,programs=[p['title'] for p in catalog['programs']],
            missing_dependencies=missing,hardware_opened=False,requires_onsite=['ASR at intended distance','speaker double-talk/AEC','smooth standby motion']),ensure_ascii=False));return 0
    if missing:parser.error('Missing optional P1 assets; run python3 tools/setup_p1.py and setup_p0.py: '+str(missing))
    if not args.listen_only:check_competitors(True)
    output=args.output or ROOT/'outputs/p1'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True,exist_ok=False)
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
    events=queue.Queue(maxsize=64);worker=asr=None;asr_log=asr_thread=None;desired=None;active_program=None
    quitting=False;fault=None;phase='starting';turn=0;started=time.monotonic();prompt=None;log=(output/'events.jsonl').open('w')
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
            phase=phase,active_program=active_program,desired_program=desired,error=fault,output=str(output))
        path=ROOT/'outputs/runtime/program_status.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,ensure_ascii=False)+'\n');temp.replace(path)
    def start_worker(target):
        nonlocal turn,active_program,phase
        turn+=1;folder=output/f'{turn:04d}_{target or "standby"}'
        program=next((p for p in catalog['programs'] if p['id']==target),None) if target else None
        command=program_command(program,robot,folder,speech,settings)
        active_program=target;phase='preparing' if target else 'returning_to_standby'
        record('start',program=target,session=folder.name);print('准备：'+(program['title'] if program else '待机'),flush=True)
        return Worker(command,folder,events)
    gate=EventGate(settings['max_event_age_s'])
    try:
        if not args.text_input:
            asr_log=(output/'asr.log').open('w')
            asr=subprocess.Popen([str(ROOT/speech['python']),'-m','vivis.speech.listen','--config',str(ROOT/settings['speech'])],
                cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONNOUSERSITE':'1'},stdout=subprocess.PIPE,
                stderr=asr_log,text=True,start_new_session=True)
            def read_asr():
                for line in asr.stdout:
                    try:event=json.loads(line)
                    except ValueError:continue
                    try:events.put_nowait(event)
                    except queue.Full:
                        try:events.get_nowait();events.put_nowait(event)
                        except (queue.Empty,queue.Full):pass
            asr_thread=threading.Thread(target=read_asr,daemon=True);asr_thread.start()
        else:
            def read_text():
                for number,line in enumerate(sys.stdin,1):
                    events.put(dict(kind='asr_final',utterance_id=number,text=line.strip(),captured_end_monotonic=time.monotonic()))
                events.put(dict(kind='shutdown'))
            threading.Thread(target=read_text,daemon=True).start()
        prompt=None if args.listen_only else PromptPlayer(robot,speech)
        ready=args.text_input
        print('P1：白骨夫人／表情跟随。语音退出节目后待机；Ctrl+C退出管理器。',flush=True)
        while True:
            now=time.monotonic()
            if args.seconds is not None and now-started>=args.seconds:quitting=True
            if (ROOT/'outputs/STOP').exists():quitting=True;fault='Emergency stop file present'
            if asr is not None and asr.poll() is not None and not quitting:quitting=True;fault='Speech process ended; see asr.log'
            for _ in range(32):
                try:event=events.get_nowait()
                except queue.Empty:break
                if event.get('kind')=='shutdown':quitting=True;continue
                if event.get('kind') in ('asr_error','asr_overload'):
                    fault='Speech input unavailable: '+str(event);quitting=True
                    if worker:worker.cancel()
                    record('speech_fault',detail=event);continue
                if event.get('kind')=='ready':
                    ready=True;record('speech_ready',detail=event)
                    print('麦克风已就绪，可以说指令。'+('本次只识别，不驱动机器人。' if args.listen_only else '请在欢迎语结束后选择节目。'),flush=True)
                    if prompt:prompt.play('welcome')
                    continue
                if event.get('kind')=='worker_ready' and worker and event['pid']==worker.process.pid:
                    phase='active' if active_program else 'standby';record('worker_ready',program=active_program);continue
                if not gate.accept(event,time.monotonic()):continue
                intent=router.parse(event.get('text',''),active_program is not None,current_program=active_program)
                stamp=event.get('speech_end_monotonic') or event['captured_end_monotonic']
                if prompt and prompt.contains(stamp) and intent['kind']!='standby':
                    record('prompt_turn_ignored',text=event['text']);continue
                record('intent',text=event['text'],intent=intent,metrics=event.get('metrics',{}))
                label={'standby':'待机','ignore':'未匹配','prompt':'节目说明'}.get(intent['kind'],intent['kind'])
                if intent['kind']=='select':label='选择'+next(p['title'] for p in catalog['programs'] if p['id']==intent['program'])
                print('识别：'+event['text']+' → '+label,flush=True)
                if args.listen_only or fault:continue
                if intent['kind'] in ('select','standby'):
                    if prompt:prompt.stop()
                    desired=intent.get('program')
                    if worker and (intent['kind']=='select' or active_program is not None):
                        worker.cancel();phase='stopping'
                        record('cancel_requested',program=active_program,reason=event['text'])
                elif intent['kind']=='prompt' and active_program is None and prompt:prompt.play(intent['name'])
            if prompt:prompt.poll()
            if worker is not None:
                code=worker.process.poll()
                if code is not None:
                    cancelled=worker.cancelled_at is not None
                    worker.kill_owned_group();worker.finish();record('worker_exit',program=active_program,code=code,cancelled=cancelled)
                    if code not in (0,130):
                        fault='Program failed; see '+str(worker.folder.with_suffix('.log'));phase='fault'
                        print('节目故障，已停止；请检查日志：'+str(worker.folder.with_suffix('.log')),flush=True)
                    if not cancelled:desired=None
                    worker=None;active_program=None
                elif worker.cancelled_at is not None and now-worker.cancelled_at>settings['cancel_timeout_s']:
                    worker.kill_owned_group();worker.process.wait(timeout=2);worker.finish();worker=None
                    fault='Worker cancellation timed out; hold position, no automatic recovery';phase='fault';quitting=True
            if quitting:
                if prompt:prompt.stop()
                if worker:worker.cancel()
                else:break
            elif not args.listen_only and ready and worker is None and fault is None:worker=start_worker(desired)
            publish();time.sleep(.02)
    except BaseException as exc:
        fault=str(exc);raise
    finally:
        cleanup_errors=[]
        def cleanup(action):
            try:action()
            except Exception as exc:cleanup_errors.append(str(exc))
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
                try:asr.wait(timeout=3)
                except subprocess.TimeoutExpired:asr.kill();asr.wait(timeout=2)
            cleanup(stop_asr)
            if asr_thread:cleanup(lambda:asr_thread.join(timeout=2))
            if asr.stdout:cleanup(asr.stdout.close)
        if asr_log:cleanup(asr_log.close)
        if cleanup_errors:fault=fault or 'Resource cleanup failed: '+'; '.join(cleanup_errors)
        phase='fault' if fault else 'stopped';cleanup(lambda:publish(True));cleanup(lambda:record('closed',error=fault));cleanup(log.close)
        cleanup(lambda:(output/'summary.json').write_text(json.dumps(dict(phase=phase,error=fault,program_sessions=turn,
            cleanup_errors=cleanup_errors,listen_only=args.listen_only,text_input=args.text_input,
            elapsed_seconds=time.monotonic()-started),ensure_ascii=False,indent=2)+'\n'))
        for s,h in handlers.items():signal.signal(s,h)
        lock.close()
    return 1 if fault else 0


if __name__=='__main__':raise SystemExit(main())
