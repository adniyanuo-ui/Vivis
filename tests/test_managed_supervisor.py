"""Exercise the real supervisor with fake voice/robot subprocesses and no devices."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from vivis.paths import ROOT
from vivis.interaction.dialogue import main


class ManagedSupervisorTests(unittest.TestCase):
    def test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(self,cloud_commands=False,extended=False,repeated=False,qwen=False,recover=False,farewell=False,early=False,slow_initial=False,faulted=False,slow_return=False,preparations=False,fail_program=False,wake_before_notice=False,notice_failed=False):
        actual_popen=subprocess.Popen;voice_starts=[];prompts=[]
        settings=json.loads((ROOT/'configs/interaction/p1.json').read_text())
        settings['realtime_voice']['control_backend']='qwen_realtime' if qwen else 'local'
        for ref in ('reception_config','memory_config'):settings['realtime_voice'][ref]=str(ROOT/settings['realtime_voice'][ref])
        speech=json.loads((ROOT/settings['speech']).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);serial=root/'serial';speaker=root/'speaker';trace=root/'trace'
            worker=root/'worker.py'
            worker.write_text('''import sys,time,os,json
from pathlib import Path
folder,serial,speaker,trace,name=map(Path,sys.argv[1:6]);folder.mkdir()
early=sys.argv[6]=='True'
slow_initial=sys.argv[7]=='True'
faulted=sys.argv[8]=='True'
slow_return=sys.argv[9]=='True'
fail_program=sys.argv[10]=='True'
preparations=sys.argv[11]=='True'
assert str(name)=='standby' or ((early or preparations) and str(name)=='interaction') or not speaker.exists(),'Cloud still owns speaker'
serial.open('x').close()
action=os.environ.get('VIVIS_MOTION_ACTION')
with trace.with_suffix('.actions').open('a') as f:f.write(json.dumps(dict(name=str(name),action=action))+'\\n')
if slow_initial and action=='startup':
 time.sleep(.5)
 assert not folder.with_suffix('.cancel').exists(),'Wake bypassed mandatory boot initialization'
with trace.open('a') as f:f.write('start '+str(name)+'\\n')
with speaker.with_suffix('.timeline').open('a') as f:f.write('worker '+str(name)+'\\n')
if fail_program and str(name)=='baigufuren':
 serial.unlink()
 with trace.open('a') as f:f.write('end '+str(name)+'\\n')
 raise RuntimeError('Official SDK ID 2: read timeout')
if early and str(name)=='interaction':time.sleep(.35)
if slow_return and action=='standby' and folder.name.startswith('0005_'):
 time.sleep(.8)
 assert not folder.with_suffix('.cancel').exists(),'Wake cancelled standby homing'
 assert speaker.exists(),'Wake speech still waited for standby homing'
 with trace.with_suffix('.return').open('w') as f:f.write('home completed after speech enabled')
if faulted and folder.name.startswith('0005_'):
 print('STANDBY FAULT injected standby fault',flush=True)
 time.sleep(1.25)
 print('STANDBY RECOVERED injected recovery',flush=True)
print('STANDBY START' if str(name)=='standby' else 'INTERACTION START' if str(name)=='interaction' else 'ROUND 1/1 START',flush=True)
start=time.monotonic()
while not folder.with_suffix('.cancel').exists() and not(str(name)=='baigufuren' and time.monotonic()-start>.2):time.sleep(.005)
serial.unlink()
with trace.open('a') as f:f.write('end '+str(name)+'\\n')
with speaker.with_suffix('.timeline').open('a') as f:f.write('end '+str(name)+'\\n')
''')
            listener=root/'voice.py'
            listener.write_text('''import sys,time,json,threading
from pathlib import Path
speaker=Path(sys.argv[1]);lock=threading.Lock();epoch=0;revision=0;reply=None;motion_ready=False;replies=set();notices={}
def emit(**e):
 with lock:print(json.dumps(e,ensure_ascii=False),flush=True)
def finish_reply():
 if reply and reply not in replies and speaker.exists():
  replies.add(reply)
  reply_id,observed_ready,observed_epoch=reply,motion_ready,epoch
  def begin():
   with speaker.with_suffix('.timeline').open('a') as f:f.write('audible '+reply_id+'\\n')
   emit(kind='voice_reply_started',epoch=observed_epoch,reply_id=reply_id,dac_monotonic=time.monotonic())
  def complete():
   with speaker.with_suffix('.farewells').open('a') as f:f.write(json.dumps(dict(reply=reply_id,motion_ready=observed_ready))+'\\n')
   with speaker.with_suffix('.timeline').open('a') as f:f.write('reply '+reply_id+'\\n')
   emit(kind='cloud_transcript',epoch=observed_epoch,role='assistant',text=notices.get(reply_id,'好，您忙。'))
   emit(kind='voice_reply_done',epoch=observed_epoch,reply_id=reply_id,reason='speech_failed' if sys.argv[11]=='True' else 'played')
  if sys.argv[9]=='True':
   if sys.argv[11]!='True':threading.Timer(.35 if sys.argv[10]=='True' else .06,begin).start()
   threading.Timer(.45 if sys.argv[10]=='True' else .12,complete).start()
  else:begin();complete()
def commands():
 global epoch,revision,reply,motion_ready
 for line in sys.stdin:
  c=json.loads(line)
  if c['kind']=='voice_execution_notice':
   notices[c['reply_id']]=c['text']
   with speaker.with_suffix('.failures').open('a') as f:f.write(c['text']+'\\n')
   continue
  if c['kind']=='voice_context':
   revision=c['revision'];reply=c.get('transition_reply_id');motion_ready=c['ready'];finish_reply();continue
  if c['kind']!='voice_mode':continue
  if c['enabled']:
   c['motion_ready']=motion_ready
   with speaker.with_suffix('.contexts').open('a') as f:f.write(json.dumps(c,ensure_ascii=False)+'\\n')
  epoch=c['epoch']
  time.sleep(.04)
  if c['enabled']:
   speaker.open('x').close();emit(kind='voice_state',epoch=c['epoch'],status='ready',speaker_owned=True)
   finish_reply()
   emit(kind='cloud_transcript',epoch=c['epoch'],role='assistant',text='播放白骨夫人')
   if sys.argv[5]=='True' and c['epoch']==1:
    emit(kind='voice_warning',epoch=c['epoch'],message='本句缓冲超限，交互继续')
    emit(kind='cloud_transcript',epoch=c['epoch']-1,role='user',text='过期识别')
    emit(kind='cloud_transcript',epoch=c['epoch'],role='user',text='   ')
    emit(kind='cloud_transcript',epoch=c['epoch'],role='user',text='停止',identity_label='小王-other')
   if c['epoch']==1:emit(kind='asr_control',control_sequence=1,text='结束',captured_end_monotonic=time.monotonic())
  else:
   speaker.unlink(missing_ok=True);emit(kind='voice_state',epoch=c['epoch'],status='off',speaker_owned=False)
threading.Thread(target=commands,daemon=True).start()
emit(kind='ready');start=time.monotonic()
turns=[(.2,'小帅'),(.8,'你是谁'),(1.2,'播放白骨夫人'),(2.,'自己玩'),(2.5,'小帅'),(3.,'停止'),(3.3,'待机')]
if sys.argv[3]=='True':turns=[(.2,'小帅'),(.8,'呃播放剧本'),(1.2,'停止'),(1.8,'模仿我的表情'),(2.2,'待机'),(2.7,'小帅'),(3.2,'播放白骨夫人'),(4.1,'待机')]
if sys.argv[4]=='True':turns=[(.2,'小帅'),(.8,'播放宇飞剧本'),(1.4,'停止'),(2.1,'再演一遍'),(2.8,'退出'),(3.4,'待机')]
if sys.argv[6]=='True':turns=[(.2,'小帅'),(.8,'断网'),(1.1,'播放白骨夫人'),(1.4,'恢复'),(2.,'小帅'),(2.6,'待机')]
if sys.argv[10]=='True':turns=[(.2,'小帅'),(.8,'待机'),(1.,'小帅')]
for n,(deadline,text) in enumerate(turns,1):
 time.sleep(max(0,start+deadline-time.monotonic()))
 if text in ('断网','恢复'):
  emit(kind='voice_health',available=text=='恢复',reason='synthetic network');continue
 if sys.argv[5]=='True':
  actions={'小帅':('wake',''),'播放白骨夫人':('play','baigufuren'),'呃播放剧本':('play','wz2'),
   '模仿我的表情':('play','expression-follow'),'自己玩':('standby',''),'停止':('stop',''),'待机':('standby','')}
  if text in actions:
   action,program=actions[text];emit(kind='cloud_action',revision=revision,arguments=dict(action=action,program=program,
    **({'explicit_wake':True} if sys.argv[8]=='True' and action=='wake' else {})),text=text,
    **({'transition_reply_id':'reply-'+str(n)} if action!='wake' and (sys.argv[9]=='True' or action=='standby' and sys.argv[7]=='True') else {}))
  else:emit(kind='asr_final',utterance_id=n,text=text,captured_end_monotonic=time.monotonic())
  continue
 if sys.argv[3]=='True' and text=='停止':
  emit(kind='asr_final',utterance_id=n,text='实时识别没有听对',captured_end_monotonic=time.monotonic())
  emit(kind='asr_control',control_sequence=99,text='退出',captured_end_monotonic=time.monotonic())
  continue
 if sys.argv[2]=='True' and text in ('播放白骨夫人','自己玩'):
  emit(kind='asr_final',utterance_id=n,text='本地识别错误',captured_end_monotonic=time.monotonic())
  emit(kind='cloud_transcript',epoch=epoch,role='user',text='排骨夫人' if text=='播放白骨夫人' else text)
 else:emit(kind='asr_final',utterance_id=n,text=text,captured_end_monotonic=time.monotonic())
time.sleep(10)
''')
            for name in ('programs','semantic','conversation'):
                settings[name]=str(ROOT/settings[name])
            settings['realtime_voice']['config']=str(ROOT/settings['realtime_voice']['config'])
            config=root/'p1.json';config.write_text(json.dumps(settings))
            class Prompt:
                def __init__(self,*_):self.clock=None
                def play(self,name):
                    if speaker.exists():raise AssertionError('Prompt competed with cloud speaker')
                    prompts.append(name)
                def stop(self):pass
                def poll(self):pass
                def contains(self,*_):return False
            def program(program,robot,folder,*_):
                return [sys.executable,str(worker),str(folder),str(serial),str(speaker),str(trace),program['id'] if program else 'standby',str(early),str(slow_initial),str(faulted),str(slow_return),str(fail_program),str(preparations)]
            def popen(command,**kwargs):
                if 'vivis.speech.managed' in command:
                    voice_starts.append(command)
                    return actual_popen([sys.executable,str(listener),str(speaker),str(cloud_commands),str(extended),str(repeated),str(qwen),str(recover),str(farewell),str(early),str(preparations),str(wake_before_notice),str(notice_failed)],**kwargs)
                return actual_popen(command,**kwargs)
            with patch('vivis.interaction.dialogue.ROOT',root), \
                 patch('vivis.interaction.dialogue.dependencies',return_value=(speech,[])), \
                 patch('vivis.interaction.dialogue.check_competitors'), \
                 patch('vivis.interaction.dialogue.preview_state',return_value=None), \
                 patch('vivis.interaction.dialogue.check_serial_access'), \
                 patch('vivis.interaction.dialogue.PromptPlayer',Prompt), \
                 patch('vivis.interaction.dialogue.PersistentVision') as camera_owner, \
                 patch('vivis.interaction.dialogue.program_command',side_effect=program), \
                 patch('vivis.interaction.dialogue.subprocess.Popen',side_effect=popen), \
                 patch.dict('os.environ',{'DASHSCOPE_API_KEY':'unit-test-placeholder'}),redirect_stdout(io.StringIO()) as console:
                camera_owner.return_value.environment.return_value={}
                camera_owner.return_value.poll.return_value=None
                self.assertEqual(main(['--config',str(config),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),
                    '--seconds','4.6' if extended else '3.8','--output',str(root/'run')]),1 if fail_program else 0,console.getvalue())
            self.assertEqual(len(voice_starts),1)
            self.assertEqual(camera_owner.call_count,1 if qwen else 0)
            if qwen:camera_owner.return_value.close.assert_called_once()
            lines=trace.read_text().splitlines()
            expected=(['standby','interaction','wz2','interaction','wz2','interaction','standby'] if repeated else
                      ['standby','interaction','wz2','interaction','expression-follow','standby','interaction','baigufuren','interaction','standby']
                      if extended else ['standby','interaction','baigufuren','interaction','standby','interaction','standby'])
            if recover:expected=['standby','interaction','standby','interaction','standby']
            if faulted:expected=['standby','interaction','baigufuren','interaction','standby']
            if fail_program:expected=['standby','interaction','baigufuren']
            if wake_before_notice:expected=['standby','interaction','standby','interaction']
            self.assertEqual([s[6:] for s in lines if s.startswith('start ')],expected)
            if recover:
                self.assertIn('管理器继续运行',console.getvalue())
                self.assertIn('语音链路已恢复',console.getvalue())
            for a,b in zip(lines[::2],lines[1::2]):self.assertEqual(a.replace('start ','end '),b)
            if qwen:
                self.assertEqual(prompts,[])
                self.assertIn('语音提示：本句缓冲超限，交互继续',console.getvalue())
                self.assertEqual([line for line in console.getvalue().splitlines() if line.startswith('小王-other（识别）：')],
                    ['小王-other（识别）：停止'])
            if farewell:
                self.assertIn('小帅：好，您忙。',console.getvalue())
                records=[json.loads(s) for s in (root/'run/events.jsonl').read_text().splitlines()]
                done=[s for s in records if s['kind']=='voice_reply_done']
                self.assertEqual(len(done),2)
                replies=[json.loads(s) for s in speaker.with_suffix('.farewells').read_text().splitlines()]
                self.assertTrue(any(not r['motion_ready'] for r in replies))
            if faulted:
                self.assertIn('STANDBY RECOVERED injected recovery',console.getvalue())
                records=[json.loads(s) for s in (root/'run/events.jsonl').read_text().splitlines()]
                self.assertTrue(any(s['kind']=='standby_fault' for s in records))
                self.assertTrue(any(s['kind']=='cloud_action_blocked_motion_fault' for s in records))
            if slow_return:self.assertTrue(trace.with_suffix('.return').exists())
            self.assertNotIn('identity',prompts)  #Natural cloud answer, no duplicate cached voice.
            actions=[json.loads(s) for s in trace.with_suffix('.actions').read_text().splitlines()]
            self.assertEqual([a['action'] for a in actions].count('startup'),1)
            self.assertTrue(all(a['action']=='resume' for a in actions if a['name'] not in ('standby',)))
            if early:
                contexts=[json.loads(s) for s in speaker.with_suffix('.contexts').read_text().splitlines()]
                self.assertTrue(any(not c['motion_ready'] for c in contexts))
            if preparations and not fail_program and not wake_before_notice:
                timeline=speaker.with_suffix('.timeline').read_text().splitlines()
                acknowledgement='reply reply-7' if extended else 'reply reply-3'
                self.assertLess(timeline.index(acknowledgement),timeline.index('worker baigufuren'))
                if extended:
                    self.assertLess(timeline.index('reply reply-2'),timeline.index('worker wz2'))
                    self.assertLess(timeline.index('end wz2'),timeline.index('reply reply-3'))
                    self.assertLess(timeline.index('reply reply-4'),timeline.index('worker expression-follow'))
                rest_reply='reply-5' if extended else 'reply-4'
                rest_start=timeline.index(('reply ' if notice_failed else 'audible ')+rest_reply)
                home_start=timeline.index('worker standby',timeline.index('end interaction')+1)
                self.assertLess(rest_start,home_start)
            if fail_program:
                self.assertIn('通信读取超时',speaker.with_suffix('.failures').read_text())
                self.assertIn('SERIAL_TIMEOUT',console.getvalue())
                self.assertIn('小帅：任务失败',console.getvalue())
            effective=(root/'run/effective.json').read_text();self.assertNotIn('unit-test-placeholder',effective)
            if repeated:
                contexts=[json.loads(s) for s in speaker.with_suffix('.contexts').read_text().splitlines()]
                results=[c['last_program'] for c in contexts if 'last_program' in c]
                self.assertEqual(results,[dict(id='wz2',title='宇飞剧本',outcome='interrupted')]*2)

    def test_verified_voice_wake_enables_speaker_during_interaction_preparation(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,early=True)

    def test_wake_during_cold_boot_waits_for_mandatory_startup(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,slow_initial=True)

    def test_wake_can_speak_while_return_home_finishes_without_cancelling_motion(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,early=True,slow_return=True)

    def test_standby_fault_keeps_manager_alive_and_rejects_wake_or_program(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,faulted=True)

    def test_network_outage_holds_motion_and_requires_fresh_request_after_recovery(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,recover=True)

    def test_farewell_owns_speaker_once_in_ready_standby_then_releases(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,farewell=True)

    def test_farewell_precedes_slow_home_and_new_wake_does_not_cancel_home(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,farewell=True,early=True,slow_return=True)

    def test_qwen_can_start_expression_follow_as_a_program_not_conversation(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,extended=True)

    def test_preparation_reply_finishes_before_song_acquires_speaker(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,preparations=True)

    def test_each_program_has_preparation_reply_and_stop_remains_immediate(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,preparations=True,extended=True)

    def test_failed_program_reports_voice_reason_then_exits_without_replay(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,preparations=True,fail_program=True)

    def test_wake_before_farewell_audio_still_completes_required_standby_home(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,preparations=True,wake_before_notice=True)

    def test_failed_farewell_does_not_block_required_standby_home(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,preparations=True,notice_failed=True)

    def test_qwen_tools_alone_execute_modes_and_cached_prompts_are_disabled(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True)

    def test_cloud_user_asr_fallback_executes_real_modes_but_assistant_cannot(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(cloud_commands=True)

    def test_script_expression_song_completion_and_direct_standby_release_resources(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(extended=True)

    def test_repeat_script_after_exit_starts_second_worker_and_keeps_actual_context(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(repeated=True)


if __name__=='__main__':unittest.main()
