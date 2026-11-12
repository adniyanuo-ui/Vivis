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
    def test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(self,cloud_commands=False,extended=False,repeated=False,qwen=False,recover=False):
        actual_popen=subprocess.Popen;voice_starts=[];prompts=[]
        settings=json.loads((ROOT/'configs/interaction/p1.json').read_text())
        settings['realtime_voice']['control_backend']='qwen_realtime' if qwen else 'local'
        for ref in ('reception_config','memory_config'):settings['realtime_voice'][ref]=str(ROOT/settings['realtime_voice'][ref])
        speech=json.loads((ROOT/settings['speech']).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);serial=root/'serial';speaker=root/'speaker';trace=root/'trace'
            worker=root/'worker.py'
            worker.write_text('''import sys,time
from pathlib import Path
folder,serial,speaker,trace,name=map(Path,sys.argv[1:]);folder.mkdir()
assert not speaker.exists(),'Cloud still owns speaker'
serial.open('x').close()
with trace.open('a') as f:f.write('start '+str(name)+'\\n')
print('STANDBY START' if str(name)=='standby' else 'INTERACTION START' if str(name)=='interaction' else 'ROUND 1/1 START',flush=True)
start=time.monotonic()
while not folder.with_suffix('.cancel').exists() and not(str(name)=='baigufuren' and time.monotonic()-start>.2):time.sleep(.005)
serial.unlink()
with trace.open('a') as f:f.write('end '+str(name)+'\\n')
''')
            listener=root/'voice.py'
            listener.write_text('''import sys,time,json,threading
from pathlib import Path
speaker=Path(sys.argv[1]);lock=threading.Lock();epoch=0;revision=0
def emit(**e):
 with lock:print(json.dumps(e,ensure_ascii=False),flush=True)
def commands():
 global epoch,revision
 for line in sys.stdin:
  c=json.loads(line)
  if c['kind']=='voice_context':revision=c['revision'];continue
  if c['kind']!='voice_mode':continue
  if c['enabled']:
   with speaker.with_suffix('.contexts').open('a') as f:f.write(json.dumps(c,ensure_ascii=False)+'\\n')
  epoch=c['epoch']
  time.sleep(.04)
  if c['enabled']:
   speaker.open('x').close();emit(kind='voice_state',epoch=c['epoch'],status='ready',speaker_owned=True)
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
if sys.argv[3]=='True':turns=[(.2,'小帅'),(.8,'呃播放剧本'),(1.2,'停止'),(1.8,'模仿我的表情'),(2.2,'待机'),(2.7,'小帅'),(3.2,'播放歌曲'),(4.1,'待机')]
if sys.argv[4]=='True':turns=[(.2,'小帅'),(.8,'播放宇飞剧本'),(1.4,'停止'),(2.1,'再演一遍'),(2.8,'退出'),(3.4,'待机')]
if sys.argv[6]=='True':turns=[(.2,'小帅'),(.8,'断网'),(1.1,'播放白骨夫人'),(1.4,'恢复'),(2.,'小帅'),(2.6,'待机')]
for n,(deadline,text) in enumerate(turns,1):
 time.sleep(max(0,start+deadline-time.monotonic()))
 if text in ('断网','恢复'):
  emit(kind='voice_health',available=text=='恢复',reason='synthetic network');continue
 if sys.argv[5]=='True':
  actions={'小帅':('wake',''),'播放白骨夫人':('play','baigufuren'),'自己玩':('standby',''),'停止':('stop',''),'待机':('standby','')}
  if text in actions:
   action,program=actions[text];emit(kind='cloud_action',revision=revision,arguments=dict(action=action,program=program),text=text)
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
                return [sys.executable,str(worker),str(folder),str(serial),str(speaker),str(trace),program['id'] if program else 'standby']
            def popen(command,**kwargs):
                if 'vivis.speech.managed' in command:
                    voice_starts.append(command)
                    return actual_popen([sys.executable,str(listener),str(speaker),str(cloud_commands),str(extended),str(repeated),str(qwen),str(recover)],**kwargs)
                return actual_popen(command,**kwargs)
            with patch('vivis.interaction.dialogue.ROOT',root), \
                 patch('vivis.interaction.dialogue.dependencies',return_value=(speech,[])), \
                 patch('vivis.interaction.dialogue.check_competitors'), \
                 patch('vivis.interaction.dialogue.preview_state',return_value=None), \
                 patch('vivis.interaction.dialogue.check_serial_access'), \
                 patch('vivis.interaction.dialogue.PromptPlayer',Prompt), \
                 patch('vivis.interaction.dialogue.program_command',side_effect=program), \
                 patch('vivis.interaction.dialogue.subprocess.Popen',side_effect=popen), \
                 patch.dict('os.environ',{'DASHSCOPE_API_KEY':'unit-test-placeholder'}),redirect_stdout(io.StringIO()) as console:
                self.assertEqual(main(['--config',str(config),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),
                    '--seconds','4.6' if extended else '3.8','--output',str(root/'run')]),0)
            self.assertEqual(len(voice_starts),1)
            lines=trace.read_text().splitlines()
            expected=(['standby','interaction','wz2','interaction','wz2','interaction','standby'] if repeated else
                      ['standby','interaction','wz2','interaction','expression-follow','standby','interaction','baigufuren','interaction','standby']
                      if extended else ['standby','interaction','baigufuren','interaction','standby','interaction','standby'])
            if recover:expected=['standby','interaction','standby','interaction','standby']
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
            self.assertNotIn('identity',prompts)  #Natural cloud answer, no duplicate cached voice.
            effective=(root/'run/effective.json').read_text();self.assertNotIn('unit-test-placeholder',effective)
            if repeated:
                contexts=[json.loads(s) for s in speaker.with_suffix('.contexts').read_text().splitlines()]
                results=[c['last_program'] for c in contexts if 'last_program' in c]
                self.assertEqual(results,[dict(id='wz2',title='宇飞剧本',outcome='interrupted')]*2)

    def test_network_outage_holds_motion_and_requires_fresh_request_after_recovery(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True,recover=True)

    def test_qwen_tools_alone_execute_modes_and_cached_prompts_are_disabled(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(qwen=True)

    def test_cloud_user_asr_fallback_executes_real_modes_but_assistant_cannot(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(cloud_commands=True)

    def test_script_expression_song_completion_and_direct_standby_release_resources(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(extended=True)

    def test_repeat_script_after_exit_starts_second_worker_and_keeps_actual_context(self):
        self.test_state_transitions_keep_one_capture_and_release_cloud_before_other_audio(repeated=True)


if __name__=='__main__':unittest.main()
