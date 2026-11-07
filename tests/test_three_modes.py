"""Wake-gated standby, interactive waiting and task exits with one live listener."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from vivis.paths import ROOT
from vivis.interaction.intents import IntentRouter,resolve_candidates
from vivis.interaction.session import SessionPolicy,STANDBY,INTERACTION
from vivis.interaction.dialogue import main,program_command
from vivis.runtime.robot import Robot
from vivis.motion.mapping import read_json


def catalog():return read_json(ROOT/'configs/interaction/programs.json')


class ThreeModePolicyTests(unittest.TestCase):
    def setUp(self):
        self.router=IntentRouter(catalog())
        self.session=SessionPolicy(p['id'] for p in catalog()['programs'])

    def say(self,text):
        s=self.session
        intent=self.router.parse(text,s.mode=='task',current_program=s.desired if s.mode=='task' else None,state=s.mode)
        s.request(intent);return intent

    def test_asleep_only_wake_can_activate_even_with_program_names_or_questions(self):
        for text in ['播放白骨夫人','你是谁','你能帮我做什么','停止','不用播了','他说小帅了','小帅是什么意思','不要唤醒小帅']:
            self.assertEqual(self.say(text)['kind'],'ignore',text);self.assertEqual(self.session.mode,STANDBY)
        self.say('小帅呀');self.assertEqual(self.session.mode,INTERACTION)
        self.assertEqual(self.say('你能帮我做什么'),dict(kind='prompt',name='menu'))

    def test_exit_returns_interaction_and_sleep_goes_directly_from_both_awake_states(self):
        for suffix in ('','吧','呀','哈','哟','吧呀哈哟'):
            self.say('小帅');self.say('播放白骨夫人')
            self.assertEqual(self.session.mode,'task')
            self.say('不用播了'+suffix);self.assertEqual(self.session.mode,INTERACTION)
            self.say('自己玩去'+suffix);self.assertEqual(self.session.mode,STANDBY)
            self.say('小帅');self.say('开始表情跟随')
            self.say('待机'+suffix);self.assertEqual(self.session.mode,STANDBY)

    def test_wake_and_followup_work_in_one_utterance_without_waking_on_quoted_or_negative_content(self):
        self.say('小帅，你是谁？');self.assertEqual(self.session.mode,INTERACTION)
        self.assertEqual(self.session.pending_prompt,'identity')
        self.say('待机');self.say('小帅，播放白骨夫人')
        self.assertEqual(self.session.desired,'baigufuren');self.assertIsNone(self.session.pending_prompt)
        self.say('不要停止');self.assertEqual(self.session.mode,'task')
        self.say('自己玩');self.say('小帅不要播放白骨夫人');self.assertEqual(self.session.mode,STANDBY)

    def test_normal_completion_returns_interaction_and_cancelled_old_child_cannot_override_newer_sleep(self):
        self.say('小帅');self.say('播放白骨夫人');self.session.completed('baigufuren')
        self.assertEqual(self.session.mode,INTERACTION)
        self.say('开始表情跟随');self.say('待机')
        self.session.completed('expression-follow',cancelled=True);self.assertEqual(self.session.mode,STANDBY)
        self.session.completed(STANDBY);self.assertEqual(self.session.mode,STANDBY)
        self.say('小帅');self.session.completed(INTERACTION);self.assertEqual(self.session.mode,INTERACTION)

    def test_prompts_wait_for_current_ready_child_and_sleep_never_starts_welcome(self):
        self.assertIsNone(self.session.take_prompt(STANDBY,True))
        self.say('小帅');self.assertIsNone(self.session.take_prompt(INTERACTION,False))
        self.assertEqual(self.session.take_prompt(INTERACTION,True),'welcome')
        self.assertIsNone(self.session.take_prompt(INTERACTION,True))
        self.say('你是谁');self.say('自己玩去吧')
        self.assertIsNone(self.session.take_prompt(INTERACTION,True))
        self.assertEqual(self.session.take_prompt(STANDBY,True),'standby')

    def test_wake_words_are_configurable_and_complete_asr_hypotheses_preserve_them(self):
        other=IntentRouter(catalog(),wake_words=['墨甲'])
        self.assertEqual(other.parse('小帅',state=STANDBY)['kind'],'ignore')
        self.assertEqual(other.parse('墨甲，开始表情跟随',state=STANDBY)['kind'],'wake')
        text,_=resolve_candidates(self.router,'你是谁','小帅你是谁')
        self.assertEqual(self.router.parse(text,state=STANDBY)['kind'],'wake')
        text,_=resolve_candidates(self.router,'不要停止','停止')
        self.assertEqual(self.router.parse(text)['kind'],'ignore')

    def test_child_commands_use_camera_free_standby_and_separate_interaction_configuration(self):
        settings=read_json(ROOT/'configs/interaction/p1.json');speech=read_json(ROOT/settings['speech'])
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        sleep=program_command(None,robot,Path('/tmp/one'),speech,settings)
        awake=program_command(dict(kind='interaction'),robot,Path('/tmp/two'),speech,settings)
        for command,mode in [(sleep,STANDBY),(awake,INTERACTION)]:
            cfg=read_json(Path(command[command.index('--config')+1]))
            self.assertEqual(cfg['behavior'],mode)
            self.assertEqual(cfg['tracking_enabled'],mode==INTERACTION)
            self.assertEqual(cfg['social_response']['enabled'],mode==INTERACTION)
        self.assertEqual(read_json(ROOT/settings['interaction'])['social_response']['strength'],.5)


class ThreeModeSupervisorTests(unittest.TestCase):
    def test_external_camera_owner_is_not_silently_reused_by_managed_sessions(self):
        from vivis.runtime.developer import vision_service
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'tracking.json';config.write_text(json.dumps(robot.tracking()))
            with patch.dict('os.environ',{'VIVIS_MANAGED_SESSION':'1'}), \
                 patch('vivis.runtime.developer.preview_state',return_value={'external':True}), \
                 patch('vivis.runtime.developer.subprocess.Popen') as child:
                with self.assertRaisesRegex(RuntimeError,'独立持有相机'):
                    with vision_service(config,root,True):self.fail('External camera reused')
                child.assert_not_called()

    def test_existing_external_camera_blocks_manager_before_microphone_or_movement(self):
        speech=read_json(ROOT/'configs/perception/speech.json')
        with patch('vivis.interaction.dialogue.dependencies',return_value=(speech,[])), \
             patch('vivis.interaction.dialogue.check_competitors'), \
             patch('vivis.interaction.dialogue.preview_state',return_value={'external':True}), \
             patch('vivis.interaction.dialogue.subprocess.Popen') as child:
            with self.assertRaisesRegex(RuntimeError,'已有独立视觉服务'):main(['--text-input'])
            child.assert_not_called()

    def test_microphone_process_is_not_restarted_across_wake_completion_exit_and_sleep(self):
        actual_popen=subprocess.Popen;launches=[];prompts=[]
        class Prompts:
            def __init__(self,*_):pass
            def play(self,name):prompts.append(name)
            def stop(self):pass
            def poll(self):pass
            def contains(self,*_):return False
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'worker.py';flag=root/'active';trace=root/'trace'
            script.write_text('''import sys,time
from pathlib import Path
folder,flag,trace,name=map(Path,sys.argv[1:]);folder.mkdir();cancel=folder.with_suffix('.cancel')
if cancel.exists():sys.exit(130)
flag.open('x').close()
with trace.open('a') as f:f.write('start '+str(name)+'\\n')
print('STANDBY START' if str(name)=='standby' else 'INTERACTION START' if str(name)=='interaction' else 'EXPRESSION FOLLOW START',flush=True)
start=time.monotonic()
while not cancel.exists() and not (str(name)=='baigufuren' and time.monotonic()-start>.25):time.sleep(.005)
time.sleep(.02);flag.unlink()
with trace.open('a') as f:f.write('end '+str(name)+'\\n')
sys.exit(130 if cancel.exists() else 0)
''')
            asr=root/'asr.py';asr.write_text('''import json,time
def emit(**event):print(json.dumps(event,ensure_ascii=False),flush=True)
emit(kind='ready');start=time.monotonic()
for i,(deadline,text) in enumerate([(0.20,'播放白骨夫人'),(.4,'小帅'),(.7,'你是谁'),(1.0,'播放白骨夫人'),(1.7,'自己玩去吧'),(2.0,'小帅呀'),(2.3,'开始表情跟随'),(2.6,'不用学了'),(2.9,'开始表情跟随'),(3.2,'待机哟'),(3.5,'你是谁')],1):
 time.sleep(max(0.,start+deadline-time.monotonic()))
 emit(kind='asr_final',utterance_id=i,text=text,captured_end_monotonic=time.monotonic())
time.sleep(10)
''')
            settings=read_json(ROOT/'configs/interaction/p1.json');settings['programs']=str(ROOT/settings['programs'])
            config=root/'p1.json';config.write_text(json.dumps(settings))
            def command(program,robot,folder,speech,settings):
                return [sys.executable,str(script),str(folder),str(flag),str(trace),program['id'] if program else STANDBY]
            def launch(command,**kwargs):
                if 'vivis.speech.listen' in command:
                    launches.append(command)
                    return actual_popen([sys.executable,str(asr)],**kwargs)
                return actual_popen(command,**kwargs)
            with patch('vivis.interaction.dialogue.ROOT',root), \
                 patch('vivis.interaction.dialogue.dependencies',return_value=(read_json(ROOT/settings['speech']),[])), \
                 patch('vivis.interaction.dialogue.check_competitors'),patch('vivis.interaction.dialogue.check_serial_access'), \
                 patch('vivis.interaction.dialogue.PromptPlayer',Prompts), \
                 patch('vivis.interaction.dialogue.program_command',side_effect=command), \
                 patch('vivis.interaction.dialogue.subprocess.Popen',side_effect=launch),redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config',str(config),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),
                    '--seconds','3.9','--output',str(root/'run')]),0)
            self.assertEqual(len(launches),1)
            lines=trace.read_text().splitlines()
            for a,b in zip(lines[::2],lines[1::2]):self.assertEqual(a.replace('start ','end '),b)
            self.assertEqual([x[6:] for x in lines[::2]],
                [STANDBY,INTERACTION,'baigufuren',INTERACTION,STANDBY,INTERACTION,'expression-follow',INTERACTION,'expression-follow',STANDBY])
            self.assertIn('welcome',prompts);self.assertIn('identity',prompts)
            self.assertEqual(prompts[0],'welcome')
            speech=read_json(root/'run/speech_effective.json');self.assertEqual(speech['wake_words'],['小帅'])
            self.assertFalse(flag.exists())
