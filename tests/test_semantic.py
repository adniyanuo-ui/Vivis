"""Text-only semantics cannot block local controls or invent hardware actions."""
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock,patch

from vivis.paths import ROOT
from vivis.interaction.semantic import SemanticRouter,SemanticWorker
from vivis.interaction.intents import IntentRouter
from vivis.interaction.session import SessionPolicy


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.catalog=json.loads((ROOT/'configs/interaction/programs.json').read_text())
        self.config=json.loads((ROOT/'configs/interaction/semantic.json').read_text())
        self.router=SemanticRouter(self.config,self.catalog)

    def test_schema_and_validation_allow_only_existing_program_ids_and_no_hardware_operations(self):
        self.assertEqual(self.router.validate(dict(action='select',program='baigufuren',reply='好。'))['intent'],
                         dict(kind='select',program='baigufuren'))
        for result in [dict(action='write_register',program='',reply=''),dict(action='select',program='shell',reply=''),
                       dict(action='chat',program='baigufuren',reply=''),dict(action='select',program=[],reply=''),
                       dict(action='chat',program='',reply='x'*101),dict(action='chat',program='',reply='',command='rm')]:
            with self.assertRaises(ValueError):self.router.validate(result)
        payload=self.router.payload('能不能给我来个魏总的表演？')
        self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
        self.assertIn('不确定',payload['messages'][0]['content'])
        self.assertEqual(payload['messages'][-1]['content'],'能不能给我来个魏总的表演？')

    def test_http_request_is_bounded_and_validated_before_it_can_change_state(self):
        data=dict(choices=[dict(message=dict(content=json.dumps(dict(action='menu',program='',reply='我有三个节目。'))))])
        response=Mock();response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        response.read.return_value=json.dumps(data).encode()
        with patch('urllib.request.urlopen',return_value=response) as request:
            result=self.router.resolve('都有什么好玩的？')
        self.assertEqual(result['intent'],dict(kind='prompt',name='menu'))
        self.assertEqual(request.call_args.kwargs['timeout'],self.config['timeout_s'])
        response.read.return_value=b'x'*65537
        with patch('urllib.request.urlopen',return_value=response),self.assertRaisesRegex(ValueError,'Oversize'):
            self.router.resolve('你好')

    def test_stop_is_immediate_while_semantic_is_stuck_and_late_selection_is_discarded(self):
        entered=threading.Event();release=threading.Event();events=[]
        router=Mock(config=self.config)
        def resolve(text):
            entered.set();release.wait(2.)
            return dict(intent=dict(kind='select',program='baigufuren'),reply='好。')
        router.resolve.side_effect=resolve;worker=SemanticWorker(router,events.append)
        session=SessionPolicy(p['id'] for p in self.catalog['programs']);session.request(dict(kind='wake'))
        try:
            self.assertTrue(worker.offer(dict(utterance_id=1,text='来个好听的'),session.mode))
            self.assertTrue(entered.wait(1.))
            start=time.monotonic();worker.invalidate()
            intent=IntentRouter(self.catalog).parse('待机吧',state=session.mode);session.request(intent)
            self.assertLess(time.monotonic()-start,.05);self.assertEqual(session.mode,'standby')
            self.assertFalse(worker.offer(dict(utterance_id=2,text='新的问题'),'interaction'))
            release.set();worker.job.wait(2.)
            self.assertFalse(worker.accepts(events[0],session.mode))
            self.assertFalse(worker.accepts(events[0],'interaction'))  # Also invalid after another wake.
            self.assertEqual(session.mode,'standby')
        finally:release.set();worker.close()

    def test_semantic_failure_is_a_result_not_a_microphone_fault(self):
        events=[];router=Mock(config=self.config);router.resolve.side_effect=TimeoutError('slow optional model')
        worker=SemanticWorker(router,events.append)
        try:
            self.assertFalse(worker.offer(dict(utterance_id=1,text='唱歌'),'standby'))
            self.assertFalse(worker.offer(dict(utterance_id=1,text='唱歌'),'task'))
            router.resolve.assert_not_called()
            worker.offer(dict(utterance_id=1,text='你好'),'interaction');worker.job.wait(1.)
            self.assertEqual(events[0]['kind'],'semantic_result');self.assertIn('error',events[0])
            self.assertTrue(worker.accepts(events[0],'interaction'))
            events[0]['expires_monotonic']=0.;self.assertFalse(worker.accepts(events[0],'interaction'))
        finally:worker.close()

    def test_manager_processes_sleep_and_new_wake_before_a_late_semantic_answer(self):
        from vivis.interaction.dialogue import main
        actual_popen=subprocess.Popen
        cfg=json.loads((ROOT/'configs/interaction/p1.json').read_text())
        speech=json.loads((ROOT/cfg['speech']).read_text())
        cfg['programs']=str(ROOT/cfg['programs']);cfg['semantic']=str(ROOT/cfg['semantic']);cfg['conversation']=str(ROOT/cfg['conversation'])
        def resolve(_,text):
            time.sleep(.35)
            return dict(intent=dict(kind='select',program='baigufuren'),reply='好。')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'p1.json';config.write_text(json.dumps(cfg))
            script=root/'asr.py';script.write_text('''import json,time
def emit(**event):print(json.dumps(event,ensure_ascii=False),flush=True)
emit(kind='ready');start=time.monotonic()
for i,(delay,text) in enumerate([(.05,'小帅'),(.10,'能不能来点有意思的'),(.15,'待机'),(.20,'小帅')],1):
 time.sleep(max(0.,start+delay-time.monotonic()))
 emit(kind='asr_final',utterance_id=i,text=text,captured_end_monotonic=time.monotonic())
time.sleep(3)
''')
            def launch(command,**kwargs):return actual_popen([sys.executable,str(script)],**kwargs)
            with patch('vivis.interaction.dialogue.ROOT',root), \
                 patch('vivis.interaction.dialogue.dependencies',return_value=(speech,[])), \
                 patch('vivis.interaction.dialogue.subprocess.Popen',side_effect=launch), \
                 patch.object(SemanticRouter,'resolve',resolve),redirect_stdout(io.StringIO()):
                code=main(['--config',str(config),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),
                           '--listen-only','--semantic','on','--seconds','.8','--output',str(root/'run')])
            self.assertEqual(code,0)
            summary=json.loads((root/'run/summary.json').read_text());self.assertEqual(summary['desired_mode'],'interaction')
            events=[json.loads(line) for line in (root/'run/events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['kind']=='semantic_stale' for e in events))
            sleep=next(e for e in events if e['kind']=='intent' and e.get('text')=='待机')
            stale=next(e for e in events if e['kind']=='semantic_stale')
            self.assertLess(sleep['monotonic'],stale['monotonic'])

    def test_unaccepted_local_model_can_propose_but_cannot_start_a_program(self):
        from vivis.interaction.dialogue import main
        actual_popen=subprocess.Popen
        self.assertFalse(self.config['enabled']);self.assertFalse(self.config['allow_program_selection'])
        cfg=json.loads((ROOT/'configs/interaction/p1.json').read_text());speech=json.loads((ROOT/cfg['speech']).read_text())
        for key in ('programs','semantic','conversation'):cfg[key]=str(ROOT/cfg[key])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'p1.json';config.write_text(json.dumps(cfg))
            script=root/'asr.py';script.write_text('''import json,time
print(json.dumps(dict(kind='ready')),flush=True)
for i,text in enumerate(['小帅','能不能来点有意思的'],1):
 time.sleep(.05)
 print(json.dumps(dict(kind='asr_final',utterance_id=i,text=text,captured_end_monotonic=time.monotonic())),flush=True)
time.sleep(3)
''')
            def launch(command,**kwargs):return actual_popen([sys.executable,str(script)],**kwargs)
            with patch('vivis.interaction.dialogue.ROOT',root), \
                 patch('vivis.interaction.dialogue.dependencies',return_value=(speech,[])), \
                 patch('vivis.interaction.dialogue.subprocess.Popen',side_effect=launch), \
                 patch.object(SemanticRouter,'resolve',return_value=dict(intent=dict(kind='select',program='baigufuren'),reply='开演了')), \
                 redirect_stdout(io.StringIO()):
                code=main(['--config',str(config),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),
                           '--listen-only','--semantic','on','--seconds','.4','--output',str(root/'run')])
            self.assertEqual(code,0)
            summary=json.loads((root/'run/summary.json').read_text());self.assertEqual(summary['desired_mode'],'interaction')
            self.assertIn('semantic_selection_blocked',(root/'run/events.jsonl').read_text())


if __name__=='__main__':unittest.main()
