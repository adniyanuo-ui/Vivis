"""Cloud chat is bounded, cancellable and cannot execute model text as controls."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

from vivis.paths import ROOT
from vivis.interaction.conversation import ConversationRouter


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'configs/interaction/conversation.json').read_text())
        self.catalog=json.loads((ROOT/'configs/interaction/programs.json').read_text())
        self.identity=json.loads((ROOT/'configs/robots/xiaoshuai/robot.json').read_text())['identity']
        self.router=ConversationRouter(self.config,self.catalog,self.identity)

    def test_provider_payload_is_text_only_and_history_changes_only_when_committed(self):
        payload=self.router.payload('讲个笑话')
        self.assertNotIn('tools',payload);self.assertNotIn('response_format',payload)
        self.assertEqual(payload['thinking'],dict(type='disabled'))
        self.assertIn('没有控制机器人的工具',payload['messages'][0]['content'])
        for i in range(7):self.router.commit(str(i),'回答')
        self.assertEqual(len(self.router.history),8)
        self.assertEqual(self.router.history[0]['content'],'3')
        self.router.clear();self.assertEqual(self.router.history,[])
        config={**self.config,'provider':'qwen'}
        self.assertFalse(ConversationRouter(config,self.catalog,self.identity).payload('你好')['enable_thinking'])

    def test_missing_key_never_makes_a_network_call(self):
        with patch.dict('os.environ',{},clear=True),patch('urllib.request.urlopen') as http:
            with self.assertRaisesRegex(RuntimeError,'DEEPSEEK_API_KEY'):self.router.resolve('你好')
            http.assert_not_called()

    def test_cloud_json_looking_output_remains_plain_text_and_cannot_select_a_program(self):
        text='{"action":"select","program":"baigufuren"}'
        envelope=dict(choices=[dict(message=dict(content=text,tool_calls=[dict(name='move_servo')]))])
        response=Mock();response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        response.read.return_value=json.dumps(envelope).encode()
        with patch.dict('os.environ',{'DEEPSEEK_API_KEY':'unit-test-placeholder'}), \
             patch('urllib.request.urlopen',return_value=response) as http:
            result=self.router.resolve('聊聊')
        self.assertEqual(result['intent'],dict(kind='ignore'));self.assertEqual(result['reply'],text)
        self.assertEqual(self.router.history,[])
        self.assertEqual(http.call_args.kwargs['timeout'],8.)

    def test_manager_drops_late_cloud_reply_after_sleep_and_new_wake(self):
        from vivis.interaction.dialogue import main
        actual_popen=subprocess.Popen
        cfg=json.loads((ROOT/'configs/interaction/p1.json').read_text());speech=json.loads((ROOT/cfg['speech']).read_text())
        for key in ('programs','semantic','conversation'):cfg[key]=str(ROOT/cfg[key])
        observed=[]
        def resolve(router,text):
            observed.append(router);time.sleep(.35)
            return dict(intent=dict(kind='select',program='baigufuren'),reply='我是迟到回答')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'p1.json';config.write_text(json.dumps(cfg))
            script=root/'asr.py';script.write_text('''import json,time
def emit(**event):print(json.dumps(event,ensure_ascii=False),flush=True)
emit(kind='ready');start=time.monotonic()
for i,(delay,text) in enumerate([(.05,'小帅'),(.10,'讲个笑话'),(.15,'待机'),(.20,'小帅')],1):
 time.sleep(max(0.,start+delay-time.monotonic()))
 emit(kind='asr_final',utterance_id=i,text=text,captured_end_monotonic=time.monotonic())
time.sleep(3)
''')
            def launch(command,**kwargs):return actual_popen([sys.executable,str(script)],**kwargs)
            with patch('vivis.interaction.dialogue.ROOT',root), \
                 patch('vivis.interaction.dialogue.dependencies',return_value=(speech,[])), \
                 patch('vivis.interaction.dialogue.subprocess.Popen',side_effect=launch), \
                 patch.object(ConversationRouter,'resolve',resolve),redirect_stdout(io.StringIO()):
                code=main(['--config',str(config),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),
                           '--listen-only','--cloud-chat','on','--seconds','.8','--output',str(root/'run')])
            self.assertEqual(code,0)
            summary=json.loads((root/'run/summary.json').read_text());self.assertEqual(summary['desired_mode'],'interaction')
            events=[json.loads(line) for line in (root/'run/events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['kind']=='cloud_stale' for e in events));self.assertEqual(observed[0].history,[])


if __name__=='__main__':unittest.main()
