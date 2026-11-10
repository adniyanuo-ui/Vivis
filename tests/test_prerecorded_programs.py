"""Prepared tasks reuse the static runtime and keep managed cancellation/AEC."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.show import main as show
from vivis.interaction.dialogue import program_command,dependencies
from vivis.interaction.intents import IntentRouter
from vivis.interaction.session import SessionPolicy


def read(path):return json.loads((ROOT/path).read_text())


class PreparedProgramTests(unittest.TestCase):
    def setUp(self):
        self.robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        self.catalog=read('configs/interaction/programs.json')
        self.settings=read('configs/interaction/p1.json')
        self.speech=read(self.settings['speech'])

    def test_both_artistic_tasks_use_static_backend_without_social_or_a2f_flags(self):
        tasks=[p for p in self.catalog['programs'] if p['kind']=='prerecorded']
        self.assertEqual({p['work'] for p in tasks},{'0001','wz2'})
        for p in tasks:
            self.assertFalse(p['realtime_response'])
            command=program_command(p,self.robot,Path('/tmp/round1'),self.speech,self.settings)
            self.assertIn('vivis.interaction.prerecorded',command)
            for flag in ('--session-stop-file','--audio-reference-port','--session-id','--soft-audio-cancel-ms'):
                self.assertIn(flag,command)
            self.assertNotIn('--smile',command)
            self.assertNotIn('vivis.interaction.singing',command)
        # Missing native A2F cannot block the static program manager.
        self.settings['realtime_voice']['lip_sync']['enabled']=False
        with patch('pathlib.Path.is_file',return_value=False):
            _,missing=dependencies(self.settings)
        self.assertFalse(any('/a2f/' in value for value in missing))

    def test_weizong_aliases_have_no_program_two_collision_and_completion_returns_interaction(self):
        router=IntentRouter(self.catalog);session=SessionPolicy(p['id'] for p in self.catalog['programs'])
        session.request(router.parse('小帅',state=session.mode))
        self.assertEqual(next(p for p in self.catalog['programs'] if p['id']=='wz2')['title'],'宇飞剧本')
        for text in ('播放宇飞剧本','播放宇飞的剧本','播放魏宇飞的剧本','播放魏总剧本','播放魏总剧本表演','播放魏总朗诵','节目二','播放wz2'):
            intent=router.parse(text,state='interaction');self.assertEqual(intent,dict(kind='select',program='wz2'))
        session.request(router.parse('播放魏总剧本',state=session.mode))
        self.assertEqual(session.mode,'task');session.completed('wz2')
        self.assertEqual(session.mode,'interaction')

    def test_repeat_needs_a_controller_record_and_only_runs_in_interaction(self):
        router=IntentRouter(self.catalog)
        for phrase in ('再演一遍','再来一次','请再播放一遍吧','小帅，再演一遍'):
            self.assertEqual(router.parse(phrase,state='interaction',previous_program='wz2'),dict(kind='select',program='wz2'))
            self.assertEqual(router.parse(phrase,state='interaction'),dict(kind='prompt',name='clarify'))
            self.assertNotEqual(router.parse(phrase,state='task',previous_program='wz2').get('kind'),'select')
        for phrase in ('不要再演一遍','再演一遍是什么意思','他说再演一遍'):
            self.assertNotEqual(router.parse(phrase,state='interaction',previous_program='wz2').get('kind'),'select')
        self.assertNotEqual(router.parse('你表演未等剧本吧',state='interaction',allow_title_correction=True).get('kind'),'select')

    def test_generic_song_and_script_requests_resolve_by_catalog_category(self):
        import copy
        router=IntentRouter(self.catalog)
        for phrase,program in [('呃播放剧本','wz2'),('播放预设剧本','wz2'),('给我表演一段剧本吧','wz2'),
                               ('播放歌曲','baigufuren'),('唱唱唱唱首歌','baigufuren')]:
            self.assertEqual(router.parse(phrase,state='interaction'),dict(kind='select',program=program),phrase)
        multiple=copy.deepcopy(self.catalog)
        multiple['programs'].append(dict(id='another-script',title='另一个剧本',kind='prerecorded',category='script',aliases=['另一个剧本']))
        router=IntentRouter(multiple)
        self.assertEqual(router.parse('播放剧本',state='interaction'),dict(kind='prompt',name='clarify'))
        self.assertEqual(router.parse('播放宇飞剧本',state='interaction'),dict(kind='select',program='wz2'))
        for phrase in ('不要播放剧本','播放剧本是什么意思','他说播放剧本','没有歌曲吗','你会哪首歌'):
            self.assertNotEqual(router.parse(phrase,state='interaction').get('kind'),'select',phrase)

    def test_static_assets_and_both_robot_retargets_pass_existing_runtime_validation(self):
        for name in ('0001','wz2'):
            work=self.robot.work(name)
            source=work['source_clip']
            payload=json.loads(work['targets'].read_text())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),payload['source_clip_sha256'])
            if name=='wz2':self.assertFalse(set(payload['motor_ids'])&{9,24,29,35,36})
            with redirect_stdout(io.StringIO()) as output:
                show(['--robot',str(self.robot.path),'--work',name,'--repeat','1','--output','/unused','--check'])
            self.assertTrue(json.loads(output.getvalue())['offline_valid'])

    def test_cancel_before_preparation_opens_no_devices_and_keeps_audio_reference_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'show';stop=Path(temp)/'cancel';stop.touch()
            with patch('vivis.runtime.show.prepare_round') as prepare,redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'Operator stop'):
                    show(['--robot',str(self.robot.path),'--work','wz2','--repeat','1','--output',str(output),
                          '--session-stop-file',str(stop),'--audio-reference-port','14791','--session-id','test1',
                          '--soft-audio-cancel-ms','40','--compact-logs'])
            prepare.assert_not_called()
            p=json.loads((output/'profile.json').read_text())
            self.assertEqual(p['session_stop_file'],str(stop))
            self.assertEqual(p['audio_reference'],dict(host='127.0.0.1',port=14791,session_id='test1'))
            self.assertEqual(p['soft_audio_cancel_ms'],40)
            self.assertEqual(p['retain_event_limit'],600)

    def test_overlapped_vision_waits_after_homing_and_before_static_play(self):
        order=[];vision=MagicMock();vision.wait_ready.side_effect=lambda guard:order.append('vision-ready')
        def prepare(*args):order.append('home');args[8]['homing']={}
        def play(*args,**kwargs):
            order.append('play')
            return dict(completed=True,full_clip_requested=True,audio_seconds_at_stop=27.4333,sent_frames=1646,skipped_source_frames=0)
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()), \
                patch('vivis.runtime.show.prepare_round',side_effect=prepare), \
                patch('vivis.runtime.show.TargetReceiver'),patch('vivis.runtime.show.MojiaPlaybackGaze'), \
                patch('vivis.runtime.show.play',side_effect=play):
            show(['--robot',str(self.robot.path),'--work','wz2','--repeat','1','--output',str(Path(temp)/'show')],
                 vision_readiness=vision)
        self.assertEqual(order,['home','vision-ready','play']);self.assertGreater(vision.check.call_count,0)


if __name__=='__main__':unittest.main()
