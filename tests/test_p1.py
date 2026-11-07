"""P1 intent, cancellation, exclusive program switching, idle and echo contracts."""
from contextlib import nullcontext,redirect_stdout,redirect_stderr
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from vivis.paths import ROOT
from vivis.interaction.intents import IntentRouter
from vivis.interaction.dialogue import EventGate,program_command,Worker,main
from vivis.runtime.cancellation import operator_cancelled


def read(path):return json.loads((ROOT/path).read_text())


class IntentTests(unittest.TestCase):
    def test_complete_hypothesis_arbitration_preserves_negation_and_ambiguity(self):
        from vivis.interaction.intents import resolve_candidates
        text,source=resolve_candidates(self.router,'播放白古夫人','播放白骨夫人')
        self.assertEqual(self.router.parse(text)['program'],'baigufuren')
        text,_=resolve_candidates(self.router,'不要停止','停止');self.assertEqual(self.router.parse(text)['kind'],'ignore')
        text,_=resolve_candidates(self.router,'不用播放白骨夫人','播放白骨夫人')
        self.assertEqual(self.router.parse(text)['kind'],'ignore')
        text,_=resolve_candidates(self.router,'我不想退出','退出')
        self.assertEqual(self.router.parse(text)['kind'],'ignore')
        text,_=resolve_candidates(self.router,'播放白骨夫人','开始表情跟随')
        self.assertEqual(self.router.parse(text)['name'],'clarify')
        self.assertEqual(self.router.parse('表情跟随',True,current_program='baigufuren')['program'],'expression-follow')

    def setUp(self):self.router=IntentRouter(read('configs/interaction/programs.json'))
    def test_all_exit_aliases_and_polite_forms_go_to_standby(self):
        for phrase in read('configs/interaction/programs.json')['standby_phrases']:
            self.assertEqual(self.router.parse(phrase,True)['kind'],'standby',phrase)
        for phrase in ('请你停止吧','小帅，请停止播放。','结束当前节目','退出这个节目','请你休息一下'):
            self.assertEqual(self.router.parse(phrase,True)['kind'],'standby',phrase)

    def test_negation_quoted_explanation_and_unrelated_speech_never_switch(self):
        for phrase in ('不要停止','别停止','我不想退出','停止是什么意思','他说停止了','今天天气很好','不要播放白骨夫人'):
            self.assertEqual(self.router.parse(phrase,True)['kind'],'ignore',phrase)

    def test_two_programs_selection_and_removed_recitation(self):
        for phrase,target in [('播放白骨夫人','baigufuren'),('开始表情跟随','expression-follow'),
                              ('切换到表情跟随','expression-follow'),('我想听白骨夫人','baigufuren')]:
            self.assertEqual(self.router.parse(phrase,True),dict(kind='select',program=target))
        self.assertEqual(self.router.parse('播放魏总朗诵')['kind'],'prompt')
        self.assertEqual(self.router.parse('白骨夫人',True)['kind'],'ignore')
        self.assertEqual(self.router.parse('白骨夫人',False)['program'],'baigufuren')
        self.assertEqual(self.router.parse('播放白骨夫人或者表情跟随')['name'],'clarify')

    def test_menu_and_identity_are_local_responses(self):
        for text,prompt in [('有哪些节目','menu'),('你能做什么','menu'),('你是谁','identity')]:
            self.assertEqual(self.router.parse(text),dict(kind='prompt',name=prompt))

    def test_only_fresh_final_utterances_once_in_sequence(self):
        gate=EventGate(2.)
        event=dict(kind='asr_final',utterance_id=1,captured_end_monotonic=10.)
        self.assertTrue(gate.accept(event,10.2));self.assertFalse(gate.accept(event,10.3))
        for other in [{**event,'kind':'asr_partial','utterance_id':2},{**event,'utterance_id':2,'captured_end_monotonic':5.},
                      {**event,'utterance_id':2,'captured_end_monotonic':20.}]:self.assertFalse(gate.accept(other,10.4))
        self.assertTrue(gate.accept({**event,'utterance_id':3},10.5))
        self.assertFalse(gate.accept({**event,'utterance_id':2},10.6))


class CancellationTests(unittest.TestCase):
    def test_session_stop_fades_but_emergency_stop_aborts_immediately(self):
        from test_runtime import PlaybackTests
        from vivis.runtime.playback import play
        for emergency in (False,True):
            profile,bus,clock=PlaybackTests().fixture();profile['soft_audio_cancel_ms']=40
            with tempfile.TemporaryDirectory() as tmp:
                stop=Path(tmp)/'STOP';cancel=Path(tmp)/'CANCEL';profile['session_stop_file']=str(cancel)
                def sample(*_):
                    (stop if emergency else cancel).touch();return np.array([2047.])
                with patch('vivis.runtime.playback.Bus',return_value=bus),patch('vivis.runtime.playback.AudioClock',return_value=clock):
                    with self.assertRaisesRegex(RuntimeError,'Operator stop'):
                        play(profile,[2],np.full((3,1),2047.),60,'fake',Path(tmp)/'log.json',stop,
                             frame_source=SimpleNamespace(sample=sample))
                if emergency:clock.close.assert_called_once();clock.close_gracefully.assert_not_called()
                else:clock.close_gracefully.assert_called_once_with(40)
                bus.close.assert_called_once()

    def test_audio_cancel_envelope_is_continuous_and_reference_uses_faded_pcm(self):
        from vivis.runtime.playback import AudioClock
        class Done(Exception):pass
        clock=AudioClock.__new__(AudioClock);clock.data=np.full((100,1),.2,np.float32)
        clock.cursor=0;clock.first_dac=None;clock.fault=None;clock.sd=SimpleNamespace(CallbackStop=Done)
        clock.fade_total=40;clock.fade_remaining=40;reference=[]
        clock.reference=SimpleNamespace(send=lambda data,timing:reference.append(data.copy()))
        timing=SimpleNamespace(outputBufferDacTime=1.,currentTime=.9)
        first=np.empty((20,1),np.float32);second=np.empty((20,1),np.float32)
        clock.callback(first,20,timing,None)
        with self.assertRaises(Done):clock.callback(second,20,timing,None)
        values=np.r_[first[:,0],second[:,0]]
        self.assertTrue(np.all(np.diff(values)<=0));self.assertAlmostEqual(values[0],.2,places=6)
        np.testing.assert_array_equal(reference[1],second)

    def test_idle_rolling_log_retains_total_submission_count(self):
        from test_runtime import PlaybackTests
        from vivis.runtime.playback import play
        profile,bus,clock=PlaybackTests().fixture();profile['retain_event_limit']=2;clock.duration=.15
        with tempfile.TemporaryDirectory() as tmp,patch('vivis.runtime.playback.Bus',return_value=bus), \
             patch('vivis.runtime.playback.AudioClock',return_value=clock):
            result=play(profile,[2],np.full((9,1),2047.),60,'fake',Path(tmp)/'log.json',Path(tmp)/'STOP')
        self.assertEqual(len(result['events']),2);self.assertGreater(result['sent_frames'],2)
        self.assertTrue(result['event_log_truncated'])

    def test_all_existing_cancel_phases_are_controlled_stops(self):
        for message in ('Operator stop','Operator stop during scheduled feedback','Operator stop during startup feedback',
                        'Stopped during preflight','Stopped before audio'):
            self.assertTrue(operator_cancelled(RuntimeError(message)))
        for message in ('Feedback watchdog','23: invalid single-turn encoder position','Servo 20: status0x01'):
            self.assertFalse(operator_cancelled(RuntimeError(message)))
        self.assertFalse(operator_cancelled(ValueError('Operator stop')))

    def test_expression_feedback_cancel_returns130_and_cleans_up(self):
        from vivis.interaction.follow import main as follow
        receiver=SimpleNamespace(wait_ready=lambda:None,close=lambda:None,max_age=.3)
        with tempfile.TemporaryDirectory() as tmp,patch.object(Path,'is_file',return_value=True), \
             patch('vivis.interaction.follow.check_competitors'),patch('vivis.interaction.follow.vision_service',return_value=nullcontext()), \
             patch('vivis.interaction.follow.TargetReceiver',return_value=receiver), \
             patch('vivis.interaction.follow.prepare_round'),patch('vivis.interaction.follow.play',side_effect=RuntimeError('Operator stop during scheduled feedback')), \
             redirect_stdout(io.StringIO()):
            folder=Path(tmp)/'run'
            self.assertEqual(follow(['--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),'--output',str(folder)]),130)
            self.assertIn('Operator stop',json.loads((folder/'summary.json').read_text())['error'])

    def test_early_cancel_does_not_create_child_output_or_affect_new_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'worker'
            code='import time,sys; time.sleep(.08); from pathlib import Path; p=Path(sys.argv[1]); p.mkdir(); sys.exit(130 if p.with_suffix(".cancel").exists() else 0)'
            worker=Worker([sys.executable,'-c',code,str(folder)],folder,queue.Queue())
            worker.cancel();self.assertFalse(folder.exists())
            self.assertEqual(worker.process.wait(timeout=3),130);worker.finish()
            self.assertTrue(folder.with_suffix('.cancel').is_file())
            self.assertFalse(folder.with_name('next').with_suffix('.cancel').exists())


class IdleTests(unittest.TestCase):
    def test_idle_prepares_canonical_startup_settings_before_motion(self):
        from vivis.interaction.standby import main as standby
        from vivis.runtime.robot import Robot
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()), \
             patch('vivis.interaction.standby.prepare_round') as prepare, \
             patch('vivis.interaction.standby.play',return_value={'completed':True}):
            self.assertEqual(standby(['--robot',str(robot.path),'--seconds','1','--output',str(Path(tmp)/'run')]),0)
            _,profile,ids,_,preset,*_=prepare.call_args.args
            self.assertEqual(preset,robot.startup(active_ids={m['id']:m for m in profile['motors']},inventory_ids=ids))
            self.assertIsNotNone(preset)

    def test_idle_no_person_tracking_bounds_low_blink_and_small_glances(self):
        from vivis.interaction.standby import plan,IdleFrames
        from vivis.runtime.robot import Robot
        c=read('configs/interaction/standby.json')
        for name in ('xiaoshuai','mojia'):
            p,ids,home,mapper,_,_=plan(Robot(ROOT/f'configs/robots/{name}/robot.json'),c,1.)
            idle=IdleFrames(mapper,c,seed=42);motors={m['id']:m for m in p['motors']}
            start=idle.sample(0.,0.)
            np.testing.assert_array_equal(start,mapper.neutral)
            for step in range(1,3601):
                goals=idle.sample(step/60,step/60)
                for sid,v in zip(ids,goals):self.assertTrue(motors[sid]['safe_min']<=v<=motors[sid]['safe_max'])
                for sid in (33,34):self.assertEqual(goals[ids.index(sid)],2047.)
                self.assertLessEqual(abs(goals[ids.index(32)]-2047),c['neck_yaw_span_counts'])
            self.assertTrue(3<=idle.blinks<=7);self.assertTrue(1<=idle.looks<=4)


class EchoTests(unittest.TestCase):
    def test_reference_resampling_follows_timestamped_clock_drift(self):
        from vivis.speech.echo import ReferenceBuffer
        b=ReferenceBuffer();actual_rate=48000*1.0002
        for begin in range(0,48000,960):
            t=10+begin/actual_rate;data=.2*np.sin(2*np.pi*440*(np.arange(960)+begin)/actual_rate)
            b.accept(self.packet(t,data),10.9)
        v,coverage=b.sample(10.1,8000,16000)
        expected=.2*np.sin(2*np.pi*440*(.1+np.arange(8000)/16000))
        self.assertEqual(coverage,1.);self.assertLess(np.max(np.abs(v-expected)),.002)

    def packet(self,stamp,pcm,session='a',rate=48000):
        return json.dumps(dict(version=1,session_id=session,timestamp=stamp,rate=rate,n=len(pcm))).encode()+b'\n'+np.asarray(pcm,dtype='<f4').tobytes()

    def test_dac_pcm_reference_aligns_timestamp_clocks_and_rejects_stale(self):
        from vivis.speech.echo import ReferenceBuffer
        b=ReferenceBuffer();values=np.linspace(0,.5,960).astype(np.float32)
        b.accept(self.packet(10.,values),10.01)
        result,coverage=b.sample(10.,320,16000)
        np.testing.assert_allclose(result,values[::3],atol=1e-6);self.assertEqual(coverage,1.)
        b.accept(self.packet(1.,values),10.02);self.assertEqual(b.rejected,1)
        b.accept(b'broken',10.02);self.assertEqual(b.rejected,2)

    def test_cancel_removes_unplayed_future_reference_only_of_its_session(self):
        from vivis.speech.echo import ReferenceBuffer
        b=ReferenceBuffer();b.accept(self.packet(10.,np.ones(960)*.2),10.)
        b.accept(json.dumps(dict(version=1,end=True,session_id='a',timestamp=10.01)).encode()+b'\n',10.01)
        _,coverage=b.sample(10.01,160,16000);self.assertEqual(coverage,0.)
        b.accept(self.packet(10.02,np.ones(960)*.3,session='new'),10.02)
        v,coverage=b.sample(10.02,320,16000);self.assertEqual(coverage,1.);self.assertAlmostEqual(v[0],.3,places=6)


class SupervisorTests(unittest.TestCase):
    def test_fast_failed_child_reports_self_check_and_rejects_later_selection(self):
        class Prompts:
            def __init__(self,*_):pass
            def poll(self):pass
            def stop(self):pass
            def contains(self,*_):return False
        class Input:
            def __iter__(self):
                time.sleep(.3);yield '开始表情跟随'
                time.sleep(.1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings=read('configs/interaction/p1.json');settings['programs']=str(ROOT/settings['programs'])
            config=root/'p1.json';config.write_text(json.dumps(settings))
            command=[sys.executable,'-c',"print('SELF-CHECK START test',flush=True); print('SELF-CHECK FAIL ID5: outside limits',flush=True); raise SystemExit(1)"]
            with patch('vivis.interaction.dialogue.ROOT',root), \
                 patch('vivis.interaction.dialogue.dependencies',return_value=({},[])), \
                 patch('vivis.interaction.dialogue.check_competitors'),patch('vivis.interaction.dialogue.check_serial_access'), \
                 patch('vivis.interaction.dialogue.PromptPlayer',Prompts),patch('sys.stdin',Input()), \
                 patch('vivis.interaction.dialogue.program_command',return_value=command) as commands,redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['--config',str(config),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),'--text-input']),1)
            self.assertEqual(commands.call_count,1)
            status=json.loads((root/'outputs/runtime/program_status.json').read_text())
            self.assertEqual(status['self_check']['state'],'failed')
            self.assertIn('SELF-CHECK FAIL ID5',output.getvalue());self.assertIn('未执行',output.getvalue())

    def test_serial_permission_error_precedes_microphone_prompts_and_worker(self):
        from vivis.hardware.feetech.protocol import ProtocolError
        with patch('vivis.interaction.dialogue.dependencies',return_value=({},[])), \
             patch('vivis.interaction.dialogue.check_competitors'), \
             patch('vivis.interaction.dialogue.check_serial_access',side_effect=ProtocolError('Serial device access denied')) as access, \
             patch('subprocess.Popen') as process,patch('vivis.interaction.dialogue.PromptPlayer') as prompt, \
             redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as stopped:main([])
        self.assertEqual(stopped.exception.code,2)
        access.assert_called_once_with('/dev/ttyCH343USB0')
        process.assert_not_called();prompt.assert_not_called()
        self.assertIn('Serial device access denied',error.getvalue())

    def test_prompt_cleanup_error_does_not_leave_motion_worker_running(self):
        calls=[]
        class Prompts:
            def __init__(self,*_):self.polls=0
            def poll(self):
                self.polls+=1
                if self.polls>1:raise RuntimeError('audio failed')
            def stop(self):raise RuntimeError('audio cleanup failed')
        class Input:
            def __iter__(self):time.sleep(.2);return iter(())
        worker=SimpleNamespace(process=SimpleNamespace(poll=lambda:None,wait=lambda **_:calls.append('wait')),
            cancel=lambda:calls.append('cancel'),kill_owned_group=lambda:calls.append('kill_owned'),
            finish=lambda:calls.append('finish'),cancelled_at=None)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings=read('configs/interaction/p1.json');settings['programs']=str(ROOT/settings['programs'])
            config=root/'p1.json';config.write_text(json.dumps(settings))
            with patch('vivis.interaction.dialogue.ROOT',root), \
                 patch('vivis.interaction.dialogue.dependencies',return_value=(read('configs/perception/speech.json'),[])), \
                 patch('vivis.interaction.dialogue.check_competitors'),patch('vivis.interaction.dialogue.check_serial_access'), \
                 patch('vivis.interaction.dialogue.PromptPlayer',Prompts), \
                 patch('vivis.interaction.dialogue.Worker',return_value=worker),patch('sys.stdin',Input()),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'audio failed'):
                    main(['--config',str(config),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),'--text-input'])
            self.assertEqual(calls,['cancel','wait','kill_owned','finish'])

    def test_existing_serial_owner_blocks_manager_before_audio_or_asr(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);lock_path=root/'outputs/developer/control.lock';lock_path.parent.mkdir(parents=True)
            config=read('configs/interaction/p1.json');config['programs']=str(ROOT/config['programs'])
            file=root/'p1.json';file.write_text(json.dumps(config))
            with lock_path.open('a') as held:
                fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch('vivis.interaction.dialogue.ROOT',root),patch('vivis.interaction.dialogue.dependencies',return_value=({},[])), \
                     patch('vivis.interaction.dialogue.check_competitors'),patch('vivis.interaction.dialogue.check_serial_access'), \
                     patch('vivis.interaction.dialogue.PromptPlayer') as prompt, \
                     patch('subprocess.Popen') as process:
                    with self.assertRaisesRegex(RuntimeError,'控制权'):
                        main(['--config',str(file),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),'--text-input'])
                    process.assert_not_called();prompt.assert_not_called()

    def test_checks_and_simulation_never_open_devices_or_models(self):
        with patch('subprocess.Popen',side_effect=AssertionError('device/model opened')),redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['--check']),0)
        self.assertFalse(json.loads(out.getvalue())['hardware_opened'])
        with patch('subprocess.Popen',side_effect=AssertionError('device opened')),redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['--simulate-text','播放白骨夫人','--simulate-text','开始表情跟随','--simulate-text','休息一下']),0)
        self.assertTrue(all(not json.loads(s)['hardware_opened'] for s in out.getvalue().splitlines()))

    def test_cancel_transition_waits_for_old_worker_before_new_worker(self):
        from vivis.runtime.robot import Robot
        class Prompts:
            def __init__(self,*_):pass
            def play(self,*_):pass
            def stop(self):pass
            def poll(self):pass
            def contains(self,*_):return False
        class Input:
            def __iter__(self):
                for text in ('播放白骨夫人','开始表情跟随','休息一下'):
                    time.sleep(.25);yield text
                time.sleep(.25)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'worker.py';flag=root/'active';trace=root/'trace'
            script.write_text('''import sys,time
from pathlib import Path
folder,flag,trace,name=map(Path,sys.argv[1:])
cancel=folder.with_suffix('.cancel')
folder.mkdir()
if cancel.exists():sys.exit(130)
handle=flag.open('x');handle.close()
with trace.open('a') as f:f.write('start '+str(name)+'\\n')
print('STANDBY START' if str(name)=='standby' else 'EXPRESSION FOLLOW START',flush=True)
while not cancel.exists():time.sleep(.005)
time.sleep(.03)
flag.unlink()
with trace.open('a') as f:f.write('end '+str(name)+'\\n')
sys.exit(130)
''')
            settings=read('configs/interaction/p1.json');settings['programs']=str(ROOT/settings['programs'])
            config=root/'p1.json';config.write_text(json.dumps(settings))
            def command(program,robot,folder,speech,settings):
                return [sys.executable,str(script),str(folder),str(flag),str(trace),program['id'] if program else 'standby']
            with patch('vivis.interaction.dialogue.ROOT',root),patch('vivis.interaction.dialogue.dependencies',return_value=(read('configs/perception/speech.json'),[])), \
                 patch('vivis.interaction.dialogue.check_competitors'),patch('vivis.interaction.dialogue.check_serial_access'), \
                 patch('vivis.interaction.dialogue.PromptPlayer',Prompts), \
                 patch('vivis.interaction.dialogue.program_command',side_effect=command),patch('sys.stdin',Input()),redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config',str(config),'--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),'--text-input']),0)
            lines=trace.read_text().splitlines();self.assertEqual(len(lines),8)
            for i in range(0,len(lines),2):self.assertEqual(lines[i].replace('start ','end '),lines[i+1])
            self.assertEqual([x[6:] for x in lines[::2]],['standby','baigufuren','expression-follow','standby'])
            self.assertFalse(flag.exists())


class SegmentationTests(unittest.TestCase):
    def test_short_commands_keep_prefix_and_split_before_next_command(self):
        from vivis.speech.segment import Segmenter
        config=read('configs/perception/speech.json')['segmentation']
        flags=[False]*10+[True]*8+[False]*16+[True]*7+[False]*18
        iterator=iter(flags);s=Segmenter(config,detector=lambda pcm:next(iterator));segments=[]
        for i in range(len(flags)):
            result=s.feed(np.full(320,i/100,np.float32),10+i*.02)
            if result:segments.append(result)
        self.assertEqual(len(segments),2)
        self.assertLessEqual(segments[0]['start'],10.2)
        self.assertAlmostEqual(segments[0]['speech_end'],10.36)
        self.assertLess(segments[0]['end'],segments[1]['speech_end'])
        self.assertTrue(np.isclose(segments[0]['audio'][::320],.1).any())


class SelfCheckReportingTests(unittest.TestCase):
    def test_pre_motion_failure_reports_axis_without_position_or_enable_writes(self):
        from test_cold_start import ColdAxis
        from vivis.runtime.robot import Robot
        from vivis.runtime.show import prepare_round
        from unittest.mock import Mock
        robot=Robot();profile=robot.motion_profile();motor=next(m for m in profile['motors'] if m['id']==21)
        profile['motors']=[motor];profile['diagnostic_command_ids']=[21]
        axes=[a for a in robot.home()['axes'] if a['id']==21]
        bus=ColdAxis(position=motor['device_max']+90,sid=21,low=motor['device_min'],high=motor['device_max']);bus.close=Mock()
        args=SimpleNamespace(usb_volume_times_three=False,home_only=True,home_speed=60,authorized_tongue_observation=False)
        report={}
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()) as out, \
             patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus):
            with self.assertRaisesRegex(RuntimeError,'ID21: home position outside'):
                prepare_round(args,profile,[21],axes,None,None,robot.telemetry(),lambda:None,report,Path(tmp),lambda *_:None)
        self.assertEqual(report['self_check']['state'],'failed');self.assertEqual(report['self_check']['failed_axis'],21)
        self.assertFalse(report['preparation_completed']);self.assertIn('SELF-CHECK FAIL',out.getvalue())
        self.assertFalse(any(event[0] in ('write','goal') for event in bus.events));bus.close.assert_called_once()

    def test_homing_failure_keeps_successful_read_only_self_check_distinct(self):
        from test_cold_start import ColdAxis
        from vivis.runtime.robot import Robot
        from vivis.runtime.show import prepare_round
        from unittest.mock import Mock
        robot=Robot();profile=robot.motion_profile();motor=next(m for m in profile['motors'] if m['id']==21)
        profile['motors']=[motor];profile['diagnostic_command_ids']=[21]
        axes=[a for a in robot.home()['axes'] if a['id']==21]
        bus=ColdAxis(position=2047,enabled=1,sid=21,low=motor['device_min'],high=motor['device_max']);bus.close=Mock()
        args=SimpleNamespace(usb_volume_times_three=False,home_only=True,home_speed=60,authorized_tongue_observation=False)
        report={}
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()) as out, \
             patch('vivis.hardware.feetech.sdk.OfficialSDKBus',return_value=bus), \
             patch('vivis.runtime.show.home_all',side_effect=RuntimeError('ID21: home did not settle at neutral')):
            with self.assertRaisesRegex(RuntimeError,'home did not settle'):
                prepare_round(args,profile,[21],axes,None,None,robot.telemetry(),lambda:None,report,Path(tmp),lambda *_:None)
        self.assertEqual(report['self_check']['state'],'passed');self.assertEqual(report['self_check']['checked_ids'],[21])
        self.assertEqual(report['preparation_phase'],'homing');self.assertFalse(report['preparation_completed'])
        self.assertIn('SELF-CHECK PASS',out.getvalue());self.assertIn('PREPARE FAILED',out.getvalue())
        self.assertNotIn('SELF-CHECK FAIL',out.getvalue())


if __name__=='__main__':unittest.main()
