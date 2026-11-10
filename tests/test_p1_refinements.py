"""P1 source gains, deferred readiness and home-before-program contracts."""
from contextlib import contextmanager,ExitStack,redirect_stdout,redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.interaction.animation import FrameMapper,speech_channel_gains


def config(name):return json.loads((ROOT/name).read_text())


class SpeechGainTests(unittest.TestCase):
    def test_isolated_jaw_gain_scales_linear_stroke_keeps_neutral_and_clamps_at_endpoint(self):
        from vivis.interaction.singing import plan
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json');settings=config('configs/interaction/p0.json')
        p=plan(robot,'0001','eyes-neck',settings)
        gains=speech_channel_gains({**settings,'jaw_open_channel_gain':1.8,'round_lip_channel_gain':1.8},p['order'])
        mapper=FrameMapper(p['order'],p['shapes'],p['profile'],p['ids'],channel_gains=gains)
        raw=[0.]*52;index=p['ids'].index(31);endpoint=p['shapes']['JawOpen'][31]
        neutral=mapper.map_weights(raw).copy();self.assertEqual(neutral[index],2047.)
        raw[p['order'].index('JawOpen')]=.25
        result=mapper.map_weights(raw);self.assertAlmostEqual(result[index],2047+.25*1.8*(endpoint-2047))
        for j,sid in enumerate(p['ids']):
            if sid!=31:self.assertEqual(result[j],neutral[j])
        raw[p['order'].index('JawOpen')]=1.
        self.assertEqual(mapper.map_weights(raw)[index],endpoint)
        self.assertEqual(settings,config('configs/interaction/p0.json'))

    def test_trial_cli_changes_only_three_source_gains_without_opening_hardware_or_saving_defaults(self):
        from vivis.interaction.singing import main
        original=(ROOT/'configs/interaction/p0.json').read_bytes()
        base=['--robot',str(ROOT/'configs/robots/xiaoshuai/robot.json'),'--smile','off','--check']
        reports=[]
        with patch('vivis.interaction.stream.subprocess.Popen',side_effect=AssertionError('model opened')), \
             patch('vivis.hardware.feetech.sdk.OfficialSDKBus',side_effect=AssertionError('serial opened')):
            for options in ([],['--jaw-open-gain','1.8','--round-lip-gain','1.8']):
                with redirect_stdout(io.StringIO()) as out:main(base+options)
                reports.append(json.loads(out.getvalue()))
            for value in ('nan','0','3.1'):
                with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(base+['--jaw-open-gain',value])
        before,after=[r['source_channel_gains'] for r in reports]
        self.assertEqual({n for n in before if before[n]!=after[n]},{'JawOpen','MouthFunnel','MouthPucker'})
        self.assertTrue(all(after[n]==1.8 for n in ('JawOpen','MouthFunnel','MouthPucker')))
        self.assertEqual(reports[0]['inference_audio'],reports[1]['inference_audio'])
        self.assertEqual(reports[0]['commanded'],reports[1]['commanded'])
        self.assertEqual((ROOT/'configs/interaction/p0.json').read_bytes(),original)

    def test_larger_social_increment_moves_eligible_smile_axes_without_changing_jaw(self):
        from vivis.interaction.singing import plan
        p=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),'0001','eyes-neck',config('configs/interaction/p0.json'))
        low=FrameMapper(p['order'],p['shapes'],p['profile'],p['ids'],.15,channel_gains=p['source_gains'])
        high=FrameMapper(p['order'],p['shapes'],p['profile'],p['ids'],.30,channel_gains=p['source_gains'])
        raw=[0.]*52;raw[p['order'].index('JawOpen')]=.2
        a=low.map_weights(raw,1.);b=high.map_weights(raw,1.)
        self.assertEqual(a[p['ids'].index(31)],b[p['ids'].index(31)])
        self.assertGreater(abs(high.last_social_delta[p['ids'].index(11)]),abs(low.last_social_delta[p['ids'].index(11)]))
        self.assertFalse({9,24}&set(p['ids']))

    def test_absolute_channel_gains_do_not_stack_or_change_unrelated_source_channels(self):
        from vivis.interaction.singing import plan
        p=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),'0001','eyes-neck',config('configs/interaction/p0.json'))
        gains=speech_channel_gains({'mouth_channel_gain':1.2,'round_lip_channel_gain':1.5},p['order'])
        self.assertEqual(gains['MouthFunnel'],1.5);self.assertEqual(gains['JawOpen'],1.2)
        self.assertNotIn('EyeBlinkLeft',gains);self.assertNotIn('BrowInnerUp',gains)
        mapper=FrameMapper(p['order'],p['shapes'],p['profile'],p['ids'],channel_gains=gains)
        original=np.random.default_rng(8).random(52).tolist();saved=list(original)
        expected=[min(1.,value*gains.get(name,1.)) for name,value in zip(p['order'],original)]
        reference=FrameMapper(p['order'],p['shapes'],p['profile'],p['ids'])
        np.testing.assert_array_equal(mapper.map_weights(original),reference.map_weights(expected))
        self.assertEqual(original,saved);self.assertEqual(mapper.last_fused,expected)
        self.assertTrue(all(g==1. for g in speech_channel_gains(config('configs/interaction/p0.json'),p['order']).values()))
        with self.assertRaises(ValueError):speech_channel_gains({'mouth_channel_gain':float('nan')},p['order'])

    def test_prompt_cache_rejects_old_voice_or_old_level_processing(self):
        from vivis.speech.prompts import PROCESSING_VERSION,TEXTS,cache_matches
        speech=config('configs/perception/speech.json')
        metadata=dict(processing_version=PROCESSING_VERSION,tts=copy.deepcopy(speech['tts']),
                      level={'config':copy.deepcopy(speech['prompt_level'])},texts={**TEXTS,**speech.get('prompt_text_overrides',{})},
                      timing=copy.deepcopy(speech.get('prompt_timing')))
        self.assertTrue(cache_matches(speech,metadata))
        self.assertFalse(cache_matches(speech,[]))
        metadata['tts']['speaker_id']=3;self.assertFalse(cache_matches(speech,metadata))
        metadata['tts']=speech['tts'];metadata['processing_version']='old';self.assertFalse(cache_matches(speech,metadata))
        metadata['processing_version']=PROCESSING_VERSION
        metadata['timing']=None if speech.get('prompt_timing') is not None else {'clause_pause_ms':180}
        self.assertFalse(cache_matches(speech,metadata))


class PreparationOverlapTests(unittest.TestCase):
    def test_deferred_camera_yields_before_ready_and_checks_cancellation_without_pose_commands(self):
        from vivis.runtime.developer import vision_service
        import hashlib
        from test_developer import DeveloperResourcesTests
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=Robot().tracking();file=root/'tracking.json';file.write_text(json.dumps(data))
            digest=hashlib.sha256(file.read_bytes()).hexdigest();child=Mock();child.poll.return_value=None
            # Build the same fresh camera/transport identity as existing tests.
            ready=DeveloperResourcesTests().ready_state(data,digest)
            with patch('vivis.runtime.developer.preview_state',side_effect=[None,ready]) as preview, \
                 patch('vivis.runtime.developer.subprocess.Popen',return_value=child):
                with vision_service(file,root,True,wait_ready=False) as handle:
                    self.assertEqual(preview.call_count,1)
                    handle.check();self.assertEqual(preview.call_count,1)
                    handle.wait_ready();self.assertTrue(handle.ready)
                child.terminate.assert_called_once()

    def exercise_entry(self,song,camera_failure=False):
        import vivis.interaction.singing as singing
        import vivis.interaction.follow as follow
        module=singing if song else follow;events=[]
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        setting_name='configs/interaction/p0.json' if song else 'configs/interaction/expression_follow.json'
        settings=config(setting_name);settings['tracking_enabled']=True
        p=singing.plan(robot,'0001','eyes-neck',settings) if song else follow.plan(robot,settings,1.)
        class Vision:
            def check(self):
                if camera_failure:raise RuntimeError('Camera startup failed')
            def wait_ready(self,guard):guard();events.append('vision_ready')
        @contextmanager
        def vision(*_,**kwargs):
            self.assertFalse(kwargs['wait_ready']);events.append('vision_started');yield Vision()
        class Source:
            error=None
            def __init__(source,*args,**__):
                if song:self.assertEqual(args[2],p['inference_audio'])
                events.append('model_started')
            def wait_ready(self,guard):guard();events.append('a2f_ready')
            def close(self):events.append('model_closed')
            def summary(self):return {}
        receiver=Mock();receiver.wait_ready.side_effect=lambda:events.append('target_ready')
        def home(*args):args[7]();events.append('home_done')
        def play(*_,**__):events.append('play');return {'completed':True}
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            stack.enter_context(patch.object(module,'ROOT',Path(tmp)))
            stack.enter_context(patch.object(module,'plan',return_value=p))
            stack.enter_context(patch.object(module,'vision_service',side_effect=vision))
            stack.enter_context(patch.object(module,'TargetReceiver',return_value=receiver))
            stack.enter_context(patch.object(module,'MojiaPlaybackGaze'))
            stack.enter_context(patch.object(module,'prepare_round',side_effect=home))
            stack.enter_context(patch.object(module,'play',side_effect=play))
            stack.enter_context(patch.object(module,'check_competitors'))
            stack.enter_context(patch.object(Path,'is_file',return_value=True))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if song:
                stack.enter_context(patch.object(singing,'model_audio',return_value=p['inference_audio']))
                stack.enter_context(patch.object(singing,'LiveFrames',Source))
            else:stack.enter_context(patch.object(follow,'ExpressionFrames',Source))
            args=['--robot',str(robot.path),'--config',str(ROOT/setting_name),'--output',str(Path(tmp)/'run'),'--overlap-warmup']
            if not song:args+=['--tracking','on','--seconds','1']
            if camera_failure:
                with self.assertRaisesRegex(RuntimeError,'Camera startup failed'):module.main(args)
            else:self.assertEqual(module.main(args),0)
        return events

    def test_follow_homes_while_camera_initializes_then_waits_for_fresh_target_before_play(self):
        events=self.exercise_entry(False)
        self.assertLess(events.index('home_done'),events.index('vision_ready'))
        self.assertLess(events.index('target_ready'),events.index('play'))

    def test_song_starts_model_early_but_waits_for_home_vision_and_audio_frames_before_play(self):
        events=self.exercise_entry(True)
        self.assertLess(events.index('model_started'),events.index('home_done'))
        self.assertLess(events.index('home_done'),events.index('a2f_ready'))
        self.assertLess(events.index('vision_ready'),events.index('play'))
        self.assertLess(events.index('a2f_ready'),events.index('play'))
        self.assertIn('model_closed',events)

    def test_startup_worker_fault_stops_preparation_and_never_plays(self):
        events=self.exercise_entry(True,camera_failure=True)
        self.assertNotIn('home_done',events);self.assertNotIn('play',events)
        self.assertIn('model_closed',events)


if __name__=='__main__':unittest.main()
