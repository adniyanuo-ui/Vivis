"""Virtual speech/DAC, interruption and actual calibration; no cloud or hardware."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from vivis.paths import ROOT
from vivis.interaction.conversation_expression import ConversationEnvelope, configuration
from vivis.interaction.expression_intent import configuration as expressions, atomic_json, IntentEnvelope
from vivis.interaction.standby import plan
from vivis.runtime.robot import Robot
from vivis.speech.conversation_animation import ConversationAnimation
from vivis.speech.realtime_audio import VoiceBuffer


class ConversationAnimationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.c, self.expressions = configuration(), expressions()
        self.path = Path(self.tmp.name)/'conversation_expression.json'
        self.animation = ConversationAnimation(self.path, self.c, self.expressions)
        self.animation.context(('visit', 1), True, False, 10.)
        self.scope = ('visit', 1)

    def stage(self, style='joy', at=10., rid='reply'):
        self.animation.stage(dict(expression=style, source='social'), self.scope, at)
        self.animation.register(rid, self.scope, at)

    def test_listening_thinking_and_neutral_are_bounded_and_not_mood_diagnosis(self):
        a = self.animation
        a.context(self.scope, True, True, 10.)
        packet = a.snapshot(10.1)
        self.assertEqual(packet['phase'], 'listening')
        self.assertLess(packet['smile_scale'], 1.)
        self.assertNotIn('user_emotion', packet)
        a.context(self.scope, True, False, 11.)
        self.assertEqual(a.snapshot(11.1)['phase'], 'thinking')
        self.assertEqual(a.snapshot(18.)['weights'], {})

    def test_future_dac_not_text_or_generation_end_starts_expression(self):
        self.stage()
        a = self.animation
        a.played('reply', 0, 48000, 11.)
        self.assertNotEqual(a.snapshot(10.99)['phase'], 'speaking')
        playing = a.snapshot(11.02)
        self.assertEqual(playing['phase'], 'speaking')
        self.assertEqual(playing['style'], 'joy')
        self.assertAlmostEqual(playing['played_seconds'], .02)
        self.assertGreater(playing['weights']['MouthSmileLeft'], 0.)
        self.assertEqual(a.snapshot(12.1)['weights'], {})

    def test_late_intent_skipped_without_restarting_reply(self):
        a = self.animation
        a.register('reply', self.scope, 10.)
        a.played('reply', 0, 48000, 11.)
        result = a.stage(dict(expression='surprise', source='social'), self.scope, 11.8)
        self.assertEqual(result['status'], 'skipped_late_expression')
        self.assertFalse(result['audio_blocked'])
        self.assertEqual(a.snapshot(11.9)['style'], 'neutral')
        a.register('reply', self.scope, 12.)  # Duplicate metadata cannot reset its first DAC.
        self.assertEqual(a.stage(dict(expression='joy',source='social'),self.scope,12.)['status'],
                         'skipped_late_expression')

    def test_slow_network_keeps_valid_same_turn_pose_until_actual_audio_not_a_fixed_timeout(self):
        self.stage()
        a = self.animation
        a.played('reply', 0, 48000, 40.)
        self.assertEqual(a.snapshot(40.1)['style'], 'joy')

    def test_cancel_clears_only_its_response_and_old_audio_cannot_revive_pose(self):
        self.stage()
        a = self.animation
        a.played('reply', 0, 48000, 11.)
        self.assertEqual(a.snapshot(11.1)['style'], 'joy')
        a.cancel('reply')
        a.played('reply', 48000, 96000, 11.2)
        self.assertEqual(a.snapshot(11.3)['weights'], {})
        a.register('new', self.scope, 11.3)
        a.stage(dict(expression='concern', source='social'), self.scope, 11.3)
        a.played('new', 0, 48000, 11.4)
        a.cancel('reply')
        self.assertEqual(a.snapshot(11.5)['style'], 'concern')

    def test_new_turn_person_session_and_disable_reject_old_intent_and_spans(self):
        for scope in (('visit', 2), ('other-visit', 1), ('other-person', 1)):
            self.animation.context(self.scope, True, False, 10.)
            self.stage()
            self.animation.context(scope, True, True, 11.)
            self.animation.played('reply', 0, 48000, 11.1)
            self.assertEqual(self.animation.snapshot(11.2)['phase'], 'listening')
            self.assertFalse(self.animation.register('old', self.scope, 11.3))
            with self.assertRaises(ValueError):
                self.animation.stage(dict(expression='joy', source='social'), self.scope, 11.3)
        self.animation.context(None, False, False, 12.)
        self.assertEqual(self.animation.snapshot(12.1)['phase'], 'disabled')

    def test_pause_and_resume_follow_actual_new_spans(self):
        self.stage()
        a = self.animation
        a.played('reply', 0, 48000, 11.)
        a.pause(['reply'])
        self.assertNotEqual(a.snapshot(11.1)['phase'], 'speaking')
        a.resume(['reply'])
        self.assertNotEqual(a.snapshot(11.2)['phase'], 'speaking')
        a.played('reply', 48000, 96000, 11.3)
        self.assertEqual(a.snapshot(11.4)['style'], 'joy')
        self.assertAlmostEqual(a.snapshot(11.4)['played_seconds'], 1.1)

    def test_one_intent_cannot_leak_into_a_second_reply_or_unregistered_fixed_audio(self):
        self.stage()
        a = self.animation
        a.register('second', self.scope, 10.)
        a.played('reply', 0, 4800, 11.)
        a.played('second', 0, 4800, 11.1)
        self.assertEqual(a.snapshot(11.15)['style'], 'neutral')
        self.assertIsNone(a.snapshot(11.15)['intent_id'])
        a.played('unrelated-receipt', 0, 48000, 11.2)
        self.assertEqual(a.snapshot(11.3)['phase'], 'speaking')
        self.assertEqual(a.snapshot(11.3)['weights'], {})
        self.assertIsNone(a.snapshot(11.3)['intent_id'])

    def test_serious_intent_suppresses_automatic_smile_only_during_its_audio(self):
        self.stage('concern')
        a = self.animation
        a.played('reply', 0, 48000, 11.)
        self.assertEqual(a.snapshot(11.2)['smile_scale'], 0.)
        self.assertEqual(a.snapshot(12.2)['smile_scale'], 1.)

    def test_expression_changes_with_played_samples_not_network_wait(self):
        self.stage()
        a = self.animation
        a.played('reply', 0, 48000, 11.)
        early = a.snapshot(11.1)['weights']['BrowOuterUpLeft']
        a.played('reply', 480000, 528000, 25.)
        late = a.snapshot(25.1)['weights']['BrowOuterUpLeft']
        self.assertLess(late, early)

    def test_real_voicebuffer_mouth_wait_cancel_and_optional_failure_keep_pcm_correct(self):
        self.stage()
        player = VoiceBuffer()
        player.expression_clock = self.animation
        pcm = (np.ones(2400)*5000).astype('<i2').tobytes()
        class Mouth:
            allow = False
            def add(self, *_):pass
            def ready(self, *_):return self.allow
            def played(self, *_):pass
            def finish(self, *_):pass
            def deactivate(self):pass
        mouth = Mouth()
        player.mouth = mouth
        player.add('reply', pcm)
        player.finish('reply')
        self.assertFalse(np.any(player.render(960, 11.)))
        self.assertNotEqual(self.animation.snapshot(11.01)['phase'], 'speaking')
        mouth.allow = True
        self.assertTrue(np.any(player.render(960, 11.02)))
        self.assertEqual(self.animation.snapshot(11.03)['style'], 'joy')
        player.cancel('reply')
        self.assertNotEqual(self.animation.snapshot(11.04)['phase'], 'speaking')
        self.assertFalse(player.add('reply', pcm))
        player.render(player.count, 11.1)  # Drain only the established cancellation fade.
        class Broken:
            def played(self, *_):raise OSError('optional observer failed')
        player.expression_clock = Broken()
        player.add('new', pcm)
        self.assertTrue(np.any(player.render(960, 12.)))
        self.assertGreater(player.expression_clock_errors, 0)

    def test_receiver_smooths_then_releases_cancelled_and_stale_pose(self):
        self.stage()
        a = self.animation
        a.played('reply', 0, 48000, 11.)
        receiver = ConversationEnvelope(self.path, self.c)
        for i in range(30):
            now = 11+i/60
            atomic_json(self.path, a.snapshot(now))
            before = receiver.weights.get('BrowOuterUpLeft', 0.)
            value = receiver.step(now).get('BrowOuterUpLeft', 0.)
            self.assertLess(abs(value-before), .04)
        self.assertGreater(value, .05)
        a.context(('new', 2), True, True, 11.5)
        for i in range(31):
            now = 11.5+i/60
            atomic_json(self.path, a.snapshot(now))
            receiver.step(now)
        self.assertLess(receiver.weights.get('BrowOuterUpLeft', 0.), .01)
        for i in range(50):
            receiver.step(12.5+i/60)
        self.assertFalse(receiver.diagnostic['valid'])
        self.assertLess(sum(receiver.weights.values()), .005)

    def test_receiver_rejects_future_invalid_and_out_of_order_packets(self):
        self.stage()
        self.animation.played('reply', 0, 48000, 11.)
        good = self.animation.snapshot(11.1)
        for change in (dict(timestamp_monotonic=12.), dict(weights={'JawOpen': float('nan')}),
                       dict(weights={'not-a-channel': .3}), dict(smile_scale=2.)):
            receiver = ConversationEnvelope(self.path, self.c)
            atomic_json(self.path, good)
            receiver.step(11.1)
            bad = dict(good, **change)
            if change.get('weights', {}).get('JawOpen') is not None:
                self.path.write_text(json.dumps(bad))
            else:atomic_json(self.path, bad)
            receiver.step(11.2)
            self.assertFalse(receiver.diagnostic['valid'])
        receiver = ConversationEnvelope(self.path, self.c)
        atomic_json(self.path, good)
        receiver.step(11.1)
        atomic_json(self.path, dict(good, sequence=good['sequence']-1))
        receiver.step(11.2)
        self.assertFalse(receiver.diagnostic['valid'])

    def test_disabled_module_does_not_read_a_file_or_start_a_thread(self):
        c = dict(self.c, enabled=False)
        a = ConversationAnimation(self.path, c, self.expressions)
        a.start()
        self.assertIsNone(a.thread)
        receiver = ConversationEnvelope(self.path, c)
        with patch.object(Path, 'read_text', side_effect=AssertionError('disabled I/O')):
            self.assertEqual(receiver.step(11.), {})
        self.assertEqual(receiver.smile_scale, 1.)

    def test_publisher_is_optional_and_stops_without_device_access(self):
        self.animation.start()
        deadline = time.monotonic()+1.
        while not self.path.exists() and time.monotonic()<deadline:
            time.sleep(.005)
        self.assertTrue(self.path.is_file())
        self.animation.close()
        self.assertFalse(self.animation.thread.is_alive())
        self.animation.start()
        self.assertTrue(self.animation.thread.is_alive())
        self.animation.close()

    def test_real_calibration_protects_articulation_blink_gaze_and_limits(self):
        settings = json.loads((ROOT/'configs/interaction/interaction.json').read_text())
        profile, ids, _, mapper, source, _ = plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'), settings, 1.)
        mapper.speech_owned_ids = source.speech_owned_ids
        speech = [0.]*52
        target = mapper.map_weights(speech)
        pose = self.expressions['presets']['joy']
        base = target.copy()
        result = mapper.conversation_target(base, speech, pose, speech, self.c['shared_mouth'])
        self.assertTrue(mapper.last_conversation_mouth_delta)
        mapper.prepare_expression_mapping()
        protected = set(profile.get('gaze_owner_ids', []))
        for name, axes in mapper.expression_axes.items():
            if name.startswith(('Jaw', 'Tongue')):protected |= axes
        for sid in protected & set(ids):
            self.assertAlmostEqual(result[ids.index(sid)], base[ids.index(sid)])
        mapper.speech_owned_ids=[]
        without_a2f=mapper.conversation_target(base,speech,pose,None,self.c['shared_mouth'])
        for sid in protected & set(ids):
            self.assertAlmostEqual(without_a2f[ids.index(sid)],base[ids.index(sid)])
        mapper.speech_owned_ids=source.speech_owned_ids
        speech[mapper.order.index('JawOpen')] = .8
        base = mapper.map_weights(speech)
        blink = mapper.expression_axes['EyeBlinkLeft'] | mapper.expression_axes['EyeBlinkRight']
        result = mapper.conversation_target(base, speech, pose, speech, self.c['shared_mouth'], blink)
        self.assertFalse(mapper.last_conversation_mouth_delta)
        for sid in set(mapper.speech_owned_ids) | blink:
            if sid in ids:self.assertAlmostEqual(result[ids.index(sid)], base[ids.index(sid)])
        for sid, value in zip(ids, result):
            m = next(m for m in profile['motors'] if m['id'] == sid)
            self.assertGreaterEqual(value, m['safe_min'])
            self.assertLessEqual(value, m['safe_max'])

    def test_explicit_pose_keeps_priority_over_current_semantic_overlay(self):
        from vivis.interaction.expression_intent import IntentWriter
        settings = json.loads((ROOT/'configs/interaction/interaction.json').read_text())
        robot = Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        _, _, _, mapper, with_auto, _ = plan(robot, settings, 1.)
        _, _, _, _, baseline, _ = plan(robot, settings, 1.)
        path = Path(self.tmp.name)/'explicit.json'
        writer = IntentWriter(path,self.expressions)
        writer.publish(dict(expression='joy',source='command'),1,10.)
        for source in (baseline,with_auto):
            source.expression_intent=IntentEnvelope(path,self.expressions)
        with_auto.conversation_expression=ConversationEnvelope(self.path,self.c)
        self.stage('concern')
        self.animation.played('reply',0,48000,11.)
        for i in range(30):
            now=11+i/60
            atomic_json(self.path,self.animation.snapshot(now))
            actual=with_auto.sample(i/60,now)
            expected=baseline.sample(i/60,now)
            np.testing.assert_allclose(actual,expected)
        self.assertTrue(with_auto.conversation_expression.diagnostic['suppressed_by_explicit_command'])


class ConversationCloudTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_qwen_social_tool_stages_without_control_ack_but_command_still_waits(self):
        import test_cloud_control as fixtures
        from vivis.interaction.cloud_control import SpeechPresence
        with tempfile.TemporaryDirectory() as tmp:
            s = fixtures.CloudBackendTests().service(tmp, [])
            self.addCleanup(s.log.close)
            self.addCleanup(s.recorders.close)
            self.addCleanup(s.conversation_animation.close)
            s.context = dict(mode='interaction', ready=True, target='interaction', revision=1)
            s.presence = SpeechPresence(fixtures.config('configs/perception/speech_presence.json'), tmp)
            s.presence.wake()
            s.response_revisions['tool-response']=1
            ws = AsyncMock()
            call = dict(type='response.function_call_arguments.done', name='set_expression',
                        call_id='social-1', response_id='tool-response',
                        arguments=json.dumps(dict(expression='concern', source='social')))
            with patch.object(s.expression, 'publish', side_effect=AssertionError('automatic motor ack path')):
                await s.tool(ws, call)
            outputs = [json.loads(c.args[0]) for c in ws.send.call_args_list]
            result = json.loads(outputs[-1]['item']['output'])
            self.assertTrue(result['accepted'])
            self.assertEqual(result['status'], 'scheduled_for_current_reply')
            self.assertIsNone(s.expression_pending)
            self.assertIsNone(s.expression.active)
            scope=s.sync_conversation_animation()
            s.conversation_animation.register('answer',scope)
            s.player.expression_clock=s.conversation_animation
            pcm=(np.ones(2400)*5000).astype('<i2').tobytes()
            s.player.add('answer',pcm)
            s.player.finish('answer')
            now=time.monotonic()
            s.player.render(4800,now+.1)
            self.assertNotEqual(s.conversation_animation.snapshot(now+.09)['phase'],'speaking')
            pose=s.conversation_animation.snapshot(now+.12)
            self.assertEqual(pose['style'],'concern')
            self.assertEqual(pose['smile_scale'],0.)
            s.player.cancel('answer')
            self.assertNotEqual(s.conversation_animation.snapshot(now+.13)['phase'],'speaking')
            command = dict(call, call_id='command-2',
                           arguments=json.dumps(dict(expression='joy', source='command')))
            await s.tool(ws, command)
            self.assertIsNotNone(s.expression_pending)
            self.assertEqual(s.expression.active['source'], 'command')


if __name__ == '__main__':
    unittest.main()
