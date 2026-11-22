"""Social face patterns, smooth poses and shared ownership; no hardware/models."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.interaction.expression_intent import configuration,arguments,IntentWriter,IntentEnvelope
from vivis.interaction.social_expression import observation,SocialExpressionResponse
from vivis.interaction.standby import plan,main


def settings():return json.loads((ROOT/'configs/interaction/interaction.json').read_text())


def face(kind,stamp):
    weights={name:0. for name in ('mouthSmileLeft','mouthSmileRight','browDownLeft','browDownRight','browInnerUp',
        'mouthFrownLeft','mouthFrownRight','eyeWideLeft','eyeWideRight','browOuterUpLeft','browOuterUpRight')}
    names={'joy':('mouthSmileLeft','mouthSmileRight'),
        'concern':('browInnerUp','mouthFrownLeft','mouthFrownRight'),
        'listening':('browDownLeft','browDownRight'),
        'surprise':('eyeWideLeft','eyeWideRight','browInnerUp','browOuterUpLeft','browOuterUpRight')}
    for name in names.get(kind,()):weights[name]=.9
    return dict(schema_version='vivis-face-expression-1.0',valid=True,timestamp_monotonic=stamp,weights=weights)


def receiver(kind):
    def poll(now):
        return dict(schema_version='vivis-person-target-1.0',timestamp_monotonic=now,sequence=int(now*60),
            target=dict(x=0.,y=0.),social_expression=observation(face(kind,now)),
            social_smile=dict(schema_version='vivis-smile-event-1.0',timestamp_monotonic=now,valid=True,
                strength=1. if kind=='joy' else 0.))
    return SimpleNamespace(max_age=.3,poll=poll)


class SocialExpressionTests(unittest.TestCase):
    def engine(self):
        c=configuration()
        return SocialExpressionResponse(settings()['social_response']['expression_responses'],c['presets'],c['attack_s'],c['release_s'])

    def test_each_pattern_needs_stable_observation_and_releases_without_a_face(self):
        for kind in configuration()['presets']:
            response=self.engine()
            for i in range(20):response.step(observation(face(kind,i/60)),i/60)
            self.assertEqual(response.state,'neutral')
            for i in range(20,180):response.step(observation(face(kind,i/60)),i/60)
            self.assertEqual(response.state,kind);self.assertTrue(response.weights)
            for i in range(180,660):response.step(None,i/60)
            self.assertEqual(response.state,'neutral');self.assertTrue(all(v<.001 for v in response.weights.values()))

    def test_repeated_old_frames_and_multiple_people_cannot_establish_a_pattern(self):
        response=self.engine();event=observation(face('surprise',0.))
        for i in range(100):response.step(event,i/60)
        self.assertEqual(response.state,'neutral')
        for i in range(100,280):response.step(observation(face('surprise',i/60),single=False),i/60)
        self.assertEqual(response.state,'neutral');self.assertFalse(response.weights)

    def test_bad_coefficients_are_invalid_observations(self):
        data=face('joy',0.);data['weights']['mouthSmileLeft']=float('nan')
        self.assertFalse(observation(data)['valid'])
        response=self.engine();event=observation(face('joy',4.))
        response.step(event,0.);self.assertFalse(response.diagnostic['valid'])

    def test_presets_use_only_canonical_channels_and_both_robot_calibrations(self):
        c=configuration()
        for name in ('xiaoshuai','mojia'):
            p,ids,_,mapper,source,_=plan(Robot(ROOT/f'configs/robots/{name}/robot.json'),settings(),1.)
            mapper.prepare_expression_mapping()
            for key in c['presets']:
                value=arguments(dict(expression=key,source='social'),c)
                self.assertEqual(value['weights'],c['presets'][key])
                axes=set().union(*(mapper.expression_axes[channel] for channel in value['weights']))
                self.assertTrue(axes-set(p.get('gaze_owner_ids',[])),(name,key))
            self.assertNotIn(9,ids if name=='xiaoshuai' else [])

    def test_automatic_pose_preserves_speech_and_gaze_ownership_and_blink_closure(self):
        sources=[]
        for enabled in (False,True):
            c=settings();c['social_response']['expression_responses']['enabled']=enabled
            p,ids,_,mapper,source,_=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),c,1.)
            source.receiver=receiver('surprise');source.next_blink=source.next_look=100.
            speech=[0.]*52;speech[mapper.order.index('JawOpen')]=.35
            source.mouth=SimpleNamespace(sample=lambda now:speech)
            for i in range(180):target=source.sample(i/60,10+i/60)
            sources.append((p,ids,mapper,source,target))
        plain=sources[0];rich=sources[1]
        self.assertTrue(np.any(abs(plain[-1]-rich[-1])>1))
        for sid in set(rich[2].speech_owned_ids)|set(rich[0]['gaze_owner_ids']):
            self.assertAlmostEqual(plain[-1][plain[1].index(sid)],rich[-1][rich[1].index(sid)])
        blink=[]
        for p,ids,mapper,source,_ in sources:
            source.next_blink=3.
            source.sample(3.,13.)
            blink.append(source.sample(3.24,13.24))
        axes=rich[2].expression_axes['EyeBlinkLeft']|rich[2].expression_axes['EyeBlinkRight']
        for sid in axes:self.assertAlmostEqual(blink[0][plain[1].index(sid)],blink[1][rich[1].index(sid)])
        for j,sid in enumerate(rich[1]):
            motor=next(m for m in rich[0]['motors'] if m['id']==sid)
            self.assertTrue(motor['safe_min']<=rich[-1][j]<=motor['safe_max'])

    def test_explicit_neutral_suppresses_automatic_pose_until_auto_is_restored(self):
        _,ids,_,mapper,source,_=plan(Robot(ROOT/'configs/robots/xiaoshuai/robot.json'),settings(),1.)
        source.receiver=receiver('listening');source.next_blink=source.next_look=100.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'intent.json';c=configuration();source.expression_intent=IntentEnvelope(path,c)
            writer=IntentWriter(path,c)
            for i in range(180):active=source.sample(i/60,10+i/60)
            writer.publish(dict(expression='neutral'),1,13.)
            for i in range(240):neutral=source.sample(3+i/60,13+i/60)
            self.assertLess(np.max(abs(neutral-mapper.neutral)),1.)
            writer.publish(dict(expression='auto'),1,17.)
            for i in range(240):again=source.sample(7+i/60,17+i/60)
            self.assertGreater(np.max(abs(again-mapper.neutral)),5.)

    def test_offline_switch_preserves_legacy_smile(self):
        with redirect_stdout(io.StringIO()) as output:
            main(['--config',str(ROOT/'configs/interaction/interaction.json'),'--expression-responses','off','--check'])
        self.assertTrue(json.loads(output.getvalue())['smile_enabled'])


if __name__=='__main__':unittest.main()
