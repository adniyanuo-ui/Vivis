"""Attention intentions retain real camera truth and continuous neck state."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np

from vivis.paths import ROOT
from vivis.motion.mapping import read_json
from vivis.runtime.robot import Robot
from vivis.interaction.attention import AttentionRhythm
from vivis.interaction.standby import plan,StandbyGaze


class AttentionTests(unittest.TestCase):
    def test_continuous_smiling_never_suppresses_all_gaze_breaks(self):
        c=dict(enabled=True,contact_s=[2.,2.],aversion_s=[1.,1.])
        policy=AttentionRhythm(c,seed=1);starts=[];weights=[]
        for i in range(1201):
            now=i/60;d=policy.step(True,now,smile=i%24<12)
            if d['started']:starts.append(now)
            weights.append(d['blend']);self.assertTrue(0<=d['blend']<=1)
        self.assertGreaterEqual(len(starts),4)
        self.assertLessEqual(starts[0],3.1);self.assertLessEqual(max(np.diff(starts)),4.1)
        self.assertGreater(max(weights),.95)
        self.assertLess(max(abs(np.diff(weights))),.1)

    def test_loss_of_target_fades_aversion_and_disabled_attention_never_pauses(self):
        p=AttentionRhythm(dict(enabled=True,contact_s=[2.,2.],aversion_s=[1.,1.]),seed=2)
        for i in range(151):d=p.step(True,i/60)
        before=d['blend'];self.assertGreater(before,.5)
        values=[p.step(False,2.5+i/60)['blend'] for i in range(21)]
        self.assertAlmostEqual(values[0],before);self.assertEqual(values[-1],0.)
        self.assertTrue(all(b<=a for a,b in zip(values,values[1:])))
        disabled=AttentionRhythm({'enabled':False})
        for now in range(100):self.assertFalse(disabled.step(True,float(now))['pause'])
        self.assertEqual(disabled.count,0)

    def test_both_robot_topologies_keep_sensing_while_eye_intent_averts_and_neck_brakes(self):
        for name in ('xiaoshuai','mojia'):
            c=read_json(ROOT/'configs/interaction/interaction.json')
            c['attention'].update(contact_s=[2.,2.],aversion_s=[1.,1.])
            profile,ids,_,mapper,source,neutral=plan(Robot(ROOT/f'configs/robots/{name}/robot.json'),c,30.)
            source.attention=AttentionRhythm(c['attention'],seed=1)
            receiver=SimpleNamespace(max_age=.3,event=None,poll=lambda now:receiver.event)
            source.receiver=receiver;gaze=StandbyGaze(source.tracking,profile,ids,receiver,source)
            gaze.prepare(neutral,mapper.neutral,[],0.);gaze.set_audio_time(0.,30.)
            last=mapper.neutral.copy();m={x['id']:x for x in profile['motors']};seen=False
            for i in range(1,601):
                now=i/60;visible=now<6.
                receiver.event=dict(schema_version='vivis-person-target-1.0',sequence=i,timestamp_monotonic=now,
                    target=dict(x=.25,y=.1,eye_line_roll_rad=.02) if visible else None,
                    social_smile=dict(schema_version='vivis-smile-event-1.0',timestamp_monotonic=now,valid=visible,strength=1. if visible else 0.))
                original=copy.deepcopy(receiver.event)
                for sid,value in zip(ids,last):gaze.observe(sid,dict(position=float(value)),now)
                base=source.sample(now,now);goals=gaze.apply(base,last,1/60,now)
                d=gaze.last_diagnostic
                self.assertEqual(receiver.event,original)
                if d['attention']['pause']:
                    self.assertEqual(d['state'],'tracking')
                    self.assertEqual(gaze.controller.policy.last_seen,now)
                    self.assertEqual(d['neck_assist_reason'],'attention_pause')
                    self.assertEqual(gaze.controller.upstream.last_diagnostic['requested_velocity'],0.)
                    self.assertGreater(source.smile,.99)
                    if d['attention']['blend']>.8:
                        col=ids.index(gaze.camera_eye_id);j=gaze.controller_ids.index(gaze.camera_eye_id)
                        seen|=abs(goals[col]-gaze.previous[j])>5.
                for sid,v in zip(ids,goals):self.assertTrue(m[sid]['safe_min']<=v<=m[sid]['safe_max'])
                last=goals
            self.assertTrue(seen,name)
            for sid in gaze.owned:self.assertAlmostEqual(last[ids.index(sid)],2047.,delta=1.)
