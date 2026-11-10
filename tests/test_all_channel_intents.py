"""All52direct channels, combined poses, and bounded gaze handover."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock
import numpy as np

from vivis.paths import ROOT
from vivis.interaction.expression_intent import configuration,arguments,channel_names,IntentWriter,IntentEnvelope
from vivis.interaction.standby import plan,StandbyGaze
from vivis.runtime.robot import Robot


class AllChannelTests(unittest.TestCase):
    def test_every_canonical_channel_and_custom_combination_are_validated(self):
        c=configuration()
        self.assertEqual(len(channel_names()),52)
        for name in channel_names():
            self.assertEqual(arguments(dict(expression=name),c)['weights'],{name:1.})
        value=arguments(dict(expression='custom',weights=dict(BrowInnerUp=.7,MouthFrownLeft=.6,MouthFrownRight=.6)),c)
        self.assertEqual(len(value['weights']),3)
        for invalid in (dict(expression='custom'),dict(expression='custom',weights={'HeadPitch':1}),
                        dict(expression='custom',weights={'JawOpen':float('nan')}),
                        dict(expression='JawOpen',weights={'EyeBlinkLeft':1})):
            with self.assertRaises(ValueError):arguments(invalid,c)

    def source(self,robot='xiaoshuai'):
        c=json.loads((ROOT/'configs/interaction/interaction.json').read_text())
        result=plan(Robot(ROOT/f'configs/robots/{robot}/robot.json'),c,1.)
        result[3].prepare_expression_mapping()
        return result

    def test_all52map_within_existing_limits_on_both_robots(self):
        for robot in ('xiaoshuai','mojia'):
            profile,ids,_,mapper,_,_=self.source(robot)
            for name in channel_names():
                result=mapper.apply_expression(mapper.neutral,[0.]*52,{name:1.},1.,'command')
                self.assertTrue(np.isfinite(result).all())
                for motor,position in zip((next(m for m in profile['motors'] if m['id']==sid) for sid in ids),result):
                    self.assertGreaterEqual(position,motor['safe_min']);self.assertLessEqual(position,motor['safe_max'])
            self.assertFalse(set(ids)&set(profile.get('excluded_actuator_ids',[])))

    def test_single_channel_endpoint_is_not_overwritten_by_zero_legacy_eye_angles(self):
        _,ids,_,mapper,_,_=self.source()
        self.assertTrue(mapper.expression_axes['EyeLookUpLeft'])
        result=mapper.apply_expression(mapper.neutral,[0.]*52,{'EyeLookUpLeft':1.},1.,'command')
        self.assertTrue(any(abs(result[ids.index(sid)]-mapper.neutral[ids.index(sid)])>1 for sid in mapper.expression_axes['EyeLookUpLeft']))
        self.assertTrue(mapper.last_expression_eye_ids)
        unchanged=mapper.apply_expression(mapper.neutral,[0.]*52,{'EyeLookUpLeft':1.},1.,'social')
        self.assertTrue(np.array_equal(unchanged,mapper.neutral))

    def test_combination_reaches_source_and_ack_reports_actual_mapping(self):
        _,_,_,mapper,source,_=self.source()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'intent.json';writer=IntentWriter(path,configuration())
            source.expression_intent=IntentEnvelope(path,configuration())
            request=writer.publish(dict(expression='custom',weights=dict(BrowInnerUp=.7,MouthFrownLeft=.6,MouthFrownRight=.6)),1,10)
            start=source.sample(0.,10)
            values=[source.sample(0.,10+i/60) for i in range(241)]
            self.assertTrue(np.any(abs(values[-1]-start)>5))
            ack=writer.acknowledgement(request['intent_id'])
            self.assertEqual(ack['status'],'target_accepted');self.assertTrue(ack['mapped_actuator_ids'])
            self.assertEqual(set(ack['requested_channels']),{'BrowInnerUp','MouthFrownLeft','MouthFrownRight'})

    def test_empty_or_zero_calibration_is_reported_instead_of_fake_visible_success(self):
        _,_,_,mapper,source,_=self.source()
        missing=next(name for name,axes in mapper.expression_axes.items() if not axes)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'intent.json';writer=IntentWriter(path,configuration())
            source.expression_intent=IntentEnvelope(path,configuration())
            request=writer.publish(dict(expression=missing),1,10);source.sample(0.,10)
            ack=writer.acknowledgement(request['intent_id'])
            self.assertEqual(ack['status'],'no_calibrated_displacement')
            self.assertEqual(ack['unmapped_or_zero_channels'],[missing])

    def test_explicit_eye_channel_survives_gaze_and_smoothly_returns_control(self):
        profile,ids,_,mapper,source,neutral=self.source()
        receiver=Mock();receiver.poll.return_value=None
        gaze=StandbyGaze(source.tracking,profile,ids,receiver,source);gaze.publish=Mock()
        start=time.monotonic();gaze.prepare(neutral,neutral[0],[],start);gaze.set_audio_time(0.,None)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'intent.json';writer=IntentWriter(path,configuration())
            source.expression_intent=IntentEnvelope(path,configuration())
            writer.publish(dict(expression='EyeLookUpLeft'),1,start)
            previous=neutral[0].copy()
            for i in range(241):
                now=start+i/60;base=source.sample(0.,now);previous=gaze.apply(base,previous,1/60,now)
            eye_ids=mapper.last_expression_eye_ids
            self.assertTrue(eye_ids)
            self.assertTrue(any(abs(previous[ids.index(sid)]-neutral[0,ids.index(sid)])>5 for sid in eye_ids))
            writer.publish(dict(expression='auto'),1,start+4.1)
            for i in range(361):
                now=start+4.1+i/60;base=source.sample(0.,now);previous=gaze.apply(base,previous,1/60,now)
            self.assertLess(source.expression_intent.channel_blend,.01)
