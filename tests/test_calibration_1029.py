"""The29October revision preserves accepted timing and device control tuning."""
import hashlib,json,unittest
from pathlib import Path
import numpy as np
from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.motion.expression import ExpressionMapper


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class Calibration1029Tests(unittest.TestCase):
    def setUp(self):
        self.robot=Robot();self.revision=json.loads((ROOT/'tests/fixtures/calibration_1029.json').read_text())

    def test_exact_source_revision_and_five_changed_shapes(self):
        self.assertEqual(self.robot.definition['calibration_source'],'墨甲10.29.json')
        self.assertEqual(self.robot.calibration_sha256,self.revision['calibration_sha256'])
        mapper=ExpressionMapper(self.robot)
        self.assertEqual(len(mapper.names),52)
        self.assertEqual(set(self.revision['changed_shapes']),{'JawForward','JawLeft','JawRight','TongueOut','JawOpen'})
        for name,change in self.revision['changed_shapes'].items():
            self.assertEqual(mapper.shapes[name],{int(sid):pos for sid,pos in change['after'].items()})

    def test_only_jaw_targets_change_and_authored_blink_audio_timing_are_preserved(self):
        for name,record in self.revision['works'].items():
            work=self.robot.work(name);target=json.loads(work['targets'].read_text());positions=np.asarray(target['positions'])
            self.assertEqual(target['frames'],record['frames']);self.assertEqual(target['fps'],{'num':60,'den':1})
            self.assertEqual(target['audio_sha256'],record['audio_sha256'])
            self.assertEqual(hashlib.sha256(work['audio'].read_bytes()).hexdigest(),record['audio_sha256'])
            self.assertEqual(hashlib.sha256(work['targets'].read_bytes()).hexdigest(),record['targets_sha256'])
            self.assertEqual(target['calibration_update']['changed_axes'],[31])
            columns=[target['motor_ids'].index(sid) for sid in record['unchanged_axes']]
            self.assertEqual(digest(positions[:,columns].tolist()),record['unchanged_axes_sha256'])
            self.assertEqual(digest(target['blink_adjustment']),record['blink_adjustment_sha256'])
            for sid in (29,31,35,36):self.assertEqual(record['clipped_samples_by_id'][str(sid)],0)

    def test_confirmed_limits_preserve_all_motion_tuning_and_bound_full_jaw_endpoint(self):
        allowed={'safe_min','safe_max','device_min','device_max','feedback_min','feedback_max','endpoint_envelope_not_safe_limits'}
        for sid,changes in self.revision['hardware_updates'].items():
            self.assertTrue(set(changes)<=allowed)
            for key,change in changes.items():self.assertEqual(self.robot.axes[int(sid)][key],change['after'])
        self.assertEqual({sid:(self.robot.axes[sid]['device_min'],self.robot.axes[sid]['device_max']) for sid in (29,31,35,36)},
                         {29:(1400,2050),31:(1547,2049),35:(2000,2247),36:(1800,2280)})
        mapper=ExpressionMapper(self.robot)
        self.assertEqual(mapper.shapes['JawOpen'][31],1547)  # User explicitly corrected the endpoint.
        self.assertEqual(mapper.frame({'JawOpen':1.})[31],1547.)
        self.assertTrue(all(self.robot.axes[sid]['max_speed_counts_s']==1200 for sid in self.robot.motion_profile()['mouth_motion_ids']))

    def test_corrected_jaw_curve_is_rescaled_from_neutral_without_endpoint_clipping(self):
        from vivis.motion.mapping import compile_trajectory
        mapper=ExpressionMapper(self.robot)
        values=np.zeros((4,61));values[:,mapper.channels.index('JawOpen')]=[0.,.25,.5,1.]
        ids,positions,report=compile_trajectory(values,mapper.channels,mapper.shapes,self.robot.motion_profile(),'face_eyes')
        np.testing.assert_array_equal(positions[:,ids.index(31)],[2047.,1922.,1797.,1547.])
        self.assertEqual(report['clipped_samples_by_id']['31'],0)
        for j,sid in enumerate(ids):
            if sid!=31:np.testing.assert_array_equal(positions[:,j],np.full(4,2047.))
