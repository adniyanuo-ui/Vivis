"""Synthetic dimensions exercise equations only; not Mojia commissioning data."""
import copy
import math
import unittest
import numpy as np

from vivis.motion.parallel_neck import RotaryParallelNeck,UnreachablePose,rotation,rotation_derivatives


def synthetic_geometry():
    return dict(schema_version='vivis-rotary-parallel-neck-1.0',pivot_base_mm=[0,0,0],
        pose_limits_rad=[[-.18,.18],[-.18,.18]],branches=[dict(id=sid,
            motor_pivot_mm=[x,50,-80],platform_anchor_mm=[x,50,0],
            crank_u=[0,1,0],crank_v=[0,0,1],crank_radius_mm=20,
            rod_length_mm=math.sqrt(80**2+20**2),neutral_crank_angle_rad=0.,
            encoder_sign=1,minimum_count=1000,maximum_count=3000)
            for sid,x in [(33,-40),(34,40)]])


class ParallelKinematicsTests(unittest.TestCase):
    def test_neutral_closes_both_rods(self):
        k=RotaryParallelNeck(synthetic_geometry())
        actual=k.inverse(0.,0.)
        np.testing.assert_allclose([actual[33],actual[34]],[2047,2047],atol=1e-9)
        np.testing.assert_allclose(k.forward(actual),[0.,0.],atol=1e-9)

    def test_common_and_differential_motion_are_coupled_not_independent_axes(self):
        k=RotaryParallelNeck(synthetic_geometry())
        pitch=k.inverse(.03,0.);roll=k.inverse(0.,.03)
        self.assertAlmostEqual(pitch[33],pitch[34])
        self.assertGreater(roll[33],2047);self.assertLess(roll[34],2047)
        expected=4096/(2*math.pi)*np.array([[2.5,2.], [2.5,-2.]])
        np.testing.assert_allclose(k.jacobian(0.,0.),expected,atol=1e-8)

    def test_jacobian_matches_numerical_differences_at_combined_poses(self):
        k=RotaryParallelNeck(synthetic_geometry());eps=1e-6
        for pitch,roll in [(0.,0.),(.04,-.03),(-.04,.06)]:
            numerical=np.zeros((2,2))
            for j in range(2):
                a=[pitch,roll];b=a.copy();a[j]+=eps;b[j]-=eps
                p=k.inverse(*a);m=k.inverse(*b)
                numerical[:,j]=[(p[sid]-m[sid])/(2*eps) for sid in (33,34)]
            np.testing.assert_allclose(k.jacobian(pitch,roll),numerical,rtol=1e-6,atol=1e-4)

    def test_inverse_forward_round_trip_and_encoder_polarity(self):
        geometry=synthetic_geometry();geometry['branches'][1]['encoder_sign']=-1
        k=RotaryParallelNeck(geometry)
        for pitch in [-.08,0.,.08]:
            for roll in [-.06,0.,.06]:
                np.testing.assert_allclose(k.forward(k.inverse(pitch,roll)),[pitch,roll],atol=1e-7)

    def test_geometry_must_be_measured_consistent_and_non_singular(self):
        for mutate in [lambda g:g['branches'][0].update(rod_length_mm=100),
                       lambda g:g['branches'][0].update(crank_v=[0,1,0]),
                       lambda g:g['branches'][1].update(id=33),
                       lambda g:g['branches'][0].update(encoder_sign=True),
                       lambda g:g['branches'][0].update(crank_radius_mm=-1)]:
            geometry=synthetic_geometry();mutate(geometry)
            with self.assertRaises(ValueError):RotaryParallelNeck(geometry)
        geometry=synthetic_geometry()
        geometry['branches'][1].update(motor_pivot_mm=[-40,50,-80],platform_anchor_mm=[-40,50,0])
        with self.assertRaises(UnreachablePose):RotaryParallelNeck(geometry)

    def test_coupled_joint_limits_are_checked_without_independent_clipping(self):
        geometry=synthetic_geometry()
        for b in geometry['branches']:b.update(minimum_count=2027,maximum_count=2067)
        k=RotaryParallelNeck(geometry)
        with self.assertRaises(UnreachablePose):k.inverse(.04,.03)
        counts=k.inverse(.001,.001)
        self.assertTrue(all(2027<=v<=2067 for v in counts.values()))

    def test_pose_bounds_nonfinite_and_impossible_forward_targets_are_rejected(self):
        k=RotaryParallelNeck(synthetic_geometry())
        for pose in [(float('nan'),0.),(.19,0.),(True,0.)]:
            with self.assertRaises(ValueError):k.inverse(*pose)
        with self.assertRaises(UnreachablePose):k.forward({33:4000,34:2047})

    def test_rotation_derivatives_match_matrix_differences(self):
        pitch,roll=.2,-.15;eps=1e-6
        dp,dr=rotation_derivatives(pitch,roll)
        np.testing.assert_allclose(dp,(rotation(pitch+eps,roll)-rotation(pitch-eps,roll))/(2*eps),atol=1e-9)
        np.testing.assert_allclose(dr,(rotation(pitch,roll+eps)-rotation(pitch,roll-eps))/(2*eps),atol=1e-9)
        np.testing.assert_allclose(rotation(pitch,roll).T@rotation(pitch,roll),np.eye(3),atol=1e-12)
