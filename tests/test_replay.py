"""M10 support tests: perturbation helpers must change exactly what they claim
to change, and nothing else."""

import numpy as np
import pytest
from pyquaternion import Quaternion

from adp.calib.camera import SE3
from adp.eval.replay import noisy_pose, pitch_perturbed


def rig_T():
    R_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    return SE3.from_rt(R_base, np.array([1.7, 0.0, 1.5]))


class TestPitchPerturbation:
    def test_zero_is_identity(self):
        T = rig_T()
        assert np.allclose(pitch_perturbed(T, 0.0).matrix, T.matrix)

    def test_translation_unchanged(self):
        """Pitch rotates the camera; it must not move it."""
        T = rig_T()
        assert np.allclose(pitch_perturbed(T, 2.0).translation, T.translation)

    def test_rotation_magnitude_matches(self):
        T = rig_T()
        Tp = pitch_perturbed(T, 2.0)
        dR = T.rotation.T @ Tp.rotation
        angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        assert angle == pytest.approx(2.0, abs=1e-6)

    def test_sign_is_consistent(self):
        T = rig_T()
        a = pitch_perturbed(T, 1.0).rotation
        b = pitch_perturbed(T, -1.0).rotation
        assert not np.allclose(a, b)
        # +d then -d returns to nominal
        back = pitch_perturbed(pitch_perturbed(T, 1.0), -1.0)
        assert np.allclose(back.matrix, T.matrix, atol=1e-9)

    def test_pitch_moves_ground_intersection(self):
        """A nose-down error must change the estimated range of a ground ray."""
        from adp.calib.camera import CameraModel
        from adp.lift.ipm import GroundPlaneLift
        cam = CameraModel(K=np.array([[1266.0, 0, 800.0], [0, 1266.0, 450.0],
                                      [0, 0, 1.0]]), width=1600, height=900)
        box = np.array([760.0, 480.0, 840.0, 553.8])
        r_nom = GroundPlaneLift(cam, rig_T()).lift_box_bottom(box)[1]
        r_up = GroundPlaneLift(cam, pitch_perturbed(rig_T(), 1.0)).lift_box_bottom(box)[1]
        r_dn = GroundPlaneLift(cam, pitch_perturbed(rig_T(), -1.0)).lift_box_bottom(box)[1]
        assert not np.isclose(r_up, r_nom, rtol=1e-3)
        assert (r_up - r_nom) * (r_dn - r_nom) < 0  # opposite directions


class TestPoseNoise:
    def test_zero_noise_is_identity(self):
        T = SE3.from_quat_trans([1, 0, 0, 0], [10.0, 20.0, 0.0])
        rng = np.random.default_rng(0)
        assert np.allclose(noisy_pose(T, 0.0, 0.0, rng).matrix, T.matrix)

    def test_position_noise_scales(self):
        T = SE3.from_quat_trans([1, 0, 0, 0], [0.0, 0.0, 0.0])
        rng = np.random.default_rng(1)
        offs = [np.linalg.norm(noisy_pose(T, 1.0, 0.0, rng).translation)
                for _ in range(400)]
        assert 0.8 < np.mean(offs) < 1.8  # ~Rayleigh(1.0) mean 1.25

    def test_yaw_noise_rotates_only(self):
        T = SE3.from_quat_trans([1, 0, 0, 0], [5.0, 5.0, 0.0])
        rng = np.random.default_rng(2)
        Tn = noisy_pose(T, 0.0, 3.0, rng)
        assert np.allclose(Tn.translation, T.translation)
        assert not np.allclose(Tn.rotation, T.rotation)

    def test_yaw_error_grows_with_range(self):
        """Why yaw dominates: lateral error is range * sin(dyaw)."""
        dyaw = np.deg2rad(2.0)
        R = Quaternion(axis=[0, 0, 1], radians=dyaw).rotation_matrix
        for r in (10.0, 50.0):
            p = R @ np.array([r, 0.0, 0.0])
            assert np.linalg.norm(p - np.array([r, 0.0, 0.0])) == \
                pytest.approx(2 * r * np.sin(dyaw / 2), rel=1e-6)
