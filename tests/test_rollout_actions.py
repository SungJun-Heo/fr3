"""Rollout action surface: the rotation-delta helper, the 4 tagged-action decode
paths, delta integration onto the measured state, and that apply() advances +
returns a valid observation. Uses a real SimEnv with tiny cameras.
"""

import unittest

import numpy as np

from tests.helpers import make_env
from robot import vec_to_pose
from rollout.env import _delta_R
from collection import schema


class TestDeltaR(unittest.TestCase):
    def test_zero_is_identity(self):
        np.testing.assert_allclose(_delta_R([0, 0, 0]), np.eye(3), atol=1e-12)

    def test_matrix_passthrough(self):
        R = _delta_R([0, 0, 0.1])
        np.testing.assert_allclose(_delta_R(R), R)          # 3x3 in -> same out
        np.testing.assert_allclose(_delta_R(R.ravel()), R)  # len-9 accepted too

    def test_axis_angle_rotates(self):
        R = _delta_R([0, 0, np.pi / 2])                     # +90 deg about z
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-9)

    def test_result_is_a_rotation(self):
        R = _delta_R([0.2, -0.3, 0.1])
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)


class TestDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_env("pick_cube")
        cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_joint_absolute(self):
        q = np.linspace(0.0, 0.5, 7)
        cmd, grip = self.env._decode({"joint": np.concatenate([q, [0.5]])})
        np.testing.assert_allclose(cmd.q, q)
        self.assertAlmostEqual(grip, 0.5)

    def test_joint_delta_integrates_on_measured(self):
        dq = np.full(7, 0.01)
        q_ref = self.env.robot.read_once().q
        cmd, grip = self.env._decode({"joint_delta": np.concatenate([dq, [0.3]])})
        np.testing.assert_allclose(cmd.q, q_ref + dq)
        self.assertAlmostEqual(grip, 0.3)

    def test_cartesian_absolute(self):
        pose = self.env.robot.read_once().O_T_EE
        cmd, grip = self.env._decode({"cartesian": {"pose": pose, "gripper": 0.2}})
        np.testing.assert_allclose(cmd.O_T_EE, np.asarray(pose).ravel())
        self.assertAlmostEqual(grip, 0.2)

    def test_cartesian_delta_pos_and_rot(self):
        p_ref, R_ref = vec_to_pose(self.env.robot.read_once().O_T_EE)
        cmd, grip = self.env._decode({"cartesian_delta": {
            "dpos": [0.02, 0.0, 0.0], "drot": [0, 0, 0.1], "gripper": 0.4}})
        p_t, R_t = vec_to_pose(cmd.O_T_EE)
        np.testing.assert_allclose(p_t, p_ref + [0.02, 0, 0], atol=1e-9)
        # independent angle check of the base-frame rotation delta
        rel = R_t @ R_ref.T
        angle = np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1))
        self.assertAlmostEqual(float(angle), 0.1, places=6)
        self.assertAlmostEqual(grip, 0.4)

    def test_cartesian_delta_zero_holds_pose(self):
        pose = self.env.robot.read_once().O_T_EE
        cmd, _ = self.env._decode({"cartesian_delta": {"dpos": [0, 0, 0]}})
        np.testing.assert_allclose(cmd.O_T_EE, np.asarray(pose).ravel(), atol=1e-9)

    def test_unknown_tag_raises(self):
        with self.assertRaises(ValueError):
            self.env._decode({"bogus": [1, 2, 3]})

    def test_observe_proprio_matches_schema_home(self):
        keys = set(self.env.observe()["proprio"])
        expected = set(schema.observation_from_state(
            self.env.robot.read_once(), self.env.gripper.read_once()))
        self.assertEqual(keys, expected)


class TestApplyAdvances(unittest.TestCase):
    def test_apply_returns_obs_and_moves_arm(self):
        env = make_env("pick_cube")
        env.reset()
        try:
            q0 = env.robot.read_once().q.copy()
            for _ in range(3):
                obs = env.apply({"joint_delta": np.concatenate([np.full(7, 0.05), [0.5]])})
            self.assertEqual(set(obs), {"images", "proprio", "instruction"})
            self.assertEqual(obs["images"]["front"].shape, (32, 32, 3))
            moved = float(np.linalg.norm(env.robot.read_once().q - q0))
            self.assertGreater(moved, 1e-3)   # the delta actually drove the arm
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
