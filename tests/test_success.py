"""Per-task success detection (rollout/success.py). Stages ground-truth object
poses kinematically (no physics, no policy) and asserts the criterion. The
grasp-gated pick_cube patches is_grasped to isolate the height threshold from a
real grasp; stack/bin need only geometry + released, so they stage end to end.
"""

import unittest

from tests.helpers import make_env, place_objects


class TestSuccess(unittest.TestCase):
    def test_pick_cube_needs_grasp_and_height(self):
        env = make_env("pick_cube")
        env.reset()
        try:
            place_objects(env, {"cube": [0.5, 0.0, 0.20]})   # lifted high
            self.assertFalse(env.success())                   # ...but not grasped
            env.is_grasped = lambda: True                     # now "holding" it
            self.assertTrue(env.success())                    # grasped + lifted
            place_objects(env, {"cube": [0.5, 0.0, 0.02]})    # back on the table
            self.assertFalse(env.success())                   # grasped but not lifted
        finally:
            env.close()

    def test_stack_blocks(self):
        env = make_env("stack_blocks")
        env.reset()
        try:
            place_objects(env, {"block_b": [0.5, 0.0, 0.02],
                                "block_a": [0.5, 0.0, 0.06]})  # aligned, ~4cm up
            self.assertTrue(env.success())                     # released (open gripper)
            place_objects(env, {"block_a": [0.60, 0.0, 0.06]}) # slid off in xy
            self.assertFalse(env.success())
        finally:
            env.close()

    def test_bin_picking(self):
        env = make_env("bin_picking")
        env.reset()
        try:
            b = env.robot.data.xpos[env.robot.model.body("bin").id].copy()
            place_objects(env, {"peg": [b[0], b[1], 0.04]})    # inside, low
            self.assertTrue(env.success())
            place_objects(env, {"peg": [b[0] + 0.20, b[1], 0.04]})  # outside footprint
            self.assertFalse(env.success())
        finally:
            env.close()

    def test_unknown_task_is_false(self):
        env = make_env("empty")
        try:
            self.assertFalse(env.success())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
