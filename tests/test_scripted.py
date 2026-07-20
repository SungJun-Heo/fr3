"""Scripted oracle contracts -- offline (no sim step, no GL).

Two things are worth pinning here because both fail SILENTLY rather than loudly:

  * the geometry derived from a task's object declarations (a wrong symmetry
    grips a box across its long side and the object squirts out; a wrong height
    grasps air), and
  * the skill state machine's failure paths (a skill that never reports failure
    would have its bad episodes recorded as demonstrations).
"""

import unittest

import numpy as np

from tests.helpers import fake_gripper, fake_state
from types import SimpleNamespace

from robot.gripper import MAX_WIDTH
from scene.shapes import (grasp_span, half_height, object_spec, rim_z,
                          symmetry_of)
from scene.tasks import TASKS
from scripted import (Check, Hold, JointJitter, Servo, Shove, Skill,
                      available, make_skill)
from scripted.base import HAND_YAW_OFFSET, fold
from scripted.generate import _classify, _source
from scripted.run import success_view, ticks_for
from collection import schema

# fr3_joint7 range from models/fr3_with_gripper/fr3_with_gripper.xml
JOINT7_LIMIT = 3.0159


def fake_world(task="pick_cube"):
    """Minimal stand-in for a ControlSession: enough for phases that do not
    read object poses."""
    return SimpleNamespace(
        task_name=task,
        robot=SimpleNamespace(read_once=lambda: fake_state()),
        gripper=SimpleNamespace(read_once=lambda: fake_gripper()),
    )


class TestSymmetry(unittest.TestCase):
    def test_square_box_is_four_fold(self):
        spec = dict(kind="box", size=[0.02, 0.02, 0.02])
        self.assertAlmostEqual(symmetry_of(spec), np.pi / 2)

    def test_rectangular_box_is_two_fold(self):
        """A non-square box must NOT fold at 90 deg: the fingers would end up
        across its long side and the grasp would slip."""
        spec = dict(kind="box", size=[0.015, 0.03, 0.02])
        self.assertAlmostEqual(symmetry_of(spec), np.pi)

    def test_round_shapes_are_continuous(self):
        for kind, size in (("cylinder", [0.015, 0.04]),
                           ("sphere", [0.02]),
                           ("capsule", [0.01, 0.03]),
                           ("ellipsoid", [0.02, 0.02, 0.03])):
            self.assertIsNone(symmetry_of(dict(kind=kind, size=size)),
                              f"{kind} should have continuous symmetry")

    def test_non_round_ellipsoid_is_two_fold(self):
        self.assertAlmostEqual(
            symmetry_of(dict(kind="ellipsoid", size=[0.01, 0.02, 0.03])), np.pi)


class TestFold(unittest.TestCase):
    def test_folds_into_half_period(self):
        for period in (np.pi / 2, np.pi, 2 * np.pi):
            for yaw in np.linspace(-np.pi, np.pi, 361):
                f = fold(yaw, period)
                self.assertLessEqual(abs(f), period / 2 + 1e-9,
                                     f"yaw={yaw} period={period}")

    def test_fold_preserves_the_grasp(self):
        """Folding may only shift yaw by whole periods -- otherwise it is a
        different grasp, not an equivalent one."""
        period = np.pi / 2
        for yaw in np.linspace(-np.pi, np.pi, 181):
            k = (yaw - fold(yaw, period)) / period
            self.assertAlmostEqual(k, round(k), places=9)

    def test_square_box_grasp_stays_inside_joint7(self):
        """The whole point of folding: the commanded wrist yaw (folded angle
        plus the hand's 45 deg mount offset) must stay well inside fr3_joint7's
        range for EVERY randomized object yaw."""
        for yaw in np.linspace(-np.pi, np.pi, 721):
            commanded = fold(yaw, np.pi / 2) + HAND_YAW_OFFSET
            self.assertLess(abs(commanded), JOINT7_LIMIT)


class TestDerivedGeometry(unittest.TestCase):
    def test_half_height(self):
        self.assertAlmostEqual(half_height(dict(kind="box", size=[0.02] * 3)), 0.02)
        self.assertAlmostEqual(
            half_height(dict(kind="cylinder", size=[0.015, 0.04])), 0.04)
        self.assertAlmostEqual(
            half_height(dict(kind="capsule", size=[0.01, 0.03])), 0.04)

    def test_grasp_span_uses_the_narrow_axis(self):
        self.assertAlmostEqual(
            grasp_span(dict(kind="box", size=[0.015, 0.03, 0.02])), 0.03)
        self.assertAlmostEqual(
            grasp_span(dict(kind="cylinder", size=[0.015, 0.04])), 0.03)

    def test_bin_rim_is_twice_the_declared_height(self):
        """``add_bin``'s ``height`` is a half-extent and its walls are centred at
        that height, so the rim is at 2*height -- releasing at ``height`` would
        drag the carried object through the wall."""
        self.assertAlmostEqual(rim_z(dict(kind="bin", pos=[0.6, 0.15, 0.0])), 0.10)

    def test_rim_z_rejects_non_containers(self):
        with self.assertRaises(ValueError):
            rim_z(dict(kind="box", pos=[0, 0, 0], size=[0.02] * 3))


class TestShippedTasks(unittest.TestCase):
    """Guards that a newly added task stays inside what the oracles can do."""

    def test_every_oracle_task_builds(self):
        for task in available():
            self.assertTrue(make_skill(task).phases, f"{task} has no phases")

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(KeyError):
            make_skill("no_such_task")

    def test_declared_objects_fit_the_gripper(self):
        for task in available():
            for obj in TASKS[task]["objects"]:
                if obj["kind"] == "bin":
                    continue            # a fixture, never grasped
                spec = object_spec(task, obj["name"])
                self.assertLess(grasp_span(spec), MAX_WIDTH,
                                f"{task}/{obj['name']} is too wide to grasp")
                self.assertGreater(half_height(spec), 0.0)


class TestSkillSequencing(unittest.TestCase):
    def test_phases_run_in_order_then_finish(self):
        world = fake_world()
        skill = Skill([Hold(grip=0.0, ticks=2, name="a"),
                       Hold(grip=1.0, ticks=2, name="b")])
        seen = []
        for _ in range(6):
            name = skill.phase_name       # read BEFORE stepping: a step that
            if skill.step(world) is None:  # completes a phase advances the index
                break
            seen.append(name)
        self.assertEqual(seen[0], "a")
        self.assertEqual(seen[-1], "b")
        self.assertTrue(skill.done)
        self.assertIsNone(skill.failed)

    def test_gripper_command_persists_across_phases(self):
        """A phase that leaves ``grip`` unset must not reopen the gripper --
        that would drop whatever is being carried."""
        world = fake_world()
        skill = Skill([Hold(grip=0.0, ticks=1), Servo(lambda c: np.zeros(3))])
        skill.step(world)                       # closes
        _, _, grip = skill.step(world)          # servo leaves grip=None
        self.assertEqual(grip, 0.0)

    def test_failed_check_aborts_the_skill(self):
        world = fake_world()
        skill = Skill([Check(lambda c: False, reason="nope"),
                       Hold(grip=1.0, ticks=1, name="never")])
        skill.step(world)
        self.assertTrue(skill.done)
        self.assertIn("nope", skill.failed)

    def test_phase_timeout_is_reported(self):
        """The fake world never moves, so a Servo can never converge."""
        world = fake_world()
        skill = Skill([Servo(lambda c: np.array([9.0, 9.0, 9.0]),
                             timeout=5, name="stuck")])
        for _ in range(10):
            skill.step(world)
        self.assertIsNotNone(skill.failed)
        self.assertIn("timed out", skill.failed)


def fake_robot():
    """Just the two fields ``Shove`` touches."""
    return SimpleNamespace(_hand_body=0,
                           data=SimpleNamespace(xfrc_applied=np.zeros((1, 6))))


class TestShove(unittest.TestCase):
    def test_push_is_held_for_consecutive_ticks(self):
        """A one-tick impulse is absorbed by the position servo and never moves
        the arm, so the wrench must persist across ticks."""
        shove = Shove(displacement=0.03, prob=1.0, rng=np.random.default_rng(0))
        r = fake_robot()
        seen = []
        for _ in range(shove.ticks):
            shove.apply(r)
            seen.append(r.data.xfrc_applied[0, :3].copy())
        for w in seen[1:]:
            np.testing.assert_allclose(w, seen[0])
        self.assertEqual(shove.count, 1)
        self.assertAlmostEqual(float(np.linalg.norm(seen[0])), shove.force)

    def test_wrench_is_cleared_when_the_pulse_ends(self):
        """A wrench left set would keep pushing for the rest of the episode."""
        shove = Shove(displacement=0.01, prob=1.0, rng=np.random.default_rng(0))
        r = fake_robot()
        for _ in range(shove.ticks):
            shove.apply(r)
        shove.prob = 0.0
        shove.apply(r)
        np.testing.assert_allclose(r.data.xfrc_applied[0], np.zeros(6))

    def test_inactive_clears_the_wrench(self):
        shove = Shove(displacement=0.03, prob=1.0, rng=np.random.default_rng(0))
        r = fake_robot()
        shove.apply(r)
        shove.apply(r, active=False)
        np.testing.assert_allclose(r.data.xfrc_applied[0], np.zeros(6))

    def test_disabled_when_probability_is_zero(self):
        shove = Shove(prob=0.0, rng=np.random.default_rng(0))
        r = fake_robot()
        shove.apply(r)
        np.testing.assert_allclose(r.data.xfrc_applied[0], np.zeros(6))
        self.assertEqual(shove.count, 0)

    def test_jitter_moves_joints_and_respects_limits(self):
        """A jitter must never command a configuration the arm cannot hold."""
        r = SimpleNamespace(
            _qadr=np.arange(7),
            _q_min=np.full(7, -0.1), _q_max=np.full(7, 0.1),
            data=SimpleNamespace(qpos=np.zeros(7)))
        j = JointJitter(sigma=5.0, prob=1.0, ticks=4,   # absurdly large on purpose
                        rng=np.random.default_rng(0))
        for _ in range(4):
            j.apply(r)
        self.assertEqual(j.count, 1)
        self.assertTrue(np.any(r.data.qpos != 0.0), "joints did not move")
        self.assertTrue(np.all(r.data.qpos >= r._q_min - 1e-12))
        self.assertTrue(np.all(r.data.qpos <= r._q_max + 1e-12))

    def test_jitter_is_spread_over_ticks(self):
        """One teleport would put a discontinuity in q while dq reports
        nothing -- a state no real robot can produce."""
        r = SimpleNamespace(_qadr=np.arange(7),
                            _q_min=np.full(7, -9.0), _q_max=np.full(7, 9.0),
                            data=SimpleNamespace(qpos=np.zeros(7)))
        j = JointJitter(sigma=0.07, prob=1.0, ticks=7,
                        rng=np.random.default_rng(1))
        j.apply(r)
        after_one = r.data.qpos.copy()
        for _ in range(6):
            j.apply(r)
        # each tick contributes an equal share, so the total is ticks x the first
        np.testing.assert_allclose(r.data.qpos, after_one * 7, rtol=1e-9)

    def test_jitter_off_when_inactive(self):
        r = SimpleNamespace(_qadr=np.arange(7),
                            _q_min=np.full(7, -9.0), _q_max=np.full(7, 9.0),
                            data=SimpleNamespace(qpos=np.zeros(7)))
        j = JointJitter(sigma=0.05, prob=1.0, rng=np.random.default_rng(0))
        j.apply(r, active=False)
        np.testing.assert_allclose(r.data.qpos, np.zeros(7))
        self.assertEqual(j.count, 0)

    def test_longer_displacement_means_a_longer_pulse(self):
        """The tuned quantity is pulse length -- force is fixed above the
        servo's break-away, where force/deflection is a near step."""
        self.assertLess(ticks_for(0.005), ticks_for(0.03))
        self.assertLess(ticks_for(0.03), ticks_for(0.10))
        self.assertGreaterEqual(ticks_for(0.0), 1)


class TestSuccessView(unittest.TestCase):
    def test_adapts_a_session(self):
        view = success_view(fake_world("stack_blocks"))
        self.assertEqual(view.task, "stack_blocks")
        self.assertFalse(view.is_grasped())

    def test_passes_through_an_env(self):
        env = SimpleNamespace(task="pick_cube", robot=None,
                              is_grasped=lambda: True)
        self.assertIs(success_view(env), env)


if __name__ == "__main__":
    unittest.main()


class TestGenerationProvenance(unittest.TestCase):
    """What produced an episode must be recoverable from the episode itself.

    Added late is worthless: a dataset written without it cannot be split by
    origin afterwards, so "did the disturbed episodes help?" could only be
    answered by regenerating everything."""

    def test_source_records_seed_and_disturbance(self):
        src = _source("pick_cube", 12345, True, 10.0, 2.0, 1.0)
        self.assertEqual(src["kind"], "scripted_oracle")
        self.assertEqual(src["oracle"], "pick_cube")
        self.assertEqual(src["seed"], 12345)
        self.assertTrue(src["randomized"])
        self.assertEqual(src["disturbance"]["shove_mm"], 10.0)
        self.assertEqual(src["disturbance"]["jitter_deg"], 2.0)

    def test_no_disturbance_is_recorded_as_none(self):
        """Absent must be distinguishable from zero-valued, so a later split on
        "was this episode disturbed?" is unambiguous."""
        self.assertIsNone(_source("pick_cube", 1, True, 0.0, 0.0, 1.0)["disturbance"])

    def test_meta_carries_source_and_defaults_to_none(self):
        kw = dict(task="pick_cube", instruction="x", num_frames=3, success=True,
                  keep=True,
                  session_params=dict(fps=50.0, control_dt=0.02,
                                      sim_timestep=0.002, substeps=10),
                  camera_specs={}, object_qpos0=[],
                  robot_meta_dict=dict(joint_names=[], joint_limits={},
                                       ee_site="s", gripper={}, object_names=[]))
        self.assertIsNone(schema.build_meta(**kw)["source"])
        src = _source("pick_cube", 7, True, 0.0, 2.0, 1.0)
        self.assertEqual(schema.build_meta(source=src, **kw)["source"], src)


class TestFailureClassification(unittest.TestCase):
    """The discard reasons are the operator's only signal for WHY yield is low,
    so each must land in the bucket that suggests a different fix."""

    def test_buckets(self):
        cases = {
            "safety trip: near singularity (w=0.0200)": "safety trip",
            "phase 'plan:peg': no comfortable posture grasps peg":
                "rejected before moving",
            "phase 'descend:cube' timed out after 401 ticks": "timeout",
            "exceeded 1500 ticks": "timeout",
            "task criterion not met": "finished but criterion not met",
        }
        for reason, bucket in cases.items():
            self.assertEqual(_classify(reason), bucket, reason)
