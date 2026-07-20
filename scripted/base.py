"""Closed-loop scripted oracles -- the two primitives every skill is built from.

An *oracle* is a hand-written policy that solves a task using PRIVILEGED state
(the ground-truth object poses only the sim knows). It exists to mass-produce
demonstrations without a human or a trained model.

The design point is that it is **closed-loop**. A phase does not pre-compute a
trajectory; it recomputes its goal from the live world every tick and servos the
flange toward it at a bounded speed. Two things fall out of that:

  * **Recovery data.** Shove the arm off course mid-episode (``run.Shove``) and
    the next tick simply servos back from wherever it ended up -- the recorded
    action IS the correction. A policy trained only on perfect trajectories has
    never seen the off-nominal state it lands in after its own first small
    error, which is how imitation policies actually fail.
  * **Moving targets.** An object nudged by the arm is tracked, not missed.

A skill is a list of phases run in order. Phases are task-agnostic; ``macros.py``
composes them into reusable chunks and ``skills.py`` maps tasks to those chunks,
so adding a task means adding a table entry, not a new state machine.

A skill is a pure *planner*: ``step(world)`` returns ``(pos, R, grip)`` and never
touches the controller. The caller applies it -- ``ControlSession`` via
``set_task_target``/``set_gripper_frac``, ``SimEnv`` via a ``cartesian`` action --
so the same oracle drives both the collection and the rollout stack.

``world`` is anything exposing ``.robot``, ``.gripper``, and a task name
(``.task_name`` on ``ControlSession``, ``.task`` on ``SimEnv``).
"""

import math

import numpy as np

from controller.planning import QuinticTrajectoryGenerator
from robot import vec_to_pose
from scene.shapes import object_spec, symmetry_of
from teleop.clutch import slerp_toward

# -- robot geometry (the single home for these constants) --------------------

# ``attachment_site`` (the pose O_T_EE reports) sits at the FLANGE, not between
# the fingertips. The finger pads span 0.0944..0.1114 m below it, so their centre
# is this far down the tool z-axis: grasping an object whose centre is at height
# z means commanding the flange to z + FLANGE_TO_PAD.
FLANGE_TO_PAD = 0.1029

# The ``hand`` body is mounted rotated -45 deg about z relative to the site
# (models/fr3_with_gripper/fr3_with_gripper.xml). The fingers slide along the
# HAND's y-axis, so a commanded site yaw of phi puts the finger axis at
# phi - 45 deg in the world. Add this back to aim the fingers where we mean.
HAND_YAW_OFFSET = math.pi / 4

# Top-down grasp: flange z-axis pointing at the table (180 deg about world x).
R_DOWN = np.array([[1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0]])

CONTROL_DT = 0.02        # one control tick (matches ControlSession.TICK_MS)

# How far ahead of the measured pose a streamed command may get (m). Big enough
# that the tracking error drives full commanded speed, small enough that the
# reference cannot abandon an arm that has been pushed off course.
MAX_LEAD = 0.02

# Yoshikawa manipulability a grasp pose must clear before the oracle commits to
# it, just above SimRobot's 0.02 trip floor. See ``Ctx.reachable``.
MANIP_MARGIN = 0.022


def fold(yaw, period):
    """Fold ``yaw`` into [-period/2, period/2] -- the equivalent grasp angle that
    turns the wrist least.

    A square box looks the same every 90 deg, so gripping a box at yaw=170 deg
    and at yaw=-10 deg are the SAME physical grasp; commanding 170 deg (plus the
    45 deg hand offset) would drive fr3_joint7 past its +-172.8 deg limit and
    trip the episode, while -10 deg has room to spare."""
    return yaw - period * np.round(yaw / period)


def yaw_about_z(R, yaw):
    """``R`` pre-rotated by ``yaw`` radians about the world z-axis."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R


def grasp_R(yaw=0.0):
    """Top-down grasp orientation whose FINGER AXIS lies at ``yaw`` in the world
    (the hand's -45 deg mount is compensated here, once)."""
    return yaw_about_z(R_DOWN, yaw + HAND_YAW_OFFSET)


# -- live world view ---------------------------------------------------------

class Ctx:
    """What a phase sees. Everything reads the sim FRESH -- that is what makes
    the oracle closed-loop. ``t`` counts ticks within the current phase."""

    def __init__(self, world):
        self.world = world
        self.task = getattr(world, "task_name", None) or getattr(world, "task", None)
        self.t = 0
        self.scratch = {}       # phase-to-phase handoff (e.g. the grasp posture)
        self._ee = None

    def begin_tick(self):
        """Drop the per-tick cache. ``read_once`` recomputes kinematics, and a
        phase asks for the EE pose several times per tick (target, done, goal),
        so caching it within a tick is worth the two lines."""
        self._ee = None

    @property
    def robot(self):
        return self.world.robot

    @property
    def ee(self):
        """Current flange ``(position, rotation)``."""
        if self._ee is None:
            self._ee = vec_to_pose(self.robot.read_once().O_T_EE)
        return self._ee

    @property
    def ee_pos(self):
        return self.ee[0]

    def obj(self, name):
        """Ground-truth world position of a task body (privileged)."""
        return self.robot.data.xpos[self.robot.model.body(name).id].copy()

    def obj_yaw(self, name):
        """Ground-truth yaw of a task body, folded by its own symmetry. Returns
        0.0 for continuously symmetric shapes (a cylinder has no meaningful yaw,
        so chasing the one MuJoCo reports would turn the wrist for nothing)."""
        period = symmetry_of(object_spec(self.task, name))
        if period is None:
            return 0.0
        R = self.robot.data.xmat[self.robot.model.body(name).id].reshape(3, 3)
        return float(fold(math.atan2(R[1, 0], R[0, 0]), period))

    def spec(self, name):
        return object_spec(self.task, name)

    @property
    def grasped(self):
        return bool(self.world.gripper.read_once().is_grasped)

    def reachable(self, pos, R, margin=MANIP_MARGIN):
        """Can the arm reach this pose without nearing a singularity?

        Reuses the SAME IK and manipulability floor the safety trip uses
        (``SimRobot._manip_min`` = 0.02), with a small margin -- so a pose that
        passes here will not trip mid-motion.

        The margin is deliberately thin. Over the tasks' randomization ranges,
        manipulability at the grasp pose runs 0.024 (near edge, x=0.42) to 0.086
        (far edge), so a margin of 0.03 would reject the entire near band even
        though every pose there is solvable and above the trip floor."""
        q, info = self.robot._ik.solve(np.asarray(pos, float), R,
                                       q_init=self.robot.read_once().q)
        return bool(info["converged"]) and self.robot._ik.manipulability(q) > margin


# -- phases ------------------------------------------------------------------

class Phase:
    """One step of a skill. ``target(ctx)`` runs every tick and returns the
    command ``(pos, R, grip)``; ``grip`` may be None to keep the previous one."""

    name = "phase"
    timeout = 400

    def enter(self, ctx):
        """Latch whatever must stay frozen for this phase's duration."""

    def target(self, ctx):
        raise NotImplementedError

    def done(self, ctx):
        return True


class Servo(Phase):
    """Drive the flange toward ``goal(ctx)`` at ``speed`` m/s; done within ``tol``.

    ``goal`` is a callable evaluated EVERY tick (that is the closed loop) unless
    ``latch``, in which case it is evaluated once on entry and held. Latching is
    required whenever the goal is relative to something the motion itself moves:
    lifting a grasped cube means "up from where I am now", not "up from the
    cube" -- the cube travels with the gripper, so that goal would be reached
    instantly and never rise.

    ``align_to`` names an object whose yaw the fingers should match. It is
    resolved once on entry, so a grasped object cannot drive the wrist in a
    feedback loop."""

    def __init__(self, goal, grip=None, tol=0.008, speed=0.25, latch=False,
                 align_to=None, max_lead=MAX_LEAD, timeout=400, name="servo"):
        self.goal = goal
        self.grip = grip
        self.tol = tol
        self.speed = speed
        self.latch = latch
        self.align_to = align_to
        self.max_lead = max_lead
        self.timeout = timeout
        self.name = name
        self._latched = None
        self._R_goal = R_DOWN
        self._R_cmd = R_DOWN
        self._cmd = None                 # the reference point being streamed

    def enter(self, ctx):
        self._latched = np.asarray(self.goal(ctx), float) if self.latch else None
        yaw = ctx.obj_yaw(self.align_to) if self.align_to else 0.0
        self._R_goal = grasp_R(yaw)
        pos, R = ctx.ee
        self._R_cmd = R                  # start from the current wrist, then slerp
        self._cmd = pos.copy()           # ...and from the current position

    def _goal_now(self, ctx):
        if self._latched is not None:
            return self._latched
        return np.asarray(self.goal(ctx), float)

    def target(self, ctx):
        """Advance the streamed reference one tick-step toward the goal.

        The reference advances on its OWN, not relative to the measured pose: a
        command placed a fixed step ahead of where the arm currently is can
        never be caught, so the arm settles at whatever speed closes that gap --
        far slower than asked. Streaming an independent reference makes ``speed``
        mean what it says.

        ``max_lead`` then caps how far ahead of the arm the reference may get.
        That is what keeps the loop closed: shove the arm off course and the
        reference cannot run away and abandon it, so the arm is pulled back
        toward a goal that is itself still being recomputed from the live scene.

        The command LEADS the measured pose, which is exactly the truthful
        action the recorder stores as ``O_T_EE_d``. Orientation is slerped so
        the wrist does not jerk when a phase starts with a large yaw change."""
        goal = self._goal_now(ctx)
        step = self.speed * CONTROL_DT

        d = goal - self._cmd
        n = float(np.linalg.norm(d))
        self._cmd = goal.copy() if n <= step else self._cmd + d * (step / n)

        lead = self._cmd - ctx.ee_pos
        L = float(np.linalg.norm(lead))
        if L > self.max_lead:
            self._cmd = ctx.ee_pos + lead * (self.max_lead / L)

        self._R_cmd = slerp_toward(self._R_cmd, self._R_goal, 0.15)
        return self._cmd.copy(), self._R_cmd, self.grip

    def done(self, ctx):
        return float(np.linalg.norm(self._goal_now(ctx) - ctx.ee_pos)) < self.tol


class Hold(Phase):
    """Hold the pose reached on entry for ``ticks`` while the gripper moves.

    Closing is not instantaneous and the arm must not drift during it, so the
    pose is latched rather than re-servoed. ``Gripper.set_target_width`` (which
    ``ControlSession.set_gripper_frac`` drives) is non-blocking, so the squeeze
    happens inside the recorded control loop."""

    def __init__(self, grip, ticks=30, name="hold"):
        self.grip = grip
        self.ticks = ticks
        self.name = name
        self.timeout = ticks + 10
        self._pos = None
        self._R = None

    def enter(self, ctx):
        self._pos, self._R = ctx.ee

    def target(self, ctx):
        return self._pos, self._R, self.grip

    def done(self, ctx):
        return ctx.t >= self.ticks


class Plan(Phase):
    """Zero-motion phase: solve the grasp posture ONCE and remember it.

    Which of the arm's many configurations reaches a pose is not decided by a
    Cartesian command -- the streaming DLS controller just takes local steps, so
    the posture it ends up in is whatever it drifted into. That matters because
    a descent can strand the arm in a near-singular configuration it cannot
    escape, even when a comfortable posture for the same pose exists (the
    one-shot IK finds it; local steps cannot reach it).

    So the posture is chosen here, at the pose that needs it most -- the grasp --
    and the approach is then planned to arrive ALREADY in that posture (see
    ``Transit``). Stored under ``key`` in ``ctx.scratch``."""

    def __init__(self, pose, key="q_goal", margin=MANIP_MARGIN, reason=None,
                 name="plan"):
        self.pose = pose            # (ctx) -> (pos, R)
        self.key = key
        self.margin = margin
        self.reason = reason or "no comfortable posture reaches the target"
        self.name = name
        self.timeout = 2
        self._failed = None

    def enter(self, ctx):
        pos, R = self.pose(ctx)
        q, info = ctx.robot._ik.solve(np.asarray(pos, float), R,
                                      q_init=ctx.robot.read_once().q)
        w = ctx.robot._ik.manipulability(q)
        if not info["converged"] or w <= self.margin:
            self._failed = (f"{self.reason} "
                            f"(converged={bool(info['converged'])}, w={w:.4f})")
            return
        self._failed = None
        ctx.scratch[self.key] = q

    def target(self, ctx):
        pos, R = ctx.ee
        return pos, R, None

    def done(self, ctx):
        if self._failed:
            raise SkillAborted(self._failed)
        return True


class Transit(Phase):
    """Free-space move planned in JOINT space, streamed as Cartesian poses.

    Free space has no path requirement, so it is planned where posture is
    controllable: a quintic joint interpolation from the current configuration
    to one that reaches ``goal``. Seeding that IK with ``from_key`` (the posture
    ``Plan`` chose for the grasp) makes the arm arrive in the SAME branch the
    following descent needs, so the descent is a small local motion instead of a
    posture change the controller cannot make.

    It is still streamed as Cartesian poses (FK of the interpolated joints) so
    the skill keeps its single command type and the recorded action stays
    comparable across phases."""

    def __init__(self, goal, grip=None, speed=0.25, align_to=None,
                 from_key=None, min_duration=0.8, tol=0.02, name="transit"):
        self.goal = goal
        self.grip = grip
        self.speed = speed
        self.align_to = align_to
        self.from_key = from_key
        self.min_duration = min_duration
        self.tol = tol
        self.name = name
        self.timeout = 400
        self._traj = None
        self._dur = 0.0
        self._t = 0.0
        self._failed = None

    def enter(self, ctx):
        goal = np.asarray(self.goal(ctx), float)
        yaw = ctx.obj_yaw(self.align_to) if self.align_to else 0.0
        R = grasp_R(yaw)
        seed = ctx.scratch.get(self.from_key) if self.from_key else None
        q_now = ctx.robot.read_once().q
        q_goal, info = ctx.robot._ik.solve(
            goal, R, q_init=(seed if seed is not None else q_now))
        if not info["converged"]:
            self._failed = f"no IK solution for the transit goal {np.round(goal, 3)}"
            return
        self._failed = None
        dist = float(np.linalg.norm(goal - ctx.ee_pos))
        self._dur = max(self.min_duration, dist / max(self.speed, 1e-6))
        self._traj = QuinticTrajectoryGenerator()
        self._traj.InitTrajectory(q_now, q_goal, 0.0, self._dur)
        self._t = 0.0
        self.timeout = int(self._dur / CONTROL_DT) + 120

    def target(self, ctx):
        if self._failed:
            pos, R = ctx.ee
            return pos, R, self.grip
        self._t = min(self._t + CONTROL_DT, self._dur)
        q = self._traj.getPositionTrajectory(self._t)
        pos, R = ctx.robot._ik.fk_pose(q)
        return pos, R, self.grip

    def done(self, ctx):
        if self._failed:
            raise SkillAborted(self._failed)
        if self._t < self._dur:
            return False
        # The joint plan has played out; wait for the arm to actually settle
        # there (the controller tracks with a lag).
        return float(np.linalg.norm(self.goal(ctx) - ctx.ee_pos)) < self.tol


class Check(Phase):
    """Zero-motion gate: abort the skill if ``ok(ctx)`` is False.

    Used before committing to a descent -- an unreachable grasp pose should end
    the episode immediately rather than after the arm has driven into a
    singularity and tripped."""

    def __init__(self, ok, reason="precondition failed", name="check"):
        self.ok = ok
        self.reason = reason
        self.name = name
        self.timeout = 2
        self._passed = None

    def enter(self, ctx):
        self._passed = bool(self.ok(ctx))

    def target(self, ctx):
        pos, R = ctx.ee
        return pos, R, None

    def done(self, ctx):
        if not self._passed:
            raise SkillAborted(self.reason)
        return True


class SkillAborted(Exception):
    """A ``Check`` phase decided the skill cannot proceed."""


# -- skill -------------------------------------------------------------------

class Skill:
    """A sequence of phases, stepped one control tick at a time.

    ``step(world)`` returns ``(pos, R, grip)`` for this tick, or None once
    finished. ``failed`` is set when a phase times out or a ``Check`` aborts
    (arm stuck, pose unreachable, object knocked out of range) -- the caller
    should DISCARD that episode rather than record a demonstration of failing."""

    def __init__(self, phases, name="skill"):
        self.phases = list(phases)
        self.name = name
        self.i = 0
        self.failed = None
        self._ctx = None
        self._grip = 1.0        # gripper starts open; phases may leave it unset

    @property
    def done(self):
        return self.i >= len(self.phases)

    @property
    def phase_name(self):
        return self.phases[self.i].name if not self.done else None

    def step(self, world):
        if self._ctx is None:
            self._ctx = Ctx(world)
            if self.phases:
                self.phases[0].enter(self._ctx)
        if self.done:
            return None

        ctx = self._ctx
        ctx.begin_tick()
        ph = self.phases[self.i]
        pos, R, grip = ph.target(ctx)
        if grip is not None:
            self._grip = grip
        ctx.t += 1

        try:
            finished = ph.done(ctx)
        except SkillAborted as e:
            self._abort(f"phase '{ph.name}': {e}")
            return pos, R, self._grip

        if finished:
            self._advance(ctx)
        elif ctx.t > ph.timeout:
            self._abort(f"phase '{ph.name}' timed out after {ctx.t} ticks")
        return pos, R, self._grip

    def _abort(self, reason):
        self.failed = reason
        self.i = len(self.phases)

    def _advance(self, ctx):
        self.i += 1
        ctx.t = 0
        if not self.done:
            self.phases[self.i].enter(ctx)
