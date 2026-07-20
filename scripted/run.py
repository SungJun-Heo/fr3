"""Drive a scripted skill through a ControlSession (or SimEnv) tick loop.

``run_skill`` is the seam everything else plugs into:

  * ``on_tick``  -- where a ``Collector`` goes, so the autonomous data-generation
    loop is this function plus a randomize/keep/discard wrapper around it.
  * ``shove``    -- the recovery-data generator. See ``Shove``.

It also owns the two ways an episode can be worthless and must be discarded: a
safety trip (collision reflex / IK singularity / joint limit -- motions the real
robot would fault on, so recording them poisons the dataset) and a skill abort
(unreachable pose, phase timeout).
"""

from dataclasses import dataclass

import numpy as np

from rollout.success import task_success


@dataclass
class Result:
    """Outcome of one scripted episode."""
    success: bool
    ticks: int
    reason: str = ""        # why it failed; empty on success
    shoves: int = 0         # perturbations injected (recovery segments recorded)


class _SuccessView:
    """Adapt a ``ControlSession`` to what ``rollout.success`` expects.

    The success checks were written against ``SimEnv`` (``.task``, ``.robot``,
    ``.is_grasped()``); a session spells the first and third differently. One
    adapter here beats duplicating the criteria."""

    def __init__(self, session):
        self.task = session.task_name
        self.robot = session.robot
        self._gripper = session.gripper

    def is_grasped(self):
        return bool(self._gripper.read_once().is_grasped)


def success_view(world):
    """``world`` itself if it already satisfies the success interface, else an
    adapter. Lets the same oracle be scored on a session or a ``SimEnv``."""
    if hasattr(world, "task") and hasattr(world, "is_grasped"):
        return world
    return _SuccessView(world)


# -- disturbance calibration -------------------------------------------------
#
# The arm is a stiff position servo (kp=2000, kv=200), which defeats the two
# obvious disturbance channels: below roughly 150 N an external push barely
# moves it (100 N shifts the flange <0.1 mm), and a joint-velocity impulse is
# damped out within a few ticks (a kick sized for 40 mm yields 1.6 mm). Above
# that knee the joint actuators saturate and the arm gives way -- but the
# force/deflection curve there is a near step, useless as a dial.
#
# So the magnitude is FIXED just above the knee and the pulse LENGTH is the
# tuned quantity, which IS close to linear (measured on models/fr3_with_gripper
# at 180 N, deflection of the flange from its commanded pose):
#
#     ticks    2      4      6     10     15
#     mm     4.3   15.0   30.5   68.3  105.1
#
# Re-measure if the arm, its gains, or the control rate change.
SHOVE_FORCE = 180.0        # N, above the servo's break-away
_MM_PER_TICK = 7.2         # slope of the fit above
_MM_INTERCEPT = 10.0


def ticks_for(displacement):
    """Pulse length (control ticks) that displaces the flange by ``displacement``
    metres, from the calibration above."""
    mm = max(0.0, float(displacement)) * 1000.0
    return int(np.clip(round((mm + _MM_INTERCEPT) / _MM_PER_TICK), 1, 40))


class Shove:
    """Occasionally push the arm off course with an EXTERNAL FORCE.

    This is what turns a scripted oracle into a source of RECOVERY data. A
    policy trained only on perfect trajectories has never seen the state it
    lands in after its own first small error, so it has nothing to do there. By
    displacing the arm and letting the (closed-loop) oracle correct, the
    recorded action during the correction IS the recovery behaviour.

    The disturbance is applied to the PHYSICS (MuJoCo's ``xfrc_applied`` on the
    hand body), never to the command. That distinction is the whole point. The
    recorder logs the commanded pose as the action label (``O_T_EE_d``), so
    perturbing the command would write the disturbance itself into the label --
    teaching a policy to emit a random displacement in a state that looks
    perfectly normal. Pushing the robot instead leaves the command as the
    oracle's pure intent, so the label during a shove is the CORRECTION. (This
    is the split noise-injected DAgger / DART rely on: noise in execution,
    supervision from the expert.)

    ``displacement`` is how far (metres) the flange should be pushed off its
    commanded pose; it sets the pulse length via the calibration above. The push
    is held for those consecutive ticks rather than resampled every tick: a
    one-tick impulse is absorbed by the position servo and never actually moves
    the arm, so it would teach nothing."""

    # A force is transmitted through contacts, so an object in the gripper comes
    # along with the hand. Safe to apply at any time.
    while_grasped = True

    def __init__(self, displacement=0.02, prob=0.01, force=SHOVE_FORCE, rng=None):
        self.displacement = float(displacement)
        self.prob = float(prob)
        self.force = float(force)
        self.ticks = ticks_for(displacement)
        self.rng = np.random.default_rng() if rng is None else rng
        self.count = 0
        self._left = 0
        self._dir = np.zeros(3)

    def reset(self):
        self.count = 0
        self._left = 0
        self._dir = np.zeros(3)

    def apply(self, robot, active=True):
        """Set (or clear) this tick's external wrench on the hand.

        Call once per control tick BEFORE stepping: ``xfrc_applied`` persists,
        so it acts over every substep of the tick and must be cleared again."""
        if not active:
            return self.clear(robot)
        if self._left <= 0 and self.rng.random() < self.prob:
            d = self.rng.normal(0.0, 1.0, 3)
            n = float(np.linalg.norm(d))
            # Fixed magnitude, random direction: the slider then means one
            # thing, instead of a magnitude that varies run to run.
            self._dir = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
            self._left = self.ticks
            self.count += 1
        if self._left <= 0:
            return self.clear(robot)
        self._left -= 1
        robot.data.xfrc_applied[robot._hand_body, :3] = self._dir * self.force
        robot.data.xfrc_applied[robot._hand_body, 3:] = 0.0

    def clear(self, robot):
        robot.data.xfrc_applied[robot._hand_body] = 0.0


class JointJitter:
    """Push the arm off its nominal CONFIGURATION by displacing the joints.

    A complement to ``Shove``, not a replacement -- the two perturb different
    subspaces. Pushing the flange moves the arm in task space while leaving the
    elbow near its nominal posture; jittering the joints reaches configurations
    a Cartesian push never visits. That matters here specifically: every hard
    failure in this oracle came from POSTURE (a descent stranding the arm near a
    singularity, which is why ``Plan``/``Transit`` exist), so a policy that has
    only ever seen nominal postures is fragile exactly where this arm is.

    Like ``Shove`` it perturbs the physics and never the command, so the
    recorded action stays the oracle's own intent -- the label during a jitter
    is the correction.

    The offset is spread over ``ticks`` rather than applied as one jump: a
    single teleport puts a discontinuity in the recorded ``q`` while ``dq``
    reports nothing, which is a state no real robot can produce. Spread thin
    (sigma/ticks per tick) the residual inconsistency is far below normal
    motion. Targets are clamped to the joint limits, so a jitter can never
    command a configuration the arm could not hold.

    NOT applied while the gripper holds something (``while_grasped``). Writing
    joint positions displaces the ARM, but an object held only by contact does
    not come along -- it is left behind and squeezed out through the fingers.
    Measured: a 2 deg jitter during a carry jumps the cube 4-10 mm in one tick
    and every episode that took one failed, against none that did not. Free
    space is also where posture diversity is worth having, so the restriction
    costs nothing."""

    while_grasped = False

    def __init__(self, sigma=0.02, prob=0.01, ticks=10, rng=None):
        self.sigma = float(sigma)          # rad, per joint, of the total offset
        self.prob = float(prob)
        self.ticks = int(ticks)
        self.rng = np.random.default_rng() if rng is None else rng
        self.count = 0
        self._left = 0
        self._step = np.zeros(7)

    def reset(self):
        self.count = 0
        self._left = 0
        self._step = np.zeros(7)

    def apply(self, robot, active=True):
        """Nudge the arm's joints for this tick. Call before stepping."""
        if not active:
            self._left = 0      # abandon the pulse; do not resume it later,
            return              # which would straddle a grasp
        if self._left <= 0 and self.rng.random() < self.prob:
            total = self.rng.normal(0.0, self.sigma, 7)
            self._step = total / max(self.ticks, 1)
            self._left = self.ticks
            self.count += 1
        if self._left <= 0:
            return
        self._left -= 1
        q = robot.data.qpos[robot._qadr] + self._step
        robot.data.qpos[robot._qadr] = np.clip(q, robot._q_min, robot._q_max)

    def clear(self, robot):
        """Nothing persists -- present so callers can treat both the same."""


class SkillRunner:
    """Drive a skill ONE control tick at a time.

    Stepwise rather than a loop because a GUI owns its own tick loop (tkinter's
    ``after()``) and cannot block inside one. ``run_skill`` is this class plus a
    while loop, so headless runs and the GUI button share one implementation --
    the alternative, two loops that must stay in sync about trips, settling and
    success, is exactly the kind of duplication that silently diverges.

    ``settle_ticks`` keeps streaming the final command after the skill finishes
    so physics resolves before success is judged: a dropped peg is still in the
    air the tick the skill ends, and a just-released block is still settling."""

    def __init__(self, session, skill, max_ticks=1500, settle_ticks=60,
                 shove=None, jitter=None):
        self.session = session
        self.skill = skill
        self.max_ticks = max_ticks
        self.settle_ticks = settle_ticks
        self.shove = shove
        self.jitter = jitter
        self.ticks = 0
        self.result = None
        self._view = success_view(session)
        self._cmd = None
        self._settling = False
        self._settle_left = 0
        for d in self._disturbances:
            d.reset()

    @property
    def _disturbances(self):
        return [d for d in (self.shove, self.jitter) if d is not None]

    @property
    def disturbance_count(self):
        """How many perturbations were injected -- each one a recovery segment."""
        return sum(d.count for d in self._disturbances)

    @property
    def active(self):
        return self.result is None

    @property
    def phase(self):
        """A short label for a status line."""
        if self.result is not None:
            return "done"
        return "settling" if self._settling else (self.skill.phase_name or "-")

    def tick(self):
        """Advance one control tick. Returns a ``Result`` once finished, else
        None. Steps the session itself, so the caller must not also step it."""
        if self.result is not None:
            return self.result
        if self.ticks >= self.max_ticks:
            return self._finish(False, f"exceeded {self.max_ticks} ticks")

        if not self._settling:
            step = self.skill.step(self.session)
            if self.skill.failed:
                return self._finish(False, self.skill.failed)
            if step is None:
                self._settling = True
                self._settle_left = self.settle_ticks
            else:
                self._cmd = step

        if self._cmd is not None:
            pos, R, grip = self._cmd
            self.session.set_task_target(pos, R)
            self.session.set_gripper_frac(grip)

        # Disturbances go into the physics, never the command above -- so the
        # pose just streamed (and recorded as the action) stays the oracle's
        # intent. Settling is left undisturbed so success is judged fairly, and
        # a disturbance that cannot survive a grasp is held off while carrying.
        if self._disturbances:
            grasped = self._view.is_grasped()
            for d in self._disturbances:
                ok = not self._settling and (d.while_grasped or not grasped)
                d.apply(self.session.robot, active=ok)

        if not self.session.step():
            return self._finish(False, "viewer closed")
        self.ticks += 1

        if self.session.trip:
            return self._finish(False, f"safety trip: {self.session.trip}")

        if self._settling:
            self._settle_left -= 1
            if self._settle_left <= 0:
                won = bool(task_success(self._view))
                return self._finish(won, "" if won else "task criterion not met")
        return None

    def _finish(self, success, reason):
        for d in self._disturbances:
            d.clear(self.session.robot)      # a wrench left set would persist
        self.result = Result(success, self.ticks, reason, self.disturbance_count)
        return self.result


def run_skill(session, skill, max_ticks=1500, settle_ticks=60, on_tick=None,
              shove=None, jitter=None):
    """Run ``skill`` to completion on ``session``. Returns a ``Result``."""
    runner = SkillRunner(session, skill, max_ticks, settle_ticks, shove, jitter)
    while True:
        result = runner.tick()
        if result is not None:
            return result
        if on_tick is not None:
            on_tick(session)
