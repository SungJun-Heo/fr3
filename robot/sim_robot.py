"""Sim robot -- state reading + joint-position control primitives.

``SimRobot`` loads a task scene and exposes the robot in the same shape
libfranka/pylibfranka does (see ``camel-franka``'s ``print_robot_state.py`` and
``_store_leftarm_state``). Alongside ``read_once()`` it provides an external
control loop -- ``start_joint_position_control()`` returns an ``ActiveControl``
driven by ``readOnce()``/``writeOnce()`` -- plus the real robot's safety
lifecycle (``set_collision_behavior``, ``automatic_error_recovery``, ``stop``)
including a collision reflex: after each control step the external force is
estimated from MuJoCo contacts, low-pass filtered, and compared to the
thresholds; exceeding them latches ``_has_error`` (motion refused until
``automatic_error_recovery``), mirroring the real robot tripping on impact.

Field/convention parity with pylibfranka's ``RobotState``:
  * ``q, dq``           -- 7 arm-joint positions / velocities, in joint order.
  * ``q_d``             -- controller's *desired* joint positions: the arm
                           actuators' current command target (``data.ctrl``),
                           i.e. what the position servos are tracking.
  * ``tau_J``           -- joint torques. We report the actuator-applied
                           generalized force, the closest sim analog to the
                           real arm's measured torque.
  * ``O_T_EE``          -- end-effector pose as a **column-major** 4x4 flattened
                           to length 16, so translation lives at indices
                           12,13,14 -- exactly how camel-franka indexes it.
  * ``O_T_EE_d``        -- desired EE pose; mirrors ``O_T_EE`` for now.
  * ``tau_ext_hat_filtered`` -- filtered external joint torque, estimated from
                           MuJoCo contact/constraint forces.
  * ``O_F_ext_hat_K``,  -- filtered external EE wrench (base frame). K-frame
    ``K_F_ext_hat_K``      copy is a skeleton stand-in (EE-frame rotation TODO).
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import mujoco

# Reuse the shared scene-composition code (the ``scene`` package) so the sim
# robot is built exactly like the viewer -- one source of truth for "what a
# task is".
sys.path.insert(0, str(Path(__file__).parent.parent))
from scene import build_task, initial_state
from robot.types import ControllerMode, Duration, JointPositions, CartesianPose
from controller.kinematics import DLSIKSolver

# Arm joints in kinematic order. The end-effector reference frame is the flange
# ``attachment_site``; note this is the flange, not yet the between-the-fingers
# TCP, so it differs from the real robot's ``F_T_EE`` offset -- a convention to
# reconcile later when we care about absolute EE poses.
ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
EE_SITE = "attachment_site"


@dataclass
class RobotState:
    """A snapshot of the arm, named to match pylibfranka's ``RobotState``."""
    q: np.ndarray            # (7,) joint positions
    dq: np.ndarray           # (7,) joint velocities
    q_d: np.ndarray          # (7,) desired joint positions
    tau_J: np.ndarray        # (7,) joint torques
    tau_ext_hat_filtered: np.ndarray  # (7,) filtered external joint torque
    O_T_EE: np.ndarray       # (16,) column-major 4x4 EE pose in base frame
    O_T_EE_d: np.ndarray     # (16,) desired EE pose
    O_F_ext_hat_K: np.ndarray  # (6,) external wrench, base frame
    K_F_ext_hat_K: np.ndarray  # (6,) external wrench, stiffness frame


class SimRobot:
    """MuJoCo-backed stand-in for ``pylibfranka.Robot`` (read-only for now)."""

    def __init__(self, task="empty"):
        # Single-arg construction mirrors ``Robot(ip)`` on the real side.
        self.model, self._object_names = build_task(task)
        self.data = initial_state(self.model, self._object_names)

        # Cache addresses once; a hinge joint occupies one qpos and one dof slot.
        self._qadr = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self._vadr = np.array([self.model.joint(n).dofadr[0] for n in ARM_JOINTS])
        self._act_arm = np.array([self.model.actuator(n).id for n in ARM_JOINTS])
        self._ee_site = self.model.site(EE_SITE).id

        # Safety lifecycle state (mirrors the real robot's reflex system).
        # set_collision_behavior stores the thresholds; _has_error is the
        # reflex/error latch a later safety step sets on over-force and
        # automatic_error_recovery() clears; _active is the running loop.
        self._collision_thresholds = None
        self._has_error = False
        self._error_reason = ""
        self._active = None
        # How Cartesian tracking handles an unsafe step: "trip" (default) faults
        # like the real robot; "clamp" instead brakes/limits and keeps going --
        # smooth for teleop, where a hard stop at a singularity feels like a
        # stutter. Set by start_cartesian_pose_control(safety=...).
        self._cart_safety = "trip"

        # External-force estimate (filled after each control step) and its
        # low-pass state. Filtering mirrors the real robot's *_hat_filtered
        # signals: a brief tap should not trip, a sustained push should.
        self._hand_body = self.model.body("hand").id
        self._ext_alpha = 0.1
        self._tau_ext_filt = np.zeros(7)
        self._O_F_ext_filt = np.zeros(6)

        # IK backend for Cartesian pose commands, plus the joint-limit and
        # singularity guards its output is checked against (fidelity: trip when
        # the real robot would fault, rather than silently doing something bad).
        self._ik = DLSIKSolver(self.model, EE_SITE, ARM_JOINTS, damping=0.05)
        jids = [self.model.joint(n).id for n in ARM_JOINTS]
        self._q_min = self.model.jnt_range[jids, 0].copy()
        self._q_max = self.model.jnt_range[jids, 1].copy()
        self._manip_min = 0.02  # Yoshikawa manipulability floor (near-singularity)

        # Free-joint addresses of the task objects, so they can be snapped back
        # to their declared start pose (qpos0, where scene.initial_state placed
        # them) on demand -- teleop knocks things out of reach and you want them
        # back without restarting. Static fixtures (no joint) are skipped.
        self._object_qadr = []
        self._object_vadr = []
        for name in self._object_names:
            body = self.model.body(name)
            if body.jntnum[0] > 0:
                jadr = body.jntadr[0]
                self._object_qadr.append(int(self.model.jnt_qposadr[jadr]))
                self._object_vadr.append(int(self.model.jnt_dofadr[jadr]))

    def read_once(self):
        """Return the current ``RobotState`` without advancing the sim."""
        # Recompute kinematics so site poses reflect the current qpos.
        mujoco.mj_forward(self.model, self.data)
        d = self.data
        q = d.qpos[self._qadr].copy()
        O_T_EE = self._ee_pose()
        return RobotState(
            q=q,
            dq=d.qvel[self._vadr].copy(),
            q_d=d.ctrl[self._act_arm].copy(),
            tau_J=d.qfrc_actuator[self._vadr].copy(),
            tau_ext_hat_filtered=self._tau_ext_filt.copy(),
            O_T_EE=O_T_EE,
            O_T_EE_d=O_T_EE.copy(),
            O_F_ext_hat_K=self._O_F_ext_filt.copy(),
            K_F_ext_hat_K=self._O_F_ext_filt.copy(),  # skeleton: EE-frame rot TODO
        )

    def _ee_pose(self):
        """EE pose as a column-major length-16 vector (libfranka's O_T_EE)."""
        d = self.data
        T = np.eye(4)
        T[:3, :3] = d.site_xmat[self._ee_site].reshape(3, 3)
        T[:3, 3] = d.site_xpos[self._ee_site]
        return T.flatten(order="F")  # column-major: translation at [12,13,14]

    # --- control loop --------------------------------------------------

    def start_joint_position_control(self, mode=ControllerMode.JointImpedance):
        """Begin an external joint-position control loop (mirrors libfranka's
        ``Robot.start*Control``). Drive the returned ``ActiveControl`` with
        ``readOnce()``/``writeOnce()``."""
        self._active = ActiveControl(self, mode)
        return self._active

    def start_cartesian_pose_control(self, mode=ControllerMode.CartesianImpedance,
                                     safety="trip"):
        """Begin an external Cartesian-pose control loop. Stream ``CartesianPose``
        commands via ``writeOnce``; each is tracked with one DLS IK step.

        On the real robot the firmware does this conversion; here the sim does
        it, so the same streaming code (VLA, teleop, a scripted path) runs on
        both.

        ``safety`` picks what an unsafe step does:
          * ``"trip"`` (default) -- fault like the real robot: NaN, a
            singularity, or a joint limit latches an error and ``writeOnce``
            raises (recover with ``automatic_error_recovery``).
          * ``"clamp"`` -- no fault: the DLS damping already brakes near
            singularities and the output is clipped to joint limits, so tracking
            just slows/limits smoothly. Meant for teleop, where a hard stop at a
            singularity reads as a stutter."""
        self._cart_safety = safety
        self._active = ActiveControl(self, mode)
        return self._active

    # --- safety lifecycle (real robot parity) --------------------------

    def set_collision_behavior(self, lower_torque_thresholds,
                               upper_torque_thresholds,
                               lower_force_thresholds,
                               upper_force_thresholds):
        """Store the contact/collision reflex thresholds (libfranka signature).

        Real: exceeding ``lower_*`` reports contact, exceeding ``upper_*`` trips
        a reflex and stops the robot. Here we only store them; the comparison is
        wired up in a later safety step."""
        self._collision_thresholds = dict(
            lower_torque=np.asarray(lower_torque_thresholds, float),
            upper_torque=np.asarray(upper_torque_thresholds, float),
            lower_force=np.asarray(lower_force_thresholds, float),
            upper_force=np.asarray(upper_force_thresholds, float),
        )

    def automatic_error_recovery(self):
        """Clear a tripped reflex/error so control can resume (mirrors
        libfranka; real acknowledges recoverable errors after a collision)."""
        self._has_error = False
        self._error_reason = ""

    def reset_objects(self):
        """Snap every task object back to its declared start pose (qpos0) and
        zero its velocity; the arm is left untouched. This is the sim-only
        escape hatch for teleop (no real-robot analog): when an object gets
        knocked out of reach, put it back without restarting."""
        for qadr in self._object_qadr:
            self.data.qpos[qadr:qadr + 7] = self.model.qpos0[qadr:qadr + 7]
        for vadr in self._object_vadr:
            self.data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset_home(self, q=None):
        """Snap the arm *instantly* to a joint configuration (default: the HOME
        keyframe), zeroing velocity and pointing the actuators at it so it holds
        there. Unlike a streamed quintic HOME this needs no control loop to play
        out -- a synchronous escape hatch for resetting between imitation-learning
        episodes. Also clears any latched reflex/error. Sim-only (no real-robot
        analog), like ``reset_objects``."""
        q = self.model.key_qpos[0][:7] if q is None else q
        q = np.asarray(q, dtype=float)
        self.data.qpos[self._qadr] = q
        self.data.qvel[self._vadr] = 0.0
        self.data.ctrl[self._act_arm] = q  # hold HOME, don't pull back
        self.automatic_error_recovery()
        mujoco.mj_forward(self.model, self.data)

    # --- collision reflex ----------------------------------------------

    def _update_external_estimate(self):
        """Refresh the filtered external-force estimate from MuJoCo contacts.

        Joint space uses the generalized constraint force (``qfrc_constraint``);
        Cartesian uses the net external spatial force on the hand
        (``cfrc_ext``), reordered from MuJoCo's [torque, force] to libfranka's
        [force, torque]. Both are exponentially low-pass filtered."""
        d = self.data
        a = self._ext_alpha
        tau_raw = d.qfrc_constraint[self._vadr]
        c = d.cfrc_ext[self._hand_body]
        wrench_raw = np.concatenate([c[3:6], c[0:3]])
        self._tau_ext_filt += a * (tau_raw - self._tau_ext_filt)
        self._O_F_ext_filt += a * (wrench_raw - self._O_F_ext_filt)

    def _check_collision(self):
        """Latch the reflex if a filtered estimate exceeds an upper threshold.

        No thresholds set (``set_collision_behavior`` not called) -> no check,
        so control code that skips setup still runs."""
        th = self._collision_thresholds
        if th is None or self._has_error:
            return
        tau = np.abs(self._tau_ext_filt)
        force = np.abs(self._O_F_ext_filt)
        if np.any(tau > th["upper_torque"]):
            j = int(np.argmax(tau - th["upper_torque"]))
            self._has_error = True
            self._error_reason = (f"joint{j+1} ext torque {tau[j]:.1f} > "
                                  f"{th['upper_torque'][j]:.1f} Nm")
        elif np.any(force > th["upper_force"]):
            i = int(np.argmax(force - th["upper_force"]))
            self._has_error = True
            self._error_reason = (f"EE wrench[{i}] {force[i]:.1f} > "
                                  f"{th['upper_force'][i]:.1f}")

    # --- cartesian control (IK backend + safety) -----------------------

    def _cartesian_to_ctrl(self, command):
        """Turn a ``CartesianPose`` into an arm ctrl target via one DLS IK step.

        In ``"trip"`` mode (default, fidelity): if the step would produce NaN,
        drive near a singularity, or exceed a joint limit, latch ``_has_error``
        and raise. In ``"clamp"`` mode (teleop): never fault -- hold on NaN, hold
        at the near-singularity floor (soft wall), else clip to joint limits, so
        tracking slows/holds smoothly instead of stopping or sagging."""
        T = np.asarray(command.O_T_EE, dtype=float).reshape(4, 4, order="F")
        dq, info = self._ik.velocity_step(self.data, T[:3, 3], T[:3, :3])
        q_target = self.data.qpos[self._qadr] + dq
        if self._cart_safety == "clamp":
            if not np.all(np.isfinite(q_target)):
                return self.data.ctrl[self._act_arm].copy()  # hold on NaN/inf
            # Soft wall: DLS damping alone still lets an unreachable target drag
            # the arm to full extension (w->0), where it sags below the command
            # and reads as "going limp". Once below the manipulability floor,
            # hold the last ctrl target for any step heading *deeper* in -- the
            # stiff servo then holds the last reachable pose. Steps that raise w
            # (retreating out) still pass, so the arm can always escape.
            w_next = self._ik.manipulability(q_target)
            if w_next < self._manip_min and w_next < info["manipulability"] - 1e-4:
                return self.data.ctrl[self._act_arm].copy()
            return np.clip(q_target, self._q_min, self._q_max)
        self._check_ik_safety(q_target, info)
        if self._has_error:
            raise RuntimeError(f"IK safety trip: {self._error_reason}")
        return np.clip(q_target, self._q_min, self._q_max)

    def _check_ik_safety(self, q_target, info):
        """Latch ``_has_error`` if a Cartesian-tracking step is unsafe."""
        if not np.all(np.isfinite(q_target)):
            self._has_error = True
            self._error_reason = "IK produced NaN/inf"
            return
        # Near-singularity: only trip if this step drives *deeper* into it. That
        # way the arm can still hold or escape a low-manipulability pose (as the
        # DLS brake intends) -- so recovery actually resumes control instead of
        # instantly re-tripping.
        w_now = info["manipulability"]
        w_next = self._ik.manipulability(q_target)
        if w_next < self._manip_min and w_next < w_now - 1e-4:
            self._has_error = True
            self._error_reason = (f"near singularity (w={w_next:.4f} "
                                  f"< {self._manip_min}, heading deeper)")
            return
        below = self._q_min - q_target
        above = q_target - self._q_max
        if np.any(below > 0) or np.any(above > 0):
            j = int(np.argmax(np.maximum(below, above)))
            self._has_error = True
            self._error_reason = (f"joint{j+1} would exceed limit "
                                  f"({np.degrees(q_target[j]):.1f} deg)")

    def stop(self):
        """End the running control loop (mirrors libfranka's ``Robot.stop``)."""
        if self._active is not None:
            self._active._finished = True


class ActiveControl:
    """External control-loop handle, mirroring libfranka's ``ActiveControl``.

    Real libfranka runs the arm at 1 kHz and ``readOnce()`` blocks until the
    next tick, the robot advancing in real time. Sim has no real time, so the
    step is explicit: ``writeOnce`` applies the command and advances the sim by
    one control period. One readOnce/writeOnce pair == one sim step.
    """

    def __init__(self, robot, mode):
        self._robot = robot
        self.mode = mode          # ControllerMode -- stored for parity
        self._finished = False

    def readOnce(self):
        """Return ``(RobotState, Duration)`` for the current tick; no stepping."""
        state = self._robot.read_once()
        dt = Duration(self._robot.model.opt.timestep)
        return state, dt

    def writeOnce(self, command):
        """Apply a command and advance the sim one step.

        ``JointPositions`` writes the targets straight to the actuators;
        ``CartesianPose`` is first converted to a joint target by one DLS IK
        step (with joint-limit/singularity safety). One writeOnce == one step."""
        robot = self._robot
        if robot._has_error:
            raise RuntimeError(
                "robot in reflex/error state; call automatic_error_recovery()")
        if isinstance(command, JointPositions):
            q_target = np.asarray(command.q, dtype=float)
        elif isinstance(command, CartesianPose):
            q_target = robot._cartesian_to_ctrl(command)  # IK + safety; may raise
        else:
            raise TypeError(f"unsupported command: {type(command).__name__}")
        # Write the 7 arm targets; leave the gripper ctrl slot untouched.
        robot.data.ctrl[robot._act_arm] = q_target
        mujoco.mj_step(robot.model, robot.data)
        # After stepping, refresh the external-force estimate and trip the
        # reflex if it exceeds the collision thresholds.
        robot._update_external_estimate()
        robot._check_collision()
        if getattr(command, "motion_finished", False):
            self._finished = True
