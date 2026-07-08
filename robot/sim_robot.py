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
from robot.types import ControllerMode, Duration

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

        # External-force estimate (filled after each control step) and its
        # low-pass state. Filtering mirrors the real robot's *_hat_filtered
        # signals: a brief tap should not trip, a sustained push should.
        self._hand_body = self.model.body("hand").id
        self._ext_alpha = 0.1
        self._tau_ext_filt = np.zeros(7)
        self._O_F_ext_filt = np.zeros(6)

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
        """Apply a ``JointPositions`` command and advance the sim one step."""
        if self._robot._has_error:
            raise RuntimeError(
                "robot in reflex/error state; call automatic_error_recovery()")
        robot = self._robot
        # Write the 7 arm targets; leave the gripper ctrl slot untouched.
        robot.data.ctrl[robot._act_arm] = command.q
        mujoco.mj_step(robot.model, robot.data)
        # After stepping, refresh the external-force estimate and trip the
        # reflex if it exceeds the collision thresholds.
        robot._update_external_estimate()
        robot._check_collision()
        if getattr(command, "motion_finished", False):
            self._finished = True
