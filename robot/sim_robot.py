"""Step 1 -- read-only state.

``SimRobot`` loads a task scene and exposes the robot's current state in the
same shape libfranka/pylibfranka does (see ``camel-franka``'s
``print_robot_state.py`` and ``_store_leftarm_state``). No control yet: this
step only proves we can pull MuJoCo state out in the real robot's vocabulary.

Field/convention parity with pylibfranka's ``RobotState``:
  * ``q, dq``           -- 7 arm-joint positions / velocities, in joint order.
  * ``q_d``             -- controller's *desired* joint positions. In sim there
                           is no separate desired yet, so we mirror ``q``; this
                           becomes meaningful once control loops set targets
                           (step 2).
  * ``tau_J``           -- joint torques. We report the actuator-applied
                           generalized force, the closest sim analog to the
                           real arm's measured torque.
  * ``O_T_EE``          -- end-effector pose as a **column-major** 4x4 flattened
                           to length 16, so translation lives at indices
                           12,13,14 -- exactly how camel-franka indexes it.
  * ``O_T_EE_d``        -- desired EE pose; mirrors ``O_T_EE`` for now.
  * ``O_F_ext_hat_K``,  -- external wrench estimates. Zero for now (sim force
    ``K_F_ext_hat_K``      estimation is a later step); kept so the recorder
                           schema matches real from the start.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import mujoco

# Reuse the existing scene-composition code so the sim robot is built the same
# way ``view_scene.py`` builds it -- one source of truth for "what a task is".
sys.path.insert(0, str(Path(__file__).parent.parent))
from view_scene import build_task, initial_state

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
        self._ee_site = self.model.site(EE_SITE).id

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
            q_d=q.copy(),
            tau_J=d.qfrc_actuator[self._vadr].copy(),
            O_T_EE=O_T_EE,
            O_T_EE_d=O_T_EE.copy(),
            O_F_ext_hat_K=np.zeros(6),
            K_F_ext_hat_K=np.zeros(6),
        )

    def _ee_pose(self):
        """EE pose as a column-major length-16 vector (libfranka's O_T_EE)."""
        d = self.data
        T = np.eye(4)
        T[:3, :3] = d.site_xmat[self._ee_site].reshape(3, 3)
        T[:3, 3] = d.site_xpos[self._ee_site]
        return T.flatten(order="F")  # column-major: translation at [12,13,14]
