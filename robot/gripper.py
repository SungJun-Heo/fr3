"""Franka Hand gripper, mirroring pylibfranka's ``Gripper``.

Same API the real gripper exposes (``homing`` / ``move`` / ``grasp`` / ``stop``
/ ``read_once`` -> ``GripperState``), backed by the shared MuJoCo model. Because
sim has a single step loop, the blocking calls (move/grasp/homing) step the sim
to completion themselves, holding the arm at its current command -- so scripted
"move arm, then grasp" code reads the same as on the real robot.

Sim conventions (measured from the model):
  * width  = finger_joint1 + finger_joint2  (0 .. 0.08 m).
  * ctrl   = width / max_width * 255         (the gripper actuator's range).
  * is_grasped = a finger is pressing an *external* object (contact force),
    which also survives after grasp() until the object is released/dropped.
"""

from dataclasses import dataclass

import numpy as np
import mujoco

from robot.types import Duration

MAX_WIDTH = 0.08          # m, both fingers fully open
GRASP_FORCE_MIN = 1.0     # N, finger-object contact above which is_grasped
_SETTLE_STEPS = 600       # sim steps a blocking gripper call runs (~1.2 s)


@dataclass
class GripperState:
    """Mirrors pylibfranka's ``GripperState``."""
    width: float          # current opening (m)
    max_width: float      # opening after homing (m)
    is_grasped: bool      # holding an object under force
    temperature: float    # deg C (constant in sim)
    time: Duration        # timestamp


class Gripper:
    """MuJoCo-backed stand-in for ``pylibfranka.Gripper``."""

    def __init__(self, robot):
        # Shares the sim robot's model/data (as Gripper(ip) shares the real
        # robot the arm connects to).
        self.model, self.data = robot.model, robot.data
        self._act = self.model.actuator("gripper").id
        self._fj1 = self.model.joint("finger_joint1").qposadr[0]
        self._fj2 = self.model.joint("finger_joint2").qposadr[0]
        finger = {self.model.body("left_finger").id, self.model.body("right_finger").id}
        self._finger_bodies = finger
        self._own_bodies = finger | {self.model.body("hand").id}
        self.max_width = MAX_WIDTH

    # -- state ---------------------------------------------------------

    def width(self):
        return float(self.data.qpos[self._fj1] + self.data.qpos[self._fj2])

    def _object_contact_force(self):
        """Total force where a finger touches something that is not the hand or
        the other finger (i.e. a grasped object)."""
        total = 0.0
        f = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if b1 in self._finger_bodies or b2 in self._finger_bodies:
                other = b2 if b1 in self._finger_bodies else b1
                if other not in self._own_bodies:
                    mujoco.mj_contactForce(self.model, self.data, i, f)
                    total += float(np.linalg.norm(f[:3]))
        return total

    def read_once(self):
        return GripperState(
            width=self.width(),
            max_width=self.max_width,
            is_grasped=self._object_contact_force() > GRASP_FORCE_MIN,
            temperature=25.0,
            time=Duration(float(self.data.time)),
        )

    # -- motion (blocking, like the real gripper) ----------------------

    def _drive_to(self, width):
        """Command a target width and step the sim until it settles."""
        self.data.ctrl[self._act] = float(np.clip(width / self.max_width * 255.0,
                                                   0.0, 255.0))
        for _ in range(_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

    def homing(self):
        """Open fully and (re)estimate the maximum width."""
        self._drive_to(MAX_WIDTH)
        self.max_width = self.width()
        return True

    def move(self, width, speed):
        """Move the fingers to ``width`` (``speed`` accepted for parity)."""
        self._drive_to(width)
        return True

    def grasp(self, width, speed, force=60.0, epsilon_inner=0.005,
              epsilon_outer=0.005):
        """Close toward ``width`` and report whether an object was grasped.

        Commanding a width at or below the object squeezes it; ``is_grasped``
        then reflects the finger-object contact (matching camel-franka's
        ``grasp(0.0, ...)`` = 'succeeds when an object is detected')."""
        self._drive_to(width)
        return self.read_once().is_grasped

    def stop(self):
        """Stop the current gripper motion (parity no-op: leaves fingers put)."""
