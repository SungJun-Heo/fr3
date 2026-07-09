"""Control command / mode types mirroring pylibfranka.

camel-franka pulls these from one namespace (``from pylibfranka import
ControllerMode, JointPositions``); we mirror the names and shapes so control
loops read the same on the sim side. These are deliberately thin value holders
-- no behavior lives here.
"""

from enum import Enum

import numpy as np


class ControllerMode(Enum):
    """Which internal impedance controller the real robot runs while a motion
    generator streams targets. Sim has no such switch -- the model's position
    actuators are the controller -- so this is stored for API parity only.
    """
    JointImpedance = 0
    CartesianImpedance = 1


class Duration:
    """Elapsed time for a control tick; mirrors ``franka::Duration``."""

    def __init__(self, seconds):
        self._s = float(seconds)

    def to_sec(self):
        return self._s

    def to_msec(self):
        return self._s * 1000.0


class JointPositions:
    """A joint-position command: 7 target angles + a finish flag.

    Mirrors ``franka::JointPositions``. The control loop sets
    ``motion_finished = True`` on the final command to end the session, exactly
    as camel-franka's ``_run_home`` does.
    """

    def __init__(self, q):
        self.q = np.asarray(q, dtype=float)
        self.motion_finished = False


class CartesianPose:
    """A Cartesian-pose command: the 4x4 EE pose as a **column-major** length-16
    vector (libfranka's ``O_T_EE``) + a finish flag. Mirrors
    ``franka::CartesianPose``. On the real robot the firmware tracks this; in
    sim it is tracked by one DLS IK step per tick (see ``SimRobot``).
    """

    def __init__(self, O_T_EE):
        self.O_T_EE = np.asarray(O_T_EE, dtype=float)
        self.motion_finished = False
