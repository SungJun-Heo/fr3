"""Simulated robot backend.

Mirrors the ``pylibfranka`` API surface (the raw-libfranka Python binding the
real-robot project ``camel-franka`` uses) so that higher-level code -- state
reading, control loops, data recording -- can be written once and run against
either MuJoCo (here) or the real FR3 (there). This is the "shim" seam: same
method names, same ``RobotState`` field names, same conventions (e.g. ``O_T_EE``
as a column-major 4x4), a different backend underneath.
"""

from robot.sim_robot import SimRobot, RobotState

__all__ = ["SimRobot", "RobotState"]
