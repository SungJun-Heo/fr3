"""Rest-to-rest joint trajectories.

Mirrors camel-franka's ``controller/planning/path_planner.py`` so the same
motion code reads identically on both sides. ``QuinticTrajectoryGenerator``
gives a minimum-jerk (5th-order) interpolation between two joint configurations
with zero velocity *and* acceleration at both endpoints -- a smooth start and
stop, which is what you want when commanding a real arm to a pose.
"""

import numpy as np


class QuinticTrajectoryGenerator:
    """Per-joint 5th-order polynomial from a start to a goal configuration."""

    def __init__(self):
        self.start_time = 0.0
        self.duration = 0.0
        # Boundary conditions in the order the coefficient matrix expects:
        # [start_q, goal_q, start_qd, goal_qd, start_qdd, goal_qdd]. Velocities
        # and accelerations stay zero -> rest-to-rest.
        self.boundary = {
            "start_q": np.zeros(7), "goal_q": np.zeros(7),
            "start_qd": np.zeros(7), "goal_qd": np.zeros(7),
            "start_qdd": np.zeros(7), "goal_qdd": np.zeros(7),
        }
        # Maps boundary conditions -> polynomial coefficients, for normalized
        # time s in [0, 1]. Same matrix camel-franka uses.
        self._M = np.array([
            [ -6.0,   6.0,  -3.0,  -3.0,  -0.5,   0.5],
            [ 15.0, -15.0,   8.0,   7.0,   1.5,  -1.0],
            [-10.0,  10.0,  -6.0,  -4.0,  -1.5,   0.5],
            [  0.0,   0.0,   0.0,   0.0,   0.5,   0.0],
            [  0.0,   0.0,   1.0,   0.0,   0.0,   0.0],
            [  1.0,   0.0,   0.0,   0.0,   0.0,   0.0],
        ])
        self._coeff = np.zeros((6, 7))

    def InitTrajectory(self, start_position, goal_position, start_time, duration):
        """Set up a trajectory from ``start_position`` to ``goal_position``."""
        self.start_time = start_time
        self.duration = duration
        self.boundary["start_q"] = np.asarray(start_position, dtype=float)
        self.boundary["goal_q"] = np.asarray(goal_position, dtype=float)
        self._coeff = self._M @ np.array(list(self.boundary.values()))

    def getPositionTrajectory(self, current_time):
        """Desired 7-joint position at ``current_time`` (clamped to the window)."""
        t = np.clip(current_time, self.start_time, self.start_time + self.duration)
        s = (t - self.start_time) / self.duration
        powers = np.array([s**5, s**4, s**3, s**2, s, 1.0])
        return powers @ self._coeff
