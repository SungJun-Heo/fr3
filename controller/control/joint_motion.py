"""Joint-space motions built on the control primitives.

``move_to_joint`` drives the arm to a goal configuration with a quintic
trajectory -- camel-franka's ``_run_home`` pattern. Reused by move-to-HOME and,
after an IK solve, by move-to-Cartesian-pose (the "goto pose" path).
"""

import time

from robot import ControllerMode, JointPositions
from controller.planning import QuinticTrajectoryGenerator


def move_to_joint(robot, goal_q, duration, viewer=None):
    """Drive the arm from its current target to ``goal_q`` over ``duration`` s.

    Pass a live passive ``viewer`` to watch it (paced to ~real time)."""
    traj = QuinticTrajectoryGenerator()
    ac = robot.start_joint_position_control(ControllerMode.CartesianImpedance)
    local_time = 0.0
    motion_finished = False
    while not motion_finished:
        state, dt = ac.readOnce()
        local_time += dt.to_sec()
        if local_time <= dt.to_sec():  # first tick: seed from current target
            traj.InitTrajectory(state.q_d, goal_q, 0.0, duration)
        cmd = JointPositions(traj.getPositionTrajectory(local_time))
        if local_time >= duration:
            cmd.motion_finished = True
            motion_finished = True
        ac.writeOnce(cmd)
        if viewer is not None:
            viewer.sync()
            time.sleep(dt.to_sec())
