"""Step 3 -- move to HOME with a quintic trajectory (assembly demo).

Mirrors camel-franka's ``_run_home``: ``start_joint_position_control`` -> a
``readOnce()``/``writeOnce()`` loop fed by a ``QuinticTrajectoryGenerator``.
This is the first step where the arm actually *moves* somewhere on purpose.

Run headless to print the final tracking error, or with ``--view`` to watch it.

Usage:  python examples/move_home.py [--view]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, ControllerMode, JointPositions
from controller.planning.path_planner import QuinticTrajectoryGenerator

# An off-home start pose (all joints within limits) so the motion is visible.
START_Q = np.array([0.8, 0.3, 0.0, -0.6, 0.0, 1.5, 0.0])
MOVE_DURATION = 3.0


def move_to(robot, goal_q, duration, viewer=None):
    """Drive the arm to ``goal_q`` over ``duration`` s -- mirror of camel's
    ``_run_home``: seed the trajectory from the current target on the first
    tick, then stream the quintic setpoints until the window closes."""
    traj = QuinticTrajectoryGenerator()
    ac = robot.start_joint_position_control(ControllerMode.CartesianImpedance)
    local_time = 0.0
    motion_finished = False
    while not motion_finished:
        state, dt = ac.readOnce()
        local_time += dt.to_sec()
        if local_time <= dt.to_sec():  # first tick only: seed from current q_d
            traj.InitTrajectory(state.q_d, goal_q, 0.0, duration)
        cmd = JointPositions(traj.getPositionTrajectory(local_time))
        if local_time >= duration:
            cmd.motion_finished = True
            motion_finished = True
        ac.writeOnce(cmd)
        if viewer is not None:
            viewer.sync()
            time.sleep(dt.to_sec())  # pace to ~real time so it's watchable


def set_start_pose(robot, q):
    """Place the arm and its hold target at ``q`` so the demo starts off-home."""
    robot.data.qpos[:7] = q
    robot.data.ctrl[:7] = q
    robot.data.qvel[:7] = 0.0
    mujoco.mj_forward(robot.model, robot.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="show the viewer")
    args = parser.parse_args()

    robot = SimRobot("empty")
    goal_q = robot.model.key_qpos[0][:7].copy()  # 'home' keyframe = camel HOME_POSE
    set_start_pose(robot, START_Q)

    np.set_printoptions(precision=4, suppress=True)
    print("start q :", robot.read_once().q)
    print("goal  q :", goal_q)

    if args.view:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            move_to(robot, goal_q, MOVE_DURATION, viewer)
            end = time.time() + 2.0  # linger so the final pose is visible
            while viewer.is_running() and time.time() < end:
                viewer.sync()
                time.sleep(0.02)
    else:
        move_to(robot, goal_q, MOVE_DURATION)

    final = robot.read_once().q
    err = float(np.abs(final - goal_q).max())
    print("final q :", final)
    print("max |final - goal| :", round(err, 5), "rad")


if __name__ == "__main__":
    main()
