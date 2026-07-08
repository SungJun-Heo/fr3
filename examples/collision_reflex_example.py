"""Safety step -- collision reflex (dangerous-motion detection).

Commands the arm to a target whose EE lies *below* the table, so the motion
drives it into the tabletop. The sim mirrors the real robot's reflex: the
external force is estimated from contacts, filtered, and compared to the
thresholds from set_collision_behavior; on impact it latches an error and
refuses further motion until automatic_error_recovery() -- exactly the failure
we want a VLA policy to trip in sim *before* it ever reaches real hardware.

Verifies the whole skeleton: trip -> control refused -> recover -> resume.

Usage:  python examples/collision_reflex_example.py [--view]
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

# camel-franka's collision thresholds (per-joint torque [Nm], EE wrench [N/Nm]).
LOWER_TORQUE = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
UPPER_TORQUE = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
LOWER_FORCE = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
UPPER_FORCE = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

TARGET_Q = np.array([0.0, 1.6, 0.0, -1.73, 0.0, 2.73, 0.79])  # EE at z = -0.29
MOVE_DURATION = 3.0


def ee_z(state):
    return float(state.O_T_EE.reshape(4, 4, order="F")[2, 3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="show the viewer")
    args = parser.parse_args()

    robot = SimRobot("empty")
    robot.set_collision_behavior(LOWER_TORQUE, UPPER_TORQUE, LOWER_FORCE, UPPER_FORCE)
    np.set_printoptions(precision=2, suppress=True)

    viewer = None
    if args.view:
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

    # Drive toward the below-table target; the reflex should trip on contact.
    traj = QuinticTrajectoryGenerator()
    ac = robot.start_joint_position_control(ControllerMode.CartesianImpedance)
    local_time = 0.0
    tripped = False
    print(f"commanding arm toward a target with EE z = -0.29 (into the table)...")
    while local_time < MOVE_DURATION + 1.0:
        state, dt = ac.readOnce()
        local_time += dt.to_sec()
        if local_time <= dt.to_sec():
            traj.InitTrajectory(state.q_d, TARGET_Q, 0.0, MOVE_DURATION)
        ac.writeOnce(JointPositions(traj.getPositionTrajectory(local_time)))
        if viewer is not None:
            viewer.sync()
            time.sleep(dt.to_sec())
        if robot._has_error:
            post = robot.read_once()
            print(f"\n[REFLEX TRIPPED] t = {local_time:.2f}s, EE z = {ee_z(post):+.3f}")
            print(f"  reason       : {robot._error_reason}")
            print(f"  tau_ext_filt : {post.tau_ext_hat_filtered}")
            tripped = True
            break

    if not tripped:
        print("\n[WARN] never tripped -- arm reached target without exceeding thresholds")
        return

    # 1) control must be refused while latched in error
    try:
        ac.writeOnce(JointPositions(robot.read_once().q))
        print("  [FAIL] writeOnce did not raise while in error state")
    except RuntimeError as e:
        print(f"  control refused while in error: {e}")

    # 2) recover, then 3) confirm control resumes
    robot.automatic_error_recovery()
    print(f"recovered: _has_error = {robot._has_error}")
    ac2 = robot.start_joint_position_control(ControllerMode.JointImpedance)
    s, _ = ac2.readOnce()
    ac2.writeOnce(JointPositions(s.q))  # holds current pose; must not raise
    print("control resumed OK after recovery")

    if viewer is not None:
        end = time.time() + 2.0
        while viewer.is_running() and time.time() < end:
            viewer.sync()
            time.sleep(0.02)
        viewer.close()


if __name__ == "__main__":
    main()
