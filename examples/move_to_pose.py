"""Move the EE to a Cartesian pose (step 4b, "goto pose").

The path we discussed: solve IK once for the target pose -> get a joint goal ->
reuse the joint-space quintic move. The EE path in between is a joint-space
curve (not a straight line -- that needs per-tick streaming, the next step), but
the *endpoint* lands on the commanded Cartesian pose.

Reuses everything already built: DLSIKSolver + QuinticTrajectoryGenerator +
joint position control.

Usage:  python examples/move_to_pose.py [--view]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, vec_to_pose
from robot.sim_robot import ARM_JOINTS, EE_SITE
from controller.kinematics import DLSIKSolver
from controller.control import move_to_joint

TARGET_OFFSET = np.array([0.15, 0.10, -0.15])  # from the home EE position
MOVE_DURATION = 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="show the viewer")
    args = parser.parse_args()

    robot = SimRobot("empty")
    solver = DLSIKSolver(robot.model, EE_SITE, ARM_JOINTS, damping=0.05)
    np.set_printoptions(precision=4, suppress=True)

    q0 = robot.read_once().q
    p0, R0 = vec_to_pose(robot.read_once().O_T_EE)
    target_pos = p0 + TARGET_OFFSET  # keep orientation, shift position

    # 1) IK once: Cartesian target -> joint goal
    q_goal, info = solver.solve(target_pos, R0, q_init=q0)
    print(f"IK: converged={info['converged']} iters={info['iters']} "
          f"pos_err={info['pos_err']*1000:.3f}mm")
    print(f"target EE pos : {target_pos}")
    print(f"q_goal        : {q_goal}")
    if not info["converged"]:
        print("[WARN] IK did not converge; aborting")
        return

    # 2) reuse the joint-space quintic move to the IK goal
    viewer = mujoco.viewer.launch_passive(robot.model, robot.data) if args.view else None
    move_to_joint(robot, q_goal, MOVE_DURATION, viewer)

    # 3) verify: did the EE actually reach the commanded Cartesian pose?
    pf, Rf = vec_to_pose(robot.read_once().O_T_EE)
    print(f"final EE pos  : {pf}")
    print(f"pos error     : {np.linalg.norm(pf - target_pos)*1000:.3f} mm")

    if viewer is not None:
        end = time.time() + 2.0
        while viewer.is_running() and time.time() < end:
            viewer.sync()
            time.sleep(0.02)
        viewer.close()


if __name__ == "__main__":
    main()
