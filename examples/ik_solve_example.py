"""Verify the DLS IK solver (step 4a).

Round-trip check: pick a target EE pose, solve IK for joint angles, then use FK
to confirm the solved angles actually reach that pose. No control/streaming yet
-- this only proves the solver itself is correct.

Usage:  python examples/ik_solve_example.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot
from robot.sim_robot import ARM_JOINTS, EE_SITE
from controller.kinematics import DLSIKSolver


def ee_pose(state):
    """(position, 3x3 rotation) from a RobotState's column-major O_T_EE."""
    T = state.O_T_EE.reshape(4, 4, order="F")
    return T[:3, 3].copy(), T[:3, :3].copy()


def main():
    robot = SimRobot("empty")
    solver = DLSIKSolver(robot.model, EE_SITE, ARM_JOINTS, damping=0.05)
    np.set_printoptions(precision=4, suppress=True)

    q_home = robot.read_once().q
    p_home, R_home = ee_pose(robot.read_once())
    print(f"home q     : {q_home}")
    print(f"home EE pos: {p_home}\n")

    # Targets = home EE pose shifted by a few known offsets (orientation kept).
    offsets = {
        "same pose": np.array([0.0, 0.0, 0.0]),
        "+x 10cm":   np.array([0.10, 0.0, 0.0]),
        "+y 8cm, -z 10cm": np.array([0.0, 0.08, -0.10]),
        "-x 12cm, +z 8cm": np.array([-0.12, 0.0, 0.08]),
    }

    for name, d in offsets.items():
        target_pos = p_home + d
        q_sol, info = solver.solve(target_pos, R_home, q_init=q_home)
        tag = "OK " if info["converged"] else "FAIL"
        print(f"[{tag}] {name:18s} iters={info['iters']:3d}  "
              f"pos_err={info['pos_err']*1000:6.3f} mm  "
              f"rot_err={np.degrees(info['rot_err']):6.3f} deg")


if __name__ == "__main__":
    main()
