"""Redundant-DOF (null-space) control: hold the EE fixed, move the other joints.

FR3 has 7 joints for a 6-DOF task -> 1 redundant DOF. This streams a null-space
velocity that reconfigures the arm WITHOUT moving the end-effector:

    dq = J⁺ e              (task: hold the EE at its start pose)
       + (I - J⁺J) q̇0     (null-space: swing the elbow, no EE motion)

In --view you see the elbow/arm sweep while the hand stays on the green marker.
Headless prints that the EE barely moves while some joints swing a lot.

Usage:  python examples/nullspace_control.py [--view]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, JointPositions, vec_to_pose
from viz import add_marker

N_STEPS = 3000
VEL_AMP = 1.0       # rad/s, null-space velocity amplitude
OMEGA = np.pi       # rad/s, oscillation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="show the viewer")
    args = parser.parse_args()

    robot = SimRobot("empty")
    ik = robot._ik
    home_pos, home_R = vec_to_pose(robot.read_once().O_T_EE)
    dt = robot.model.opt.timestep

    viewer = mujoco.viewer.launch_passive(robot.model, robot.data) if args.view else None
    if viewer is not None:
        add_marker(viewer.user_scn, home_pos, [0.1, 0.9, 0.1, 1.0], size=0.015)  # the fixed EE goal

    ac = robot.start_joint_position_control()
    d = np.ones(7)          # a driving vector; its null-space component is used
    q_cmd = robot.read_once().q.copy()   # integrated commanded config
    t = 0.0
    ee_err = np.empty(N_STEPS)
    q_hist = np.empty((N_STEPS, 7))
    for i in range(N_STEPS):
        ik._fk(q_cmd)                                    # evaluate at the commanded config
        J = ik._jacobian(ik.data)
        e = ik._pose_error(ik.data, home_pos, home_R)
        dq_task = ik.dls_step(e, J)                      # keep the commanded EE at home
        Jpinv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
        null = np.eye(7) - Jpinv @ J                     # null-space projector
        nd = null @ d
        nd = nd / (np.linalg.norm(nd) + 1e-9)
        dq_null = nd * VEL_AMP * np.sin(OMEGA * t) * dt  # self-motion, no EE change
        q_cmd = q_cmd + dq_task + dq_null
        ac.writeOnce(JointPositions(q_cmd))
        t += dt

        post = robot.read_once()
        ee_err[i] = np.linalg.norm(vec_to_pose(post.O_T_EE)[0] - home_pos)
        q_hist[i] = post.q
        if viewer is not None:
            viewer.sync()
            time.sleep(dt)

    print("null-space control: EE held fixed while the arm reconfigures")
    print(f"  EE position error : max {ee_err.max()*1000:.2f} mm, "
          f"mean {ee_err.mean()*1000:.2f} mm  (stays small)")
    rng = np.degrees(q_hist.max(0) - q_hist.min(0))
    print(f"  joint swing (deg) : {np.round(rng, 1)}  (redundancy -> joints move, EE does not)")

    if viewer is not None:
        end = time.time() + 3.0
        while viewer.is_running() and time.time() < end:
            viewer.sync()
            time.sleep(0.02)
        viewer.close()


if __name__ == "__main__":
    main()
