"""Gripper control -- sim mirror of camel-franka's examples/move_gripper.py.

Exercises the pylibfranka Gripper API on the sim: homing / move / read_once /
grasp. The grasp-with-object case places a cube between the fingers (gravity
frozen so the free cube stays put) to check that is_grasped detects contact.

Usage:  python examples/move_gripper.py
"""

import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, Gripper


def show(g, label):
    st = g.read_once()
    print(f"  {label:16s} width={st.width:.4f}  is_grasped={st.is_grasped}  "
          f"max_width={st.max_width:.4f}")


def main():
    np.set_printoptions(precision=4, suppress=True)

    # --- mechanics (no object) ---
    print("[mechanics]")
    robot = SimRobot("empty")
    g = Gripper(robot)
    g.homing();        show(g, "homing (open)")
    g.move(0.04, 0.1); show(g, "move 0.04")
    g.move(0.0, 0.1);  show(g, "move 0.00 (closed)")
    print(f"  grasp with no object -> is_grasped={g.grasp(0.0, 0.1, 60.0)}  (expect False)")

    # --- grasp WITH object ---
    print("\n[grasp an object]")
    robot = SimRobot("pick_cube")
    m, d = robot.model, robot.data
    g = Gripper(robot)
    g.homing()  # open around the cube

    # Drop the cube onto the grasp center between the fingers. Gravity is frozen
    # only for this isolated grasp-detection check (a real pick brings the arm
    # down to a resting object instead).
    lf, rf = m.body("left_finger").id, m.body("right_finger").id
    center = 0.5 * (d.xpos[lf] + d.xpos[rf])
    cube_q = m.jnt_qposadr[m.body("cube").jntadr[0]]
    d.qpos[cube_q:cube_q + 3] = center
    d.qpos[cube_q + 3:cube_q + 7] = [1, 0, 0, 0]
    g0 = m.opt.gravity.copy()
    m.opt.gravity[:] = 0.0
    mujoco.mj_forward(m, d)

    grasped = g.grasp(0.0, 0.1, 60.0)
    print(f"  cube at {np.round(center, 3)}")
    print(f"  grasp -> is_grasped={grasped}  final width={g.width():.4f}  (expect True)")
    m.opt.gravity[:] = g0


if __name__ == "__main__":
    main()
