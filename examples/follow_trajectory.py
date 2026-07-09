"""Follow a Cartesian trajectory, with the path drawn in the MuJoCo viewer.

Streams an EE trajectory (starting at the home pose) through
start_cartesian_pose_control -> per-tick DLS IK, so the arm tracks the commanded
path. In --view the target path is drawn as green markers and the actual EE path
as red markers, so you can see the tracking (red lags green = servo lag).

Shapes: circle (default), line (out-and-back), square.

Usage:  python examples/follow_trajectory.py [--shape circle|line|square] [--view]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, CartesianPose

N_STEPS = 4000      # points along the path (slower = tighter servo tracking)
MARKER_EVERY = 50   # draw a marker every N steps
RADIUS = 0.12       # m, circle radius
SIDE = 0.20         # m, square side
LINE_DELTA = np.array([0.0, 0.25, -0.10])  # m, line out-and-back displacement


def ee_pose(state):
    T = state.O_T_EE.reshape(4, 4, order="F")
    return T[:3, 3].copy(), T[:3, :3].copy()


def make_pose_vec(pos, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T.flatten(order="F")


def make_path(shape, home, n):
    """(n, 3) target positions, all starting/looping at ``home``."""
    if shape == "circle":  # vertical (x-z plane) circle
        th = np.linspace(0.0, 2.0 * np.pi, n)
        return home + RADIUS * np.stack([np.cos(th) - 1, np.zeros(n), np.sin(th)], 1)
    if shape == "line":    # out along LINE_DELTA and back
        half = n // 2
        s = np.concatenate([np.linspace(0, 1, half, endpoint=False),
                            np.linspace(1, 0, n - half, endpoint=False)])
        return home + s[:, None] * LINE_DELTA
    if shape == "square":  # square in the y-z plane
        corners = home + np.array([[0, 0, 0], [0, SIDE, 0], [0, SIDE, -SIDE],
                                   [0, 0, -SIDE], [0, 0, 0]], float)
        per = n // 4
        return np.vstack([np.linspace(corners[i], corners[i + 1], per, endpoint=False)
                          for i in range(4)])
    raise ValueError(f"unknown shape: {shape}")


def add_marker(scene, pos, rgba, size=0.006):
    """Add a small sphere to the viewer's user scene (skipped if full)."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, 0.0, 0.0]), np.asarray(pos, float),
        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=["circle", "line", "square"],
                        default="circle")
    parser.add_argument("--view", action="store_true", help="show the viewer")
    args = parser.parse_args()

    robot = SimRobot("empty")
    p0, R0 = ee_pose(robot.read_once())
    target = make_path(args.shape, p0, N_STEPS)
    n = len(target)
    print(f"trajectory: {args.shape}, {n} steps, start={np.round(p0, 3)}")

    viewer = mujoco.viewer.launch_passive(robot.model, robot.data) if args.view else None
    if viewer is not None:
        for p in target[::MARKER_EVERY]:  # draw the whole target path up front
            add_marker(viewer.user_scn, p, [0.1, 0.9, 0.1, 1.0])

    ac = robot.start_cartesian_pose_control()
    errs = np.empty(n)
    for i in range(n):
        ac.writeOnce(CartesianPose(make_pose_vec(target[i], R0)))
        actual, _ = ee_pose(robot.read_once())
        errs[i] = np.linalg.norm(actual - target[i])
        if viewer is not None:
            if i % MARKER_EVERY == 0:
                add_marker(viewer.user_scn, actual, [0.9, 0.1, 0.1, 1.0])
            viewer.sync()
            time.sleep(robot.model.opt.timestep)

    print(f"  max tracking error : {errs.max() * 1000:.2f} mm")
    print(f"  mean tracking error: {errs.mean() * 1000:.2f} mm")

    if viewer is not None:
        end = time.time() + 3.0
        while viewer.is_running() and time.time() < end:
            viewer.sync()
            time.sleep(0.02)
        viewer.close()


if __name__ == "__main__":
    main()
