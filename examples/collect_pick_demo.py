"""Collect one episode headless with a scripted pick -- the end-to-end proof of
the VLA-format-free data pipeline (no VR headset, no GUI).

A canned top-down pick trajectory is streamed through the same
``ControlSession`` the GUI/VR use (task/Cartesian mode), while a ``Collector``
records every tick into the raw IR episode directory
(``<root>/pick_cube/episode_XXXX/`` = ``meta.json`` + ``data.npz`` + per-camera
JPEGs). The gripper is driven NON-blocking (``set_gripper_frac``) so the whole
motion stays inside the recorded control loop -- a blocking ``Gripper.grasp``
would step the sim outside it and desync the frames.

After saving, it reads the episode back and prints a shape/field summary, so a
single run exercises: sim -> observation+truthful action -> render -> IR write
-> IR read.

Usage:
  python examples/collect_pick_demo.py [--view] [--root DIR] [--instruction STR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from gui.session import ControlSession
from collection import CollectionConfig, Collector
from robot import vec_to_pose

# Top-down grasp orientation: flange z-axis pointing straight down (world -z).
# A proper rotation (180 deg about x): x->x, y->-y, z->-z.
R_DOWN = np.array([[1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0]])

HOVER_Z = 0.30    # approach/lift height for the flange (m)
GRASP_Z = 0.135   # flange height at grasp (~fingertips at the cube)


def build_trajectory(start_pos, cube_xy):
    """A list of ``(flange_pos, grip_frac)`` per tick for a top-down pick.

    ``grip_frac``: 1 = open, 0 = closed. Segments: approach above the cube ->
    descend -> close the gripper (holding position) -> lift. Waypoints are
    linearly interpolated for a smooth, IK-friendly path."""
    hover = np.array([cube_xy[0], cube_xy[1], HOVER_Z])
    grasp = np.array([cube_xy[0], cube_xy[1], GRASP_Z])
    segments = [
        (hover, 45, 1.0, 1.0),   # move above the cube, gripper open
        (grasp, 45, 1.0, 1.0),   # descend to the cube
        (grasp, 30, 1.0, 0.0),   # close the gripper (hold pose)
        (hover, 45, 0.0, 0.0),   # lift, gripper closed
    ]
    traj, cur = [], np.asarray(start_pos, float).copy()
    for goal, n, g0, g1 in segments:
        for i in range(1, n + 1):
            a = i / n
            traj.append((cur * (1 - a) + goal * a, g0 * (1 - a) + g1 * a))
        cur = goal.copy()
    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="show the viewer")
    parser.add_argument("--root", default="data/raw", help="dataset root dir")
    parser.add_argument("--instruction", default="pick up the red cube",
                        help="language instruction stored with the episode")
    args = parser.parse_args()
    np.set_printoptions(precision=4, suppress=True)

    session = ControlSession(task="pick_cube", view=args.view)
    session.set_mode("task")
    collector = Collector(session, CollectionConfig(root=args.root))

    cube_pos = session.data.xpos[session.model.body("cube").id].copy()
    start_pos, _ = vec_to_pose(session.robot.read_once().O_T_EE)
    traj = build_trajectory(start_pos, cube_pos[:2])
    print(f"cube at {cube_pos} | {len(traj)} ticks scripted")

    # -- record the scripted pick, one frame per tick --
    collector.start_episode(args.instruction)
    for pos, grip in traj:
        session.set_task_target(pos, R_DOWN)
        session.set_gripper_frac(grip)
        if not session.step():        # viewer closed
            break
        collector.on_tick(session)

    grasped = bool(session.gripper.read_once().is_grasped)
    if session.trip:
        print(f"[note] safety trip during scripted motion: {session.trip}")
    path = collector.keep(success=grasped)
    print(f"grasped={grasped} -> saved episode: {path}")

    _read_back(path, session)
    collector.close()
    session.close()


def _read_back(path, session):
    """Load the episode we just wrote and print a shape/field summary."""
    import json
    meta = json.loads((path / "meta.json").read_text())
    d = np.load(path / "data.npz")
    T = d["q"].shape[0]
    print(f"\n[read-back] {path.name}: T={T} frames, "
          f"cameras={list(meta['cameras'])}, fps={meta['fps']:.1f}")
    for k in ("q", "q_d", "O_T_EE", "O_T_EE_d", "gripper_width",
              "gripper_width_d", "cam_extrinsic_wrist"):
        print(f"    {k:20s} {tuple(d[k].shape)}")
    for cam in meta["cameras"]:
        n = len(list((path / "images" / cam).glob("*.jpg")))
        assert n == T, f"{cam}: {n} jpgs != {T} rows"
    print(f"    images: {T} jpg/camera (rows == frames) OK")
    # truthful action: the commanded EE leads the measured EE while descending
    lead = np.linalg.norm(vec_to_pose(d["O_T_EE_d"][T // 2])[0]
                          - vec_to_pose(d["O_T_EE"][T // 2])[0])
    print(f"    action leads state by {lead*1000:.1f} mm mid-episode "
          f"(O_T_EE_d is the truthful command)")


if __name__ == "__main__":
    main()
