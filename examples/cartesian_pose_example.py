"""Streaming Cartesian pose control (step 4c) -- the VLA-relevant path.

Sim mirror of camel-franka's cartesian_pose_example: start_cartesian_pose_control
then stream CartesianPose targets, one per tick. Here each command is tracked by
one DLS IK step, so the EE follows the *commanded path* (unlike goto-pose, whose
in-between path is an uncontrolled joint-space curve). Any source can drive it --
a scripted line here, a VLA later.

Two demos:
  1. safe straight line     -> the EE tracks the line closely (no trip)
  2. reach too far / extend -> the safety guards trip (singularity / joint limit)

Usage:  python examples/cartesian_pose_example.py [--view]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, CartesianPose, vec_to_pose
from overlay import add_marker, TARGET_RGBA, ACTUAL_RGBA

MARKER_EVERY = 50   # in --view, drop a commanded/actual marker every N steps


def stream_line(robot, target_pos, n_steps, viewer=None):
    """Stream a straight position line from the current EE to target_pos.

    Returns the per-tick (commanded, actual) EE positions. Raises on a safety
    trip (the caller decides whether that was expected). In --view, draws the
    commanded path (green) vs the actual EE path (red) so tracking is visible."""
    ac = robot.start_cartesian_pose_control()
    state, _ = ac.readOnce()
    p_start, R = vec_to_pose(state.O_T_EE)           # keep orientation fixed
    track = []
    for i in range(1, n_steps + 1):
        s = i / n_steps
        p_cmd = p_start + s * (target_pos - p_start)  # straight line
        cmd = CartesianPose.from_matrix(p_cmd, R)
        if i == n_steps:
            cmd.motion_finished = True
        ac.writeOnce(cmd)                             # per-tick DLS IK + safety
        p_act, _ = vec_to_pose(robot.read_once().O_T_EE)
        track.append((p_cmd, p_act))
        if viewer is not None:
            if i % MARKER_EVERY == 0:
                add_marker(viewer.user_scn, p_cmd, TARGET_RGBA)  # commanded
                add_marker(viewer.user_scn, p_act, ACTUAL_RGBA)  # actual EE
            viewer.sync()
            time.sleep(robot.model.opt.timestep)
    return track


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="show the viewer")
    args = parser.parse_args()
    np.set_printoptions(precision=4, suppress=True)

    # ---- Demo 1: safe straight line, measure tracking --------------------
    robot = SimRobot("empty")
    viewer = mujoco.viewer.launch_passive(robot.model, robot.data) if args.view else None
    p0, _ = vec_to_pose(robot.read_once().O_T_EE)
    target = p0 + np.array([0.0, 0.20, 0.0])          # 20 cm sideways in +y
    print(f"[demo 1] streaming a straight line: EE {p0} -> {target}")
    track = stream_line(robot, target, n_steps=1500, viewer=viewer)

    cmd = np.array([c for c, _ in track])
    act = np.array([a for _, a in track])
    track_err = np.linalg.norm(act - cmd, axis=1)     # actual vs commanded, per tick
    print(f"  max tracking error : {track_err.max()*1000:.2f} mm")
    print(f"  final EE           : {act[-1]}  (target {target})")
    print(f"  final error        : {np.linalg.norm(act[-1] - target)*1000:.2f} mm")
    if viewer is not None:
        viewer.close()

    # ---- Demo 2: reach too far -> safety trip ----------------------------
    robot = SimRobot("empty")
    p0, _ = vec_to_pose(robot.read_once().O_T_EE)
    far = p0 + np.array([0.55, 0.0, -0.15])           # far forward: forces extension
    print(f"\n[demo 2] streaming toward an unreachable/extended target: {far}")
    try:
        stream_line(robot, far, n_steps=1500)
        print("  [WARN] never tripped -- reached the far target")
    except RuntimeError as e:
        p_now, _ = vec_to_pose(robot.read_once().O_T_EE)
        print(f"  [SAFETY TRIP] {e}")
        print(f"  stopped at EE      : {p_now}")
        robot.automatic_error_recovery()
        print(f"  recovered          : _has_error = {robot._has_error}")


if __name__ == "__main__":
    main()
