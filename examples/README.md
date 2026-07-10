# Examples

Each script is a small, self-verifying step that mirrors a `camel-franka`
example, built in the order the control stack came together. Read them top to
bottom as a tutorial: every one prints a result (tracking error, trip reason,
grasp success) so it doubles as a check that the layer it exercises works.

Most run **headless** and print numbers; pass `--view` (where offered) to watch
it in the MuJoCo viewer. Run from the repo root:

```bash
python examples/<name>.py [--view]
```

## Learning path

| # | Script | What it shows | Layer |
|---|--------|---------------|-------|
| 1 | `view_scene.py` | Load and view a task scene (`[task]` arg) | `scene/` |
| 2 | `print_robot_state.py` | Read a `RobotState` once (q, dq, `O_T_EE`) | `robot/` state |
| 3 | `joint_position_example.py` | Joint-position control primitives — the `readOnce`/`writeOnce` loop | `robot/` control |
| 4 | `move_home.py` | Quintic move to HOME (composes `move_to_joint`) | `controller/` planning |
| 5 | `collision_reflex_example.py` | Drive into the table → reflex trips → recover → resume | `robot/` safety |
| 6 | `ik_solve_example.py` | DLS IK round-trip: solve a pose, FK-check it | `controller/kinematics` |
| 7 | `move_to_pose.py` | Goto-pose: IK once → joint move to the goal | IK + planning |
| 8 | `cartesian_pose_example.py` | Streaming Cartesian: per-tick DLS IK tracks a line; far target trips safety | streaming (VLA path) |
| 9 | `follow_trajectory.py` | Follow a Cartesian path (`--shape circle\|line\|square`), draw target vs actual | streaming |
| 10 | `nullspace_control.py` | Redundant DOF: reconfigure the arm while the EE stays fixed | kinematics |
| 11 | `move_gripper.py` | Gripper API: homing / move / grasp, `is_grasped` detection | `robot/` gripper |

## Apps / tools

| Script | What it is |
|--------|-----------|
| `control_gui.py` | Tkinter hand-control panel (joint / task space + gripper). Also `python main.py --mode gui`. |
| `vr_monitor.py` | Taps the real VR teleop loop and reports input fps / frame age / EE lag. Needs a VR client (or `python -m teleop.mock_vr_client`). |

Viewer-overlay helpers (`add_marker`, `add_frame`) live in `viz.py` at the repo
root, shared with `teleop`. The pose/command conventions — `pose_to_vec` /
`vec_to_pose` / `CartesianPose.from_matrix` — live in the `robot` package, since
they're library concerns, not example glue.
