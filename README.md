# fr3 — a MuJoCo FR3 that mirrors `pylibfranka`

A simulated Franka Research 3 (MuJoCo) whose control API is a **drop-in mirror of
`pylibfranka`** — the raw-libfranka Python binding the real-robot rig
(`camel-franka`) uses. The point: write control code once — a VLA policy, teleop,
a data-collection loop — and run it against **either** this sim or the real FR3
without changing the code. Validate in sim, transfer to hardware.

This is the "shim seam": same method names (`Robot(ip)` → `SimRobot(task)`), same
control loop (`start_*_control()` → `ActiveControl.readOnce()/writeOnce()`), same
`RobotState` fields, same conventions (`O_T_EE` as a column-major 4×4) — a MuJoCo
backend underneath instead of firmware.

**Fidelity rule:** the sim should *trip or flag whatever would fail on the real
robot*. The collision reflex and the IK singularity / joint-limit trips exist so a
policy hits those failures here, before hardware.

## Quick start

```bash
# 1. models come from the mujoco_menagerie submodule
git submodule update --init

# 2. env (conda): Python 3.11, MuJoCo 3.8.1
conda create -n fr3_sim python=3.11
conda activate fr3_sim
pip install mujoco numpy          # tkinter (stdlib) is needed for the GUIs

# 3. run
python main.py --mode gui         # hand-control GUI (joint / task space + gripper)
python main.py --mode vr          # VR teleop (Meta Quest over TCP) — default mode
```

No headset? Drive the VR pipeline with the mock client:

```bash
python main.py --mode vr &
python -m teleop.mock_vr_client
```

`--task` picks the scene for either mode: `empty` (default), `pick_cube`,
`stack_blocks`, `bin_picking`. VR flags: `--hand`, `--host/--port`, `--scale`,
`--smooth-tau`, `--stats`, `--no-gui`, `--no-view` (see `python main.py -h`).

## Layout

| Path | What lives there |
|------|------------------|
| `robot/` | `SimRobot` — the `pylibfranka` mirror: state read, joint + Cartesian control, safety lifecycle, collision reflex. `Gripper`. `types.py` — command types + the `O_T_EE` pose convention. |
| `controller/` | `kinematics/` (DLS IK solver), `planning/` (quintic trajectories), `control/` (`move_to_joint`). Mirrors `camel-franka/controller/`. |
| `scene/` | Task registry + object library + an `mjSpec` builder. One source of truth for "what a task is", shared by the viewer and `SimRobot`. |
| `teleop/` | VR teleop: Quest → TCP JSON → relative-clutch Cartesian control. Server, control loop, and a headless-friendly mock client. |
| `models/` | `fr3_with_gripper` scene (arm + Franka Hand + table). |
| `examples/` | Tutorial scripts, one per build step — see [examples/README.md](examples/README.md). |
| `docs/` | Study notes (e.g. the Cartesian-IK control derivation). |
| `mujoco_menagerie/` | Upstream MuJoCo model submodule (the FR3 source). |

## Key concepts

- **The control loop.** `start_joint_position_control()` /
  `start_cartesian_pose_control()` return an `ActiveControl`; you drive it with
  `readOnce()` (get `RobotState`) and `writeOnce(command)`. Real libfranka runs at
  1 kHz and `readOnce` blocks; sim has no real time, so **one `writeOnce` applies
  the command and advances the sim exactly one step**.
- **`O_T_EE` convention.** The EE pose is a 4×4 transform flattened
  **column-major** to length 16 (translation at indices 12,13,14) — how
  `camel-franka` indexes it. It has one home: `pose_to_vec` / `vec_to_pose` /
  `CartesianPose.from_matrix` in `robot/types.py`. Nothing re-derives `order="F"`.
- **Cartesian streaming = per-tick DLS IK.** On the real robot the firmware
  converts a streamed `CartesianPose` to joint motion; here `SimRobot` does it with
  one damped-least-squares step per tick. Two safety modes:
  `"trip"` (fidelity — faults on NaN / singularity / joint limit, like the real
  robot) and `"clamp"` (teleop — brakes smoothly instead of stuttering).
- **Collision reflex.** After each step the external force is estimated from MuJoCo
  contacts, low-pass filtered, and compared to `set_collision_behavior` thresholds;
  exceeding them latches an error until `automatic_error_recovery()` — mirroring the
  real robot tripping on impact.
- **Sim-only escape hatches.** `reset_objects()` and `reset_home()` snap objects /
  arm back instantly (no real-robot analog) — handy between imitation-learning
  episodes when teleop knocks something out of reach.

## Code shape

Higher-level behavior is **composed from primitives**, and each capability has one
home:
- `Gripper.homing/move/grasp` are built from `_set_force_limit` + `_drive_to`.
- `move_to_pose` = `DLSIKSolver.solve` + `move_to_joint`; `move_home` reuses the
  same `move_to_joint`.
- The `O_T_EE` pack/unpack and the Yoshikawa manipulability formula each exist in
  exactly one place and everything else calls into them.

## Status

**Working:** state read · joint-position control · quintic moves · collision
reflex · DLS IK · goto-pose · streaming Cartesian control · null-space demo ·
gripper (force-limited grasp vs position-only move) · hand-control GUI · VR teleop.

**Next:** a VLA runner (observation → policy → `CartesianPose` stream), and IK
quality work (null-space toward a reference posture + LPF smoothing, per the
`camel-RBY1` OSMC controller).

## Related

- `camel-franka` — the real FR3 teleop / data-collection rig (`pylibfranka`, real
  hardware only). This project mirrors its API.
- `camel-RBY1` — humanoid rig whose OSMC controller is the production version of the
  same DLS-IK-with-safety problem; reference for the IK-quality roadmap.
