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

**Scope.** fr3 does two things: **collect** demonstration data (a VLA-format-free
intermediate representation) and **consume** a VLA policy (roll it out in the sim
and score it). Per-model dataset **conversion, training, and inference** live in
*separate* projects — fr3 stays sim-focused and format-agnostic, so one collected
dataset can feed GR00T, π0, or anything else.

## Quick start

```bash
# 1. models come from the mujoco_menagerie submodule
git submodule update --init

# 2. env (conda): Python 3.11, MuJoCo 3.8.1
conda create -n fr3_sim python=3.11
conda activate fr3_sim
pip install mujoco numpy          # tkinter (stdlib) is needed for the GUIs

# 3. run — the unified control GUI (joint / task / VR teleop in one window)
python main.py
```

Pick the control mode in the panel: **JOINT** (7 angle sliders), **TASK** (EE-pose
sliders + DLS IK), or **VR** (Meta Quest over TCP). No headset? Select **VR** and
drive the pipeline with the mock client:

```bash
python main.py &
python -m teleop.mock_vr_client
```

The settings row adjusts everything at runtime — the **task/scene** (`empty`,
`pick_cube`, `stack_blocks`, `bin_picking`), the VR position **scale**, the VR
**smooth-tau**, and a **markers** toggle for the overlay. The viewer overlays the
**commanded** EE pose (translucent frame) vs the **actual** EE pose (solid frame)
so you can see how well the arm tracks (in TASK / VR modes); the status line also
shows the tracking error in mm.

## Layout

| Path | What lives there |
|------|------------------|
| `robot/` | `SimRobot` — the `pylibfranka` mirror: state read, joint + Cartesian control, safety lifecycle, collision reflex. `Gripper`. `types.py` — command types + the `O_T_EE` pose convention. |
| `controller/` | `kinematics/` (DLS IK solver), `planning/` (quintic trajectories), `control/` (`move_to_joint`). Mirrors `camel-franka/controller/`. |
| `scene/` | Task registry + object library + an `mjSpec` builder. One source of truth for "what a task is", shared by the viewer and `SimRobot`. |
| `gui/` | The unified control GUI. `ControlSession` (UI-agnostic tick loop: joint / task / VR modes, HOME, reset, gripper, overlay, telemetry) + `UnifiedGUI` (the Tkinter panel). |
| `teleop/` | VR input: Quest → TCP JSON → relative-clutch EE mapping. Server (`vr_server`), the clutch mechanism (`clutch`), and a headless-friendly mock client. The GUI's "vr" mode drives it. |
| `collection/` | **Raw-data collection** (VLA-format-free): `SimCameraRenderer` (render model cameras), `EpisodeRecorder` (raw IR = `meta.json` + `data.npz` + JPEG frames), `Collector` (record off a session tick), `EpisodePlayer` (exact kinematic replay). Plus domain randomization + episode delete/reindex. |
| `rollout/` | **VLA-policy consumer**: `SimEnv` (produce observations → consume one tagged action, absolute/delta × joint/Cartesian → step), `task_success` (per-task ground-truth), `evaluate` (rollout success rate over N episodes). |
| `overlay/` | Viewer-overlay debug drawing (markers, pose frames). Neutral layer shared by `teleop` and `examples`. |
| `models/` | `fr3_with_gripper` scene (arm + Franka Hand + table). |
| `examples/` | Tutorial scripts, one per build step — see [examples/README.md](examples/README.md). |
| `tests/` | `unittest` suite (no pytest dependency). Offline: IR schema single-home, recorder round-trip, `evaluate` harness (fake env). Sim-backed (tiny 32×32 cameras): the 4 tagged-action decode paths, delta integration, per-task success. Run: `python -m unittest discover -s tests`. |
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

**Working:**
- *Sim/control:* state read · joint-position control · quintic moves · collision
  reflex · DLS IK · goto-pose · streaming Cartesian control · gripper · unified
  control GUI (joint / task / VR teleop, keyboard-free VR-button collection).
- *Data collection:* format-free IR (`meta.json` + `data.npz` + JPEG) · truthful
  commanded-action recording · per-task domain randomization · exact kinematic
  replay · episode delete/reindex.
- *VLA consumer:* `rollout.SimEnv` (observe → apply one tagged action → step;
  absolute + delta, joint + Cartesian) · per-task success detection · `evaluate`
  rollout harness (success rate).
- *Tests:* `unittest` suite covering the IR schema, recorder round-trip, the four
  tagged-action decode paths, delta integration, per-task success, and the eval
  harness (`python -m unittest discover -s tests`).

**Next:** raw-IR transfer to the training server · IK quality (null-space toward a
reference posture + LPF, per the `camel-RBY1` OSMC controller). Per-model
**conversion / training / inference** are separate projects; fr3 produces the raw
IR and hosts the rollout env.

## Related

- `camel-franka` — the real FR3 teleop / data-collection rig (`pylibfranka`, real
  hardware only). This project mirrors its API.
- `camel-RBY1` — humanoid rig whose OSMC controller is the production version of the
  same DLS-IK-with-safety problem; reference for the IK-quality roadmap.
