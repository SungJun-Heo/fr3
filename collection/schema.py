"""The IR contract -- the one home for *what a recorded frame/episode contains*.

Pure mapping, no I/O: it turns sim value objects (``RobotState`` / ``GripperState``
+ camera extrinsics) into the per-frame IR dict, and assembles the episode
``meta.json`` header. The recorder persists whatever this produces; the (deferred)
converters read this module to know how to reshape the IR into LeRobot/GR00T/pi0.

Design principles (so ANY VLA format is derivable without re-collecting):
  * state/action stored as *components* (q, gripper, O_T_EE separately), never a
    pre-concatenated vector -- a converter assembles the model's own ordering.
  * BOTH joint (q, q_d) and end-effector (O_T_EE, O_T_EE_d) are kept -- a policy
    config in either space converts losslessly.
  * actions are ABSOLUTE -- delta/relative actions derive as (q_d - q) or a pose
    difference at convert time.
  * rotations stay as the raw column-major O_T_EE (len-16); no euler/quat is
    imposed here (that is a per-model choice, see ``robot/types.pose_to_vec``).
"""

import datetime
import socket
import subprocess

import numpy as np
import mujoco

# Bump when the field set / conventions below change in a non-additive way.
# 1.1: added meta.object_names + meta.object_qpos0 (initial movable-object poses).
# 1.2: added per-frame object_qpos (full object trajectory) so replay is EXACT --
#      it kinematically re-displays the recorded ground truth (no re-simulation
#      drift). meta.object_qpos0 kept as the initial-layout summary.
SCHEMA_VERSION = "1.2"

# How to read the raw arrays. Copied verbatim into every ``meta.json`` so a
# reader (or converter) needs no external doc.
CONVENTIONS = dict(
    pose="4x4 homogeneous transform, column-major flattened to len-16 (libfranka "
         "O_T_EE); translation at indices 12,13,14",
    rotation="raw rotation matrix inside O_T_EE; NO euler/quaternion imposed",
    wrench_order="[force(3), torque(3)] in the base frame",
    units="m, rad, N, Nm, s",
    frame="robot base frame (== world for the fixed-base FR3)",
    action_alignment="per row: observation = post-step state sampled at sim_time; "
                     "action (q_d / O_T_EE_d / gripper_width_d) = the command "
                     "applied during the step that produced this state",
    gripper="gripper_width is the continuous opening in metres (0..max_width); "
            "gripper_is_grasped is finger-object contact; binarise/normalise at "
            "convert time using meta.gripper.max_width",
)


def frame_from_state(state, gripper_state, gripper_width_d, sim_time, wall_time,
                     cam_extrinsics, object_qpos):
    """One IR frame (dict of arrays/scalars) from the sim's value objects.

    ``state`` is a ``robot.RobotState`` (observation q/dq/tau/O_T_EE/wrench AND
    the action q_d/O_T_EE_d). ``gripper_state`` is a ``GripperState``;
    ``gripper_width_d`` the commanded width (``Gripper.target_width()``).
    ``cam_extrinsics`` maps a *moving* camera name -> its (4,4) world pose this
    frame (static cameras live in ``meta.json`` instead). ``object_qpos`` is the
    movable task objects' ground-truth poses ``(n_obj, 7)`` this frame -- stored
    every frame so replay can re-display the exact recorded scene (arm + objects)
    with zero re-simulation drift. The recorder stacks these into ``(T, ...)``.
    """
    frame = dict(
        sim_time=float(sim_time),
        wall_time=float(wall_time),
        # observation -- measured state
        q=np.asarray(state.q, np.float64),
        dq=np.asarray(state.dq, np.float64),
        tau_J=np.asarray(state.tau_J, np.float64),
        tau_ext_hat_filtered=np.asarray(state.tau_ext_hat_filtered, np.float64),
        O_T_EE=np.asarray(state.O_T_EE, np.float64),
        O_F_ext_hat_K=np.asarray(state.O_F_ext_hat_K, np.float64),
        K_F_ext_hat_K=np.asarray(state.K_F_ext_hat_K, np.float64),
        gripper_width=float(gripper_state.width),
        gripper_is_grasped=bool(gripper_state.is_grasped),
        # action -- commanded targets this step
        q_d=np.asarray(state.q_d, np.float64),
        O_T_EE_d=np.asarray(state.O_T_EE_d, np.float64),
        gripper_width_d=float(gripper_width_d),
        # ground-truth object poses this frame (for exact kinematic replay)
        object_qpos=np.asarray(object_qpos, np.float64),
    )
    for name, T in cam_extrinsics.items():
        frame[f"cam_extrinsic_{name}"] = np.asarray(T, np.float64)
    return frame


def robot_meta(robot, gripper):
    """Static robot description for ``meta.json`` (joint names/limits, EE site,
    gripper). Sourced from the sim robot so it always matches the data."""
    from robot.sim_robot import ARM_JOINTS, EE_SITE
    return dict(
        joint_names=list(ARM_JOINTS),
        joint_limits=dict(lower=np.asarray(robot._q_min).tolist(),
                          upper=np.asarray(robot._q_max).tolist()),
        ee_site=EE_SITE,   # flange attachment_site, not the between-fingers TCP
        gripper=dict(max_width=float(gripper.max_width)),
        # movable objects, in the order object_qpos columns are stored
        object_names=robot.movable_object_names,
    )


def build_meta(task, instruction, num_frames, success, keep, session_params,
               camera_specs, robot_meta_dict, object_qpos0):
    """Assemble the ``meta.json`` header (everything but ``field_index``, which
    the recorder fills from the finished ``data.npz``). ``object_qpos0`` is the
    movable objects' initial poses ``(n_obj, 7)`` -- the scene layout replay
    reconstructs before re-simulating."""
    return dict(
        schema_version=SCHEMA_VERSION,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hostname=socket.gethostname(),
        task=task,
        language_instruction=instruction,
        success=success,
        keep=keep,
        fps=float(session_params["fps"]),
        control_dt=float(session_params["control_dt"]),
        sim_timestep=float(session_params["sim_timestep"]),
        substeps=int(session_params["substeps"]),
        num_frames=int(num_frames),
        cameras=camera_specs,
        joint_names=robot_meta_dict["joint_names"],
        joint_limits=robot_meta_dict["joint_limits"],
        ee_site=robot_meta_dict["ee_site"],
        gripper=robot_meta_dict["gripper"],
        object_names=robot_meta_dict["object_names"],
        object_qpos0=np.asarray(object_qpos0, np.float64).tolist(),
        conventions=CONVENTIONS,
        provenance=provenance(),
    )


def field_index(arrays):
    """Self-describing index of the npz payload: ``{key: {shape, dtype}}``.
    Lets a reader/converter validate the file without opening every array."""
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype))
            for k, v in arrays.items()}


def provenance():
    """Who/what/when produced this episode."""
    return dict(
        fr3_git_commit=_git_commit(),
        mujoco_version=mujoco.__version__,
        numpy_version=np.__version__,
        lerobot_version=_optional_version("lerobot"),
        schema_version=SCHEMA_VERSION,
    )


def _git_commit():
    try:
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _optional_version(module):
    try:
        return __import__(module).__version__
    except Exception:
        return None
