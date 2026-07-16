"""Shared test fixtures/helpers -- dependency-light (stdlib + numpy).

Offline tests build fake ``RobotState`` / ``GripperState`` value objects so they
need no MuJoCo/GL. Sim tests spin up a real ``SimEnv`` with tiny 32x32 cameras so
rendering stays cheap. Importing this also puts the repo root on ``sys.path`` so
tests run from anywhere (``python -m unittest`` from the repo root, or a single
file).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# identity O_T_EE (column-major len-16), the neutral EE pose
I16 = np.eye(4).flatten(order="F")

# the 9 measured-observation fields (collection.schema.observation_from_state)
OBS_KEYS = {"q", "dq", "tau_J", "tau_ext_hat_filtered", "O_T_EE",
            "O_F_ext_hat_K", "K_F_ext_hat_K", "gripper_width", "gripper_is_grasped"}
# what frame_from_state adds on top of the observation subset
ACTION_GT_KEYS = {"q_d", "O_T_EE_d", "gripper_width_d", "object_qpos"}


def fake_state(**over):
    """A stand-in ``RobotState`` with just the fields schema.py reads."""
    d = dict(
        q=np.arange(7, dtype=float),
        dq=np.zeros(7),
        tau_J=np.zeros(7),
        tau_ext_hat_filtered=np.zeros(7),
        O_T_EE=I16.copy(),
        O_F_ext_hat_K=np.zeros(6),
        K_F_ext_hat_K=np.zeros(6),
        q_d=np.arange(7, dtype=float) + 0.1,
        O_T_EE_d=I16.copy(),
    )
    d.update(over)
    return SimpleNamespace(**d)


def fake_gripper(width=0.04, is_grasped=False):
    """A stand-in ``GripperState``."""
    return SimpleNamespace(width=width, is_grasped=is_grasped)


def tiny_meta_inputs():
    """Minimal (camera_specs, robot_meta, session_params) for a recorder round
    trip without a live sim -- values are shape-plausible, not physical."""
    camera_specs = {"front": {"width": 8, "height": 6}}
    robot_meta = dict(
        joint_names=[f"j{i}" for i in range(7)],
        joint_limits=dict(lower=[-1.0] * 7, upper=[1.0] * 7),
        ee_site="attachment_site",
        gripper=dict(max_width=0.08),
        object_names=[],
    )
    session_params = dict(fps=50.0, control_dt=0.02, sim_timestep=0.001, substeps=20)
    return camera_specs, robot_meta, session_params


def make_env(task="pick_cube", **kw):
    """A real ``SimEnv`` with tiny cameras so per-tick rendering is cheap."""
    from rollout import SimEnv
    kw.setdefault("img_w", 32)
    kw.setdefault("img_h", 32)
    return SimEnv(task, **kw)


def place_objects(env, poses_by_name):
    """Kinematically move named movable objects to ``pos(3)`` (identity quat),
    leaving every other object where it is -- for staging success scenarios.
    Objects without a free joint (e.g. a fixed ``bin``) are ignored here; read
    their pose from ``env.robot.data.xpos`` instead."""
    names = env.robot.movable_object_names
    obj = env.robot.object_qpos().copy()               # keep current poses
    if obj.shape[0] != len(names):                     # empty/mismatch safety
        obj = np.tile([0, 0, 0, 1, 0, 0, 0], (len(names), 1)).astype(float)
    for name, pos in poses_by_name.items():
        if name in names:
            i = names.index(name)
            obj[i, :3] = pos
            obj[i, 3:] = [1.0, 0.0, 0.0, 0.0]          # identity quat (wxyz)
    env.robot.set_replay_state(env.robot.read_once().q, obj)
