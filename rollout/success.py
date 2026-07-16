"""Per-task success detection from the sim state -- for rollout evaluation.

A rollout needs to know "did the policy accomplish the task?", which is
task-specific and reads the ground-truth object poses (only available in sim).
Kept next to the rollout env; each task has one criterion. ``task_success(env)``
dispatches on ``env.task`` and returns a bool (False for unknown/empty tasks).
"""

import numpy as np


def _xyz(env, name):
    return env.robot.data.xpos[env.robot.model.body(name).id].copy()


def task_success(env):
    fn = _CHECKS.get(env.task)
    return bool(fn(env)) if fn else False


def _pick_cube(env):
    """Cube grasped and lifted clear of the table (top at z=0 + margin)."""
    return env.is_grasped() and _xyz(env, "cube")[2] > 0.08


def _stack_blocks(env):
    """block_a resting on block_b: xy-aligned, ~one block-height above, released."""
    a, b = _xyz(env, "block_a"), _xyz(env, "block_b")
    aligned = np.linalg.norm(a[:2] - b[:2]) < 0.03      # centres within 3 cm
    on_top = 0.03 <= (a[2] - b[2]) <= 0.055             # ~4 cm (2 half-heights)
    return aligned and on_top and not env.is_grasped()


def _bin_picking(env):
    """peg dropped inside the bin: within the inner footprint, low, released."""
    peg, b = _xyz(env, "peg"), _xyz(env, "bin")
    inside = bool(np.all(np.abs(peg[:2] - b[:2]) < 0.08))   # bin inner half = 0.08 m
    return inside and peg[2] < 0.06 and not env.is_grasped()


_CHECKS = {
    "pick_cube": _pick_cube,
    "stack_blocks": _stack_blocks,
    "bin_picking": _bin_picking,
}
