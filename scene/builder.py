"""Scene builder: compose a task into a compiled MuJoCo model.

A task = a base scene (robot + environment) + task-specific objects, composed at
runtime with MuJoCo's model-editing API (mjSpec). See ``tasks.py`` / ``objects.py``.
This is the single source of truth for turning a task spec into something the
viewer and the sim robot can both run.
"""

import sys
from pathlib import Path

import mujoco

from scene.tasks import TASKS, BASES
from scene.objects import add_object

# Base-scene paths in ``tasks.py`` are relative to the repo root. This file lives
# in ``<repo>/scene/``, so the repo root is one directory up.
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_task(task_name):
    """Compose base scene + task objects into a compiled model.

    Returns (model, object_body_names). The object names let us place each
    free object at its intended initial pose after resetting to the keyframe.
    """
    if task_name not in TASKS:
        sys.exit(f"unknown task '{task_name}'. choose from: {', '.join(TASKS)}")

    task = TASKS[task_name]
    spec = mujoco.MjSpec.from_file(str(REPO_ROOT / BASES[task["base"]]))

    object_names = []
    for obj in task["objects"]:
        body = add_object(spec, **obj)
        object_names.append(body.name)

    return spec.compile(), object_names


def initial_state(model, object_names):
    data = mujoco.MjData(model)

    # Reset the robot to its "home" keyframe (this also sets ctrl so the arm
    # holds pose instead of collapsing under gravity). Keyframe padding leaves
    # added free objects at the origin, so...
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    # ...restore each free object to the pose declared in its task spec, which
    # the compiler stored in qpos0.
    for name in object_names:
        body = model.body(name)
        if body.jntnum[0] > 0:  # free objects have a joint; static fixtures don't
            qadr = model.jnt_qposadr[body.jntadr[0]]
            data.qpos[qadr:qadr + 7] = model.qpos0[qadr:qadr + 7]

    mujoco.mj_forward(model, data)
    return data
