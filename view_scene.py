"""Load and view a task scene.

Usage:  python view_scene.py [task]     (default: "empty")

A task = a base scene (robot + environment) + task-specific objects composed at
runtime with MuJoCo's model-editing API (mjSpec). See tasks.py / objects.py.
"""

import sys
import mujoco
import mujoco.viewer
from pathlib import Path

from tasks import TASKS, BASES
from objects import add_object

ROOT = Path(__file__).parent


def build_task(task_name):
    """Compose base scene + task objects into a compiled model.

    Returns (model, object_body_names). The object names let us place each
    free object at its intended initial pose after resetting to the keyframe.
    """
    if task_name not in TASKS:
        sys.exit(f"unknown task '{task_name}'. choose from: {', '.join(TASKS)}")

    task = TASKS[task_name]
    spec = mujoco.MjSpec.from_file(str(ROOT / BASES[task["base"]]))

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


def main():
    task_name = sys.argv[1] if len(sys.argv) > 1 else "empty"
    model, object_names = build_task(task_name)
    data = initial_state(model, object_names)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
