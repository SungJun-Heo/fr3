"""Load and view a task scene.

Usage:  python examples/view_scene.py [task]     (default: "empty")

A task = a base scene (robot + environment) + task-specific objects composed at
runtime with MuJoCo's model-editing API (mjSpec). Scene composition lives in the
``scene`` package; this script is just the interactive viewer front end.
"""

import sys
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))
from scene import build_task, initial_state


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
