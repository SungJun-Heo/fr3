"""Scene composition: object library, task registry, and scene builder.

A *task* (``tasks.py``) names a base scene and lists the objects to drop into
it; ``objects.py`` is the reusable object library; ``builder.py`` composes and
compiles a task into a MuJoCo model. Together they are the single source of
truth for "what a task is" -- shared by the viewer and the sim robot.
"""

from scene.objects import add_object
from scene.tasks import TASKS, BASES
from scene.builder import build_task, initial_state

__all__ = ["add_object", "TASKS", "BASES", "build_task", "initial_state"]
