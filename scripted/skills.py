"""Task -> oracle table. The ONLY place that knows what a task is.

Adding a task touches exactly three files, none of which need new motion code:

  1. ``scene/tasks.py``      -- the scene: objects + their randomization ranges
  2. ``rollout/success.py``  -- the success criterion (ground-truth check)
  3. here                    -- one entry composing macros

For example, "put the red cube in the bin" is::

    "cube_in_bin": lambda: pick("cube") + drop_in("cube", "bin"),

Grasp width, grasp height, wrist alignment and release height all follow from
the object's declared ``kind``/``size`` (``scene/shapes.py``), so they are not
restated here. A task that needs a motion the macros do not cover gets ONE new
macro in ``macros.py`` and still only one line here.
"""

from scripted.base import Skill
from scripted.macros import drop_in, pick, place_on

# Each value builds a fresh phase list (phases hold per-run latched state, so
# they must not be shared between episodes).
SKILLS = {
    "pick_cube": lambda: pick("cube"),

    "stack_blocks": lambda: pick("block_a") + place_on("block_a", "block_b"),

    "bin_picking": lambda: pick("peg") + drop_in("peg", "bin"),
}


def make_skill(task):
    """A fresh oracle for ``task``. Raises KeyError for a task with no oracle."""
    if task not in SKILLS:
        raise KeyError(f"no oracle for task {task!r}; have {sorted(SKILLS)}")
    return Skill(SKILLS[task](), name=task)


def available():
    """Task names that have an oracle."""
    return sorted(SKILLS)
