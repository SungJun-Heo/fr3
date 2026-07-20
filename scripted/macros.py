"""Reusable manipulation chunks -- the vocabulary tasks are written in.

``pick`` appears in ALL THREE shipped tasks, so it belongs here rather than in
any one of them. Each macro returns a list of phases, so a task is macros
concatenated (see ``skills.py``).

Nothing here names a task, and nothing takes a hand-tuned height: grasp and
release heights are derived from the object's live pose plus its declared shape
(``scene/shapes.py``). That is what keeps a new task down to one table entry --
declare the object in ``tasks.py`` and its geometry is already known here.
"""

import numpy as np

from robot.gripper import MAX_WIDTH
from scene.shapes import grasp_span, half_height, rim_z
from scripted.base import (FLANGE_TO_PAD, Check, Hold, Plan, Servo, Transit,
                           grasp_R)

HOVER_Z = 0.30           # flange height for free-space transit (m)
DESCEND_SPEED = 0.12     # slower than transit: precision near contact (m/s)
GRIP_TICKS = 30          # ticks held while the fingers close or open

# Fingers must leave some slack around the object or the approach clips it.
SPAN_MARGIN = 0.008

# Free-space legs only need to arrive roughly; insisting on millimetres there
# stalls, because a loaded arm settles with a steady-state droop that no amount
# of extra time removes.
TRANSIT_TOL = 0.02


# -- goal helpers (evaluated fresh every tick -> closed loop) -----------------

def over(name, height):
    """The object's live xy, at an absolute flange ``height``."""
    return lambda c: np.array([c.obj(name)[0], c.obj(name)[1], float(height)])


def grasp_pose(name):
    """Flange position that puts the finger pads at the object's centre."""
    def goal(c):
        p = c.obj(name)
        return np.array([p[0], p[1], p[2] + FLANGE_TO_PAD])
    return goal


def straight_up(height):
    """Directly above the CURRENT flange position. Always used with
    ``latch=True`` -- see ``Servo``."""
    return lambda c: np.array([c.ee_pos[0], c.ee_pos[1], float(height)])


# -- macros ------------------------------------------------------------------

def pick(name, hover=HOVER_Z):
    """Grasp ``name`` top-down and lift it clear.

    Order matters: the grasp POSTURE is chosen first (``Plan``), the approach
    then flies to the hover pose in that same posture (``Transit`` seeded with
    it), and only the short final descent is a Cartesian servo. Approaching in a
    posture that does not suit the grasp is what strands the arm near a
    singularity part-way down -- a failure the streaming controller cannot back
    out of, because escaping needs a posture change it will not make.

    Aborts before moving if the object is too wide for the gripper or no
    comfortable posture reaches it: failing on tick 1 beats tripping half-way
    through a recorded episode."""
    grasp_key = f"q_grasp:{name}"
    return [
        Check(lambda c: grasp_span(c.spec(name)) < MAX_WIDTH - SPAN_MARGIN,
              reason=f"{name} is too wide for the gripper",
              name=f"check:{name}:span"),
        Plan(lambda c: (grasp_pose(name)(c), grasp_R(c.obj_yaw(name))),
             key=grasp_key,
             reason=f"no comfortable posture grasps {name}",
             name=f"plan:{name}"),
        Transit(over(name, hover), grip=1.0, align_to=name, from_key=grasp_key,
                name=f"approach:{name}"),
        Servo(grasp_pose(name), grip=1.0, align_to=name, speed=DESCEND_SPEED,
              name=f"descend:{name}"),
        Hold(grip=0.0, ticks=GRIP_TICKS, name=f"close:{name}"),
        Servo(straight_up(hover), latch=True, tol=TRANSIT_TOL,
              name=f"lift:{name}"),
    ]


def place_on(carried, target, hover=HOVER_Z):
    """Release the carried object resting on top of ``target``.

    The release height is derived: top of ``target`` + half of ``carried`` +
    the flange-to-pad offset. No per-task constant."""
    def release(c):
        t = c.obj(target)
        z = (t[2] + half_height(c.spec(target))
             + half_height(c.spec(carried)) + FLANGE_TO_PAD)
        return np.array([t[0], t[1], z])

    key = f"q_place:{target}"
    return [
        Plan(lambda c: (release(c), grasp_R(c.obj_yaw(target))), key=key,
             reason=f"no comfortable posture places on {target}",
             name=f"plan:{target}"),
        Transit(over(target, hover), from_key=key, name=f"carry:{target}"),
        Servo(release, speed=DESCEND_SPEED, name=f"lower:{carried}"),
        Hold(grip=1.0, ticks=GRIP_TICKS, name=f"release:{carried}"),
        Servo(straight_up(hover), latch=True, tol=TRANSIT_TOL,
              name=f"retreat:{carried}"),
    ]


def drop_in(carried, container, clearance=0.04, hover=HOVER_Z):
    """Release the carried object above ``container`` so it falls inside.

    Not a lowering motion: the walls make descending into the cavity a collision
    risk, so the object is carried over the rim and dropped. The height clears
    the rim by the carried object's own half-height plus ``clearance``."""
    def drop(c):
        b = c.obj(container)
        z = (rim_z(c.spec(container)) + half_height(c.spec(carried))
             + FLANGE_TO_PAD + clearance)
        return np.array([b[0], b[1], z])

    key = f"q_drop:{container}"
    return [
        Plan(lambda c: (drop(c), grasp_R(0.0)), key=key,
             reason=f"no comfortable posture reaches over {container}",
             name=f"plan:{container}"),
        Transit(drop, from_key=key, name=f"carry:{container}"),
        Hold(grip=1.0, ticks=GRIP_TICKS, name=f"drop:{carried}"),
        Servo(straight_up(hover), latch=True, tol=TRANSIT_TOL,
              name=f"retreat:{carried}"),
    ]
