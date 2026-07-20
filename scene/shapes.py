"""Geometry facts derived from a task's object declarations.

``tasks.py`` already states each object's ``kind`` and ``size``. Everything a
grasp needs follows from those two fields, so this module DERIVES them instead
of making each task restate them:

  * ``symmetry_of``  -- the rotation that maps the object onto itself, used to
    pick the equivalent grasp that turns the wrist least.
  * ``half_height``  -- centre-to-top, for stacking and for placing.
  * ``grasp_span``   -- how wide the fingers must open to clear the object.
  * ``rim_z``        -- top of a container's wall (bins), for release height.

Why derivation matters: an oracle that hard-codes "a cube has 90 degrees
symmetry" breaks silently the first time a task declares a non-square box (the
fingers would close across the LONG side and the grasp slips). Reading it from
the spec means adding a task never involves restating geometry -- see
``scripted/macros.py``, which is the only consumer.

The spec lookup mirrors ``SimRobot.__init__`` (``robot/sim_robot.py``), which
reads the same ``TASKS[task]["objects"]`` list for randomization ranges.
"""

import numpy as np

from scene.tasks import TASKS

# Mirrored from ``scene.objects.add_bin`` defaults. A bin's walls are built as
# geoms of half-height ``height`` centred at z = ``height``, so the rim sits at
# 2 * height above the body origin -- not at ``height``.
BIN_HEIGHT = 0.05
BIN_WALL = 0.005

# Kinds with continuous rotational symmetry about z: every yaw gives the same
# grasp, so the wrist should not turn at all.
_CONTINUOUS = ("cylinder", "sphere", "capsule")


def object_spec(task, name):
    """The declaration dict for one object of ``task`` (from ``tasks.py``)."""
    for obj in TASKS.get(task, {}).get("objects", []):
        if obj["name"] == name:
            return obj
    raise KeyError(f"task {task!r} declares no object named {name!r}")


def symmetry_of(spec):
    """Rotation period about z that leaves the object looking identical (rad).

    ``None`` means continuous symmetry -- there is no preferred yaw, so a grasp
    should not rotate the wrist at all. ``2*pi`` means no symmetry: the grasp
    must match the object's yaw exactly."""
    kind = spec["kind"]
    if kind in _CONTINUOUS:
        return None
    size = spec.get("size", ())
    if kind in ("box", "ellipsoid"):
        sx, sy = float(size[0]), float(size[1])
        square = abs(sx - sy) < 1e-9
        if kind == "ellipsoid" and square:
            return None                 # a spheroid: continuous about z
        return np.pi / 2 if square else np.pi
    return 2.0 * np.pi                  # unknown/composite: assume asymmetric


def half_height(spec):
    """Centre-to-top along z (m). What to add to a resting surface to find the
    centre height of an object placed on it."""
    kind, size = spec["kind"], spec.get("size", ())
    if kind == "box" or kind == "ellipsoid":
        return float(size[2])
    if kind == "cylinder":
        return float(size[1])           # (radius, half-length)
    if kind == "capsule":
        return float(size[0]) + float(size[1])
    if kind == "sphere":
        return float(size[0])
    raise ValueError(f"half_height undefined for kind {kind!r}")


def grasp_span(spec):
    """Width the fingers must span to grip the object across its narrow axis (m).

    A square box is gripped across a face (2 * half-extent); a rectangular one
    across its SHORT side, which is why the narrow axis is what matters."""
    kind, size = spec["kind"], spec.get("size", ())
    if kind in ("box", "ellipsoid"):
        return 2.0 * min(float(size[0]), float(size[1]))
    if kind in ("cylinder", "capsule"):
        return 2.0 * float(size[0])     # diameter
    if kind == "sphere":
        return 2.0 * float(size[0])
    raise ValueError(f"grasp_span undefined for kind {kind!r}")


def rim_z(spec):
    """World height of a container's wall top (m) -- what a carried object must
    clear before it can be released inside. Bins only."""
    if spec["kind"] != "bin":
        raise ValueError(f"rim_z is only defined for containers, got {spec['kind']!r}")
    height = float(spec.get("height", BIN_HEIGHT))
    return float(spec["pos"][2]) + 2.0 * height


def cavity_floor_z(spec):
    """World height of a container's inner floor (m)."""
    if spec["kind"] != "bin":
        raise ValueError(f"cavity_floor_z is only defined for containers, got {spec['kind']!r}")
    wall = float(spec.get("wall", BIN_WALL))
    return float(spec["pos"][2]) + 2.0 * wall
