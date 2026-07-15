"""Task registry.

A *base* is a robot + environment scene (floor, table, lights, camera) with no
task objects. A *task* picks a base and lists the objects to drop into it.

Add a new task by adding one entry to ``TASKS``; reuse object kinds from
``objects.py`` (box / sphere / cylinder / capsule / ellipsoid / bin). Table top
surface is at z = 0, so a body sits flush when ``pos_z == its half-height``.

An object may carry an optional ``rand`` dict giving per-axis sampling ranges for
domain randomization -- ``x``/``y``/``z`` in metres and ``yaw`` in radians, e.g.
``rand=dict(x=(0.42, 0.60), y=(-0.18, 0.18), yaw=(-math.pi, math.pi))``. Axes not
listed keep the declared ``pos``/orientation. ``SimRobot.randomize_objects`` (the
GUI "Randomize" button) draws a fresh layout from these ranges; ``rand`` is not a
geometry field (the builder strips it).
"""

import math

# Base scenes: name -> scene file (relative to this repo root). SimRobot needs
# the Franka Hand (it reads the ``hand`` body for the collision reflex), so a
# base must include the gripper.
BASES = {
    "fr3_with_gripper": "models/fr3_with_gripper/scene.xml",    # gripper + table
}

# Convenient colors.
RED = (0.85, 0.15, 0.15, 1)
GREEN = (0.15, 0.6, 0.2, 1)
BLUE = (0.2, 0.35, 0.8, 1)

TASKS = {
    # Just the base scene, nothing to manipulate.
    "empty": dict(base="fr3_with_gripper", objects=[]),

    # Single cube to pick up. Sized well within the 0.08 m gripper opening.
    "pick_cube": dict(base="fr3_with_gripper", objects=[
        dict(kind="box", name="cube", pos=[0.5, 0.0, 0.02],
             size=[0.02, 0.02, 0.02], rgba=RED,
             rand=dict(x=(0.42, 0.60), y=(-0.18, 0.18), yaw=(-math.pi, math.pi))),
    ]),

    # Two blocks to stack (kept in separate y-halves so a random layout never
    # overlaps them).
    "stack_blocks": dict(base="fr3_with_gripper", objects=[
        dict(kind="box", name="block_a", pos=[0.45, 0.12, 0.02],
             size=[0.02, 0.02, 0.02], rgba=RED,
             rand=dict(x=(0.42, 0.58), y=(0.05, 0.20), yaw=(-math.pi, math.pi))),
        dict(kind="box", name="block_b", pos=[0.55, -0.12, 0.02],
             size=[0.02, 0.02, 0.02], rgba=BLUE,
             rand=dict(x=(0.42, 0.58), y=(-0.20, -0.05), yaw=(-math.pi, math.pi))),
    ]),

    # A cylinder to drop into a bin (the bin is a static fixture, not randomized).
    "bin_picking": dict(base="fr3_with_gripper", objects=[
        dict(kind="bin", name="bin", pos=[0.6, 0.15, 0.0]),
        dict(kind="cylinder", name="peg", pos=[0.45, -0.1, 0.04],
             size=[0.015, 0.04], rgba=GREEN,
             rand=dict(x=(0.40, 0.55), y=(-0.18, 0.05), yaw=(-math.pi, math.pi))),
    ]),
}
