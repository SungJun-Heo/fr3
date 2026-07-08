"""Task registry.

A *base* is a robot + environment scene (floor, table, lights, camera) with no
task objects. A *task* picks a base and lists the objects to drop into it.

Add a new task by adding one entry to ``TASKS``; reuse object kinds from
``objects.py`` (box / sphere / cylinder / capsule / ellipsoid / bin). Table top
surface is at z = 0, so a body sits flush when ``pos_z == its half-height``.
"""

# Base scenes: name -> scene file (relative to this repo root).
BASES = {
    "fr3": "models/fr3/scene.xml",                              # arm only, no table
    "fr3_with_gripper": "models/fr3_with_gripper/scene.xml",    # gripper + table
}

# Convenient colors.
RED = (0.85, 0.15, 0.15, 1)
GREEN = (0.15, 0.6, 0.2, 1)
BLUE = (0.2, 0.35, 0.8, 1)

TASKS = {
    # Just the base scene, nothing to manipulate.
    "empty": dict(base="fr3_with_gripper", objects=[]),

    # Single cube to pick up.
    "pick_cube": dict(base="fr3_with_gripper", objects=[
        dict(kind="box", name="cube", pos=[0.5, 0.0, 0.03],
             size=[0.03, 0.03, 0.03], rgba=RED),
    ]),

    # Two blocks to stack.
    "stack_blocks": dict(base="fr3_with_gripper", objects=[
        dict(kind="box", name="block_a", pos=[0.45, 0.12, 0.03],
             size=[0.03, 0.03, 0.03], rgba=RED),
        dict(kind="box", name="block_b", pos=[0.55, -0.12, 0.03],
             size=[0.03, 0.03, 0.03], rgba=BLUE),
    ]),

    # A cylinder to drop into a bin.
    "bin_picking": dict(base="fr3_with_gripper", objects=[
        dict(kind="bin", name="bin", pos=[0.6, 0.15, 0.0]),
        dict(kind="cylinder", name="peg", pos=[0.45, -0.1, 0.05],
             size=[0.02, 0.05], rgba=GREEN),
    ]),
}
