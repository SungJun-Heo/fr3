"""Viewer-overlay debug drawing, shared by ``teleop`` and the examples.

Helpers that push geoms into a passive viewer's ``user_scn`` so you can *see*
what the controller is doing -- path markers, and pose frames for comparing a
commanded (target) pose against the actual one. Pure visualization: nothing here
touches robot state or the physics, so it sits above every package and below
none.
"""

import numpy as np
import mujoco


def add_marker(scene, pos, rgba, size=0.006):
    """Add a small sphere to a viewer's user scene (skipped if it's full).

    Used to draw target vs actual EE paths in the trajectory demos."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, 0.0, 0.0]), np.asarray(pos, float),
        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def add_frame(scene, pos, R, length=0.1, width=0.005, alpha=1.0):
    """Draw a coordinate frame (x=red, y=green, z=blue) at ``pos`` / ``R``.

    Three capsules from ``pos`` along the rotation's columns -- the readable way
    to show a full pose (position *and* orientation) in the viewer. Pass
    ``alpha`` < 1 for a translucent 'ghost', e.g. a commanded target drawn behind
    the solid actual pose so you can watch one track the other."""
    pos = np.asarray(pos, float)
    R = np.asarray(R, float)
    for i, rgb in enumerate(((1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.3, 0.5, 1.0))):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray((*rgb, alpha), np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             pos, pos + length * R[:, i])
        scene.ngeom += 1
