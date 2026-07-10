"""Small helpers shared by the example scripts.

Only *viewer-side* conveniences live here (drawing debug markers). The pose /
command conventions are library concerns and live in ``robot`` -- import
``pose_to_vec`` / ``vec_to_pose`` / ``CartesianPose.from_matrix`` from there.
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
