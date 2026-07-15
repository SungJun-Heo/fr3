"""Sim camera rendering -- the one home for "render named model cameras".

``SimCameraRenderer`` wraps a single ``mujoco.Renderer`` and turns the model's
named cameras (``front``, ``wrist``, ...) into RGB frames, plus exposes their
calibration (intrinsic fovy + extrinsic pose) for the episode metadata. It is
the sim analog of camel-franka's RealSense ``camera.py`` and of camel-RBY1's
per-task hand-eye JSON -- except here every number is exact, straight from the
compiled model.

Camera pose handling:
  * A *static* camera (parent = the world body, e.g. ``front``) has a constant
    world pose -> stored once in ``specs()``.
  * A *moving* camera (mounted on a body, e.g. ``wrist`` on the ``d435i`` link)
    moves with the arm -> its world pose is read per frame via
    ``extrinsics_world()`` and logged as a ``(T,4,4)`` array by the recorder.

Rendering returns RGB (MuJoCo's native order), so downstream writes with PIL
directly -- no BGR<->RGB juggling (the channel-order bug camel-franka guards
against only exists because RealSense delivers BGR).
"""

import mujoco
import numpy as np


class SimCameraRenderer:
    """Render + calibrate a fixed set of model cameras from live ``data``."""

    def __init__(self, model, data, cameras=("front", "wrist"), width=640,
                 height=480):
        self.cameras = tuple(cameras)
        self.width = int(width)
        self.height = int(height)
        self._renderer = None
        self.rebind(model, data)   # binds model/data, cam ids, and the GL context

    def rebind(self, model, data):
        """(Re)bind to a model/data pair, rebuilding the GL context.

        Called on construction and after a task reload (``reload_task`` rebuilds
        the model, staling the old renderer's context and camera ids)."""
        self.close()
        self.model = model
        self.data = data
        # model.camera(name) raises if a configured camera is missing -> fail loud.
        self._cam_ids = {name: model.camera(name).id for name in self.cameras}
        self._renderer = mujoco.Renderer(model, self.height, self.width)

    def render(self):
        """Render every configured camera -> ``{name: (H,W,3) uint8 RGB}``.

        Copies each frame out of the renderer's reused internal buffer."""
        out = {}
        for name in self.cameras:
            self._renderer.update_scene(self.data, camera=name)
            out[name] = self._renderer.render().copy()
        return out

    def extrinsics_world(self):
        """World-frame 4x4 pose of each camera from the *current* ``data``.

        Requires up-to-date kinematics (true right after ``mj_step``/
        ``mj_forward`` -- the state the control loop leaves ``data`` in). For a
        moving camera this differs every frame."""
        out = {}
        for name in self.cameras:
            cid = self._cam_ids[name]
            T = np.eye(4)
            T[:3, 3] = self.data.cam_xpos[cid]
            T[:3, :3] = self.data.cam_xmat[cid].reshape(3, 3)
            out[name] = T
        return out

    def specs(self):
        """Per-camera calibration for ``meta.json``.

        Each entry: render size, ``fovy`` (vertical FoV, deg -> the intrinsic),
        mount (``world`` vs a body), the parent body, the camera pose *relative*
        to that parent (constant), and ``moving``. A static camera also carries
        its constant ``extrinsic_world_4x4``; a moving one's world pose is logged
        per frame by the recorder instead."""
        ext = self.extrinsics_world()
        specs = {}
        for name in self.cameras:
            cid = self._cam_ids[name]
            body_id = int(self.model.cam_bodyid[cid])
            moving = body_id != 0      # body 0 is the world body -> static camera
            spec = dict(
                width=self.width,
                height=self.height,
                fovy=float(self.model.cam_fovy[cid]),
                mount="body" if moving else "world",
                parent_body=self.model.body(body_id).name,
                cam_pos_rel=self.model.cam_pos[cid].tolist(),
                cam_quat_rel=self.model.cam_quat[cid].tolist(),
                moving=moving,
            )
            if not moving:
                spec["extrinsic_world_4x4"] = ext[name].tolist()
            specs[name] = spec
        return specs

    def close(self):
        """Release the GL context (avoids the EGL teardown warning at GC)."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
