"""VR teleoperation for the FR3 sim.

Ports camel-RBY1's VR teleop (Unity/Meta-Quest -> TCP JSON -> coordinate
transform -> relative-clutch EE mapping) onto fr3's single-arm Cartesian control
path. ``vr_server`` is the input transport; ``clutch`` maps a hand pose to a
commanded EE pose (the reusable mechanism the unified GUI's ``ControlSession``
drives in "vr" mode); ``mock_vr_client`` drives the pipeline without a headset.
"""

from teleop.vr_server import (
    VRState, VRSnapshot, VRTeleopServer, meta_pose_to_robot, quat_xyzw_to_mat,
)
from teleop.clutch import VRClutch, slerp_toward, GRIP_ENGAGE, SMOOTH_TAU

__all__ = [
    "VRState", "VRSnapshot", "VRTeleopServer",
    "meta_pose_to_robot", "quat_xyzw_to_mat",
    "VRClutch", "slerp_toward", "GRIP_ENGAGE", "SMOOTH_TAU",
]
