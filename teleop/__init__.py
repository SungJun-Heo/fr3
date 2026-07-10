"""VR teleoperation for the FR3 sim.

Ports camel-RBY1's VR teleop (Unity/Meta-Quest -> TCP JSON -> coordinate
transform -> relative-clutch EE mapping) onto fr3's single-arm Cartesian control
path. See ``vr_server`` (input) and ``vr_teleop`` (control loop); ``mock_vr_client``
drives the pipeline without a headset.
"""

from teleop.vr_server import (
    VRState, VRSnapshot, VRTeleopServer, meta_pose_to_robot, quat_xyzw_to_mat,
)
from teleop.vr_teleop import VRTeleop

__all__ = [
    "VRState", "VRSnapshot", "VRTeleopServer", "VRTeleop",
    "meta_pose_to_robot", "quat_xyzw_to_mat",
]
