"""VR relative-clutch teleop mechanism -- the reusable core of the old VRTeleop.

Maps a Meta-Quest hand pose (delivered over TCP as a ``VRSnapshot``; see
``teleop/vr_server.py``) to a commanded end-effector pose using a grip-triggered
clutch:

  * grip trigger = clutch. Hold to drive the arm; release to freeze it in place.
    On each fresh press we re-anchor (capture the hand pose and the current EE
    pose), so the arm never jumps and you can re-grip from a comfortable spot.
  * While engaged we map the *relative* motion from the anchor:
        EE_pos = EE_anchor + scale * (hand_now - hand_anchor)
        EE_R   = (hand_now_R . hand_anchor_R^T) . EE_anchor_R
  * an optional low-pass (``alpha`` per substep) smooths bursty VR input.

This owns no control loop, viewer, or sim -- the unified GUI's ``ControlSession``
drives it in its "vr" mode (``update`` per tick, ``advance`` per substep,
``command`` to get the ``CartesianPose`` to stream).
"""
import math

import numpy as np

from robot import CartesianPose

GRIP_ENGAGE = 0.5       # grip trigger above this = clutch engaged
SMOOTH_TAU = 0.0        # default command low-pass time constant (s); 0 = off.
                        # The sim's position servo already smooths, so this is
                        # off by default (adds latency); raise it (e.g. 0.05) if
                        # a real headset's input jitter shows through.


def slerp_toward(R_cur, R_tgt, a):
    """Rotate ``R_cur`` a fraction ``a`` in [0,1] of the way toward ``R_tgt``.

    Geodesic (shortest-arc) step: take the relative rotation, scale its angle by
    ``a`` (Rodrigues), and apply it. Used to low-pass the commanded orientation
    so bursty VR input becomes smooth wrist motion. Per-tick deltas are tiny, so
    the near-180 degrees degeneracy is not a concern here."""
    R_rel = R_tgt @ R_cur.T
    cos_ang = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    ang = math.acos(cos_ang)
    if ang < 1e-8:
        return R_tgt.copy()
    axis = np.array([R_rel[2, 1] - R_rel[1, 2],
                     R_rel[0, 2] - R_rel[2, 0],
                     R_rel[1, 0] - R_rel[0, 1]]) / (2.0 * math.sin(ang))
    da = a * ang
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    R_step = np.eye(3) + math.sin(da) * K + (1.0 - math.cos(da)) * (K @ K)
    return R_step @ R_cur


class VRClutch:
    """Grip-clutched mapping from VR hand pose to a commanded EE pose.

    Holds two poses: ``cmd_pos``/``cmd_R`` is the raw clutch target; the
    ``*_filt`` pair is the low-passed pose actually streamed to the arm (equal to
    the target when ``alpha == 1``, i.e. smoothing off).
    """

    def __init__(self, position_scale=1.0, alpha=1.0):
        self.position_scale = float(position_scale)
        self.alpha = float(alpha)
        self.engaged = False
        self._anchor = None                 # (hand_tf, ee_pos, ee_R) at engage
        self.cmd_pos = self.cmd_R = None
        self.cmd_pos_filt = self.cmd_R_filt = None

    def reset(self, ee_pos, ee_R):
        """Point the commanded pose (target + filtered) at the current EE, so
        teleop resumes from HOME/recover/reset without a jump, and drop the
        anchor so the next grip re-anchors cleanly."""
        self.cmd_pos = np.asarray(ee_pos, float).copy()
        self.cmd_R = np.asarray(ee_R, float).copy()
        self.cmd_pos_filt = self.cmd_pos.copy()
        self.cmd_R_filt = self.cmd_R.copy()
        self.engaged = False
        self._anchor = None

    def update(self, snap, ee_pos, ee_R, active=True):
        """Per tick: capture the anchor on the grip's rising edge, then map the
        hand's relative motion onto the commanded EE pose. ``active=False`` (e.g.
        a HOME move is playing) forces disengage so we re-anchor cleanly after."""
        engaged = bool(active and snap.tracking and snap.grip > GRIP_ENGAGE)
        if engaged and not self.engaged:
            self._anchor = (snap.hand_tf.copy(),
                            np.asarray(ee_pos, float).copy(),
                            np.asarray(ee_R, float).copy())
        if engaged:
            hand0, ee_pos0, ee_R0 = self._anchor
            d_pos = snap.hand_tf[:3, 3] - hand0[:3, 3]
            self.cmd_pos = ee_pos0 + self.position_scale * d_pos
            R_delta = snap.hand_tf[:3, :3] @ hand0[:3, :3].T
            self.cmd_R = R_delta @ ee_R0
        self.engaged = engaged

    def advance(self):
        """Per substep: slew the filtered command toward the target (position:
        exponential lerp; orientation: geodesic slerp)."""
        a = self.alpha
        self.cmd_pos_filt += a * (self.cmd_pos - self.cmd_pos_filt)
        self.cmd_R_filt = slerp_toward(self.cmd_R_filt, self.cmd_R, a)

    def command(self):
        """The filtered commanded pose as a ``CartesianPose`` to stream."""
        return CartesianPose.from_matrix(self.cmd_pos_filt, self.cmd_R_filt)
