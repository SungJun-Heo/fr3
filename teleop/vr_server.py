"""VR controller TCP server + shared state -- the fr3 equivalent of camel-RBY1's
``OPRVRJoystickServer`` and ``shm.VRData``.

A Unity (Meta Quest) app streams one JSON object per line over TCP: headset and
per-hand pose (position + quaternion), the index/grip triggers, thumbstick, and
A/B/X/Y buttons. This server reads ONE hand (single-arm FR3), converts its pose
from the Meta coordinate frame to the robot/MuJoCo base frame, and publishes the
result to a ``VRState`` that the teleop control loop reads (see
``teleop/vr_teleop.py``).

Why this is a near-copy of RBY1 but smaller:
  * RBY1 is dual-arm and routes VR into a ``SharedMemory`` object consumed by a
    250 Hz controller thread that runs its own IK. Here the arm is single, and
    ``SimRobot``'s Cartesian path already does the IK + safety (mirroring the
    real robot's firmware), so the server only has to produce *one hand pose*.
  * RBY1's "SharedMemory" is not real inter-process memory -- the VR server runs
    as a daemon *thread* in the same process and writes a plain Python object
    (``shm.VRData``) lock-free. We do the same, except ``VRState`` takes a lock
    so the control loop always reads a consistent single-frame snapshot.

Coordinate mapping (identical convention to RBY1):
  * Meta axes (x:right, y:up, z:back) -> robot base via the basis matrix ``A``.
    Position becomes ``A @ p`` = ``[z, -x, y]``; rotation becomes ``A R Aᵀ``.
  * Unity reports a lost hand as position exactly ``(0,0,0)``; on that frame we
    keep the previous pose (so the robot freezes instead of snapping to base).

Note on the grip-alignment offset: RBY1 right-multiplies a fixed rotation onto
the published hand pose to line the controller grip up with the EE. The teleop
consumer here maps *relative* rotation (ΔR = R_now · R_refᵀ from the clutch
anchor), and any constant right-multiplied offset cancels in that difference --
so we deliberately do NOT apply one here.
"""

from dataclasses import dataclass
import json
import socket
import threading

import numpy as np

# Meta -> robot base change-of-basis. Applied to a Meta vector it yields
# [z, -x, y] (the position mapping); applied as ``A R Aᵀ`` it re-expresses a
# Meta rotation in the robot frame. Same matrix RBY1 calls ``self.A``.
META_TO_ROBOT = np.array([[0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])

# Keys for each hand in the Unity JSON. Selecting a hand picks its pose, its two
# triggers, and the button we treat as "go HOME / recover" (B for the right
# controller, Y for the left -- the buttons that fall under the thumb).
_HAND_KEYS = {
    "right": dict(pos="rightHandPos", rot="rightHandRot",
                  grip="rightGripTrigger", index="rightIndexTrigger",
                  home="buttonB"),
    "left": dict(pos="leftHandPos", rot="leftHandRot",
                 grip="leftGripTrigger", index="leftIndexTrigger",
                 home="buttonY"),
}


def quat_xyzw_to_mat(x, y, z, w):
    """Unit-ish quaternion ``(x, y, z, w)`` -> 3x3 rotation matrix.

    Meta/Unity reports quaternions in ``(x, y, z, w)`` order. We do this in
    numpy (rather than pull in scipy, which fr3 does not depend on); the formula
    is the standard one and tolerates a non-normalized input by dividing by the
    squared norm."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def meta_pose_to_robot(pos, rot):
    """Unity hand pose dicts -> ``(T, tracked)``.

    ``pos``/``rot`` are the ``{x,y,z}`` / ``{x,y,z,w}`` dicts from the JSON.
    Returns a 4x4 homogeneous pose in the robot base frame and a ``tracked``
    flag that is False when Unity signalled tracking loss (position exactly
    zero). On a lost frame the returned ``T`` is identity and should be ignored
    by the caller (which holds the previous pose)."""
    x = pos.get("x", 0.0)
    y = pos.get("y", 0.0)
    z = pos.get("z", 0.0)
    if x == 0.0 and y == 0.0 and z == 0.0:
        return np.eye(4), False
    T = np.eye(4)
    T[:3, 3] = META_TO_ROBOT @ np.array([x, y, z])
    R_meta = quat_xyzw_to_mat(rot.get("x", 0.0), rot.get("y", 0.0),
                              rot.get("z", 0.0), rot.get("w", 1.0))
    T[:3, :3] = META_TO_ROBOT @ R_meta @ META_TO_ROBOT.T
    return T, True


@dataclass
class VRSnapshot:
    """One consistent frame of VR input, handed to the control loop.

    ``hand_tf`` is the mapped hand pose (robot frame). ``grip`` is the clutch
    trigger (hold to move the robot); ``trigger`` is the index trigger (gripper).
    ``home`` requests HOME/recover. ``connected`` / ``tracking`` let the loop
    decide whether the pose is trustworthy this tick."""
    hand_tf: np.ndarray
    grip: float
    trigger: float
    home: bool
    connected: bool
    tracking: bool


class VRState:
    """Thread-shared VR input -- the fr3 stand-in for RBY1's ``shm.VRData``.

    The server thread calls ``publish`` each frame; the control loop calls
    ``snapshot`` each tick. A single lock guards all fields so the loop never
    reads position from a new frame together with rotation from an old one (the
    one correctness upgrade over RBY1's lock-free in-place writes)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._hand_tf = np.eye(4)
        self._grip = 0.0
        self._trigger = 0.0
        self._home = False
        self._connected = False
        self._tracking = False

    def publish(self, hand_tf, grip, trigger, home, tracking):
        with self._lock:
            self._hand_tf = hand_tf
            self._grip = float(grip)
            self._trigger = float(trigger)
            self._home = bool(home)
            self._tracking = bool(tracking)

    def set_connected(self, flag):
        """Flag (dis)connection. On disconnect we also clear ``tracking`` so a
        dropped client cannot leave the clutch believing the hand is live."""
        with self._lock:
            self._connected = bool(flag)
            if not flag:
                self._tracking = False

    def snapshot(self):
        with self._lock:
            return VRSnapshot(self._hand_tf.copy(), self._grip, self._trigger,
                              self._home, self._connected, self._tracking)


class VRTeleopServer:
    """TCP listener that parses one hand's Unity stream into a ``VRState``.

    Mirrors ``OPRVRJoystickServer.openServer``: bind, accept a single client,
    and split the byte stream on newlines into JSON frames. Unlike RBY1 it loops
    back to ``accept`` after a disconnect so the headset can reconnect without
    restarting the process, and it honours ``stop()`` for a clean shutdown."""

    def __init__(self, state: VRState, hand="right", host="0.0.0.0", port=8081):
        if hand not in _HAND_KEYS:
            raise ValueError(f"hand must be 'right' or 'left', got {hand!r}")
        self.state = state
        self.hand = hand
        self.host = host
        self.port = port
        self._keys = _HAND_KEYS[hand]
        # Last valid hand pose, held across tracking-loss frames so the robot
        # freezes (rather than jumping) when the hand leaves the headset's view.
        self._last_tf = np.eye(4)
        self._shutdown = threading.Event()

    def serve_forever(self):
        """Accept clients and process their frames until ``stop()``.

        Runs as a daemon thread from the teleop loop. The listen socket has a
        1 s timeout so the accept loop can notice ``stop()``; each connection's
        recv also times out so a silent client cannot wedge shutdown."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            s.settimeout(1.0)
            print(f"[vr] waiting for a VR client on {self.host}:{self.port} "
                  f"({self.hand} hand)...")
            while not self._shutdown.is_set():
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue
                print(f"[vr] connected: {addr}")
                self.state.set_connected(True)
                try:
                    self._handle_client(conn)
                finally:
                    conn.close()
                    self.state.set_connected(False)
                    self._last_tf = np.eye(4)  # forget stale pose on disconnect
                    print("[vr] client disconnected")

    def stop(self):
        self._shutdown.set()

    def _handle_client(self, conn):
        conn.settimeout(1.0)
        buffer = ""
        while not self._shutdown.is_set():
            try:
                data = conn.recv(4096)
            except socket.timeout:
                continue
            if not data:
                return  # peer closed the socket
            buffer += data.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                try:
                    self._process(json.loads(line))
                except json.JSONDecodeError:
                    # A truncated/partial line -- drop it; the next one recovers.
                    pass

    def _process(self, vr):
        """Turn one decoded JSON frame into a ``VRState.publish`` call."""
        k = self._keys
        tf, tracked = meta_pose_to_robot(vr.get(k["pos"], {}), vr.get(k["rot"], {}))
        if tracked:
            self._last_tf = tf
        # grip = clutch (hold to move the robot); index = gripper; button = HOME.
        # Triggers/button are read every frame so releasing still registers even
        # if the hand is momentarily untracked.
        self.state.publish(
            hand_tf=self._last_tf.copy(),
            grip=vr.get(k["grip"], 0.0),
            trigger=vr.get(k["index"], 0.0),
            home=vr.get(k["home"], False),
            tracking=tracked,
        )
