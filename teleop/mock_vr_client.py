"""Synthetic Meta-Quest client -- exercise the VR teleop path without a headset.

Connects to ``VRTeleopServer`` and streams the same newline-delimited JSON a
Unity app would, driving the right hand through a scripted motion so you can see
the FR3 follow. It is both a manual sanity tool (run ``python main.py``, select
the "VR" mode, then run this in a second terminal) and the input side of the
end-to-end test.

Timeline (default): a settle phase with the clutch released (robot must stay
put), then the grip is held and the hand traces a circle in the Meta y/z plane
(so the EE sweeps a circle in robot z/x) while the index trigger pulses closed
halfway through (so the gripper visibly closes). Poses are sent in the Meta
frame (x:right, y:up, z:back); the server maps them to the robot frame.
"""

import argparse
import json
import math
import socket
import time

# A plausible resting hand pose in the Meta frame: out in front, chest height.
# Deliberately non-zero so the server never reads it as tracking-loss (0,0,0).
_HAND_BASE = (0.10, 1.10, -0.30)
_HEADSET = {"x": 0.0, "y": 1.60, "z": 0.0}
_IDENT_QUAT = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def _frame(hand_pos, grip, index, home):
    """Build one Unity-shaped JSON frame (right hand active, left hand neutral)."""
    return {
        "headsetPos": _HEADSET,
        "headsetRot": _IDENT_QUAT,
        "rightHandPos": {"x": hand_pos[0], "y": hand_pos[1], "z": hand_pos[2]},
        "rightHandRot": _IDENT_QUAT,
        "rightIndexTrigger": index,
        "rightGripTrigger": grip,
        "rightThumbstick": {"x": 0.0, "y": 0.0},
        "buttonA": False,
        "buttonB": home,
        "rightThumbstickClick": False,
        # Left hand neutral/untracked -- the server only reads the active hand.
        "leftHandPos": {"x": 0.0, "y": 0.0, "z": 0.0},
        "leftHandRot": _IDENT_QUAT,
        "leftIndexTrigger": 0.0,
        "leftGripTrigger": 0.0,
        "leftThumbstick": {"x": 0.0, "y": 0.0},
        "buttonX": False,
        "buttonY": False,
        "leftThumbstickClick": False,
    }


def run(host="127.0.0.1", port=8081, duration=6.0, hz=50.0, radius=0.08,
        settle=1.0, connect_timeout=10.0):
    """Stream the scripted motion for ``duration`` seconds.

    Retries the connection for ``connect_timeout`` seconds so the client can be
    started before or after the server. ``radius`` is the circle size in metres
    (Meta frame); ``settle`` seconds run with the clutch released first."""
    deadline = time.time() + connect_timeout
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=2.0)
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.2)
    print(f"[mock] connected to {host}:{port}")

    dt = 1.0 / hz
    t0 = time.time()
    try:
        with sock:
            while True:
                t = time.time() - t0
                if t >= duration:
                    break
                engaged = t >= settle
                grip = 1.0 if engaged else 0.0
                # Trace a circle only once engaged; index trigger closes the
                # gripper for the second half of the engaged window.
                if engaged:
                    ang = 2.0 * math.pi * 0.25 * (t - settle)  # 0.25 Hz
                    hand = (_HAND_BASE[0],
                            _HAND_BASE[1] + radius * math.sin(ang),
                            _HAND_BASE[2] + radius * (math.cos(ang) - 1.0))
                    index = 1.0 if (t - settle) > (duration - settle) / 2 else 0.0
                else:
                    hand = _HAND_BASE
                    index = 0.0
                frame = _frame(hand, grip, index, home=False)
                sock.sendall((json.dumps(frame) + "\n").encode("utf-8"))
                time.sleep(dt)
    except (BrokenPipeError, ConnectionResetError):
        print("[mock] server closed the connection")
    print("[mock] done")


def main():
    parser = argparse.ArgumentParser(description="Mock Meta-Quest VR client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--radius", type=float, default=0.08)
    parser.add_argument("--settle", type=float, default=1.0)
    args = parser.parse_args()
    run(host=args.host, port=args.port, duration=args.duration, hz=args.hz,
        radius=args.radius, settle=args.settle)


if __name__ == "__main__":
    main()
