#!/usr/bin/env python3
"""Project entry point -- launch the unified control GUI.

Opens the MuJoCo viewer plus a Tkinter panel that does all three control modes in
one window:

  * JOINT -- 7 joint-angle sliders.
  * TASK  -- 6 EE-pose sliders (x, y, z + roll, pitch, yaw), tracked via DLS IK.
  * VR    -- a Meta-Quest controller drives the arm over TCP (relative clutch).

plus Execute (quintic move to the slider targets), HOME, Recover, Reset objects,
Reset ALL, and a gripper slider. The viewer overlays the commanded vs actual EE.

The task/scene, the VR position scale, the VR smoothing time constant, and the
overlay toggle are set at runtime in the panel's settings row -- there are no
CLI flags.

Usage:
  python main.py

For VR: select the "VR" mode in the panel, then point a Meta-Quest Unity app (or
``python -m teleop.mock_vr_client``) at this host on TCP port 8081.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gui.app import main

if __name__ == "__main__":
    main()
