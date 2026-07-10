#!/usr/bin/env python3
"""Project entry point.

Usage:
  python main.py --mode gui [--task empty]     # hand-control GUI (joint/task)
  python main.py --mode vr  [--task empty]     # VR teleop (Meta Quest over TCP)

For VR: this launches the viewer + TCP server, then point a Meta-Quest Unity app
(or ``python -m teleop.mock_vr_client``) at ``<this host>:<--port>``.

Add a mode by writing a ``run_*`` function and registering it in ``MODES``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_gui(args):
    """Launch the joint/task-space hand-control GUI."""
    from examples.control_gui import ControlGUI
    ControlGUI(args.task).run()


def run_vr(args):
    """Launch VR teleoperation (Meta Quest streams hand pose over TCP)."""
    from teleop import VRTeleop
    VRTeleop(task=args.task, hand=args.hand, host=args.host, port=args.port,
             position_scale=args.scale, smooth_tau=args.smooth_tau,
             view=not args.no_view).run()


MODES = {
    "gui": run_gui,
    "vr": run_vr,
}


def main():
    parser = argparse.ArgumentParser(description="fr3 sim entry point")
    parser.add_argument("--mode", choices=list(MODES), default="gui",
                        help="what to run (default: gui)")
    parser.add_argument("--task", default="empty",
                        help="scene/task name (default: empty)")
    # VR-mode options (ignored by other modes).
    parser.add_argument("--hand", default="right", choices=["right", "left"],
                        help="[vr] which controller drives the arm")
    parser.add_argument("--host", default="0.0.0.0",
                        help="[vr] TCP bind address for the VR server")
    parser.add_argument("--port", type=int, default=8081,
                        help="[vr] TCP port for the VR server")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="[vr] hand->EE position scale")
    parser.add_argument("--smooth-tau", type=float, default=0.0,
                        help="[vr] command low-pass time constant (s); 0 disables")
    parser.add_argument("--no-view", action="store_true",
                        help="[vr] run headless (no viewer)")
    args = parser.parse_args()
    MODES[args.mode](args)


if __name__ == "__main__":
    main()
