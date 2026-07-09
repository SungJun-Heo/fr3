#!/usr/bin/env python3
"""Project entry point.

Usage:
  python main.py --mode gui [--task empty]     # hand-control GUI (joint/task)

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


MODES = {
    "gui": run_gui,
}


def main():
    parser = argparse.ArgumentParser(description="fr3 sim entry point")
    parser.add_argument("--mode", choices=list(MODES), default="gui",
                        help="what to run (default: gui)")
    parser.add_argument("--task", default="empty",
                        help="scene/task name (default: empty)")
    args = parser.parse_args()
    MODES[args.mode](args)


if __name__ == "__main__":
    main()
