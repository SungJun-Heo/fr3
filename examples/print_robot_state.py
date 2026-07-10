"""Print the sim robot's state once -- the sim mirror of camel-franka's
``examples/print_robot_state.py``.

Usage:  python examples/print_robot_state.py [task]   (default: "empty")
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, vec_to_pose


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "empty"
    robot = SimRobot(task)
    s = robot.read_once()

    np.set_printoptions(precision=4, suppress=True)
    print(f"task: {task}")
    print(f"q      : {s.q}")
    print(f"dq     : {s.dq}")
    print(f"tau_J  : {s.tau_J}")
    pos, _ = vec_to_pose(s.O_T_EE)  # column-major O_T_EE -> (position, rotation)
    print(f"EE pos : {pos}")
    print("O_T_EE (4x4):")
    print(s.O_T_EE.reshape(4, 4, order="F"))  # raw datatype, shown as a matrix


if __name__ == "__main__":
    main()
