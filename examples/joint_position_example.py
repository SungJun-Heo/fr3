"""Step 2 -- joint-position control primitives (plumbing check).

Sim mirror of camel-franka's ``examples/joint_position_example.py``. There is
no trajectory generator yet (that is step 3): we drive the ActiveControl
read/write loop with plain *step* inputs to prove the plumbing works --
``writeOnce`` -> ``data.ctrl`` -> ``mj_step`` -> ``readOnce``.

Usage:  python examples/joint_position_example.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, ControllerMode, JointPositions


def command_constant(robot, target, n_steps):
    """Hold a constant joint target for ``n_steps``; return (final_state, elapsed).

    This is the whole control loop in miniature: read a tick, write the same
    command, let the sim advance one step -- repeated.
    """
    ac = robot.start_joint_position_control(ControllerMode.JointImpedance)
    elapsed = 0.0
    for i in range(n_steps):
        _, dt = ac.readOnce()
        cmd = JointPositions(target)
        if i == n_steps - 1:
            cmd.motion_finished = True  # end the session on the last tick
        ac.writeOnce(cmd)
        elapsed += dt.to_sec()
    return robot.read_once(), elapsed


def main():
    robot = SimRobot("empty")

    # Mirror camel-franka's setup: configure collision thresholds. Stored for
    # now (the reflex check itself lands in a later safety step) but this proves
    # the real robot's setup call runs unchanged against the sim.
    robot.set_collision_behavior(
        [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
        [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
        [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
        [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
    )

    np.set_printoptions(precision=4, suppress=True)
    q0 = robot.read_once().q
    print("start q :", q0)

    # 1) HOLD: command the current pose -> arm stays put, sim time advances.
    final, elapsed = command_constant(robot, q0, 500)
    print("\n[hold] commanded = start q")
    print("  final q       :", final.q)
    print("  |q - q0| max  :", np.abs(final.q - q0).max())
    print("  sim elapsed   :", round(elapsed, 4), "s  (= 500 x 0.002)")

    # 2) NUDGE: a +0.1 rad step on joint 1 (near-zero gravity load) -> the
    #    position actuator should converge the joint onto the commanded target.
    target = q0.copy()
    target[0] += 0.1
    final, _ = command_constant(robot, target, 1500)
    print("\n[nudge] joint1 target += 0.1 rad")
    print("  target j1     :", round(float(target[0]), 4))
    print("  final  j1     :", round(float(final.q[0]), 4))
    print("  tracking err  :", round(abs(float(final.q[0] - target[0])), 5))
    print("  q_d (desired) :", final.q_d)  # q_d now reflects the ctrl target


if __name__ == "__main__":
    main()
