"""VR teleoperation loop for the FR3 sim -- the fr3 answer to camel-RBY1's
``VR_TELEOPERATION_CONTROL`` path.

RBY1 splits VR teleop across three threads: a VR server fills ``shm.VRData``, a
250 Hz controller maps the input to an EE pose and solves IK to joint refs
(``set_vr_trajectory`` -> ``calculate_task_space_q_ref``), and a 500 Hz comm
thread streams those joint refs to the robot. On fr3 the last two collapse into
one call: ``SimRobot.start_cartesian_pose_control()`` +
``writeOnce(CartesianPose)`` already does the per-tick DLS IK, joint-limit /
singularity / collision safety, and the sim step (the real robot's firmware does
the same). So the only piece we port is RBY1's ``set_vr_trajectory`` -- the
mapping from VR input to a commanded EE pose -- and hand it to that existing path.

Control scheme (single right hand by default):
  * grip trigger = clutch. Hold to drive the arm; release to freeze it in place.
    On each fresh press we re-anchor (capture the hand pose and the current EE
    pose), so the arm never jumps and you can re-grip from a comfortable spot --
    exactly RBY1's rotation clutch, extended to cover position too.
  * While engaged we map the *relative* motion from the anchor:
        EE_pos = EE_anchor + scale * (hand_now - hand_anchor)
        EE_R   = (hand_now_R · hand_anchor_Rᵀ) · EE_anchor_R
  * index trigger = gripper (squeeze to close), non-blocking so it tracks
    alongside the arm.
  * B button = smooth quintic move HOME + clear any safety trip (the escape
    hatch, like RBY1's B button).

The VR server runs in a daemon thread and publishes to a ``VRState``; this loop
reads a locked snapshot each tick. Runs with the MuJoCo viewer (``view=True``)
or headless (for tests / a display-less host).
"""

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, JointPositions, CartesianPose, Gripper
from controller.planning import QuinticTrajectoryGenerator
from teleop.vr_server import VRState, VRTeleopServer

TICK_MS = 20            # control/UI tick period (matches control_gui)
HOME_DURATION = 2.0     # seconds for the HOME motion
GRIP_ENGAGE = 0.5       # grip trigger above this = clutch engaged


def make_pose_vec(pos, R):
    """position + 3x3 rotation -> column-major length-16 O_T_EE (libfranka)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T.flatten(order="F")


class VRTeleop:
    def __init__(self, task="empty", hand="right", host="0.0.0.0", port=8081,
                 position_scale=1.0, view=True):
        self.robot = SimRobot(task)
        self.model, self.data = self.robot.model, self.robot.data
        self.gripper = Gripper(self.robot)
        self.ac = self.robot.start_cartesian_pose_control()
        self.position_scale = float(position_scale)
        self.substeps = max(1, round(TICK_MS / 1000 / self.model.opt.timestep))

        # Clutch state. ``_engaged`` tracks the grip edge; the anchors are the
        # hand pose and EE pose captured the instant the clutch last engaged.
        self._engaged = False
        self._vr_anchor = None
        self._ee_anchor_pos = None
        self._ee_anchor_R = None

        # Persistent commanded pose. When disengaged we keep streaming this
        # (frozen) pose so the arm holds; it starts at the current EE pose.
        p, R = self._ee_pose()
        self._cmd_pos = p
        self._cmd_R = R

        self._trip = None        # safety-trip message, or None
        self._home_traj = None   # active quintic while HOME runs
        self._home_t = 0.0
        self._prev_home_btn = False

        # VR input pipeline: server thread -> VRState -> this loop.
        self.state = VRState()
        self.server = VRTeleopServer(self.state, hand=hand, host=host, port=port)
        self._server_thread = threading.Thread(target=self.server.serve_forever,
                                                daemon=True)
        self._server_thread.start()

        self.viewer = (mujoco.viewer.launch_passive(self.model, self.data)
                       if view else None)

    # -- helpers -------------------------------------------------------

    def _ee_pose(self):
        """Current EE (position, rotation) from the sim."""
        T = self.robot.read_once().O_T_EE.reshape(4, 4, order="F")
        return T[:3, 3].copy(), T[:3, :3].copy()

    def _resync_cmd(self):
        """Point the commanded pose at the current EE (no jump on resume)."""
        self._cmd_pos, self._cmd_R = self._ee_pose()

    # -- clutch (the ported set_vr_trajectory) -------------------------

    def _update_clutch(self, snap):
        """Map VR input to the commanded EE pose using the clutch anchor.

        Engaged only while the hand is tracked, the grip is held, no trip is
        latched, and no HOME motion is running. On the rising edge we capture
        the anchors; while held we apply the relative position/rotation delta;
        on release we simply stop updating, leaving the last command frozen."""
        engaged = (snap.tracking and snap.grip > GRIP_ENGAGE
                   and self._trip is None and self._home_traj is None)

        if engaged and not self._engaged:
            self._vr_anchor = snap.hand_tf.copy()
            self._ee_anchor_pos, self._ee_anchor_R = self._ee_pose()

        if engaged:
            d_pos = snap.hand_tf[:3, 3] - self._vr_anchor[:3, 3]
            self._cmd_pos = self._ee_anchor_pos + self.position_scale * d_pos
            R_delta = snap.hand_tf[:3, :3] @ self._vr_anchor[:3, :3].T
            self._cmd_R = R_delta @ self._ee_anchor_R

        self._engaged = engaged

    # -- HOME / recover ------------------------------------------------

    def _go_home(self):
        """Clear any trip and start a smooth quintic motion to the HOME key."""
        self.robot.automatic_error_recovery()
        self._trip = None
        home = self.model.key_qpos[0][:7].copy()
        traj = QuinticTrajectoryGenerator()
        traj.InitTrajectory(self.robot.read_once().q, home, 0.0, HOME_DURATION)
        self._home_traj = traj
        self._home_t = 0.0
        self._engaged = False  # force a fresh anchor when teleop resumes

    # -- control tick --------------------------------------------------

    def _tick(self):
        snap = self.state.snapshot()

        # Gripper: index trigger 0..1 -> width (squeeze to close). Non-blocking.
        self.gripper.set_target_width((1.0 - snap.trigger) * self.gripper.max_width)

        # HOME on the button's rising edge (so holding it fires once).
        if snap.home and not self._prev_home_btn:
            self._go_home()
        self._prev_home_btn = snap.home

        self._update_clutch(snap)

        for _ in range(self.substeps):
            try:
                self._control_substep()
            except RuntimeError as e:
                self._trip = str(e)

    def _control_substep(self):
        """One sim step: hold on a trip, stream HOME if running, else track the
        commanded (clutch) pose through the existing Cartesian IK+safety path."""
        if self._trip is not None:
            mujoco.mj_step(self.model, self.data)
            return
        if self._home_traj is not None:
            self._home_t += self.model.opt.timestep
            q = self._home_traj.getPositionTrajectory(self._home_t)
            self.ac.writeOnce(JointPositions(q))
            if self._home_t >= HOME_DURATION:
                self._home_traj = None
                self._resync_cmd()  # resume teleop from HOME without a jump
            return
        self.ac.writeOnce(CartesianPose(make_pose_vec(self._cmd_pos, self._cmd_R)))

    def _mode_tag(self, connected):
        if not connected:
            return "no client"
        if self._trip is not None:
            return "TRIP"
        if self._home_traj is not None:
            return "HOME"
        return "engaged" if self._engaged else "idle"

    def _print_stats(self, dt, ticks, work, prev_frames, sim_per_tick):
        """One-line health readout (once/sec) -- the numbers that localise a
        stutter: loop rate, per-tick work, sim real-time factor, VR input rate.

        Read it as: ``rt`` well below 1.0 => the loop can't keep real time (work
        too heavy, or rendering); ``input`` far below the loop rate => the
        network/headset is the bottleneck, not us; ``work`` near the tick period
        => raise TICK_MS or cut substeps."""
        snap = self.state.snapshot()
        loop_fps = ticks / dt
        work_ms = 1000.0 * work / max(ticks, 1)
        in_fps = (snap.frames - prev_frames) / dt
        rt = loop_fps * sim_per_tick  # sim seconds advanced per wall second
        print(f"[vr] loop {loop_fps:4.0f}Hz  work {work_ms:4.1f}ms  rt {rt:.2f}x "
              f"| input {in_fps:4.0f}fps | {self._mode_tag(snap.connected)}",
              flush=True)

    def run(self, max_ticks=None, on_tick=None):
        """Run the teleop loop at a steady ~1/TICK_MS rate.

        Each iteration is paced by a work-compensated sleep, so the sim tracks
        real time instead of drifting slower (and jittering) under per-tick load
        -- the naive ``sleep(period)`` makes the loop run at ``1/(work+period)``,
        below target and wobbling with work. A 1 Hz stats line reports where any
        stutter lives (see ``_print_stats``).

        ``max_ticks`` stops after N ticks (tests); ``on_tick(self)`` is called
        each tick after stepping (lets a test observe EE/clutch state)."""
        print("[vr] teleop running. Hold grip to move, index to grip, B for HOME.")
        period = TICK_MS / 1000
        sim_per_tick = self.substeps * self.model.opt.timestep
        tick = 0
        stat_t0 = time.perf_counter()
        stat_ticks = 0
        stat_work = 0.0
        stat_frames = self.state.snapshot().frames
        try:
            while True:
                if self.viewer is not None and not self.viewer.is_running():
                    break
                t0 = time.perf_counter()
                self._tick()
                if self.viewer is not None:
                    self.viewer.sync()
                if on_tick is not None:
                    on_tick(self)
                stat_work += time.perf_counter() - t0
                stat_ticks += 1

                now = time.perf_counter()
                if now - stat_t0 >= 1.0:
                    self._print_stats(now - stat_t0, stat_ticks, stat_work,
                                      stat_frames, sim_per_tick)
                    stat_t0, stat_ticks, stat_work = now, 0, 0.0
                    stat_frames = self.state.snapshot().frames

                sleep = period - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
                tick += 1
                if max_ticks is not None and tick >= max_ticks:
                    break
        finally:
            self.server.stop()
            if self.viewer is not None:
                self.viewer.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FR3 VR teleoperation")
    parser.add_argument("--task", default="empty")
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="hand->EE position scale")
    parser.add_argument("--no-view", action="store_true", help="run headless")
    args = parser.parse_args()
    VRTeleop(task=args.task, hand=args.hand, host=args.host, port=args.port,
             position_scale=args.scale, view=not args.no_view).run()


if __name__ == "__main__":
    main()
