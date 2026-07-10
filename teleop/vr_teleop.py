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
SMOOTH_TAU = 0.0        # default command low-pass time constant (s); 0 = off.
                        # The sim's position servo already smooths, so this is
                        # off by default (adds latency); raise it (e.g. 0.05) if
                        # a real headset's input jitter shows through.


def make_pose_vec(pos, R):
    """position + 3x3 rotation -> column-major length-16 O_T_EE (libfranka)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T.flatten(order="F")


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


class VRTeleop:
    def __init__(self, task="empty", hand="right", host="0.0.0.0", port=8081,
                 position_scale=1.0, smooth_tau=SMOOTH_TAU, view=True,
                 show_stats=True):
        self.show_stats = show_stats
        self.task_name = task
        self.robot = SimRobot(task)
        self.model, self.data = self.robot.model, self.robot.data
        self.gripper = Gripper(self.robot)
        # "clamp" safety: near a singularity or joint limit the arm slows/limits
        # smoothly instead of hard-tripping -- a trip there reads as a stutter.
        self.ac = self.robot.start_cartesian_pose_control(safety="clamp")
        self.position_scale = float(position_scale)
        self.substeps = max(1, round(TICK_MS / 1000 / self.model.opt.timestep))

        # Command low-pass: each substep the pose actually sent to the arm slews
        # a fraction toward the clutch target, so bursty/jittery VR input (WiFi
        # delivery, 72->50 Hz undersampling) comes out as smooth motion. alpha
        # is per substep; smooth_tau=0 sends the raw target (no smoothing).
        self._cmd_alpha = (1.0 - math.exp(-self.model.opt.timestep / smooth_tau)
                           if smooth_tau > 0 else 1.0)

        # Clutch state. ``_engaged`` tracks the grip edge; the anchors are the
        # hand pose and EE pose captured the instant the clutch last engaged.
        self._engaged = False
        self._vr_anchor = None
        self._ee_anchor_pos = None
        self._ee_anchor_R = None

        # Persistent commanded pose. When disengaged we keep streaming this
        # (frozen) pose so the arm holds; it starts at the current EE pose.
        # ``_cmd_*_filt`` is the low-passed pose actually sent to the arm.
        p, R = self._ee_pose()
        self._cmd_pos = p
        self._cmd_R = R
        self._cmd_pos_filt = p.copy()
        self._cmd_R_filt = R.copy()

        # Soft-wall state: True on the ticks a near-singularity / joint-limit
        # holds the arm. Unlike a latched fault it clears itself the moment the
        # hand steers back into a reachable pose (see _control_substep).
        self._blocked = False
        self._block_reason = ""
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
        """Point the commanded pose (and its filtered copy) at the current EE,
        so teleop resumes from HOME/recover without a jump."""
        self._cmd_pos, self._cmd_R = self._ee_pose()
        self._cmd_pos_filt = self._cmd_pos.copy()
        self._cmd_R_filt = self._cmd_R.copy()

    # -- clutch (the ported set_vr_trajectory) -------------------------

    def _update_clutch(self, snap):
        """Map VR input to the commanded EE pose using the clutch anchor.

        Engaged only while the hand is tracked, the grip is held, no trip is
        latched, and no HOME motion is running. On the rising edge we capture
        the anchors; while held we apply the relative position/rotation delta;
        on release we simply stop updating, leaving the last command frozen."""
        # Note: a soft-wall block does NOT disengage the clutch -- we keep the
        # anchor so that steering the hand back out resumes tracking at once.
        engaged = (snap.tracking and snap.grip > GRIP_ENGAGE
                   and self._home_traj is None)

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
        """Clear any block and start a smooth quintic motion to the HOME key."""
        self.robot.automatic_error_recovery()
        self._blocked = False
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

        self._blocked = False
        for _ in range(self.substeps):
            self._control_substep()

    def _control_substep(self):
        """One sim step: stream HOME if running, else track the commanded
        (clutch) pose through the existing Cartesian IK+safety path.

        Near-singularity / joint-limit trips are handled as a *soft wall*: the
        Cartesian safety raises, we clear it right away and just hold this step.
        The arm therefore refuses to push deeper into the singular region but
        resumes the instant the hand steers back to a reachable pose -- no HOME
        needed. (Latching the trip is what made low-manipulability teleop
        stutter to a dead stop.) SimRobot itself still latches, so scripted /
        VLA control keeps the faithful fault behaviour; only teleop softens it."""
        if self._home_traj is not None:
            self._home_t += self.model.opt.timestep
            q = self._home_traj.getPositionTrajectory(self._home_t)
            self.ac.writeOnce(JointPositions(q))
            if self._home_t >= HOME_DURATION:
                self._home_traj = None
                self._resync_cmd()  # resume teleop from HOME without a jump
            return
        # Low-pass the commanded pose toward the clutch target before sending it
        # (position: exponential lerp; orientation: geodesic slerp). This is the
        # smoothing that turns jittery VR input into steady arm motion.
        a = self._cmd_alpha
        self._cmd_pos_filt += a * (self._cmd_pos - self._cmd_pos_filt)
        self._cmd_R_filt = slerp_toward(self._cmd_R_filt, self._cmd_R, a)
        try:
            self.ac.writeOnce(CartesianPose(
                make_pose_vec(self._cmd_pos_filt, self._cmd_R_filt)))
        except RuntimeError as e:
            self.robot.automatic_error_recovery()
            mujoco.mj_step(self.model, self.data)  # hold pose, keep time moving
            self._blocked = True
            self._block_reason = str(e)

    def _mode_tag(self, connected):
        if not connected:
            return "no client"
        if self._home_traj is not None:
            return "HOME"
        if self._blocked:
            return "singular/limit (hold)"
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
        # Yoshikawa manipulability: how far from a singularity the arm is. In
        # clamp mode the arm slows smoothly as this drops (no stutter); watch it
        # dip when you steer into an extended/awkward pose.
        J = self.robot._ik._jacobian(self.data)
        w = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
        tag = self._mode_tag(snap.connected)
        if w < 0.04:
            tag += " near-sing"
        print(f"[vr] loop {loop_fps:4.0f}Hz  work {work_ms:4.1f}ms  rt {rt:.2f}x "
              f"| input {in_fps:4.0f}fps  w={w:.3f} | {tag}", flush=True)

    # -- reset GUI -----------------------------------------------------

    def _run_gui(self):
        """Drive the teleop loop from a small Tk panel with reset buttons.

        Mirrors control_gui's structure: the MuJoCo passive viewer plus a Tk
        window whose ``after``-scheduled tick advances the sim. Button callbacks
        run in the same (Tk main) thread as the tick, so they can touch MuJoCo
        directly -- no locking, no races. The VR server stays a daemon thread
        (it only touches VRState)."""
        import tkinter as tk

        print("[vr] teleop + reset GUI. Hold grip to move, index to grip, B for HOME.")
        self.root = tk.Tk()
        self.root.title(f"FR3 VR teleop [{self.task_name}]")
        self.root.geometry("420x460")
        btn_font = ("sans", 22, "bold")
        # Status pinned to the bottom; the big buttons fill everything above it
        # (each expands to ~1/3 of the window, so they dominate as asked).
        self._gui_status = tk.Label(self.root, text="", justify=tk.LEFT,
                                    anchor="w", font=("monospace", 11))
        self._gui_status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)
        for text, cmd, color in (
            ("Reset objects", self._gui_reset_objects, "#cfe8cf"),
            ("HOME robot", self._go_home, "#cfe0f2"),
            ("Reset ALL", self._gui_reset_all, "#f4c88a"),
        ):
            tk.Button(self.root, text=text, command=cmd, font=btn_font,
                      bg=color, activebackground=color
                      ).pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.root.after(TICK_MS, self._gui_tick)
        try:
            self.root.mainloop()
        finally:
            self.server.stop()
            if self.viewer is not None:
                self.viewer.close()

    def _gui_tick(self):
        # Closing the 3D viewer ends the session; reschedule with a
        # work-compensated delay so the loop still holds ~1/TICK_MS (a plain
        # after(TICK_MS) would run at 1/(work+TICK_MS), below real time).
        if self.viewer is not None and not self.viewer.is_running():
            self.root.destroy()
            return
        t0 = time.perf_counter()
        self._tick()
        if self.viewer is not None:
            self.viewer.sync()
        self._gui_update_status()
        delay = max(1, int(round(TICK_MS - (time.perf_counter() - t0) * 1000)))
        self.root.after(delay, self._gui_tick)

    def _gui_reset_objects(self):
        """Put task objects back at their start pose (arm untouched)."""
        self.robot.reset_objects()

    def _gui_reset_all(self):
        """Objects back to start AND a smooth HOME for the arm."""
        self.robot.reset_objects()
        self._go_home()

    def _gui_update_status(self):
        snap = self.state.snapshot()
        p = self.data.site_xpos[self.robot._ee_site]
        g = self.gripper.read_once()
        conn = "connected" if snap.connected else "waiting for VR client"
        msg = (f"{conn}  |  {self._mode_tag(snap.connected)}\n"
               f"EE=({p[0]*100:+.1f},{p[1]*100:+.1f},{p[2]*100:+.1f})cm   "
               f"grip={g.width / g.max_width:.2f}"
               f"{'  [GRASP]' if g.is_grasped else ''}")
        self._gui_status.config(text=msg)

    def run(self, max_ticks=None, on_tick=None, gui=False):
        """Run the teleop loop at a steady ~1/TICK_MS rate.

        Each iteration is paced by a work-compensated sleep, so the sim tracks
        real time instead of drifting slower (and jittering) under per-tick load
        -- the naive ``sleep(period)`` makes the loop run at ``1/(work+period)``,
        below target and wobbling with work. A 1 Hz stats line reports where any
        stutter lives (see ``_print_stats``).

        ``gui=True`` instead drives the loop from a small Tk reset panel (see
        ``_run_gui``). ``max_ticks`` stops after N ticks (tests); ``on_tick(self)``
        is called each tick after stepping (lets a test observe EE/clutch state)."""
        if gui:
            self._run_gui()
            return
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
                    if self.show_stats:
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
    parser.add_argument("--smooth-tau", type=float, default=SMOOTH_TAU,
                        help="command low-pass time constant (s); 0 disables")
    parser.add_argument("--stats", action="store_true",
                        help="print the 1 Hz loop/latency stats line")
    parser.add_argument("--gui", action="store_true",
                        help="show the reset-button GUI")
    parser.add_argument("--no-view", action="store_true", help="run headless")
    args = parser.parse_args()
    VRTeleop(task=args.task, hand=args.hand, host=args.host, port=args.port,
             position_scale=args.scale, smooth_tau=args.smooth_tau,
             view=not args.no_view, show_stats=args.stats).run(gui=args.gui)


if __name__ == "__main__":
    main()
