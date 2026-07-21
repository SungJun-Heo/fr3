"""ControlSession -- UI-agnostic control core for the unified GUI.

Owns the sim robot, the MuJoCo viewer, the gripper, and the per-tick control
loop, with three interchangeable modes:

  * ``"joint"`` -- track 7 joint-angle targets (manual sliders).
  * ``"task"``  -- track an EE pose target via DLS IK (manual sliders).
  * ``"vr"``    -- track the VR relative-clutch commanded pose (Quest over TCP).

Everything shared across the modes lives here once (this is the single home for
what used to be duplicated between ``examples/control_gui.py`` and
``teleop/vr_teleop.py``): quintic HOME / Execute moves, error recovery, object /
scene reset, the gripper, the commanded-vs-actual EE viewer overlay, and a
telemetry snapshot for the UI.

A tkinter app (``gui/app.py``) drives ``step()`` on an ``after()`` loop; ``run()``
offers the same tick headless (for tests / a display-less host). The VR TCP
server always runs (a daemon thread) so "vr" mode works the instant it is
selected; the clutch itself lives in ``teleop/clutch.py``.
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
from robot import (
    SimRobot, JointPositions, CartesianPose, Gripper, vec_to_pose,
)
from controller.planning import QuinticTrajectoryGenerator
from teleop.vr_server import VRState, VRTeleopServer
from teleop.clutch import VRClutch, SMOOTH_TAU
from overlay import add_frame

TICK_MS = 20         # control/UI tick period
HOME_DURATION = 2.0  # seconds for a HOME motion


class ControlSession:
    def __init__(self, task="empty", view=True, hand="right", host="0.0.0.0",
                 port=8081, position_scale=1.0, smooth_tau=SMOOTH_TAU,
                 show_markers=True):
        self.task_name = task
        self.robot = SimRobot(task)
        self.model, self.data = self.robot.model, self.robot.data
        self.gripper = Gripper(self.robot)
        self.show_markers = show_markers
        self.substeps = max(1, round(TICK_MS / 1000 / self.model.opt.timestep))

        # Control mode + its active-control handle. "joint" uses joint-position
        # control; "task"/"vr" use Cartesian pose control -- task with the
        # faithful "trip" safety (a reach too far faults, like control_gui), vr
        # with "clamp" so a singularity brakes smoothly instead of stuttering.
        self.mode = "joint"
        self.ac = self.robot.start_joint_position_control()

        # Live targets streamed each tick by the active mode.
        st = self.robot.read_once()
        self.joint_targets = st.q.copy()
        self.task_target_pos, self.task_target_R = vec_to_pose(st.O_T_EE)
        self.gripper_target = self.gripper.width()

        # Timed quintic move (Execute / HOME); overrides live tracking while it
        # streams. Trip/notice surface safety + IK messages to the status line.
        self.move_traj = None
        self.move_t = 0.0
        self.move_dur = 0.0
        self.trip = None
        self.notice = ""

        # VR pipeline: server daemon thread -> VRState -> this loop's clutch.
        # alpha is the per-substep low-pass strength (1.0 = no smoothing).
        alpha = (1.0 - math.exp(-self.model.opt.timestep / smooth_tau)
                 if smooth_tau > 0 else 1.0)
        self.clutch = VRClutch(position_scale=position_scale, alpha=alpha)
        self.clutch.reset(*self._ee_pose())
        self.state = VRState()
        self.server = VRTeleopServer(self.state, hand=hand, host=host, port=port)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self._prev_home_btn = False
        self._blocked = False        # True on ticks a vr-mode soft wall holds
        self._block_reason = ""

        self.viewer = (mujoco.viewer.launch_passive(self.model, self.data)
                       if view else None)

    # -- helpers -------------------------------------------------------

    def _ee_pose(self):
        """Current EE (position, rotation) from the sim."""
        return vec_to_pose(self.robot.read_once().O_T_EE)

    def _sync_targets_to_state(self):
        """Point every live target at the robot's current pose, so switching
        modes / recovering / finishing a move never causes a jump or re-trip."""
        st = self.robot.read_once()
        self.joint_targets = st.q.copy()
        self.task_target_pos, self.task_target_R = vec_to_pose(st.O_T_EE)
        self.gripper_target = self.gripper.width()
        if self.mode == "vr":
            self.clutch.reset(*self._ee_pose())

    # -- targets (set by the UI) ---------------------------------------

    def set_joint_targets(self, q_rad):
        self.joint_targets = np.asarray(q_rad, dtype=float)

    def set_task_target(self, pos, R):
        self.task_target_pos = np.asarray(pos, dtype=float)
        self.task_target_R = np.asarray(R, dtype=float)

    def set_gripper_frac(self, frac):
        """Set the gripper opening as a 0..1 fraction of max width."""
        self.gripper_target = float(np.clip(frac, 0.0, 1.0)) * self.gripper.max_width

    # -- runtime settings (adjustable while running) -------------------

    def set_position_scale(self, scale):
        """VR position gain: EE displacement per unit hand displacement."""
        self.clutch.position_scale = float(scale)

    def set_smooth_tau(self, tau):
        """VR command low-pass time constant (s); 0 disables smoothing. Recomputes
        the per-substep filter strength from the sim timestep."""
        tau = float(tau)
        self.clutch.alpha = (1.0 - math.exp(-self.model.opt.timestep / tau)
                             if tau > 0 else 1.0)

    def reload_task(self, task):
        """Switch task/scene at runtime: rebuild the model+data (same arm,
        different objects), relaunch the viewer, and re-establish control for the
        current mode. No-op if already on ``task``."""
        if task == self.task_name:
            return
        had_viewer = self.viewer is not None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.task_name = task
        self.robot = SimRobot(task)
        self.model, self.data = self.robot.model, self.robot.data
        self.gripper = Gripper(self.robot)
        self.substeps = max(1, round(TICK_MS / 1000 / self.model.opt.timestep))
        self.trip = None
        self.notice = ""
        self.move_traj = None
        self._blocked = False
        # Re-establish the active-control handle on the new robot for the current
        # mode, then point every target (and the clutch) at the fresh state.
        if self.mode == "joint":
            self.ac = self.robot.start_joint_position_control()
        elif self.mode == "task":
            self.ac = self.robot.start_cartesian_pose_control(safety="trip")
        else:
            self.ac = self.robot.start_cartesian_pose_control(safety="clamp")
        self._sync_targets_to_state()
        self.clutch.reset(*self._ee_pose())
        if had_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # -- mode / commands -----------------------------------------------

    def set_mode(self, mode):
        """Switch control mode, swapping the active-control handle and re-syncing
        targets so the arm does not jump."""
        if mode == self.mode:
            return
        self.mode = mode
        self._sync_targets_to_state()
        if mode == "joint":
            self.ac = self.robot.start_joint_position_control()
        elif mode == "task":
            self.ac = self.robot.start_cartesian_pose_control(safety="trip")
        else:  # vr
            self.ac = self.robot.start_cartesian_pose_control(safety="clamp")
            self.clutch.reset(*self._ee_pose())

    def execute(self, duration):
        """Move to the current manual targets over ``duration`` s as a smooth
        quintic joint move (JOINT: the slider angles; TASK: IK for the slider EE
        pose). No-op in VR mode. An unreachable task pose is reported and ignored."""
        if self.mode == "joint":
            q_goal = self.joint_targets.copy()
        elif self.mode == "task":
            q_goal, info = self.robot._ik.solve(
                self.task_target_pos, self.task_target_R,
                q_init=self.robot.read_once().q)
            if not info["converged"]:
                self._sync_targets_to_state()
                self.notice = (f"IK did not converge for that pose "
                               f"(pos_err={info['pos_err']*1000:.1f}mm) -- ignored")
                return
        else:
            return
        self.start_move(q_goal, duration)

    def start_move(self, q_goal, duration):
        """Begin a smooth quintic joint move from the current pose to ``q_goal``.
        Clears any trip first (Execute/HOME both double as an escape from a stop)."""
        self.robot.automatic_error_recovery()
        self.trip = None
        self.notice = ""
        traj = QuinticTrajectoryGenerator()
        traj.InitTrajectory(self.robot.read_once().q, np.asarray(q_goal, float),
                            0.0, duration)
        self.move_traj = traj
        self.move_t = 0.0
        self.move_dur = duration

    def go_home(self):
        """Smooth quintic move to the HOME keyframe (works from any mode)."""
        home = self.model.key_qpos[0][:7].copy()
        self.start_move(home, HOME_DURATION)

    def recover(self):
        """Clear a trip / abandon a move and resume live tracking."""
        self.robot.automatic_error_recovery()
        self.trip = None
        self.notice = ""
        self.move_traj = None
        self._blocked = False
        self._sync_targets_to_state()

    def reset_objects(self):
        """Put task objects back at their start pose (arm untouched)."""
        self.robot.reset_objects()

    def randomize_objects(self):
        """Randomize movable object poses within their task ranges (arm
        untouched) -- domain randomization for the next episode."""
        self.robot.randomize_objects()

    def reset_all(self):
        """Objects and arm instantly back to start (a synchronous scene reset)."""
        self.move_traj = None
        self.robot.reset_objects()
        self.robot.reset_home()
        self.gripper.reset_open()   # before the sync: gripper_target reads width()
        self.trip = None
        self._blocked = False
        self._sync_targets_to_state()

    # -- control tick --------------------------------------------------

    def step(self):
        """Advance one UI/control tick. Returns False when the viewer has been
        closed (the caller should stop), else True."""
        if self.viewer is not None and not self.viewer.is_running():
            return False

        # VR mode reads its per-tick input here (gripper from the trigger, HOME
        # on the button's rising edge, and the clutch from the hand pose).
        if self.mode == "vr":
            snap = self.state.snapshot()
            self.gripper_target = (1.0 - snap.trigger) * self.gripper.max_width
            if snap.home and not self._prev_home_btn:
                self.go_home()
            self._prev_home_btn = snap.home
            ee_pos, ee_R = self._ee_pose()
            self.clutch.update(snap, ee_pos, ee_R, active=(self.move_traj is None))

        self.gripper.set_target_width(self.gripper_target)  # non-blocking

        self._blocked = False
        for _ in range(self.substeps):
            try:
                self._control_substep()
            except RuntimeError as e:
                self.trip = str(e)

        if self.viewer is not None:
            self._draw_overlay()
            self.viewer.sync()
        return True

    def _control_substep(self):
        """Advance one sim step: hold if tripped, stream an active timed move
        (Execute/HOME) if one is running, else drive the current mode's command."""
        if self.trip is not None:
            mujoco.mj_step(self.model, self.data)
            return
        if self.move_traj is not None:
            self.move_t += self.model.opt.timestep
            q = self.move_traj.getPositionTrajectory(self.move_t)
            self.ac.writeOnce(JointPositions(q))
            if self.move_t >= self.move_dur:
                self.move_traj = None
                self._sync_targets_to_state()  # resume live tracking, no jump
            return
        if self.mode == "joint":
            self.ac.writeOnce(JointPositions(self.joint_targets))
        elif self.mode == "task":
            self.ac.writeOnce(
                CartesianPose.from_matrix(self.task_target_pos, self.task_target_R))
        else:  # vr -- low-pass the clutch target, then stream (clamp safety)
            self.clutch.advance()
            try:
                self.ac.writeOnce(self.clutch.command())
            except RuntimeError as e:
                self.robot.automatic_error_recovery()
                mujoco.mj_step(self.model, self.data)  # hold, keep time moving
                self._blocked = True
                self._block_reason = str(e)

    def _commanded_pose(self):
        """The (pos, R) the active mode is commanding, for the ghost overlay
        frame -- or None (JOINT mode commands joint angles, not an EE pose)."""
        if self.mode == "task":
            return self.task_target_pos, self.task_target_R
        if self.mode == "vr":
            return self.clutch.cmd_pos_filt, self.clutch.cmd_R_filt
        return None

    def _draw_overlay(self):
        """Overlay the commanded EE pose (translucent ghost) vs the actual EE
        pose (solid) in the viewer, so tracking is visible."""
        if self.viewer is None or not self.show_markers:
            return
        scn = self.viewer.user_scn
        scn.ngeom = 0
        cmd = self._commanded_pose()
        if cmd is not None:
            add_frame(scn, cmd[0], cmd[1], length=0.12, alpha=0.35)
        ee_pos = self.data.site_xpos[self.robot._ee_site]
        ee_R = self.data.site_xmat[self.robot._ee_site].reshape(3, 3)
        add_frame(scn, ee_pos, ee_R, length=0.09, alpha=1.0)

    # -- telemetry -----------------------------------------------------

    def _vr_tag(self, vs):
        if not vs.connected:
            return "no client"
        if self.move_traj is not None:
            return "HOME"
        if self._blocked:
            return "singular/limit (hold)"
        return "engaged" if self.clutch.engaged else "idle"

    def _fsm_state(self):
        """A concise control-state label (+ color hint) for the UI indicator --
        fr3's stand-in for camel-franka's FSM readout. Single source of truth for
        "what is the arm doing right now"."""
        if self.trip is not None:
            return "TRIP", "#e05a5a"
        if self.move_traj is not None:
            return "MOVING", "#5a8fe0"
        if self.mode == "vr":
            tag = self._vr_tag(self.state.snapshot())
            color = {"no client": "#9aa0a6", "singular/limit (hold)": "#e0a24a",
                     "engaged": "#5a8fe0", "HOME": "#5a8fe0"}.get(tag, "#4aa84a")
            return f"VR · {tag}", color
        return self.mode.upper(), "#4aa84a"

    def snapshot(self):
        """A telemetry dict for the UI status line (mode, EE pose, gripper,
        manipulability, trip/notice, move progress, and VR fields in vr mode)."""
        p = self.data.site_xpos[self.robot._ee_site]
        w = self.robot._ik.manipulability_at(self.data)
        g = self.gripper.read_once()
        fsm, fsm_color = self._fsm_state()
        snap = dict(
            mode=self.mode,
            fsm=fsm,
            fsm_color=fsm_color,
            ee_cm=(np.asarray(p) * 100.0).tolist(),
            manip=float(w),
            grip=g.width / g.max_width,
            grasped=bool(g.is_grasped),
            trip=self.trip,
            notice=self.notice,
            moving=((self.move_t, self.move_dur) if self.move_traj is not None else None),
        )
        if self.mode == "vr":
            vs = self.state.snapshot()
            cmd = self.clutch.cmd_pos_filt
            snap.update(
                vr_connected=bool(vs.connected),
                vr_tag=self._vr_tag(vs),
                cmd_cm=(np.asarray(cmd) * 100.0).tolist(),
                track_err_mm=float(np.linalg.norm(cmd - p)) * 1000.0,
            )
        return snap

    # -- headless loop (tests / display-less host) ---------------------

    def run(self, max_ticks=None, on_tick=None):
        """Run the tick loop headless at ~1/TICK_MS (work-compensated). Mirrors
        the GUI's cadence without tkinter; ``on_tick(self)`` fires each tick."""
        period = TICK_MS / 1000
        tick = 0
        try:
            while True:
                t0 = time.perf_counter()
                if not self.step():
                    break
                if on_tick is not None:
                    on_tick(self)
                tick += 1
                if max_ticks is not None and tick >= max_ticks:
                    break
                sleep = period - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            self.close()

    def close(self):
        self.server.stop()
        if self.viewer is not None:
            self.viewer.close()
