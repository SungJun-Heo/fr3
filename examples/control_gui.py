"""Simple hand-control GUI for the sim robot -- joint space or task space.

Opens the MuJoCo passive viewer plus a Tkinter slider panel:
  * JOINT mode -- 7 sliders set joint targets (clamped to limits).
  * TASK  mode -- 6 sliders set the EE pose (x, y, z + roll, pitch, yaw).

Both drive the robot through the same SimRobot control API the rest of the
project uses, so in TASK mode the DLS IK and its safety guards (singularity /
joint-limit / collision) are live -- reach too far and it trips; press Recover
or HOME to resume.

Usage:  python examples/control_gui.py [task]      (default task: "empty")
"""

import math
import sys
import tkinter as tk
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, JointPositions, CartesianPose, Gripper
from robot.sim_robot import ARM_JOINTS
from controller.planning import QuinticTrajectoryGenerator

TICK_MS = 20        # GUI/control tick period
HOME_DURATION = 2.0  # seconds for the HOME motion


def euler_to_mat(rx, ry, rz):
    """Roll-pitch-yaw (applied X then Y then Z) -> 3x3 rotation matrix."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat_to_euler(R):
    """3x3 rotation -> (roll, pitch, yaw), matching euler_to_mat."""
    sy = np.clip(-R[2, 0], -1.0, 1.0)
    return math.atan2(R[2, 1], R[2, 2]), math.asin(sy), math.atan2(R[1, 0], R[0, 0])


def make_pose_vec(pos, R):
    """position + 3x3 rotation -> column-major length-16 O_T_EE."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T.flatten(order="F")


class ControlGUI:
    def __init__(self, task="empty"):
        self.robot = SimRobot(task)
        self.model, self.data = self.robot.model, self.robot.data
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.substeps = max(1, round(TICK_MS / 1000 / self.model.opt.timestep))
        self.mode = "joint"
        self.ac = self.robot.start_joint_position_control()
        self.gripper = Gripper(self.robot)
        self.trip = None
        self._editing = False    # True while the user is typing into entries
        self._home_traj = None   # active quintic while a HOME motion runs
        self._home_t = 0.0
        self._build_ui()
        self._sync_sliders_to_state()

    # -- UI ------------------------------------------------------------

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("FR3 sim control")

        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=6, pady=4)
        self.mode_var = tk.StringVar(value="joint")
        tk.Radiobutton(top, text="Joint", variable=self.mode_var, value="joint",
                       command=self._on_mode).pack(side=tk.LEFT)
        tk.Radiobutton(top, text="Task", variable=self.mode_var, value="task",
                       command=self._on_mode).pack(side=tk.LEFT)
        tk.Button(top, text="Execute", command=self._execute).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="HOME", command=self._go_home).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Recover", command=self._recover).pack(side=tk.LEFT)

        # joint: slider + numeric entry per DOF (degrees)
        self.joint_frame = tk.LabelFrame(self.root, text="Joint targets (deg)")
        self.joint_sliders = []
        self.joint_entries = []
        for i in range(7):
            row = tk.Frame(self.joint_frame)
            row.pack(fill=tk.X, padx=6)
            s = tk.Scale(row, from_=float(np.degrees(self.robot._q_min[i])),
                         to=float(np.degrees(self.robot._q_max[i])), resolution=0.5,
                         orient=tk.HORIZONTAL, length=240, label=f"joint{i + 1}")
            s.pack(side=tk.LEFT)
            e = tk.Entry(row, width=8)
            e.pack(side=tk.LEFT, padx=4)
            e.bind("<Return>", lambda ev: self._execute())
            e.bind("<FocusIn>", lambda ev: setattr(self, "_editing", True))
            e.bind("<Escape>", lambda ev: self._cancel_edit())
            self.joint_sliders.append(s)
            self.joint_entries.append(e)

        # task: slider + numeric entry per DOF (x/y/z in cm, roll/pitch/yaw in deg)
        self.task_frame = tk.LabelFrame(self.root, text="EE pose (cm / deg)")
        self.task_sliders = {}
        self.task_entries = {}
        specs = [("x", 10, 90, 0.5), ("y", -50, 50, 0.5), ("z", 0, 100, 0.5),
                 ("roll", -180, 180, 1), ("pitch", -180, 180, 1), ("yaw", -180, 180, 1)]
        for name, lo, hi, res in specs:
            row = tk.Frame(self.task_frame)
            row.pack(fill=tk.X, padx=6)
            s = tk.Scale(row, from_=lo, to=hi, resolution=res,
                         orient=tk.HORIZONTAL, length=240, label=name)
            s.pack(side=tk.LEFT)
            e = tk.Entry(row, width=8)
            e.pack(side=tk.LEFT, padx=4)
            e.bind("<Return>", lambda ev: self._execute())
            e.bind("<FocusIn>", lambda ev: setattr(self, "_editing", True))
            e.bind("<Escape>", lambda ev: self._cancel_edit())
            self.task_sliders[name] = s
            self.task_entries[name] = e

        # gripper slider: always visible (works in either mode). Driven
        # non-blocking each tick, so it tracks alongside arm control.
        self.gripper_frame = tk.LabelFrame(self.root, text="Gripper")
        self.gripper_slider = tk.Scale(self.gripper_frame, from_=0.0, to=1.0,
                                       resolution=0.02, orient=tk.HORIZONTAL,
                                       length=320, label="open (0=closed, 1=open)")
        self.gripper_slider.pack(padx=6)
        self.gripper_frame.pack(fill=tk.X, padx=6)

        # wraplength caps the text width so a long SAFETY-TRIP message wraps to
        # another line instead of stretching the window horizontally.
        self.status = tk.Label(self.root, text="", justify=tk.LEFT, anchor="w",
                               font=("monospace", 10), wraplength=320)
        self.status.pack(fill=tk.X, padx=6, pady=4)
        self._show_frame()

    def _show_frame(self):
        self.joint_frame.pack_forget()
        self.task_frame.pack_forget()
        frame = self.joint_frame if self.mode == "joint" else self.task_frame
        frame.pack(fill=tk.X, padx=6, before=self.gripper_frame)  # keep grip below

    def _sync_sliders_to_state(self):
        """Set every slider to the robot's current pose (so switching modes or
        recovering never causes a jump/re-trip)."""
        st = self.robot.read_once()
        for i in range(7):
            self.joint_sliders[i].set(float(np.degrees(st.q[i])))
        T = st.O_T_EE.reshape(4, 4, order="F")
        for name, v in zip(("x", "y", "z"), T[:3, 3]):
            self.task_sliders[name].set(float(v) * 100.0)
        for name, v in zip(("roll", "pitch", "yaw"), mat_to_euler(T[:3, :3])):
            self.task_sliders[name].set(float(np.degrees(v)))
        self.gripper_slider.set(self.gripper.width() / self.gripper.max_width)  # m -> 0..1

    # -- buttons -------------------------------------------------------

    def _on_mode(self):
        self.mode = self.mode_var.get()
        self._sync_sliders_to_state()  # avoid a jump on switch
        self.ac = (self.robot.start_joint_position_control() if self.mode == "joint"
                   else self.robot.start_cartesian_pose_control())
        self._show_frame()

    def _execute(self):
        """Apply ALL typed numeric targets at once: parse the visible mode's
        entries and set the sliders (which drive the robot). A bad field is
        skipped. Ends editing so entries resume reflecting the live state."""
        entries = (list(zip(self.joint_sliders, self.joint_entries)) if self.mode == "joint"
                   else [(self.task_sliders[n], self.task_entries[n]) for n in self.task_entries])
        for slider, entry in entries:
            try:
                slider.set(float(entry.get()))
            except ValueError:
                pass
        self._editing = False
        self.root.focus_set()  # defocus entries so they resume reflecting state

    def _cancel_edit(self):
        """Discard in-progress typing (Escape) and resume the live readout."""
        self._editing = False
        self.root.focus_set()

    def _go_home(self):
        """Run a smooth quintic motion to HOME (not a teleport). Works from any
        mode; the per-tick loop streams the trajectory as joint commands."""
        self._recover()  # clear any trip first
        home = self.model.key_qpos[0][:7].copy()
        traj = QuinticTrajectoryGenerator()
        traj.InitTrajectory(self.robot.read_once().q, home, 0.0, HOME_DURATION)
        self._home_traj = traj
        self._home_t = 0.0

    def _recover(self):
        self.robot.automatic_error_recovery()
        self.trip = None
        self._editing = False
        self._sync_sliders_to_state()

    # -- control tick --------------------------------------------------

    def _build_command(self):
        if self.mode == "joint":
            return JointPositions(np.radians([s.get() for s in self.joint_sliders]))
        pos = np.array([self.task_sliders[n].get() for n in ("x", "y", "z")]) / 100.0
        R = euler_to_mat(*[math.radians(self.task_sliders[n].get())
                           for n in ("roll", "pitch", "yaw")])
        return CartesianPose(make_pose_vec(pos, R))

    def _tick(self):
        if not self.viewer.is_running():
            self.root.destroy()
            return
        self.gripper.set_target_width(
            self.gripper_slider.get() * self.gripper.max_width)  # 0..1 -> m, non-blocking
        for _ in range(self.substeps):
            try:
                self._control_substep()
            except RuntimeError as e:
                self.trip = str(e)
        self.viewer.sync()
        self._update_status()
        self._refresh_entries()
        self.root.after(TICK_MS, self._tick)

    def _refresh_entries(self):
        """Show the current value in every entry -- unless the user is composing
        a command (editing), in which case leave them all alone so several typed
        values persist together until Execute."""
        if self._editing:
            return
        st = self.robot.read_once()
        if self.mode == "joint":
            for i, e in enumerate(self.joint_entries):
                self._set_entry(e, f"{np.degrees(st.q[i]):.1f}")
        else:
            T = st.O_T_EE.reshape(4, 4, order="F")
            vals = dict(zip(("x", "y", "z"), T[:3, 3] * 100.0))
            vals.update(zip(("roll", "pitch", "yaw"),
                            [math.degrees(a) for a in mat_to_euler(T[:3, :3])]))
            for name, e in self.task_entries.items():
                self._set_entry(e, f"{vals[name]:.1f}")

    @staticmethod
    def _set_entry(entry, text):
        entry.delete(0, tk.END)
        entry.insert(0, text)

    def _control_substep(self):
        """Advance one sim step: hold if tripped, stream the HOME trajectory if
        one is running, else track the sliders."""
        if self.trip is not None:
            mujoco.mj_step(self.model, self.data)
            return
        if self._home_traj is not None:
            self._home_t += self.model.opt.timestep
            q = self._home_traj.getPositionTrajectory(self._home_t)
            self.ac.writeOnce(JointPositions(q))
            if self._home_t >= HOME_DURATION:
                self._home_traj = None
                self._sync_sliders_to_state()  # resume from HOME, no jump
            return
        self.ac.writeOnce(self._build_command())

    def _update_status(self):
        p = self.data.site_xpos[self.robot._ee_site]
        J = self.robot._ik._jacobian(self.data)
        w = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
        g = self.gripper.read_once()
        msg = (f"mode={self.mode:5s}  EE=({p[0]*100:+.1f},{p[1]*100:+.1f},"
               f"{p[2]*100:+.1f})cm\n"
               f"manip w={w:.3f}   grip={g.width / g.max_width:.2f}"
               f"{'  [GRASP]' if g.is_grasped else ''}")
        if self.trip:
            msg += f"\n⚠ SAFETY TRIP: {self.trip}\n   press Recover or HOME"
        self.status.config(text=msg)

    def run(self):
        self.root.after(TICK_MS, self._tick)
        self.root.mainloop()
        self.viewer.close()


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "empty"
    ControlGUI(task).run()


if __name__ == "__main__":
    main()
