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
from robot import SimRobot, JointPositions, CartesianPose
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
        self.trip = None
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
        tk.Button(top, text="HOME", command=self._go_home).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Recover", command=self._recover).pack(side=tk.LEFT)

        # joint sliders
        self.joint_frame = tk.LabelFrame(self.root, text="Joint targets (rad)")
        self.joint_sliders = []
        for i in range(7):
            s = tk.Scale(self.joint_frame, from_=float(self.robot._q_min[i]),
                         to=float(self.robot._q_max[i]), resolution=0.01,
                         orient=tk.HORIZONTAL, length=320, label=f"joint{i + 1}")
            s.pack(padx=6)
            self.joint_sliders.append(s)

        # task sliders
        self.task_frame = tk.LabelFrame(self.root, text="EE pose")
        self.task_sliders = {}
        specs = [("x", 0.1, 0.9, 0.005), ("y", -0.5, 0.5, 0.005),
                 ("z", 0.0, 1.0, 0.005), ("roll", -math.pi, math.pi, 0.02),
                 ("pitch", -math.pi, math.pi, 0.02), ("yaw", -math.pi, math.pi, 0.02)]
        for name, lo, hi, res in specs:
            s = tk.Scale(self.task_frame, from_=lo, to=hi, resolution=res,
                         orient=tk.HORIZONTAL, length=320, label=name)
            s.pack(padx=6)
            self.task_sliders[name] = s

        # wraplength caps the text width so a long SAFETY-TRIP message wraps to
        # another line instead of stretching the window horizontally.
        self.status = tk.Label(self.root, text="", justify=tk.LEFT, anchor="w",
                               font=("monospace", 10), wraplength=320)
        self.status.pack(fill=tk.X, padx=6, pady=4)
        self._show_frame()

    def _show_frame(self):
        self.joint_frame.pack_forget()
        self.task_frame.pack_forget()
        (self.joint_frame if self.mode == "joint" else self.task_frame).pack(
            fill=tk.X, padx=6)

    def _sync_sliders_to_state(self):
        """Set every slider to the robot's current pose (so switching modes or
        recovering never causes a jump/re-trip)."""
        st = self.robot.read_once()
        for i in range(7):
            self.joint_sliders[i].set(float(st.q[i]))
        T = st.O_T_EE.reshape(4, 4, order="F")
        for name, v in zip(("x", "y", "z"), T[:3, 3]):
            self.task_sliders[name].set(float(v))
        for name, v in zip(("roll", "pitch", "yaw"), mat_to_euler(T[:3, :3])):
            self.task_sliders[name].set(float(v))

    # -- buttons -------------------------------------------------------

    def _on_mode(self):
        self.mode = self.mode_var.get()
        self._sync_sliders_to_state()  # avoid a jump on switch
        self.ac = (self.robot.start_joint_position_control() if self.mode == "joint"
                   else self.robot.start_cartesian_pose_control())
        self._show_frame()

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
        self._sync_sliders_to_state()

    # -- control tick --------------------------------------------------

    def _build_command(self):
        if self.mode == "joint":
            return JointPositions([s.get() for s in self.joint_sliders])
        pos = np.array([self.task_sliders[n].get() for n in ("x", "y", "z")])
        R = euler_to_mat(*[self.task_sliders[n].get()
                           for n in ("roll", "pitch", "yaw")])
        return CartesianPose(make_pose_vec(pos, R))

    def _tick(self):
        if not self.viewer.is_running():
            self.root.destroy()
            return
        for _ in range(self.substeps):
            try:
                self._control_substep()
            except RuntimeError as e:
                self.trip = str(e)
        self.viewer.sync()
        self._update_status()
        self.root.after(TICK_MS, self._tick)

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
        msg = (f"mode={self.mode:5s}  EE=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  "
               f"manip w={w:.3f}")
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
