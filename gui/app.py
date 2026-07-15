"""Unified control GUI -- joint / task / VR teleop in one Tkinter window.

Opens the MuJoCo passive viewer plus a control panel with a mode selector:

  * JOINT -- 7 sliders set joint targets (clamped to limits).
  * TASK  -- 6 sliders set the EE pose (x, y, z + roll, pitch, yaw); DLS IK and
    its singularity / joint-limit / collision guards are live (trips on overreach;
    press Recover or HOME).
  * VR    -- a Meta-Quest controller drives the arm over TCP (relative clutch);
    the sliders are idle and the grip/trigger/B-button do the work.

Execute runs a smooth quintic move to the slider targets (JOINT: the angles;
TASK: IK for the EE pose). HOME / Recover / Reset objects / Reset ALL are always
available. The viewer overlays the commanded EE pose (translucent) vs the actual
one. All control logic lives in ``gui/session.py`` (``ControlSession``); this
file is only the Tkinter panel driving it on an ``after()`` loop.

The task/scene, the VR position scale, the VR smoothing time constant, and the
overlay toggle are all adjustable at runtime from the panel's settings row.

Usage:  python main.py
"""
import math
import sys
import tkinter as tk
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import vec_to_pose
from scene import TASKS
from teleop.clutch import SMOOTH_TAU
from gui.session import ControlSession, TICK_MS


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


# --- theme -----------------------------------------------------------------
BG     = "#eceff4"   # window background
CARD   = "#ffffff"   # panels / cards
INK    = "#232a34"   # primary text
MUTED  = "#5b6472"   # secondary text
ACCENT = "#3b7dd8"   # primary (selected mode, Execute)
BLUE   = "#3b7dd8"
GREEN  = "#2fa96b"
AMBER  = "#e0a24a"
ORANGE = "#e0703b"
IDLE   = "#d4dbe4"   # unselected mode button
TROUGH = "#d7dde6"   # slider trough

FONT       = ("sans", 12)
FONT_BOLD  = ("sans", 12, "bold")
FONT_BTN   = ("sans", 13, "bold")
FONT_MODE  = ("sans", 16, "bold")   # the big JOINT / TASK / VR buttons
FONT_IND   = ("sans", 16, "bold")   # state indicator
FONT_SMALL   = ("sans", 11)
FONT_LABEL   = ("sans", 13, "bold")   # slider labels (x/y/z, joint1, open/closed, ...)
FONT_SETTING = ("sans", 16, "bold")   # the task selector row
FONT_MANIP   = ("sans", 14, "bold")   # manipulability readout
FONT_MONO    = ("monospace", 13)


def _darken(hexc, f=0.86):
    """Return ``hexc`` scaled toward black by ``f`` -- for button hover/press."""
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    return f"#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}"


class UnifiedGUI:
    def __init__(self, task="empty", hand="right", host="0.0.0.0", port=8081,
                 position_scale=2.0, smooth_tau=SMOOTH_TAU):
        self.session = ControlSession(
            task=task, view=True, hand=hand, host=host, port=port,
            position_scale=position_scale, smooth_tau=smooth_tau)
        self._init_scale = position_scale    # seed the runtime setting entries
        self._init_tau = smooth_tau
        self._editing = False     # True while the user is typing into entries
        self._was_moving = False  # track move end so we resnap the sliders
        self._build_ui()
        self._sync_sliders_to_state()

    # -- UI build ------------------------------------------------------

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title(f"FR3 sim control [{self.session.task_name}]")
        self.root.configure(bg=BG)
        # theme the classic-tk defaults (frames, labels, menus); colored widgets
        # are styled explicitly below.
        self.root.tk_setPalette(background=BG, foreground=INK,
                                activeBackground="#dbe2ec", activeForeground=INK)

        # -- mode + action bar --
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill=tk.X, padx=12, pady=(12, 6))
        self.mode_buttons = {}
        for text, val in (("JOINT", "joint"), ("TASK", "task"), ("VR", "vr")):
            b = tk.Button(top, text=text, font=FONT_MODE, width=6, pady=10,
                          relief="flat", bd=0, cursor="hand2", highlightthickness=0,
                          command=lambda v=val: self._select_mode(v))
            b.pack(side=tk.LEFT, padx=(0, 6))
            self.mode_buttons[val] = b

        self._btn(top, "HOME", self.session.go_home, BLUE).pack(side=tk.LEFT, padx=(12, 4))
        self._btn(top, "Recover", self._recover, AMBER).pack(side=tk.LEFT, padx=4)

        # Execute + its move time: manual modes only (hidden in VR, where the
        # controller drives the arm directly).
        self.exec_group = tk.Frame(top, bg=BG)
        self.exec_group.pack(side=tk.LEFT, padx=(12, 0))
        self._btn(self.exec_group, "Execute", self._execute, ACCENT).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(self.exec_group, text="exec(s)", font=FONT_LABEL, bg=BG, fg=INK).pack(side=tk.LEFT, padx=(4, 2))
        self.exec_entry = self._mk_entry(self.exec_group, 5)
        self.exec_entry.insert(0, "2.0")
        self.exec_entry.pack(side=tk.LEFT)

        # -- state indicator (fr3's stand-in for camel-franka's FSM label) --
        self.indicator = tk.Label(self.root, text="", anchor="w", fg="white",
                                  font=FONT_IND, padx=14, pady=8)
        self.indicator.pack(fill=tk.X, padx=12, pady=(2, 6))

        # -- manipulability readout (turns red near a singularity) --
        self.manip_label = tk.Label(self.root, text="", anchor="w", font=FONT_MANIP,
                                    bg=BG, fg=INK, padx=14)
        self.manip_label.pack(fill=tk.X, padx=12, pady=(0, 4))

        # -- task selector (its own row, larger font) --
        trow = tk.Frame(self.root, bg=BG)
        trow.pack(fill=tk.X, padx=12, pady=(2, 6))
        tk.Label(trow, text="task", font=FONT_SETTING, bg=BG, fg=INK).pack(side=tk.LEFT)
        self.task_var = tk.StringVar(value=self.session.task_name)
        om = tk.OptionMenu(trow, self.task_var, *sorted(TASKS), command=self._on_task)
        om.config(font=FONT_SETTING, bg=CARD, fg=INK, relief="flat", bd=0, cursor="hand2",
                  activebackground="#e6ebf1", highlightthickness=1, highlightbackground=TROUGH)
        om["menu"].config(font=FONT_SETTING, bg=CARD, fg=INK, activebackground=ACCENT,
                          activeforeground="white")
        om.pack(side=tk.LEFT, padx=(10, 0))

        # -- mode-panel slot: an expanding frame the mode's panels pack into, so
        # the window (fixed size, see run()) never resizes when you switch modes;
        # the slot just absorbs the height difference between modes. --
        self.panel_area = tk.Frame(self.root, bg=BG)
        self.panel_area.pack(fill=tk.BOTH, expand=True, padx=12, pady=2)

        # -- joint panel: slider + numeric entry per DOF (degrees) --
        self.joint_frame = self._panel("Joint targets (deg)")
        self.joint_sliders, self.joint_entries = [], []
        for i in range(7):
            row = tk.Frame(self.joint_frame, bg=CARD)
            row.pack(fill=tk.X, padx=10, pady=1)
            s = self._slider(row, float(np.degrees(self.session.robot._q_min[i])),
                             float(np.degrees(self.session.robot._q_max[i])), 0.5,
                             f"joint{i + 1}")
            e = self._entry(row)                       # entry fixed on the right
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)  # slider fills the rest
            self.joint_sliders.append(s)
            self.joint_entries.append(e)

        # -- task panel: x/y/z in cm, roll/pitch/yaw in deg --
        self.task_frame = self._panel("EE pose (cm / deg)")
        self.task_sliders, self.task_entries = {}, {}
        specs = [("x", 10, 90, 0.5), ("y", -50, 50, 0.5), ("z", 0, 100, 0.5),
                 ("roll", -180, 180, 1), ("pitch", -180, 180, 1), ("yaw", -180, 180, 1)]
        for name, lo, hi, res in specs:
            row = tk.Frame(self.task_frame, bg=CARD)
            row.pack(fill=tk.X, padx=10, pady=1)
            s = self._slider(row, lo, hi, res, name)
            e = self._entry(row)                       # entry fixed on the right
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)  # slider fills the rest
            self.task_sliders[name] = s
            self.task_entries[name] = e

        # -- gripper: manual modes only (VR uses the controller trigger) --
        self.gripper_frame = self._panel("Gripper")
        self.gripper_slider = self._slider(self.gripper_frame, 0.0, 1.0, 0.02,
                                           "open (0=closed, 1=open)", length=360)
        self.gripper_slider.pack(fill=tk.X, expand=True, padx=10, pady=(0, 4))

        # -- VR panel: position scale + smoothing, one slider per row (shown in
        # VR mode, in place of the joint/task/gripper sliders) --
        self.vr_frame = self._panel("VR teleop")
        sr = tk.Frame(self.vr_frame, bg=CARD)
        sr.pack(fill=tk.X, padx=10, pady=1)
        self.scale_slider = self._slider(sr, 0.5, 5.0, 0.1, "position scale", length=360)
        self.scale_slider.set(self._init_scale)
        self.scale_slider.config(command=lambda v: self.session.set_position_scale(float(v)))
        self.scale_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        mr = tk.Frame(self.vr_frame, bg=CARD)
        mr.pack(fill=tk.X, padx=10, pady=1)
        self.smooth_slider = self._slider(mr, 0.0, 0.2, 0.005, "smooth-tau (s)", length=360)
        self.smooth_slider.set(self._init_tau)
        self.smooth_slider.config(command=lambda v: self.session.set_smooth_tau(float(v)))
        self.smooth_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # -- reset buttons (always available) --
        self.reset_frame = tk.Frame(self.root, bg=BG)
        self.reset_frame.pack(fill=tk.X, padx=12, pady=(4, 2))
        self._btn(self.reset_frame, "Reset objects", self.session.reset_objects, GREEN).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self._btn(self.reset_frame, "Reset ALL", self._reset_all, ORANGE).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        # -- status --
        self.status = tk.Label(self.root, text="", justify=tk.LEFT, anchor="w",
                               font=FONT_MONO, bg=CARD, fg=INK, wraplength=400,
                               padx=12, pady=10)
        self.status.pack(fill=tk.X, padx=12, pady=(4, 12))
        self._style_mode_buttons()
        self._show_frame()

    # -- styled-widget factories ---------------------------------------

    def _btn(self, parent, text, cmd, color, fg="white"):
        return tk.Button(parent, text=text, command=cmd, font=FONT_BTN, bg=color,
                         fg=fg, activebackground=_darken(color), activeforeground="white",
                         relief="flat", bd=0, padx=16, pady=9, cursor="hand2",
                         highlightthickness=0)

    def _mk_entry(self, parent, width):
        return tk.Entry(parent, width=width, font=FONT, relief="solid", bd=1,
                        bg=CARD, fg=INK, insertbackground=INK, highlightthickness=0)

    def _panel(self, title):
        return tk.LabelFrame(self.panel_area, text=title, font=FONT_BOLD, bg=CARD, fg=MUTED,
                             relief="flat", bd=0, padx=6, pady=6,
                             highlightbackground=TROUGH, highlightthickness=1)

    def _slider(self, parent, lo, hi, res, label, length=300):
        return tk.Scale(parent, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                        length=length, label=label, font=FONT_LABEL, bg=CARD, fg=INK,
                        troughcolor=TROUGH, activebackground=ACCENT, highlightthickness=0,
                        bd=0, sliderlength=38, width=28)

    def _entry(self, row):
        e = self._mk_entry(row, 7)
        e.pack(side=tk.RIGHT, padx=8)
        e.bind("<Return>", lambda ev: self._execute())
        e.bind("<FocusIn>", lambda ev: setattr(self, "_editing", True))
        e.bind("<Escape>", lambda ev: self._cancel_edit())
        return e

    def _show_frame(self):
        """Show the panels for the current mode: joint/task get their slider panel
        plus the gripper; VR gets the scale + smooth-tau sliders (no gripper)."""
        for f in (self.joint_frame, self.task_frame, self.gripper_frame, self.vr_frame):
            f.pack_forget()
        mode = self.session.mode
        if mode == "joint":
            panels = (self.joint_frame, self.gripper_frame)
        elif mode == "task":
            panels = (self.task_frame, self.gripper_frame)
        else:  # vr
            panels = (self.vr_frame,)
        for f in panels:
            f.pack(fill=tk.X, pady=(0, 6))

    # -- buttons / mode ------------------------------------------------

    def _select_mode(self, mode):
        self.session.set_mode(mode)
        self._sync_sliders_to_state()
        self._show_frame()
        self._style_mode_buttons()
        # Execute + exec(s) only make sense in the manual modes.
        if mode == "vr":
            self.exec_group.pack_forget()
        else:
            self.exec_group.pack(side=tk.LEFT, padx=(12, 0))

    def _style_mode_buttons(self):
        """Highlight the active mode button (accent) and mute the others."""
        for val, btn in self.mode_buttons.items():
            on = (val == self.session.mode)
            btn.config(bg=ACCENT if on else IDLE, fg="white" if on else INK,
                       activebackground=_darken(ACCENT) if on else _darken(IDLE),
                       activeforeground="white" if on else INK)

    def _exec_time(self):
        try:
            return max(0.05, float(self.exec_entry.get()))
        except (ValueError, AttributeError):
            return 2.0

    def _execute(self):
        if self.session.mode == "vr":
            return
        entries = (list(zip(self.joint_sliders, self.joint_entries))
                   if self.session.mode == "joint"
                   else [(self.task_sliders[n], self.task_entries[n]) for n in self.task_entries])
        for slider, entry in entries:
            try:
                slider.set(float(entry.get()))
            except ValueError:
                pass
        self._editing = False
        self.root.focus_set()
        self._push_targets()
        self.session.execute(self._exec_time())

    def _cancel_edit(self):
        self._editing = False
        self.root.focus_set()

    def _recover(self):
        # Recover and Reset ALL change the arm *instantly* (no quintic move whose
        # end would resnap the sliders), so resync the sliders here -- otherwise
        # the next tick re-pushes the stale slider values and the arm jumps back
        # to the old command instead of holding at the reset pose.
        self.session.recover()
        self._sync_sliders_to_state()

    def _reset_all(self):
        self.session.reset_all()
        self._sync_sliders_to_state()

    # -- runtime settings ----------------------------------------------

    def _on_task(self, name):
        self.session.reload_task(name)
        self._sync_sliders_to_state()
        self.root.title(f"FR3 sim control [{self.session.task_name}]")

    # -- per-tick sync -------------------------------------------------

    def _push_targets(self):
        """Push the visible sliders into the session's live targets."""
        if self.session.mode == "joint":
            self.session.set_joint_targets(np.radians([s.get() for s in self.joint_sliders]))
        elif self.session.mode == "task":
            pos = np.array([self.task_sliders[n].get() for n in ("x", "y", "z")]) / 100.0
            R = euler_to_mat(*[math.radians(self.task_sliders[n].get())
                               for n in ("roll", "pitch", "yaw")])
            self.session.set_task_target(pos, R)

    def _sync_sliders_to_state(self):
        """Set every slider to the robot's current pose (so switching modes /
        recovering / finishing a move never causes a jump/re-trip)."""
        st = self.session.robot.read_once()
        for i in range(7):
            self.joint_sliders[i].set(float(np.degrees(st.q[i])))
        pos, R = vec_to_pose(st.O_T_EE)
        for name, v in zip(("x", "y", "z"), pos):
            self.task_sliders[name].set(float(v) * 100.0)
        for name, v in zip(("roll", "pitch", "yaw"), mat_to_euler(R)):
            self.task_sliders[name].set(float(np.degrees(v)))
        g = self.session.gripper
        self.gripper_slider.set(g.width() / g.max_width)

    def _refresh_entries(self):
        """Show the current value in every entry -- unless the user is typing."""
        if self._editing or self.session.mode == "vr":
            return
        st = self.session.robot.read_once()
        if self.session.mode == "joint":
            for i, e in enumerate(self.joint_entries):
                self._set_entry(e, f"{np.degrees(st.q[i]):.1f}")
        else:
            pos, R = vec_to_pose(st.O_T_EE)
            vals = dict(zip(("x", "y", "z"), pos * 100.0))
            vals.update(zip(("roll", "pitch", "yaw"),
                            [math.degrees(a) for a in mat_to_euler(R)]))
            for name, e in self.task_entries.items():
                self._set_entry(e, f"{vals[name]:.1f}")

    @staticmethod
    def _set_entry(entry, text):
        entry.delete(0, tk.END)
        entry.insert(0, text)

    # -- control tick --------------------------------------------------

    def _tick(self):
        moving = self.session.move_traj is not None
        if self.session.mode in ("joint", "task"):
            self.session.set_gripper_frac(self.gripper_slider.get())
            if not moving and not self._editing:
                self._push_targets()

        if not self.session.step():
            self.root.destroy()
            return

        # After a timed move finishes, snap the sliders to the settled pose so
        # live tracking resumes from there instead of yanking back to old values.
        if self._was_moving and self.session.move_traj is None:
            self._sync_sliders_to_state()
        self._was_moving = self.session.move_traj is not None

        self._update_status()
        self._refresh_entries()
        self.root.after(TICK_MS, self._tick)

    def _update_status(self):
        s = self.session.snapshot()
        self.indicator.config(text=f"STATE   {s['fsm']}", bg=s["fsm_color"])
        w = s["manip"]
        self.manip_label.config(text=f"manipulability   w = {w:.3f}",
                                fg="#e05a5a" if w < 0.04 else INK)
        ex, ey, ez = s["ee_cm"]
        if s["mode"] == "vr":
            cx, cy, cz = s["cmd_cm"]
            conn = "connected" if s["vr_connected"] else "waiting for VR client"
            msg = (f"VR  |  {conn}  |  {s['vr_tag']}\n"
                   f"cmd=({cx:+.1f},{cy:+.1f},{cz:+.1f})cm\n"
                   f"act=({ex:+.1f},{ey:+.1f},{ez:+.1f})cm  err={s['track_err_mm']:4.0f}mm\n"
                   f"grip={s['grip']:.2f}{'  [GRASP]' if s['grasped'] else ''}")
        else:
            msg = (f"mode={s['mode']:5s}  EE=({ex:+.1f},{ey:+.1f},{ez:+.1f})cm\n"
                   f"grip={s['grip']:.2f}{'  [GRASP]' if s['grasped'] else ''}")
            if s["moving"] is not None:
                t, dur = s["moving"]
                msg += f"\nmoving... {t:.1f}/{dur:.1f}s"
        if s["trip"]:
            msg += f"\n⚠ SAFETY TRIP: {s['trip']}\n   press Recover or HOME"
        if s["notice"]:
            msg += f"\n{s['notice']}"
        self.status.config(text=msg)

    def run(self):
        # Lock the window to its largest layout (JOINT mode is shown at startup and
        # is the tallest) plus a generous margin, then disable resizing -- so
        # switching modes never changes the window size (the expanding panel_area
        # absorbs the per-mode height difference).
        self.root.update_idletasks()
        w = self.root.winfo_reqwidth() + 40
        h = self.root.winfo_reqheight() + 60
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(w, h)
        self.root.resizable(False, False)
        self.root.after(TICK_MS, self._tick)
        self.root.mainloop()
        self.session.close()


def main():
    # No CLI flags: task/scene, VR scale, VR smoothing, and the overlay toggle
    # are all set at runtime from the panel; the VR server binds its defaults.
    UnifiedGUI().run()


if __name__ == "__main__":
    main()
