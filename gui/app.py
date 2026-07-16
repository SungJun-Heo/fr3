"""Unified control GUI -- joint / task / VR teleop in one Tkinter window.

Opens the MuJoCo passive viewer plus a control panel with a mode selector:

  * JOINT -- 7 sliders set joint targets (clamped to limits).
  * TASK  -- 6 sliders set the EE pose (x, y, z + roll, pitch, yaw); DLS IK and
    its singularity / joint-limit / collision guards are live (trips on overreach;
    press Recover or HOME).
  * VR    -- a Meta-Quest controller drives the arm over TCP (relative clutch);
    the sliders are idle and the buttons do the work: grip = clutch, trigger =
    gripper, A = REC/PAUSE, B = HOME (teleop hand); X = Save, Y = reset-all +
    randomize (free hand).

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
import os
import sys
import time
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import vec_to_pose
from scene import TASKS, task_instruction
from teleop.clutch import SMOOTH_TAU
from gui.session import ControlSession, TICK_MS
from collection import (CollectionConfig, Collector, EpisodePlayer,
                        count_episodes, list_episodes, delete_episode,
                        episode_meta)


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
VIOLET = "#8b5cf6"    # domain-randomize objects
DANGER = "#d9534f"   # destructive action (delete episode)
IDLE   = "#d4dbe4"   # unselected mode button
TROUGH = "#d7dde6"   # slider trough

FONT       = ("sans", 12)
FONT_BOLD  = ("sans", 12, "bold")
FONT_BTN   = ("sans", 13, "bold")
FONT_MODE  = ("sans", 16, "bold")   # the big JOINT / TASK / VR buttons
FONT_IND   = ("sans", 16, "bold")   # state indicator
FONT_SMALL   = ("sans", 11)
FONT_LABEL   = ("sans", 13, "bold")   # slider labels (x/y/z, joint1, open/closed, ...)
FONT_SETTING = ("sans", 16, "bold")   # the task selector label
FONT_TASK    = ("sans", 20, "bold")   # the task dropdown (enlarged)
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
        # Data collection: a Collector is built lazily on the first Record press
        # (it opens an offscreen camera renderer, so we defer the cost until used).
        self.collector = None
        self.collect_config = CollectionConfig()   # shared by the Collector + counter
        self._last_saved = None
        self._last_collect_state = "idle"   # idle / rec / paused (for UI resync)
        # Rising-edge state for the VR face buttons routed to GUI actions
        # (A->REC/PAUSE, X->Save, Y->reset-all+randomize; B->HOME is in session).
        self._prev_record_btn = False
        self._prev_save_btn = False
        self._prev_reset_btn = False
        self.player = EpisodePlayer(self.session)  # re-simulation episode replay
        self._replaying = False   # GUI replay mode (distinct from player's cursor)
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

        # Two columns. The left "main" column holds live control (mode bar, state
        # indicator, sliders, status); the right "Session / Data" card groups
        # everything about the session/episode -- task selection, scene resets,
        # and recording -- so they sit together, apart from the control widgets.
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)
        main_col = tk.Frame(body, bg=BG)
        main_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side_col = tk.Frame(body, bg=BG)
        side_col.pack(side=tk.RIGHT, fill=tk.Y)
        self.session_card = tk.LabelFrame(
            side_col, text="Session / Data", font=FONT_BOLD, bg=CARD, fg=MUTED,
            relief="flat", bd=0, padx=12, pady=10,
            highlightbackground=TROUGH, highlightthickness=1)
        self.session_card.pack(fill=tk.Y, anchor="n", padx=(0, 12), pady=12)

        # -- mode + action bar --
        top = tk.Frame(main_col, bg=BG)
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
        self.indicator = tk.Label(main_col, text="", anchor="w", fg="white",
                                  font=FONT_IND, padx=14, pady=8)
        self.indicator.pack(fill=tk.X, padx=12, pady=(2, 6))

        # -- manipulability readout (turns red near a singularity) --
        self.manip_label = tk.Label(main_col, text="", anchor="w", font=FONT_MANIP,
                                    bg=BG, fg=INK, padx=14)
        self.manip_label.pack(fill=tk.X, padx=12, pady=(0, 4))

        # -- task selector: label on its own line, the (enlarged) dropdown below --
        tsec = tk.Frame(self.session_card, bg=CARD)
        tsec.pack(fill=tk.X, pady=(0, 2))
        tk.Label(tsec, text="task", font=FONT_SETTING, bg=CARD, fg=INK,
                 anchor="w").pack(fill=tk.X)
        self.task_var = tk.StringVar(value=self.session.task_name)
        om = tk.OptionMenu(tsec, self.task_var, *sorted(TASKS), command=self._on_task)
        om.config(font=FONT_TASK, bg=CARD, fg=INK, relief="flat", bd=0, cursor="hand2",
                  activebackground="#e6ebf1", highlightthickness=1, highlightbackground=TROUGH,
                  anchor="w", padx=14, pady=8)
        om["menu"].config(font=FONT_TASK, bg=CARD, fg=INK, activebackground=ACCENT,
                          activeforeground="white")
        om.pack(fill=tk.X, pady=(4, 0))

        # -- mode-panel slot: an expanding frame the mode's panels pack into, so
        # the window (fixed size, see run()) never resizes when you switch modes;
        # the slot just absorbs the height difference between modes. --
        self.panel_area = tk.Frame(main_col, bg=BG)
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

        # -- collected-episode counter (live) + scene reset / randomize --
        self._sep(self.session_card)
        self.episode_label = tk.Label(self.session_card, text="episodes collected: 0",
                                      font=FONT_MANIP, bg=CARD, fg=INK, anchor="w")
        self.episode_label.pack(fill=tk.X, pady=(0, 6))
        # Randomize: domain-randomize the movable objects (set up a fresh scene
        # before recording). Its own full-width row above the reset pair.
        self._btn(self.session_card, "Randomize objects", self._randomize, VIOLET).pack(
            fill=tk.X, pady=(0, 4))
        self.reset_frame = tk.Frame(self.session_card, bg=CARD)
        self.reset_frame.pack(fill=tk.X)
        self._btn(self.reset_frame, "Reset objects", self.session.reset_objects, GREEN).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self._btn(self.reset_frame, "Reset ALL", self._reset_all, ORANGE).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        # -- data collection: record episodes to the raw IR (any mode) --
        self._sep(self.session_card)
        crec = tk.Frame(self.session_card, bg=CARD)
        crec.pack(fill=tk.X)
        tk.Label(crec, text="record episodes", font=FONT_BOLD, bg=CARD, fg=MUTED,
                 anchor="w").pack(fill=tk.X, pady=(0, 4))
        irow = tk.Frame(crec, bg=CARD)
        irow.pack(fill=tk.X, pady=(0, 4))
        tk.Label(irow, text="instr", font=FONT_LABEL, bg=CARD, fg=INK).pack(side=tk.LEFT)
        self.instr_entry = self._mk_entry(irow, 16)
        self.instr_entry.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)
        self._apply_task_instruction()   # pre-fill with the task's default
        # REC/PAUSE toggle + Save. Idle->REC starts; recording->PAUSE holds the
        # buffer; paused->REC discards this take and records anew; Save writes it.
        brow = tk.Frame(crec, bg=CARD)
        brow.pack(fill=tk.X)
        self.rec_btn = self._btn(brow, "REC", self._toggle_record, GREEN)
        self.rec_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self.save_btn = self._btn(brow, "Save", self._save, BLUE)
        self.save_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        self.collect_label = tk.Label(crec, text="idle", font=FONT_SMALL, bg=CARD,
                                      fg=MUTED, anchor="w")
        self.collect_label.pack(fill=tk.X, pady=(4, 0))

        # -- replay: pick a saved episode and re-simulate its trajectory --
        self._sep(self.session_card)
        rsec = tk.Frame(self.session_card, bg=CARD)
        rsec.pack(fill=tk.X)
        tk.Label(rsec, text="replay episode", font=FONT_BOLD, bg=CARD, fg=MUTED,
                 anchor="w").pack(fill=tk.X, pady=(0, 4))
        self.episode_var = tk.StringVar(value="")
        self.episode_menu = tk.OptionMenu(rsec, self.episode_var, "")
        self.episode_menu.config(font=FONT_LABEL, bg=CARD, fg=INK, relief="flat", bd=0,
                                 cursor="hand2", activebackground="#e6ebf1",
                                 highlightthickness=1, highlightbackground=TROUGH, anchor="w")
        self.episode_menu["menu"].config(font=FONT_LABEL, bg=CARD, fg=INK,
                                         activebackground=ACCENT, activeforeground="white")
        self.episode_menu.pack(fill=tk.X, pady=(0, 4))
        rbrow = tk.Frame(rsec, bg=CARD)
        rbrow.pack(fill=tk.X)
        self.replay_btn = self._btn(rbrow, "Replay", self._replay, ACCENT)
        self.replay_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self.stop_btn = self._btn(rbrow, "Stop", self._stop_replay, ORANGE)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        self.replay_label = tk.Label(rsec, text="", font=FONT_SMALL, bg=CARD,
                                     fg=MUTED, anchor="w", justify=tk.LEFT,
                                     wraplength=300)
        self.replay_label.pack(fill=tk.X, pady=(4, 0))
        # Delete the selected episode, then reindex the rest to 0..N-1.
        self.delete_btn = self._btn(rsec, "Delete episode", self._delete_episode, DANGER)
        self.delete_btn.pack(fill=tk.X, pady=(4, 0))

        # -- status (main column) --
        self.status = tk.Label(main_col, text="", justify=tk.LEFT, anchor="w",
                               font=FONT_MONO, bg=CARD, fg=INK, wraplength=400,
                               padx=12, pady=10)
        self.status.pack(fill=tk.X, padx=12, pady=(4, 12))
        self._style_mode_buttons()
        self._sync_collection_ui()
        # refresh the frame-count readout whenever the episode selection changes
        self.episode_var.trace_add("write", lambda *_: self._update_replay_ui())
        self._update_replay_ui()
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

    def _sep(self, parent):
        """A thin horizontal rule to group sub-sections inside a card."""
        tk.Frame(parent, bg=TROUGH, height=1).pack(fill=tk.X, pady=8)

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

    def _randomize(self):
        """Domain-randomize the movable objects (arm untouched)."""
        self.session.randomize_objects()

    # -- data collection -----------------------------------------------

    def _instruction(self):
        return self.instr_entry.get().strip() or "unspecified task"

    def _apply_task_instruction(self):
        """Pre-fill the instruction input with the current task's default
        instruction (from ``scene.tasks``). Called on startup and task switch."""
        self.instr_entry.delete(0, tk.END)
        self.instr_entry.insert(0, task_instruction(self.session.task_name))

    def _toggle_record(self):
        """REC/PAUSE toggle. idle->start recording; recording->pause; paused->
        discard this take and start a fresh one (Save keeps it instead)."""
        if self._replaying:
            return                      # not while a replay is running
        if self.collector is None:
            self.collector = Collector(self.session, self.collect_config)
        c = self.collector
        if not c.active:                 # idle -> start recording
            c.start_episode(self._instruction())
        elif c.recording:                # recording -> pause (keep the buffer)
            c.pause()
        else:                            # paused -> discard, record anew
            c.discard()
            c.start_episode(self._instruction())
        self._last_saved = None
        self._sync_collection_ui()

    def _save(self):
        """Write the current episode (recording or paused) as a success."""
        if self.collector is None or not self.collector.active:
            return
        path = self.collector.keep(success=True)
        self._last_saved = path.name if path is not None else None
        self._sync_collection_ui()

    def _collect_state(self):
        c = self.collector
        if c is None or not c.active:
            return "idle"
        return "rec" if c.recording else "paused"

    def _sync_collection_ui(self):
        """Repaint the toggle + Save + status label for the current state."""
        state = self._collect_state()
        if state == "rec":               # recording -> the button pauses
            self.rec_btn.config(text="PAUSE", bg=AMBER, activebackground=_darken(AMBER))
        else:                            # idle or paused -> the button (re)records
            self.rec_btn.config(text="REC", bg=GREEN, activebackground=_darken(GREEN))
        # rec is only disabled during replay (see _replay); this path is never
        # reached while replaying, so restore it here. Save is enabled ONLY while
        # paused -- you pause the take, then Save it (or REC again to redo).
        self.rec_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL if state == "paused" else tk.DISABLED)
        self._refresh_collect_label()
        self._update_episode_count()     # a Save just landed -> refresh the count
        self._last_collect_state = state

    def _update_episode_count(self):
        """Show how many episodes are saved on disk for the current task (live),
        and refresh the replay picker to match."""
        n = count_episodes(self.collect_config.root, self.session.task_name)
        self.episode_label.config(text=f"episodes collected: {n}")
        self._refresh_episode_list()

    def _refresh_episode_list(self):
        """Rebuild the replay dropdown from the saved episodes of the current
        task, keeping the selection valid (default: the latest episode)."""
        eps = list_episodes(self.collect_config.root, self.session.task_name)
        menu = self.episode_menu["menu"]
        menu.delete(0, "end")
        for name in eps:
            menu.add_command(label=name, command=tk._setit(self.episode_var, name))
        if self.episode_var.get() not in eps:
            self.episode_var.set(eps[-1] if eps else "")
        self._update_replay_ui()   # a new/removed episode changes Replay's enable

    # -- replay --------------------------------------------------------

    def _replay(self):
        """Load the selected episode and re-simulate it (overrides live control
        until it finishes or Stop is pressed)."""
        if self._replaying:
            return
        if self.collector is not None and self.collector.active:
            return                      # can't replay while recording
        name = self.episode_var.get()
        if not name:
            return
        ep_dir = self.collect_config.root / self.session.task_name / name
        if not self.player.load(ep_dir):
            return
        self.player.start()             # reconstructs the recorded initial scene
        self._replaying = True
        self._replay_t0 = time.perf_counter()   # wall anchor for tempo pacing
        if self.session.viewer is not None:
            self.session.viewer.user_scn.ngeom = 0   # clear the stale EE overlay
        self.rec_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self._sync_sliders_to_state()   # the arm jumped to the episode start
        self._update_replay_ui()

    def _stop_replay(self):
        """End replay and resume live control from the current (replayed) pose.
        Runs both on the Stop button and on auto-finish."""
        if not self._replaying:
            return
        self._replaying = False
        self.player.stop()
        self.session.recover()          # resync control targets -> no jump
        self._sync_sliders_to_state()
        self._sync_collection_ui()      # restore rec/save button states
        self._update_replay_ui()

    def _delete_episode(self):
        """Delete the selected episode (with confirmation) and reindex the rest
        so the numbering stays contiguous."""
        if self._replaying:
            return
        name = self.episode_var.get()
        if not name:
            return
        if not messagebox.askyesno("Delete episode",
                                   f"Delete {name} from '{self.session.task_name}'?\n"
                                   "This cannot be undone; remaining episodes are renumbered."):
            return
        delete_episode(self.collect_config.root, self.session.task_name, name)
        self.episode_var.set("")            # force _refresh to pick a valid one
        self._update_episode_count()        # refresh count + dropdown + replay UI

    def _replay_delay_ms(self):
        """Delay (ms) to the next replay frame so playback matches the recorded
        wall-clock tempo. Absolute-target scheduling self-corrects drift; falls
        back to the fixed tick when the episode has no wall_time."""
        elapsed = self.player.rec_elapsed
        i, n = self.player.progress          # i = index of the next frame to show
        if elapsed is None or i >= n:
            return TICK_MS
        target = self._replay_t0 + float(elapsed[i])
        return max(1, int((target - time.perf_counter()) * 1000))

    def _update_replay_ui(self):
        """Repaint the replay picker/buttons/progress, and show the selected
        episode's frame count before playback."""
        has_ep = bool(self.episode_var.get())
        self.replay_btn.config(
            state=tk.NORMAL if (has_ep and not self._replaying) else tk.DISABLED)
        self.delete_btn.config(
            state=tk.NORMAL if (has_ep and not self._replaying) else tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL if self._replaying else tk.DISABLED)
        self.episode_menu.config(state=tk.DISABLED if self._replaying else tk.NORMAL)
        if self._replaying:
            i, n = self.player.progress
            self.replay_label.config(text=f"{self.player.name}  {i}/{n}", fg=ACCENT)
        elif has_ep:
            meta = episode_meta(self.collect_config.root, self.session.task_name,
                                self.episode_var.get()) or {}
            lines = []
            if meta.get("num_frames") is not None:
                lines.append(f"{meta['num_frames']} frames")
            if meta.get("language_instruction"):
                lines.append(f'"{meta["language_instruction"]}"')
            self.replay_label.config(text="\n".join(lines), fg=MUTED)
        else:
            self.replay_label.config(text="", fg=MUTED)

    def _refresh_collect_label(self):
        state = self._collect_state()
        if state == "rec":
            self.collect_label.config(
                text=f"REC  {self.collector.recorder.num_frames} frames", fg="#e05a5a")
        elif state == "paused":
            self.collect_label.config(
                text=f"PAUSED  {self.collector.recorder.num_frames} frames", fg=AMBER)
        elif self._last_saved:
            self.collect_label.config(text=f"saved {self._last_saved}", fg=GREEN)
        else:
            self.collect_label.config(text="idle", fg=MUTED)

    # -- runtime settings ----------------------------------------------

    def _on_task(self, name):
        self.session.reload_task(name)
        self._sync_sliders_to_state()
        self._apply_task_instruction()  # pre-fill this task's default instruction
        self._update_episode_count()   # count is per-task -> refresh for the new one
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
        # Replay overrides live control: re-simulate the selected episode frame by
        # frame, syncing the viewer, until it ends or Stop is pressed.
        if self._replaying:
            if self.session.viewer is not None and not self.session.viewer.is_running():
                self.root.destroy()
                return
            if self.player.step():
                if self.session.viewer is not None:
                    self.session.viewer.sync()
                self._update_replay_ui()
                delay = self._replay_delay_ms()
            else:
                self._stop_replay()
                delay = TICK_MS
            self._update_status()
            self.root.after(delay, self._tick)
            return

        # VR face buttons (rising edge) drive the GUI's session actions hands-free
        # -- mirroring the B-button->HOME edge handled inline in session.step().
        # These are GUI concepts (the Collector / scene resets live here, not in
        # the session), so they are read here. No-op with no VR client.
        vr = self.session.state.snapshot()
        if vr.record and not self._prev_record_btn:   # A: REC/PAUSE toggle
            self._toggle_record()
        if vr.save and not self._prev_save_btn:        # X: save the take
            self._save()
        if vr.reset and not self._prev_reset_btn:      # Y: fresh randomized scene
            self._reset_all()
            self._randomize()
        self._prev_record_btn = vr.record
        self._prev_save_btn = vr.save
        self._prev_reset_btn = vr.reset

        moving = self.session.move_traj is not None
        if self.session.mode in ("joint", "task"):
            self.session.set_gripper_frac(self.gripper_slider.get())
            if not moving and not self._editing:
                self._push_targets()

        if not self.session.step():
            self.root.destroy()
            return

        # Record a frame after the tick's physics (handles task reload internally).
        if self.collector is not None:
            self.collector.on_tick(self.session)
            if self._collect_state() != self._last_collect_state:
                self._sync_collection_ui()       # e.g. a reload discarded the take
            else:
                self._refresh_collect_label()    # live frame counter

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
            msg += f"\nSAFETY TRIP: {s['trip']}\n   press Recover or HOME"
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
        # Closing the window (X button) or the MuJoCo viewer both end mainloop;
        # either way run the same shutdown so the process always terminates.
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.root.after(TICK_MS, self._tick)
        self.root.mainloop()
        self._shutdown()

    def _shutdown(self):
        """Close the viewer + VR server, then force-terminate. os._exit avoids a
        lingering MuJoCo / GL / render thread keeping the process alive after the
        UI is closed (a recurring "mujoco won't quit" symptom)."""
        try:
            if self.collector is not None:
                self.collector.close()
            self.session.close()
        finally:
            os._exit(0)


def main():
    # No CLI flags: task/scene, VR scale, VR smoothing, and the overlay toggle
    # are all set at runtime from the panel; the VR server binds its defaults.
    UnifiedGUI().run()


if __name__ == "__main__":
    main()
