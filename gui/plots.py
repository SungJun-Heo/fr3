"""State-vs-action traces for a replayed episode, drawn on a Canvas.

Deliberately no matplotlib: fr3 installs with ``pip install mujoco numpy`` plus
stdlib tkinter, and a plotting window is not worth breaking that. The drawing
here is a few polylines, which Canvas does fine.

Each panel overlays a MEASURED signal (solid) on the COMMANDED one (dashed).
Reading the gap between them is the point -- it is the tracking error, and the
recorded action is the dashed line, not the solid one. A dataset whose action
was mistakenly taken from the state would show the two lying exactly on top of
each other, which is the fastest visual check that the IR is right.

The widget knows nothing about robots: a caller hands it named views of
``(label, measured, commanded, unit)`` tracks and it draws whichever is
selected. ``gui/app.py`` builds the joint-space and task-space views, since it
already owns the pose conventions.
"""

import tkinter as tk

import numpy as np

BG = "#ffffff"
CARD = "#ffffff"
INK = "#232a34"
MUTED = "#8a93a0"
GRID = "#e6eaf0"
IDLE = "#d4dbe4"
ACCENT = "#3b7dd8"
STATE = "#3b7dd8"      # measured
ACTION = "#e0703b"     # commanded -- the recorded action
HEAD = "#d9534f"

PAD_L, PAD_R, PAD_T, PAD_B = 74, 16, 16, 26
ROW_GAP = 6


class TracePlotWindow:
    """A resizable window of stacked traces with a replay playhead.

    ``views`` maps a name ("joint", "task") to a list of
    ``(label, measured, commanded, unit)``; ``commanded`` may be None."""

    def __init__(self, parent, on_close=None):
        self.top = tk.Toplevel(parent)
        self.top.title("episode traces")
        self.top.configure(bg=BG)
        self.top.geometry("820x900")

        bar = tk.Frame(self.top, bg=BG)
        bar.pack(fill=tk.X, padx=10, pady=(8, 2))
        self._buttons = {}
        for name, text in (("joint", "JOINT"), ("task", "TASK")):
            b = tk.Button(bar, text=text, font=("sans", 11, "bold"), width=8,
                          relief="flat", bd=0, cursor="hand2", highlightthickness=0,
                          command=lambda n=name: self.set_view(n))
            b.pack(side=tk.LEFT, padx=(0, 6))
            self._buttons[name] = b
        self.header = tk.Label(bar, text="", font=("sans", 11), bg=BG, fg=INK)
        self.header.pack(side=tk.LEFT, padx=(10, 0))
        legend = tk.Frame(bar, bg=BG)
        legend.pack(side=tk.RIGHT)
        tk.Label(legend, text="— measured", font=("sans", 10), bg=BG,
                 fg=STATE).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(legend, text="┄ commanded (action)", font=("sans", 10),
                 bg=BG, fg=ACTION).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.top, bg=BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self._on_close = on_close
        self.top.protocol("WM_DELETE_WINDOW", self.close)

        self._views = {}
        self._view = "joint"
        self._n = 0
        self._frame = 0
        self._head_ids = []
        self._title = ""

    # -- content --------------------------------------------------------

    def show(self, views, frames, title=""):
        """Load an episode's views. ``frames`` is the recorded frame count."""
        self._views = {k: v for k, v in views.items() if v}
        self._n = int(frames)
        self._frame = 0
        self._title = title
        if self._view not in self._views and self._views:
            self._view = next(iter(self._views))
        self._style_buttons()
        self._redraw()

    def set_view(self, name):
        if name not in self._views or name == self._view:
            return
        self._view = name
        self._style_buttons()
        self._redraw()

    def set_frame(self, i):
        """Move the playhead. Cheap -- only the head lines are touched."""
        if not self._views or not self._n:
            return
        self._frame = max(0, min(int(i), self._n - 1))
        x = self._x_of(self._frame)
        for item, y0, y1 in self._head_ids:
            self.canvas.coords(item, x, y0, x, y1)
        self.header.config(
            text=f"{self._title}    frame {self._frame + 1}/{self._n}")

    def close(self):
        if self.top is not None:
            self.top.destroy()
            self.top = None
        if self._on_close is not None:
            self._on_close()

    @property
    def alive(self):
        return self.top is not None

    # -- drawing --------------------------------------------------------

    def _style_buttons(self):
        for name, btn in self._buttons.items():
            on = (name == self._view)
            has = name in self._views
            btn.config(bg=ACCENT if on else IDLE,
                       fg="white" if on else (INK if has else MUTED),
                       state=tk.NORMAL if has else tk.DISABLED)

    def _x_of(self, i):
        w = max(self.canvas.winfo_width(), 200)
        return PAD_L + (w - PAD_L - PAD_R) * (i / max(self._n - 1, 1))

    def _redraw(self):
        if self.top is None:
            return
        c = self.canvas
        c.delete("all")
        self._head_ids = []
        tracks = self._views.get(self._view, [])
        if not tracks:
            return

        w = max(c.winfo_width(), 200)
        h = max(c.winfo_height(), 200)
        rows = len(tracks)
        avail = h - PAD_T - PAD_B - ROW_GAP * (rows - 1)
        row_h = max(avail / rows, 18)

        for r, (label, measured, commanded, unit) in enumerate(tracks):
            y0 = PAD_T + r * (row_h + ROW_GAP)
            y1 = y0 + row_h
            lo, hi = self._range(measured, commanded)
            c.create_rectangle(PAD_L, y0, w - PAD_R, y1, outline=GRID)
            c.create_text(PAD_L - 10, (y0 + y1) / 2 - 6, anchor="e", fill=INK,
                          font=("sans", 10, "bold"), text=label)
            c.create_text(PAD_L - 10, (y0 + y1) / 2 + 8, anchor="e", fill=MUTED,
                          font=("sans", 8), text=unit)
            c.create_text(PAD_L - 4, y0 + 7, anchor="e", fill=MUTED,
                          font=("sans", 8), text=f"{hi:.1f}")
            c.create_text(PAD_L - 4, y1 - 7, anchor="e", fill=MUTED,
                          font=("sans", 8), text=f"{lo:.1f}")
            if commanded is not None:
                self._polyline(commanded, y0, y1, lo, hi, ACTION, dash=(4, 3))
            self._polyline(measured, y0, y1, lo, hi, STATE)
            self._head_ids.append(
                (c.create_line(PAD_L, y0, PAD_L, y1, fill=HEAD, width=1), y0, y1))

        c.create_text(w - PAD_R, h - 9, anchor="e", fill=MUTED,
                      font=("sans", 9), text="x = recorded frame")
        self.set_frame(self._frame)

    @staticmethod
    def _range(measured, commanded):
        vals = [measured] if commanded is None else [measured, commanded]
        lo = float(min(np.min(v) for v in vals))
        hi = float(max(np.max(v) for v in vals))
        if hi - lo < 1e-6:            # a signal that never moved still needs a band
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    def _polyline(self, values, y0, y1, lo, hi, color, dash=None):
        w = max(self.canvas.winfo_width(), 200)
        n = len(values)
        if n < 2:
            return
        x_span = w - PAD_L - PAD_R
        scale = (y1 - y0) / (hi - lo)
        pts = []
        for i, v in enumerate(values):
            pts.append(PAD_L + x_span * (i / (n - 1)))
            pts.append(y1 - (float(v) - lo) * scale)
        self.canvas.create_line(*pts, fill=color, width=2,
                                **({"dash": dash} if dash else {}))
