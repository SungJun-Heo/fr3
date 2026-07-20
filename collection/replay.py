"""EpisodePlayer -- replay a saved episode in a session's sim.

Two modes, chosen automatically from what the episode stored:

  * "kinematic" (EXACT, the default for schema >= 1.2): the episode recorded the
    full ground-truth state per frame -- arm ``q``, ``object_qpos``, gripper
    width -- so replay simply SETS that state each frame and runs forward
    kinematics. No physics is re-run, so there is ZERO drift: what you see is
    exactly what was collected. (This is what matters for reviewing data; the
    small re-simulation drift never existed here.)

  * "resim" (FALLBACK, older episodes that stored only the initial object
    layout): reconstruct the start scene and re-drive the recorded joint command
    ``q_d`` through joint control so objects move via contact. Faithful but not
    bit-exact (open-loop re-simulation accumulates a little error).

Either way the caller (the GUI tick loop) paces one frame per call and syncs the
viewer; playback speed follows the recorded ``wall_time`` (see ``rec_elapsed``).
The player borrows the session and reads ``session.robot``/``gripper`` fresh each
step, so a task reload does not stale it.
"""

import json
from pathlib import Path

import numpy as np

from robot import JointPositions


class EpisodePlayer:
    """Replay one recorded episode's trajectory in the session's sim."""

    def __init__(self, session):
        self.session = session
        self._reset()

    def _reset(self):
        self._mode = None
        # kinematic (recorded ground truth)
        self._q = None            # (T,7) actual arm trajectory
        self._obj = None          # (T,n_obj,7) actual object trajectory, or None
        self._gw = None           # (T,) measured gripper width
        # re-sim fallback inputs
        self._q_d = None
        self._ee = None
        self._ee_d = None
        self._gw_cmd = None
        self._q0 = None
        self._dq0 = None
        self._obj0 = None
        self._obj_ok = False
        # common
        self._task = None
        self._rec_elapsed = None
        self._sub = 1
        self._ac = None
        self._i = 0
        self._n = 0
        self.playing = False
        self.name = None

    def load(self, episode_dir):
        """Load an episode dir. Returns True if it has frames to replay."""
        episode_dir = Path(episode_dir)
        d = np.load(episode_dir / "data.npz")
        meta = json.loads((episode_dir / "meta.json").read_text())
        self._q = d["q"]
        self._gw = d["gripper_width"] if "gripper_width" in d.files else None
        self._obj = d["object_qpos"] if "object_qpos" in d.files else None
        # re-sim fallback inputs (older episodes without a per-frame object track)
        self._q_d = d["q_d"] if "q_d" in d.files else self._q
        self._ee = d["O_T_EE"] if "O_T_EE" in d.files else None
        self._ee_d = d["O_T_EE_d"] if "O_T_EE_d" in d.files else None
        self._gw_cmd = d["gripper_width_d"] if "gripper_width_d" in d.files else self._gw
        self._q0 = self._q[0]
        self._dq0 = d["dq"][0] if "dq" in d.files else np.zeros(self._q.shape[1])
        obj0 = np.asarray(meta.get("object_qpos0") or [], dtype=float)
        self._obj0 = obj0 if obj0.size else None
        self._task = meta.get("task")
        # per-frame recorded wall offset, so playback runs at the captured tempo
        wall = d["wall_time"] if "wall_time" in d.files else None
        self._rec_elapsed = (wall - wall[0]) if wall is not None else None
        # EXACT kinematic replay when the full object trajectory was recorded
        self._mode = "kinematic" if self._obj is not None else "resim"
        self._n = int(len(self._q))
        self.name = episode_dir.name
        return self._n > 0

    def start(self):
        """Prepare replay: match the task, and (re-sim only) reconstruct the
        initial scene and open a joint-control loop."""
        s = self.session
        if self._task and self._task != s.task_name:
            s.reload_task(self._task)
        self._obj_ok = (self._obj is not None and self._obj.ndim == 3
                        and self._obj.shape[1] == len(s.robot.movable_object_names))
        if self._mode == "resim":
            obj0 = (self._obj0 if self._obj0 is not None
                    and len(self._obj0) == len(s.robot.movable_object_names) else None)
            s.gripper.set_kinematic_width(float(self._gw_cmd[0]))
            s.robot.set_replay_state(self._q0, obj0, dq=self._dq0)
            s.robot.automatic_error_recovery()
            s.gripper.set_target_width(float(self._gw_cmd[0]))
            self._sub = max(1, s.substeps)
            self._ac = s.robot.start_joint_position_control()
        # kinematic mode needs no setup; the first step() displays frame 0
        self._i = 0
        self.playing = True

    def step(self):
        """Advance one recorded frame. Returns False when the episode ends (sync
        the viewer only if this returned True)."""
        if not self.playing or self._i >= self._n:
            self.playing = False
            return False
        if self._mode == "kinematic":
            self._step_kinematic()
        else:
            self._step_resim()
        self._i += 1
        return True

    def _step_kinematic(self):
        """Re-display the exact recorded ground-truth state for this frame."""
        s = self.session
        i = self._i
        if self._gw is not None:
            s.gripper.set_kinematic_width(float(self._gw[i]))
        obj_i = self._obj[i] if self._obj_ok else None
        s.robot.set_replay_state(self._q[i], obj_i)   # arm + objects qpos + forward

    def _step_resim(self):
        """Re-drive the recorded joint command over the frame's substeps,
        interpolating so the within-frame speed matches the recording."""
        s = self.session
        i = self._i
        cmd_prev = self._q_d[i - 1] if i > 0 else self._q_d[0]
        cmd_now = self._q_d[i]
        s.gripper.set_target_width(float(self._gw_cmd[i]))
        for k in range(self._sub):
            a = (k + 1) / self._sub
            self._ac.writeOnce(JointPositions(cmd_prev * (1.0 - a) + cmd_now * a))

    def stop(self):
        self.playing = False

    @property
    def progress(self):
        """``(current_frame, total_frames)`` for a UI readout."""
        return (self._i, self._n)

    @property
    def traces(self):
        """Measured state vs commanded action for the loaded episode.

        For plotting the two against each other -- the loader already holds
        them, so a viewer does not need to re-open the npz. Any entry may be
        None on an older episode that did not record it."""
        return dict(q=self._q, q_d=self._q_d,
                    gripper_width=self._gw, gripper_width_d=self._gw_cmd,
                    O_T_EE=self._ee, O_T_EE_d=self._ee_d)

    @property
    def rec_elapsed(self):
        """Per-frame recorded wall-clock offset from frame 0 (seconds), or None.
        Lets the caller replay at the tempo it was captured."""
        return self._rec_elapsed
