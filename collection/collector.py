"""Collector -- drive the episode recorder off a ControlSession tick.

Composes the two primitives (``SimCameraRenderer`` + ``EpisodeRecorder``) and
plugs into the session's existing per-tick seam. It BORROWS the session (never
owns or mutates its control state), so the control core (``gui/session.py``)
stays a single-responsibility unit -- recording is bolted on from outside:

    collector = Collector(session, CollectionConfig())
    collector.start_episode("pick up the red cube")
    session.run(max_ticks=N, on_tick=lambda s: (drive(s), collector.on_tick(s)))
    collector.keep(success=True)     # or collector.discard()

In the GUI it is the same: buttons call ``start_episode``/``keep``/``discard``
and one line after ``session.step()`` calls ``collector.on_tick(session)``.

Each recorded frame is sampled AFTER the tick's physics, straight from the
public read APIs (``robot.read_once`` -> observation + truthful action,
``gripper.read_once``/``target_width``, ``renderer.render``), so the Collector
never reaches into private control state.
"""

from collection.camera import SimCameraRenderer
from collection.recorder import EpisodeRecorder
from collection import schema


class Collector:
    """Record episodes from a running ``ControlSession``."""

    def __init__(self, session, config):
        self.session = session
        self.config = config
        self._model = session.model     # to detect a task reload (model swap)
        self.renderer = SimCameraRenderer(session.model, session.data,
                                          config.cameras, config.width,
                                          config.height)
        self.recorder = EpisodeRecorder(config)
        # Which cameras move (their world pose is logged per frame; static
        # cameras' pose lives once in meta.json). Cached from specs.
        self._specs = self.renderer.specs()
        self._moving = [n for n in config.cameras if self._specs[n]["moving"]]
        # Paused = an episode buffer exists but on_tick stops appending (so the
        # operator can review it, then Save it or redo it).
        self._paused = False

    @property
    def active(self):
        """An episode buffer exists (recording OR paused)."""
        return self.recorder.active

    @property
    def paused(self):
        return self.recorder.active and self._paused

    @property
    def recording(self):
        """Actively appending frames (an episode is open and not paused)."""
        return self.recorder.active and not self._paused

    # -- episode control -----------------------------------------------

    def start_episode(self, instruction, source=None):
        """Begin recording. ``instruction`` is the language annotation both
        GR00T and pi0 consume as the task/prompt. Captures the movable objects'
        initial layout now (``object_qpos0``) so replay can reconstruct the scene."""
        rm = schema.robot_meta(self.session.robot, self.session.gripper)
        object_qpos0 = self.session.robot.object_qpos()
        self.recorder.start(self.session.task_name, instruction, self._specs,
                            rm, self._session_params(), object_qpos0, source)
        self._paused = False

    def pause(self):
        """Stop appending frames but keep the episode buffer. No-op if idle."""
        if self.recorder.active:
            self._paused = True

    def keep(self, success=True):
        """Finish and write the current episode. Returns its dir (or None)."""
        path = self.recorder.stop(keep=True, success=success)
        self._paused = False
        return path

    def discard(self):
        """Abandon the current episode (delete the staging dir)."""
        self.recorder.stop(keep=False)
        self._paused = False

    # -- per-tick sampling ---------------------------------------------

    def on_tick(self, session):
        """Sample one frame if recording. Call right after ``session.step()``.

        Guards a task reload: ``reload_task`` rebuilds the model/data, which
        stales the renderer -- rebind it, and drop any in-progress episode
        (an episode must not span a scene change)."""
        if session.model is not self._model:
            if self.recorder.active:
                print("[collect] task reloaded mid-episode -> discarding it")
                self.recorder.stop(keep=False)
                self._paused = False
            self.renderer.rebind(session.model, session.data)
            self._model = session.model
            self._specs = self.renderer.specs()
            self._moving = [n for n in self.config.cameras
                            if self._specs[n]["moving"]]
            return

        if not self.recording:      # idle or paused -> do not append a frame
            return

        state = session.robot.read_once()
        gripper_state = session.gripper.read_once()
        gripper_width_d = session.gripper.target_width()
        images = self.renderer.render()
        ext = self.renderer.extrinsics_world()
        moving_ext = {n: ext[n] for n in self._moving}
        frame = schema.frame_from_state(
            state, gripper_state, gripper_width_d,
            sim_time=float(session.data.time), wall_time=_wall_time(),
            cam_extrinsics=moving_ext, object_qpos=session.robot.object_qpos())
        self.recorder.add(frame, images)

    # -- teardown ------------------------------------------------------

    def close(self):
        if self.recorder.active:
            self.recorder.stop(keep=False)
        self.renderer.close()

    # -- helpers -------------------------------------------------------

    def _session_params(self):
        """Truthful record timing derived from the session: sim-time advanced
        per recorded tick is ``substeps * timestep`` (one frame per tick)."""
        dt = self.session.model.opt.timestep
        substeps = self.session.substeps
        control_dt = substeps * dt
        return dict(fps=1.0 / control_dt, control_dt=control_dt,
                    sim_timestep=dt, substeps=substeps)


def _wall_time():
    import time
    return time.time()
