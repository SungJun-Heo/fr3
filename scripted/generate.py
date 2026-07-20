"""Autonomous dataset generation: randomize -> solve -> judge -> keep or discard.

This is the loop the whole ``scripted`` package exists for. Each episode gets a
fresh randomized layout, the oracle solves it while the ``Collector`` records
every tick, ``rollout.success`` judges the outcome from ground truth, and only
successes are written.

Two rules make the output trustworthy, and both are about what is THROWN AWAY:

  * **Discard failures.** A demonstration of not accomplishing the task teaches
    a policy to not accomplish the task. The oracle does not succeed on every
    layout, so a run of N attempts yields fewer than N episodes -- budget
    attempts, not episodes.
  * **Discard safety trips.** A collision reflex or singularity trip means the
    arm did something the real robot would fault on. Those frames are worse
    than useless: they are confidently wrong.

Every episode records HOW it was made (``source`` in ``meta.json``): the oracle,
the seed that drew its layout, and the disturbance settings. Without that a
mixed dataset cannot be split by origin later, and questions like "did the
disturbed episodes help?" can only be answered by regenerating everything.

Seeds are drawn from one root RNG and stored per episode, so any single episode
can be reproduced exactly -- which is how you debug the one that went wrong out
of a thousand.

``DatasetGenerator`` advances one control tick at a time so a GUI can drive it
from its own tick loop; ``generate`` is that class plus a while loop, so the
headless CLI and the GUI button cannot drift apart.

Usage:
    from scripted import generate
    summary = generate("pick_cube", attempts=100, shove_mm=10, jitter_deg=2)
"""

import time
from collections import Counter

import numpy as np

from collection import CollectionConfig, Collector
from scene.tasks import task_instruction
from scripted.base import CONTROL_DT
from scripted.run import JointJitter, Shove, SkillRunner
from scripted.skills import make_skill


def _classify(reason):
    """Group a failure reason into the bucket that suggests what to do about it."""
    if reason.startswith("safety trip"):
        return "safety trip"
    if "posture" in reason or "unreachable" in reason:
        return "rejected before moving"
    if "too wide" in reason:
        return "object too wide"
    if "timed out" in reason or "exceeded" in reason:
        return "timeout"
    if "criterion" in reason:
        return "finished but criterion not met"
    return reason or "unknown"


def _source(task, seed, randomize, shove_mm, jitter_deg, rate_hz):
    """The ``source`` block stored in meta.json -- what made this episode."""
    disturbance = None
    if shove_mm > 0 or jitter_deg > 0:
        disturbance = dict(shove_mm=float(shove_mm),
                           jitter_deg=float(jitter_deg),
                           rate_hz=float(rate_hz))
    return dict(kind="scripted_oracle", oracle=task, seed=int(seed),
                randomized=bool(randomize), disturbance=disturbance)


class DatasetGenerator:
    """Run N oracle attempts against ``session``, keeping the successes.

    Stepwise (``tick``) rather than a loop, because a GUI owns its own tick loop
    and cannot block inside one -- the same split as ``SkillRunner``/``run_skill``.
    The session is BORROWED: the caller opened it and the caller closes it.
    ``close`` releases the generator's own recorder/renderer."""

    def __init__(self, session, task, attempts=50, root="data/raw",
                 instruction=None, randomize=True, shove_mm=0.0, jitter_deg=0.0,
                 rate_hz=1.0, seed=0, keep_failures=False, on_episode=None):
        session.reload_task(task)
        session.set_mode("task")
        self.session = session
        self.task = task
        self.attempts = int(attempts)
        self.randomize = randomize
        self.shove_mm = float(shove_mm)
        self.jitter_deg = float(jitter_deg)
        self.rate_hz = float(rate_hz)
        self.keep_failures = keep_failures
        self.on_episode = on_episode
        self.root = str(root)

        self.collector = Collector(session, CollectionConfig(root=root))
        self.text = instruction if instruction is not None else task_instruction(task)
        self._rng = np.random.default_rng(seed)
        self._per_tick = min(self.rate_hz * CONTROL_DT, 1.0)

        self.attempt = 0          # episodes started
        self.kept = 0
        self.reasons = Counter()
        self.paths = []
        self.summary = None
        self._runner = None
        self._t0 = time.perf_counter()

    # -- status ---------------------------------------------------------

    @property
    def active(self):
        return self.summary is None

    @property
    def phase(self):
        return self._runner.phase if self._runner is not None else "-"

    @property
    def rate(self):
        return self.kept / self.attempt if self.attempt else 0.0

    # -- driving --------------------------------------------------------

    def tick(self):
        """Advance one control tick. Returns the summary once finished, else None."""
        if self.summary is not None:
            return self.summary
        if self._runner is None:
            if self.attempt >= self.attempts:
                return self._finish()
            self._begin_episode()

        result = self._runner.tick()
        self.collector.on_tick(self.session)      # after the tick's physics
        if result is None:
            return None

        self._end_episode(result)
        if result.reason == "viewer closed":
            return self._finish()
        return None

    def stop(self):
        """Abort early, dropping the episode in flight (it is incomplete)."""
        if self._runner is not None:
            self.collector.discard()
            self._runner = None
        return self._finish()

    def close(self):
        self.collector.close()

    # -- internals ------------------------------------------------------

    def _begin_episode(self):
        # One seed per episode, drawn from the root and STORED: any single
        # episode out of a thousand can then be reproduced exactly.
        ep_seed = int(self._rng.integers(0, 2**31 - 1))
        ep_rng = np.random.default_rng(ep_seed)

        self.session.reset_all()
        if self.randomize:
            self.session.robot.randomize_objects(ep_rng)

        # Fresh disturbances per episode -- they carry pulse state.
        shove = (Shove(displacement=self.shove_mm / 1000.0, prob=self._per_tick,
                       rng=ep_rng) if self.shove_mm > 0 else None)
        jitter = (JointJitter(sigma=np.radians(self.jitter_deg),
                              prob=self._per_tick, rng=ep_rng)
                  if self.jitter_deg > 0 else None)

        self.collector.start_episode(self.text, source=_source(
            self.task, ep_seed, self.randomize, self.shove_mm, self.jitter_deg,
            self.rate_hz))
        self._runner = SkillRunner(self.session, make_skill(self.task),
                                   shove=shove, jitter=jitter)
        self.attempt += 1

    def _end_episode(self, result):
        if result.success or self.keep_failures:
            self.paths.append(self.collector.keep(success=result.success))
            self.kept += 1
        else:
            self.collector.discard()
        if not result.success:
            self.reasons[_classify(result.reason)] += 1
        self._runner = None
        if self.on_episode is not None:
            self.on_episode(self.attempt, result)

    def _finish(self):
        self.summary = {
            "task": self.task,
            "attempts": self.attempt,
            "kept": self.kept,
            "yield": self.rate,
            "failures": dict(self.reasons),
            "elapsed_s": round(time.perf_counter() - self._t0, 1),
            "root": self.root,
            "paths": [str(p) for p in self.paths if p is not None],
        }
        return self.summary


def generate(task, attempts=50, root="data/raw", instruction=None,
             randomize=True, shove_mm=0.0, jitter_deg=0.0, rate_hz=1.0,
             seed=0, view=False, keep_failures=False, session=None,
             on_episode=None, progress=True):
    """Run ``attempts`` oracle episodes, keeping the successes. Returns a summary.

    ``shove_mm`` / ``jitter_deg`` add the two disturbance channels (0 = off);
    ``rate_hz`` is how often either fires. ``keep_failures`` writes the failures
    too, flagged ``success=False`` -- off by default, and only useful if you
    intend to train something that needs negatives.

    ``session`` lets a caller lend its own live session instead of opening a
    second viewer; a lent session is not closed on the way out."""
    own_session = session is None
    if own_session:
        # Imported here, not at module scope: gui/__init__ pulls in the whole
        # app, which imports scripted -- a cycle. Only opening our own session
        # needs it, and a caller that lends one never takes this branch.
        from gui.session import ControlSession
        session = ControlSession(task=task, view=view)

    def report(i, result):
        if progress:
            mark = "keep" if result.success else "drop"
            note = "" if result.success else f"  <- {result.reason}"
            print(f"  [{i:4d}/{attempts}] {mark}  {result.ticks:4d} ticks{note}")
        if on_episode is not None:
            on_episode(i, result)

    gen = DatasetGenerator(
        session, task, attempts=attempts, root=root, instruction=instruction,
        randomize=randomize, shove_mm=shove_mm, jitter_deg=jitter_deg,
        rate_hz=rate_hz, seed=seed, keep_failures=keep_failures,
        on_episode=report)
    try:
        while True:
            summary = gen.tick()
            if summary is not None:
                return summary
    finally:
        gen.close()
        if own_session:
            session.close()
