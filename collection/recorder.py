"""EpisodeRecorder -- persist one episode of the IR to disk.

The single home for IR persistence. It buffers the low-dimensional per-frame
arrays in RAM and streams the (large) camera JPEGs straight to disk, then on
``stop(keep=True)`` stacks the buffers into ``data.npz``, writes ``meta.json``,
drops a ``READY.done`` marker, and atomically renames the ``.tmp`` staging dir
to its final name. A discarded episode just removes the staging dir.

On-disk (one self-contained, rsync-friendly directory per episode):

    <root>/<task>/episode_XXXX/
        meta.json                     # schema.build_meta + field_index
        data.npz                      # stacked (T, ...) arrays, np.savez_compressed
        images/<cam>/000000.jpg ...   # JPEG q95, index == data.npz row
        READY.done                    # written last -> "fully flushed" marker

Episode numbers are gap-filled (smallest unused ``episode_N``) so manually
deleting a file frees its slot, mirroring camel-franka's recorder.
"""

import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from collection import schema

_EPISODE_RE = re.compile(r"episode_(\d+)$")


def _episode_numbers(task_dir):
    """Set of episode indices already saved under ``task_dir`` (final dirs only;
    in-progress ``episode_*.tmp`` dirs are excluded)."""
    task_dir = Path(task_dir)
    if not task_dir.exists():
        return set()
    return {int(m.group(1)) for entry in task_dir.iterdir()
            if entry.is_dir() and (m := _EPISODE_RE.match(entry.name))}


def count_episodes(root, task):
    """Number of saved episodes for ``task`` under ``root`` -- for a live counter."""
    return len(_episode_numbers(Path(root) / task))


def list_episodes(root, task):
    """Sorted saved-episode dir names for ``task`` (for a replay picker)."""
    return [f"episode_{n:04d}" for n in sorted(_episode_numbers(Path(root) / task))]


def delete_episode(root, task, name):
    """Delete episode ``name`` under ``root/task``, then reindex the remaining
    episodes to a contiguous ``0..N-1`` sequence (no gaps). Returns the new
    episode count. Renaming ascending is collision-free: each target slot is
    lower than its source and already vacated."""
    task_dir = Path(root) / task
    target = task_dir / name
    if target.exists():
        shutil.rmtree(target)
    nums = sorted(_episode_numbers(task_dir))
    for i, num in enumerate(nums):
        if num != i:
            (task_dir / f"episode_{num:04d}").rename(task_dir / f"episode_{i:04d}")
    return len(nums)


class EpisodeRecorder:
    """Buffer frames for one episode, then write it as an IR directory."""

    def __init__(self, config):
        self.config = config
        self._reset()

    def _reset(self):
        self._active = False
        self._frames = []
        self._task = None
        self._instruction = None
        self._camera_specs = None
        self._robot_meta = None
        self._session_params = None
        self._object_qpos0 = None
        self._tmp_dir = None
        self._final_dir = None

    @property
    def active(self):
        return self._active

    @property
    def num_frames(self):
        return len(self._frames)

    # -- lifecycle -----------------------------------------------------

    def start(self, task, instruction, camera_specs, robot_meta, session_params,
              object_qpos0):
        """Begin an episode: allocate the staging dir and per-camera image dirs.

        ``camera_specs`` / ``robot_meta`` / ``session_params`` / ``object_qpos0``
        (the movable objects' initial poses) are held until ``stop`` builds
        ``meta.json`` from them (so the header always matches the data that was
        actually recorded)."""
        if self._active:
            raise RuntimeError("an episode is already recording; stop it first")
        self._reset()
        self._active = True
        self._task = task
        self._instruction = instruction
        self._camera_specs = camera_specs
        self._robot_meta = robot_meta
        self._session_params = session_params
        self._object_qpos0 = object_qpos0

        task_dir = self.config.root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        n = self._next_episode_number(task_dir)
        self._final_dir = task_dir / f"episode_{n:04d}"
        self._tmp_dir = task_dir / f"episode_{n:04d}.tmp"
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir)   # stale crash artifact -> start clean
        for cam in camera_specs:
            (self._tmp_dir / "images" / cam).mkdir(parents=True, exist_ok=True)

    def add(self, frame, images):
        """Append one IR frame (from ``schema.frame_from_state``) and write its
        camera images. Image filename index == the frame's row in ``data.npz``."""
        if not self._active:
            raise RuntimeError("start() an episode before add()")
        idx = len(self._frames)
        for cam, rgb in images.items():
            path = self._tmp_dir / "images" / cam / f"{idx:06d}.jpg"
            Image.fromarray(rgb).save(path, format="JPEG",
                                      quality=self.config.jpeg_quality)
        self._frames.append(frame)

    def stop(self, keep=True, success=None):
        """Finish the episode. ``keep=True`` writes it and returns the final dir;
        ``keep=False`` (or an empty buffer) discards the staging dir and returns
        None. Resets the recorder either way."""
        if not self._active:
            return None
        try:
            if not keep or not self._frames:
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
                return None
            return self._write(success)
        finally:
            self._reset()

    # -- writing -------------------------------------------------------

    def _write(self, success):
        arrays = self._stack()
        np.savez_compressed(self._tmp_dir / "data.npz", **arrays)

        meta = schema.build_meta(
            task=self._task, instruction=self._instruction,
            num_frames=len(self._frames), success=success,
            keep=True, session_params=self._session_params,
            camera_specs=self._camera_specs, robot_meta_dict=self._robot_meta,
            object_qpos0=self._object_qpos0)
        meta["field_index"] = schema.field_index(arrays)
        with open(self._tmp_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # written LAST: the completion marker a converter/uploader keys on.
        (self._tmp_dir / "READY.done").write_text("")

        if self._final_dir.exists():
            shutil.rmtree(self._final_dir)
        self._tmp_dir.rename(self._final_dir)   # atomic on the same filesystem
        return self._final_dir

    def _stack(self):
        """Stack the per-frame dicts into ``{key: (T, ...) ndarray}``."""
        keys = self._frames[0].keys()
        return {k: np.asarray([f[k] for f in self._frames]) for k in keys}

    @staticmethod
    def _next_episode_number(task_dir):
        used = _episode_numbers(task_dir)
        n = 0
        while n in used:
            n += 1
        return n
