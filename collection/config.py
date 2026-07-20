"""Collection settings -- one home for the data-collection knobs.

A ``CollectionConfig`` is passed to the ``Collector`` (and, through it, to the
camera renderer and episode recorder). Keeping every tunable here means no
module reaches for a hard-coded path / resolution / quality of its own.

The raw store is VLA-format-free (see ``collection/schema.py``): episodes land
under ``root/<task>/episode_XXXX/`` as ``meta.json`` + ``data.npz`` + per-camera
JPEGs, ready to be converted to LeRobot/GR00T/pi0 (or rsync'd to a train server)
later. ``root`` can point anywhere -- e.g. a mounted/external drive -- since the
local disk is limited.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CollectionConfig:
    """Where and how episodes are recorded.

    * ``root``          -- dataset root; a ``<task>/`` subdir is created under it.
    * ``cameras``       -- model camera names to render each frame (order kept).
    * ``width/height``  -- render size (<= the model's 640x480 offscreen buffer).
    * ``jpeg_quality``  -- per-frame JPEG quality.
    * ``record_every``  -- keep every Nth control tick (1 = every tick).

    These four decide the dataset's size almost entirely: the images are ~99% of
    an episode on disk (a 322-frame pick is 30 MB of JPEG against 180 kB of
    numbers), so resolution, quality, camera count and record rate ARE the
    storage budget. They must be chosen before generating in bulk -- changing
    them later means regenerating.

    ``record_every`` drops whole frames, never images alone: the IR's invariant
    is one JPEG per npz row per camera, and half-recorded frames would break
    every reader. The effective rate is recomputed from it and written to
    ``meta.json``, so ``fps`` still cannot disagree with the data.

    The defaults are 256x256 q85 at the full control rate. Resolution and
    quality are cheap to give up -- policies train at ~224-256 px, so 640x480
    was mostly storing pixels that get resized away again (measured: 28.0 ->
    6.3 MB per episode, 1000 episodes 28 GB -> 6 GB). ``record_every`` stays 1
    because it is the one-way door: 50 Hz can be subsampled later, 10 Hz cannot
    be un-thrown-away, so paying for it up front is the safe default.

    Every episode records the size it was rendered at (``cameras[*].width`` in
    meta.json), so a set mixing old 640x480 episodes with new 256x256 ones is at
    least self-describing -- but a converter has to handle it, so prefer
    regenerating over mixing.
    """
    root: Path = Path("data/raw")
    cameras: tuple = ("front", "wrist")
    width: int = 256
    height: int = 256
    jpeg_quality: int = 85
    record_every: int = 1

    def __post_init__(self):
        self.root = Path(self.root)
        self.cameras = tuple(self.cameras)
        self.record_every = max(1, int(self.record_every))

    def bytes_per_frame(self, jpeg_bytes_per_px=0.20):
        """Rough on-disk bytes per recorded frame, for a size estimate.

        The constant is empirical (640x480 q95 renders of these scenes come out
        near 62 kB), so treat the result as an order-of-magnitude guide."""
        return int(self.width * self.height * jpeg_bytes_per_px * len(self.cameras))
