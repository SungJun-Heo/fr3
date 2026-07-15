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
    * ``jpeg_quality``  -- per-frame JPEG quality (q95 by user's choice).

    The record rate (``fps`` / ``control_dt``) is NOT a knob here -- it is derived
    from the session's true per-tick sim-time advance (``substeps * timestep``)
    and written to ``meta.json`` by the ``Collector``, so it can never disagree
    with the data.
    """
    root: Path = Path("data/raw")
    cameras: tuple = ("front", "wrist")
    width: int = 640
    height: int = 480
    jpeg_quality: int = 95

    def __post_init__(self):
        self.root = Path(self.root)
        self.cameras = tuple(self.cameras)
