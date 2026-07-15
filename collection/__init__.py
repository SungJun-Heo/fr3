"""VLA-format-free demonstration data collection.

Records teleop/scripted episodes into a raw, lossless, self-describing
intermediate representation (IR) -- ``<root>/<task>/episode_XXXX/`` holding
``meta.json`` + ``data.npz`` + per-camera JPEGs. The IR is deliberately NOT any
one VLA training format; separate converters (deferred) turn it into LeRobot/
GR00T, pi0, RLDS, ACT-HDF5, etc. See ``collection/schema.py`` for the IR
contract and ``docs``/the plan for the rationale.

Primitives:
  * ``CollectionConfig``  -- where/how to record (paths, cameras, resolution).
  * ``SimCameraRenderer`` -- render named model cameras -> RGB + calibration.
  * ``EpisodeRecorder``   -- buffer frames -> write one episode dir.
  * ``Collector``         -- drive the recorder off a ``ControlSession`` tick.
"""

from collection.config import CollectionConfig
from collection.camera import SimCameraRenderer
from collection.recorder import (EpisodeRecorder, count_episodes,
                                 list_episodes, delete_episode)
from collection.collector import Collector
from collection.replay import EpisodePlayer

__all__ = ["CollectionConfig", "SimCameraRenderer", "EpisodeRecorder",
           "Collector", "EpisodePlayer", "count_episodes", "list_episodes",
           "delete_episode"]
