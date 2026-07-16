"""IR schema contract + recorder round-trip -- fully offline (no sim/GL).

Guards the "single home for the observation subset" refactor (observe() and the
recorder must agree on fields/dtype) and that a recorded episode reloads intact.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import (fake_state, fake_gripper, tiny_meta_inputs,
                           OBS_KEYS, ACTION_GT_KEYS)
from collection import schema
from collection.config import CollectionConfig
from collection.recorder import EpisodeRecorder, delete_episode, count_episodes


class TestObservationSingleHome(unittest.TestCase):
    def test_keys_and_dtype(self):
        obs = schema.observation_from_state(fake_state(), fake_gripper())
        self.assertEqual(set(obs), OBS_KEYS)
        self.assertEqual(obs["q"].dtype, np.float64)
        self.assertIsInstance(obs["gripper_width"], float)
        self.assertIsInstance(obs["gripper_is_grasped"], bool)

    def test_frame_composes_the_observation(self):
        st, g = fake_state(), fake_gripper(width=0.05, is_grasped=True)
        obs = schema.observation_from_state(st, g)
        frame = schema.frame_from_state(
            st, g, gripper_width_d=0.03, sim_time=1.0, wall_time=2.0,
            cam_extrinsics={"wrist": np.eye(4)}, object_qpos=np.zeros((0, 7)))
        # every observation field appears identically in the recorded frame
        for k in OBS_KEYS:
            np.testing.assert_array_equal(np.asarray(frame[k]), np.asarray(obs[k]))
        # the frame is a superset: action targets + ground truth + timing + cam
        self.assertTrue(ACTION_GT_KEYS <= set(frame))
        self.assertIn("cam_extrinsic_wrist", frame)
        self.assertEqual(frame["sim_time"], 1.0)
        self.assertEqual(frame["gripper_width_d"], 0.03)

    def test_conventions_pin_the_delta_contract(self):
        # the converter (derives deltas) and rollout (integrates them) read this
        self.assertIn("delta", schema.CONVENTIONS)
        self.assertIn("measured", schema.CONVENTIONS["delta"].lower())


class TestRecorderRoundTrip(unittest.TestCase):
    def _record(self, root, task="pick_cube", T=3):
        camera_specs, robot_meta, session_params = tiny_meta_inputs()
        rec = EpisodeRecorder(CollectionConfig(root=root, cameras=("front",)))
        rec.start(task, "grab it", camera_specs, robot_meta, session_params,
                  object_qpos0=np.zeros((0, 7)))
        for t in range(T):
            st = fake_state(q=np.full(7, float(t)))
            frame = schema.frame_from_state(
                st, fake_gripper(), gripper_width_d=0.02, sim_time=t * 0.02,
                wall_time=t * 0.02, cam_extrinsics={}, object_qpos=np.zeros((0, 7)))
            rec.add(frame, {"front": np.zeros((6, 8, 3), np.uint8)})
        return rec.stop(keep=True, success=True), T

    def test_write_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            out, T = self._record(d)
            self.assertIsNotNone(out)
            self.assertTrue((out / "READY.done").exists())
            self.assertFalse((out.parent / (out.name + ".tmp")).exists())  # tmp gone

            data = np.load(out / "data.npz")
            self.assertEqual(data["q"].shape, (T, 7))
            self.assertEqual(data["O_T_EE"].shape, (T, 16))
            np.testing.assert_array_equal(data["q"][1], np.full(7, 1.0))  # per-frame kept
            self.assertEqual(len(list((out / "images" / "front").glob("*.jpg"))), T)

            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(meta["num_frames"], T)
            self.assertEqual(meta["schema_version"], schema.SCHEMA_VERSION)
            self.assertTrue(meta["success"])
            self.assertIn("field_index", meta)
            self.assertEqual(meta["field_index"]["q"]["shape"], [T, 7])
            self.assertIn("delta", meta["conventions"])

    def test_discard_leaves_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            rec = EpisodeRecorder(CollectionConfig(root=d, cameras=("front",)))
            cs, rm, sp = tiny_meta_inputs()
            rec.start("pick_cube", "x", cs, rm, sp, object_qpos0=np.zeros((0, 7)))
            rec.add(schema.frame_from_state(
                fake_state(), fake_gripper(), 0.02, 0.0, 0.0, {}, np.zeros((0, 7))),
                {"front": np.zeros((6, 8, 3), np.uint8)})
            self.assertIsNone(rec.stop(keep=False))
            self.assertEqual(count_episodes(d, "pick_cube"), 0)

    def test_delete_reindexes_contiguously(self):
        with tempfile.TemporaryDirectory() as d:
            for _ in range(3):
                self._record(d, T=1)                       # episode_0000..0002
            self.assertEqual(count_episodes(d, "pick_cube"), 3)
            delete_episode(d, "pick_cube", "episode_0001")  # remove the middle
            self.assertEqual(count_episodes(d, "pick_cube"), 2)
            names = {p.name for p in (Path(d) / "pick_cube").iterdir()}
            self.assertEqual(names, {"episode_0000", "episode_0001"})  # 0002 -> 0001


if __name__ == "__main__":
    unittest.main()
