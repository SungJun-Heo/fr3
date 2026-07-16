"""evaluate() harness logic -- offline, no sim. Injects a FakeEnv in place of
SimEnv so we test the loop/aggregation (success tally, per-episode results, the
on_episode callback, max_steps timeout) without running physics or a real policy.
"""

import unittest

import rollout.eval as ev


class FakeEnv:
    """Wins after ``win_at`` apply() calls; counts steps per episode."""
    win_at = 2

    def __init__(self, task, instruction="", view=False):
        self.task = task
        self.n_apply = 0

    def reset(self, randomize=False):
        self.n_apply = 0
        return {"step": 0}

    def apply(self, action):
        self.n_apply += 1
        return {"step": self.n_apply}

    def success(self):
        return self.n_apply >= FakeEnv.win_at

    def close(self):
        pass


def _noop_policy_factory():
    return lambda obs: {"joint": [0.0] * 8}


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self._orig = ev.SimEnv
        ev.SimEnv = FakeEnv

    def tearDown(self):
        ev.SimEnv = self._orig

    def test_counts_successes_and_callback(self):
        FakeEnv.win_at = 2
        seen = []
        res = ev.evaluate(
            make_policy=_noop_policy_factory, task="pick_cube", n_episodes=3,
            randomize=False, max_steps=10,
            on_episode=lambda i, won, steps: seen.append((i, won, steps)))
        self.assertEqual(res["task"], "pick_cube")
        self.assertEqual(res["n"], 3)
        self.assertEqual(res["successes"], 3)
        self.assertEqual(res["success_rate"], 1.0)
        self.assertEqual(res["per_episode"], [True, True, True])
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0], (0, True, 2))   # won on the 2nd step

    def test_timeout_is_failure(self):
        FakeEnv.win_at = 999
        res = ev.evaluate(
            make_policy=_noop_policy_factory, task="pick_cube", n_episodes=2,
            randomize=False, max_steps=5)
        self.assertEqual(res["successes"], 0)
        self.assertEqual(res["success_rate"], 0.0)
        self.assertEqual(res["per_episode"], [False, False])


if __name__ == "__main__":
    unittest.main()
