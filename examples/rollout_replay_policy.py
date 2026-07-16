"""Drive the VLA rollout env with a DUMMY policy (replay a recorded episode).

Proves fr3's VLA-consumer surface end-to-end without any trained model: a
stand-in "policy" emits a recorded episode's actions one at a time, and the
``SimEnv`` consumes each raw action and returns the next observation -- exactly
the loop a real inference project will run (just swap ``ReplayPolicy`` for the
real policy + its de-chunker). Pins the episode's initial scene so the pick
reproduces even if it was recorded with domain randomization.

Usage:
  python examples/rollout_replay_policy.py [--episode DIR] [--view]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from rollout import SimEnv


class ReplayPolicy:
    """Dummy policy: replays a recorded episode's actions, ignoring the obs.
    Action = ``[q_d(7), gripper_width_d/max_width]`` -- the env's joint format."""

    def __init__(self, episode_dir):
        ep = Path(episode_dir)
        self.meta = json.loads((ep / "meta.json").read_text())
        d = np.load(ep / "data.npz")
        maxw = float(self.meta["gripper"]["max_width"])
        self.actions = np.concatenate(
            [d["q_d"], (d["gripper_width_d"] / maxw)[:, None]], axis=1)  # (T, 8)
        self.q0, self.dq0 = d["q"][0], d["dq"][0]
        obj0 = np.asarray(self.meta.get("object_qpos0") or [], dtype=float)
        self.obj0 = obj0 if obj0.size else None
        self._i = 0

    def __call__(self, obs):
        a = self.actions[min(self._i, len(self.actions) - 1)]
        self._i += 1
        return a

    @property
    def done(self):
        return self._i >= len(self.actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="data/raw/pick_cube/episode_0000")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()
    np.set_printoptions(precision=3, suppress=True)

    policy = ReplayPolicy(args.episode)
    task = policy.meta["task"]
    env = SimEnv(task, instruction=policy.meta.get("language_instruction", ""),
                 view=args.view)

    obs = env.reset()
    # pin the recorded initial scene so this exact scenario reproduces
    env.robot.set_replay_state(policy.q0, policy.obj0, dq=policy.dq0)
    obs = env.observe()
    print(f"task={task!r}  instruction={env.instruction!r}")
    print(f"obs: images={ {c: v.shape for c, v in obs['images'].items()} } "
          f"state{obs['state'].shape}={obs['state']}")

    obj = env.robot.movable_object_names[0]
    while not policy.done:
        action = policy(obs)          # external "policy" -> one raw action
        obs = env.apply(action)       # fr3 consumes it, advances one tick

    z = float(env.robot.data.xpos[env.robot.model.body(obj).id][2])
    print(f"\nrollout done: {policy._i} steps | grasped={env.is_grasped()} | "
          f"{obj} z={z:.3f} -> {'PICKED' if z > 0.08 else 'not lifted'}")
    env.close()


if __name__ == "__main__":
    main()
