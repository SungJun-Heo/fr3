"""Evaluate a policy over N rollout episodes and print the success rate.

Uses the DUMMY replay-policy (replays a recorded episode's actions) as a
stand-in for a real VLA -- so run it with ``--no-randomize`` (the replayed
actions only fit the scene they were recorded in). A real reactive policy from
the inference project would be evaluated WITH randomization to measure
generalization; the harness is identical, only ``make_policy`` changes.

Usage:
  python examples/rollout_eval.py [--episode DIR] [--n N] [--no-randomize]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rollout import evaluate
from examples.rollout_replay_policy import ReplayPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="data/raw/pick_cube/episode_0000")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--no-randomize", action="store_true",
                    help="fixed scene (needed for the replay-policy stand-in)")
    args = ap.parse_args()

    probe = ReplayPolicy(args.episode)
    task = probe.meta["task"]
    instr = probe.meta.get("language_instruction", "")

    res = evaluate(
        make_policy=lambda: ReplayPolicy(args.episode),   # fresh cursor per episode
        task=task, instruction=instr,
        n_episodes=args.n, randomize=not args.no_randomize,
        max_steps=len(probe.actions) + 40,
        on_episode=lambda i, won, steps: print(f"  ep{i}: {'OK ' if won else 'FAIL'} ({steps} steps)"),
    )
    print(f"\n[{task}] success {res['successes']}/{res['n']} "
          f"= {res['success_rate']*100:.0f}%  (randomize={not args.no_randomize})")


if __name__ == "__main__":
    main()
