"""Generate a demonstration dataset with the scripted oracle -- no human, no GUI.

Each attempt gets a fresh randomized layout; the oracle solves it while every
tick is recorded; ground truth judges the outcome; only successes are written.
Failures and safety trips are discarded, so ATTEMPTS is the budget, not episodes.

Usage:
  python examples/generate_dataset.py --task pick_cube -n 100
  python examples/generate_dataset.py --task stack_blocks -n 200 --shove 10 --jitter 2
  python examples/generate_dataset.py --all -n 50 --root data/raw
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripted import available, generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=available(), help="task to generate for")
    p.add_argument("--all", action="store_true", help="every task with an oracle")
    p.add_argument("-n", "--attempts", type=int, default=50,
                   help="attempts (NOT episodes -- failures are discarded)")
    p.add_argument("--root", default="data/raw", help="dataset root")
    p.add_argument("--instruction", default=None,
                   help="language instruction (default: the task's own)")
    p.add_argument("--shove", type=float, default=0.0,
                   help="EE disturbance in mm (0 = off) -- yields recovery data")
    p.add_argument("--jitter", type=float, default=0.0,
                   help="joint disturbance in deg (0 = off) -- posture diversity")
    p.add_argument("--rate", type=float, default=1.0, help="disturbances per second")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view", action="store_true", help="watch it (much slower)")
    p.add_argument("--keep-failures", action="store_true",
                   help="also write failed episodes, flagged success=false")
    p.add_argument("-q", "--quiet", action="store_true", help="no per-episode lines")
    args = p.parse_args()

    if not args.task and not args.all:
        p.error("pass --task TASK or --all")
    tasks = available() if args.all else [args.task]

    summaries = []
    for task in tasks:
        print(f"\n=== {task}: {args.attempts} attempts ===")
        summaries.append(generate(
            task, attempts=args.attempts, root=args.root,
            instruction=args.instruction, shove_mm=args.shove,
            jitter_deg=args.jitter, rate_hz=args.rate, seed=args.seed,
            view=args.view, keep_failures=args.keep_failures,
            progress=not args.quiet))

    print("\n" + "=" * 64)
    print(f"{'task':<16}{'kept':>8}{'attempts':>10}{'yield':>8}{'time':>9}")
    print("-" * 64)
    for s in summaries:
        print(f"{s['task']:<16}{s['kept']:>8}{s['attempts']:>10}"
              f"{s['yield']:>7.0%}{s['elapsed_s']:>8.0f}s")
        if s["failures"]:
            for k, v in sorted(s["failures"].items(), key=lambda kv: -kv[1]):
                print(f"    discarded: {k} x{v}")
    print(f"\nwritten under {args.root}/<task>/episode_XXXX/")
    print(json.dumps({s["task"]: {"kept": s["kept"], "yield": round(s["yield"], 3)}
                      for s in summaries}))


if __name__ == "__main__":
    main()
