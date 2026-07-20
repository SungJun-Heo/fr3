"""Verify the scripted oracles: run each task N times on randomized layouts and
report the success rate.

This is the yield measurement that decides whether the oracles are good enough
to generate a dataset -- an oracle that solves 60% of random layouts means 40%
of the compute is wasted, and the failure reasons printed here say why.

Usage:
  python examples/scripted_demo.py                         # all tasks, 20 each
  python examples/scripted_demo.py --task pick_cube -n 50
  python examples/scripted_demo.py --task stack_blocks --view -n 3
  python examples/scripted_demo.py --perturb 0.01          # with recovery shoves
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from gui.session import ControlSession
from scripted import JointJitter, Shove, available, make_skill, run_skill

TICK_PERIOD = 0.02       # the control tick the oracle plans against (s)


def pacer(speed):
    """An ``on_tick`` that slows the loop to ``speed`` x real time.

    Without it the loop runs as fast as physics computes -- fine for measuring
    yield, far too fast to watch. This rides the same ``on_tick`` seam a
    ``Collector`` would use."""
    period = TICK_PERIOD / max(speed, 1e-6)
    last = [time.perf_counter()]

    def tick(_session):
        remaining = period - (time.perf_counter() - last[0])
        if remaining > 0:
            time.sleep(remaining)
        last[0] = time.perf_counter()
    return tick


def run_task(session, task, episodes, rng, perturb, jitter_deg, verbose,
             on_tick=None):
    """Run one task ``episodes`` times on fresh random layouts."""
    session.reload_task(task)
    session.set_mode("task")

    wins, reasons, shoves = 0, Counter(), 0
    for ep in range(1, episodes + 1):
        session.reset_all()
        session.robot.randomize_objects(rng)
        shove = Shove(displacement=perturb, rng=rng) if perturb > 0 else None
        jitter = (JointJitter(sigma=np.radians(jitter_deg), rng=rng)
                  if jitter_deg > 0 else None)

        res = run_skill(session, make_skill(task), shove=shove, jitter=jitter,
                        on_tick=on_tick)
        wins += res.success
        shoves += res.shoves
        if not res.success:
            reasons[_bucket(res.reason)] += 1
        if verbose:
            mark = "OK  " if res.success else "FAIL"
            note = "" if res.success else f"  <- {res.reason}"
            print(f"    ep {ep:3d}  {mark}  {res.ticks:4d} ticks{note}")
        if res.reason == "viewer closed":
            return wins, ep, reasons, shoves
    return wins, episodes, reasons, shoves


def _bucket(reason):
    """Group failure reasons so the summary stays readable."""
    if reason.startswith("safety trip"):
        return "safety trip"
    if "unreachable" in reason:
        return "unreachable grasp"
    if "timed out" in reason or "exceeded" in reason:
        return "timeout"
    if "too wide" in reason:
        return "object too wide"
    return reason or "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=available(), help="one task (default: all)")
    parser.add_argument("-n", "--episodes", type=int, default=20)
    parser.add_argument("--view", action="store_true", help="show the viewer")
    parser.add_argument("--perturb", type=float, default=0.0,
                        help="shove the arm this far off course, in metres "
                             "(0 = off); tests closed-loop recovery")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="joint-space disturbance, degrees per joint "
                             "(0 = off); reaches postures a shove cannot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="summary only, no per-episode lines")
    parser.add_argument("--speed", type=float, default=None,
                        help="playback speed vs real time (default: 1.0 with "
                             "--view, unpaced otherwise)")
    args = parser.parse_args()

    tasks = [args.task] if args.task else available()
    rng = np.random.default_rng(args.seed)
    speed = args.speed if args.speed is not None else (1.0 if args.view else None)
    on_tick = pacer(speed) if speed else None
    session = ControlSession(task=tasks[0], view=args.view)
    session.set_mode("task")

    summary = []
    try:
        for task in tasks:
            bits = "".join([f", shove={args.perturb*1000:.0f}mm" if args.perturb else "",
                            f", jitter={args.jitter:.2f}deg" if args.jitter else ""])
            print(f"\n{task}  ({args.episodes} episodes{bits})")
            wins, n, reasons, shoves = run_task(
                session, task, args.episodes, rng, args.perturb,
                args.jitter, not args.quiet, on_tick=on_tick)
            summary.append((task, wins, n, reasons, shoves))
    finally:
        session.close()

    print("\n" + "=" * 58)
    print(f"{'task':<16}{'success':>12}{'rate':>8}   failures")
    print("-" * 58)
    for task, wins, n, reasons, shoves in summary:
        detail = ", ".join(f"{k} x{v}" for k, v in reasons.most_common()) or "-"
        print(f"{task:<16}{wins:>7}/{n:<4}{wins / n if n else 0:>7.0%}   {detail}")
    if args.perturb or args.jitter:
        total = sum(s for *_, s in summary)
        print(f"\n{total} disturbances injected -- each one is a recovery "
              f"segment the dataset would otherwise lack.")


if __name__ == "__main__":
    main()
