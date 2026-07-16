"""Rollout evaluation -- run a policy over N episodes, tally task success.

``evaluate`` treats the policy as a black box ``policy(obs) -> one raw action``
(the inference project hides its own de-chunking/reactivity behind that), resets
the env each episode (optionally domain-randomized), applies actions until the
task succeeds or ``max_steps``, and reports the success rate. ``make_policy`` is
a factory so each episode gets a fresh policy (cursor/history reset).
"""

from rollout.env import SimEnv


def evaluate(make_policy, task, n_episodes=20, randomize=True, max_steps=300,
             instruction=None, view=False, on_episode=None):
    """Return ``{task, n, successes, success_rate, per_episode}``.

    ``make_policy() -> policy``; ``policy(obs) -> action`` (raw, single step).
    ``on_episode(i, won, steps)`` is an optional per-episode callback."""
    env = SimEnv(task, instruction=(instruction or ""), view=view)
    results = []
    try:
        for i in range(n_episodes):
            obs = env.reset(randomize=randomize)
            policy = make_policy()
            won, steps = False, 0
            for steps in range(1, max_steps + 1):
                obs = env.apply(policy(obs))
                if env.success():
                    won = True
                    break
            results.append(won)
            if on_episode is not None:
                on_episode(i, won, steps)
    finally:
        env.close()
    n = len(results)
    return {
        "task": task,
        "n": n,
        "successes": int(sum(results)),
        "success_rate": (sum(results) / n) if n else 0.0,
        "per_episode": results,
    }
