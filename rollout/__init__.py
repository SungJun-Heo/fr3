"""VLA rollout -- fr3 as the environment a VLA policy acts in.

fr3 is the *consumer* of a VLA policy: it produces observations (images + state
+ instruction) and consumes the policy's raw, single-step action (``SimEnv``).
The policy itself -- inference, action chunking, rate/reactivity, and network
transport -- lives in a separate inference project; here we only expose the env.
"""

from rollout.env import SimEnv

__all__ = ["SimEnv"]
