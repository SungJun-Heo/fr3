"""Scripted oracles -- hand-written policies that solve tasks from privileged
sim state, to mass-produce demonstrations without a human or a trained model.

    from scripted import make_skill, run_skill, Shove, JointJitter

    skill = make_skill("pick_cube")
    result = run_skill(session, skill,
                       shove=Shove(displacement=0.01),   # push the EE off course
                       jitter=JointJitter(sigma=0.02))   # ...and off posture

See ``base.py`` for why the oracles are closed-loop, and ``skills.py`` for how
to add a task.
"""

from scripted.base import (Check, Ctx, Hold, Plan, Servo, Skill, SkillAborted,
                           Transit, grasp_R)
from scripted.generate import DatasetGenerator, generate
from scripted.macros import drop_in, pick, place_on
from scripted.run import (JointJitter, Result, Shove, SkillRunner, run_skill,
                          success_view, ticks_for)
from scripted.skills import SKILLS, available, make_skill

__all__ = [
    "Check", "Ctx", "Hold", "Plan", "Servo", "Skill", "SkillAborted",
    "Transit", "grasp_R",
    "DatasetGenerator", "drop_in", "generate", "pick", "place_on",
    "JointJitter", "Result", "Shove", "SkillRunner", "run_skill",
    "success_view", "ticks_for",
    "SKILLS", "available", "make_skill",
]
