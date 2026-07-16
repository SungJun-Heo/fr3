"""SimEnv -- fr3 as a VLA-policy consumer (observation producer + action sink).

fr3's role in the VLA loop: it PRODUCES observations (camera images +
proprioceptive state + language instruction) and CONSUMES the policy's **raw,
single-step** action, applying it through the existing controllers. Everything
policy-side -- inference, action chunking, rate handling, network transport --
lives in a SEPARATE project; this env only does ``observe()`` and
``apply(one raw action)``. A rollout loop is then just::

    env = SimEnv("pick_cube", instruction="pick up the red cube")
    obs = env.reset()
    while not done:
        action = policy(obs)      # external project: infer + de-chunk -> one action
        obs = env.apply(action)   # fr3: decode -> controller -> step -> next obs

Conventions match the collected data (so a policy trained on our episodes plugs
in unchanged): state/action are joint space ``[q(7), gripper(1)]`` with the
gripper normalized to ``[0,1]`` (width/max_width). Images are RAW RGB -- the
policy-side transform resizes/normalizes them (symmetry with the "raw action"
contract). One ``apply()`` advances one control tick (``substeps`` mj_steps at
the recording's 50 Hz), matching how the data was produced.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, Gripper, JointPositions
from collection.camera import SimCameraRenderer
from scene.tasks import task_instruction


class SimEnv:
    """A minimal sim environment a VLA policy acts in (joint action space)."""

    def __init__(self, task="pick_cube", instruction="", cameras=("front", "wrist"),
                 action_space="joint", control_dt=0.02, view=False,
                 img_w=640, img_h=480):
        if action_space != "joint":
            raise ValueError("only action_space='joint' is implemented (ee = TODO)")
        self.task = task
        # default the language instruction to the task's own (tasks.py) if unset
        self.instruction = instruction or task_instruction(task)
        self.action_space = action_space
        self.robot = SimRobot(task)
        self.gripper = Gripper(self.robot)
        self.cams = SimCameraRenderer(self.robot.model, self.robot.data,
                                      cameras, img_w, img_h)
        # one apply() = one control tick = substeps mj_steps (recording cadence)
        self.substeps = max(1, round(control_dt / self.robot.model.opt.timestep))
        self.ac = self.robot.start_joint_position_control()
        self.viewer = None
        if view:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.robot.model, self.robot.data)

    # -- gym-ish surface ----------------------------------------------

    def reset(self, randomize=False):
        """Reset arm to HOME and objects to their start (or a random layout).
        Returns the first observation."""
        self.robot.reset_home()
        if randomize:
            self.robot.randomize_objects()
        else:
            self.robot.reset_objects()
        self.ac = self.robot.start_joint_position_control()
        if self.viewer is not None:
            self.viewer.sync()
        return self.observe()

    def observe(self):
        """The VLA input this tick: raw camera images + proprio state + language.

        ``state`` = ``[q(7), gripper_norm(1)]`` (measured), matching the dataset's
        state definition; ``images`` = ``{cam: (H,W,3) uint8 RGB}`` (unresized)."""
        st = self.robot.read_once()
        return {
            "images": self.cams.render(),
            "state": np.concatenate(
                [st.q, [self.gripper.width() / self.gripper.max_width]]
            ).astype(np.float32),
            "instruction": self.instruction,
        }

    def apply(self, action):
        """Consume one raw action and advance one control tick.

        ``action`` = ``[q_d(7), gripper(1)]``: 7 joint targets (rad) + normalized
        gripper opening in [0,1]. Decoded to the arm/gripper controllers and held
        for ``substeps`` steps. Returns the next observation."""
        action = np.asarray(action, dtype=float)
        self.gripper.set_target_width(
            float(np.clip(action[7], 0.0, 1.0)) * self.gripper.max_width)
        cmd = JointPositions(action[:7])
        for _ in range(self.substeps):
            self.ac.writeOnce(cmd)
        if self.viewer is not None:
            self.viewer.sync()
        return self.observe()

    # -- helpers -------------------------------------------------------

    def is_grasped(self):
        return bool(self.gripper.read_once().is_grasped)

    def close(self):
        self.cams.close()
        if self.viewer is not None:
            self.viewer.close()
