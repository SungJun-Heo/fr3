"""SimEnv -- fr3 as a VLA-policy consumer (observation producer + action sink).

fr3 stays policy-agnostic, mirroring the collection IR's "keep the superset,
adapt later" philosophy:

  * observe() returns EVERYTHING it cheaply can -- raw camera images + the full
    measured proprioception keyed by the SAME names as the recorded IR
    (``q``, ``dq``, ``tau_J``, ``O_T_EE``, ``gripper_width``, ...) + the language
    instruction. The inference project's adapter picks the fields its policy was
    trained on (proprio is ~350 bytes, so passing all of it is free). Which
    proprio the policy uses is decided policy-side, not here.

  * apply() consumes ONE TAGGED action and dispatches to fr3's own control
    primitives -- it can't pure-passthrough (an action must become a controller
    command), so the inference project emits one of a small fixed vocabulary:
        {"joint":     [q(7), gripper]}                # joint position control
        {"cartesian": {"pose": O_T_EE(16), "gripper": g}}   # Cartesian (clamp IK)
    gripper is normalized [0,1]. One apply() = one 50 Hz control tick.

Policy inference, action chunking/reactivity, and network transport live in a
SEPARATE inference project; fr3 only exposes this env. A rollout is then::

    env = SimEnv("pick_cube")
    obs = env.reset(randomize=True)
    while not env.success():
        obs = env.apply(policy(obs))   # policy: pick obs fields -> tagged action
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot import SimRobot, Gripper, JointPositions, CartesianPose
from collection.camera import SimCameraRenderer
from collection.schema import robot_meta as _robot_meta, observation_from_state
from scene.tasks import task_instruction
from rollout.success import task_success


class SimEnv:
    """A minimal sim environment a VLA policy acts in (keyed obs, tagged action)."""

    def __init__(self, task="pick_cube", instruction="", cameras=("front", "wrist"),
                 control_dt=0.02, view=False, img_w=640, img_h=480):
        self.task = task
        self.instruction = instruction or task_instruction(task)
        self.robot = SimRobot(task)
        self.gripper = Gripper(self.robot)
        self.cams = SimCameraRenderer(self.robot.model, self.robot.data,
                                      cameras, img_w, img_h)
        # one apply() = one control tick = substeps mj_steps (recording cadence)
        self.substeps = max(1, round(control_dt / self.robot.model.opt.timestep))
        # one control handle serves BOTH action tags: writeOnce dispatches on the
        # command type (JointPositions vs CartesianPose); clamp = brake at
        # singularities instead of faulting (right for an arbitrary policy).
        self.ac = self.robot.start_cartesian_pose_control(safety="clamp")
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
        self.ac = self.robot.start_cartesian_pose_control(safety="clamp")
        if self.viewer is not None:
            self.viewer.sync()
        return self.observe()

    def observe(self):
        """The VLA input this tick: raw images + the FULL measured proprio (keyed
        by the IR field names) + language. The proprio is built by the SAME
        ``collection.schema.observation_from_state`` the recorder uses, so a
        policy sees identical fields/dtype at train and rollout time. The policy's
        adapter selects/normalizes what it needs (see ``metadata()`` for constants
        like max_width)."""
        return {
            "images": self.cams.render(),                # {cam: (H,W,3) uint8 RGB}
            "proprio": observation_from_state(self.robot.read_once(),
                                              self.gripper.read_once()),
            "instruction": self.instruction,
        }

    def apply(self, action):
        """Consume one TAGGED action and advance one control tick. Returns the
        next observation. ``action`` is a dict with exactly one control key:
        ``{"joint": [q(7), gripper]}`` or
        ``{"cartesian": {"pose": O_T_EE(16), "gripper": g}}`` (gripper in [0,1])."""
        cmd, grip = self._decode(action)
        if grip is not None:
            self.gripper.set_target_width(
                float(np.clip(grip, 0.0, 1.0)) * self.gripper.max_width)
        for _ in range(self.substeps):
            self.ac.writeOnce(cmd)
        if self.viewer is not None:
            self.viewer.sync()
        return self.observe()

    def _decode(self, action):
        """Tagged action -> (controller command, gripper-or-None)."""
        if "joint" in action:
            v = np.asarray(action["joint"], dtype=float)
            return JointPositions(v[:7]), (v[7] if v.shape[0] > 7 else None)
        if "cartesian" in action:
            c = action["cartesian"]
            cmd = CartesianPose(np.asarray(c["pose"], dtype=float).ravel())
            return cmd, c.get("gripper")
        raise ValueError(
            f"action needs a 'joint' or 'cartesian' key, got {list(action)}")

    # -- helpers -------------------------------------------------------

    def metadata(self):
        """Static robot description (joint names/limits, gripper max_width, ee
        site) -- the SAME dict the dataset's meta.json carries, so the inference
        adapter normalizes rollout proprio consistently with training."""
        return _robot_meta(self.robot, self.gripper)

    def success(self):
        """Task accomplished this tick? (ground-truth sim check; see
        ``rollout/success.py``). Lets a rollout terminate / be scored."""
        return task_success(self)

    def is_grasped(self):
        return bool(self.gripper.read_once().is_grasped)

    def close(self):
        self.cams.close()
        if self.viewer is not None:
            self.viewer.close()
