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
        {"joint":           [q(7), gripper]}                     # absolute joint
        {"cartesian":       {"pose": O_T_EE(16), "gripper": g}}  # absolute EE (clamp IK)
        {"joint_delta":     [dq(7), gripper]}                    # joint increment
        {"cartesian_delta": {"dpos": dp(3), "drot": r, "gripper": g}}  # EE increment
    gripper is ABSOLUTE, normalized [0,1] (even for the delta tags). One apply()
    = one 50 Hz control tick.

    Delta tags are a *representation*, not a different controller: no delta-native
    servo exists -- the increment is integrated onto the MEASURED current state
    (read at the START of this tick) to form the absolute target the SAME
    controller tracks. Conventions (must match how the training set differenced,
    also pinned in ``collection.schema.CONVENTIONS`` so converter + integrator
    agree):
        reference = measured state this tick (not a commanded accumulator)
        frame = base:  q_target = q + dq ;  p_target = p + dp ;  R_target = dR . R
        drot = axis-angle rotation vector (3,), or a 3x3 rotation matrix

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
from robot import SimRobot, Gripper, JointPositions, CartesianPose, vec_to_pose
from collection.camera import SimCameraRenderer
from collection.schema import robot_meta as _robot_meta, observation_from_state
from scene.tasks import task_instruction
from rollout.success import task_success


def _delta_R(drot):
    """A rotation-delta 3x3 from either an axis-angle rotation vector ``(3,)``
    (Rodrigues; the vector's norm is the angle in rad, its direction the axis) or
    an already-formed ``(3,3)`` / len-9 rotation matrix. The one place rollout
    turns a policy's ``drot`` into a matrix to compose with the current EE frame."""
    a = np.asarray(drot, dtype=float)
    if a.size == 9:
        return a.reshape(3, 3)
    v = a.ravel()[:3]
    theta = float(np.linalg.norm(v))
    if theta < 1e-9:
        return np.eye(3)
    k = v / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


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
        next observation. ``action`` has exactly one control key -- absolute:
        ``{"joint": [q(7), gripper]}`` / ``{"cartesian": {"pose": O_T_EE(16),
        "gripper": g}}`` -- or delta (integrated onto the measured state this
        tick, base frame): ``{"joint_delta": [dq(7), gripper]}`` /
        ``{"cartesian_delta": {"dpos": dp(3), "drot": r, "gripper": g}}``. gripper
        is absolute in [0,1]. See the module docstring for the delta conventions."""
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
        """Tagged action -> (controller command, gripper-or-None). Absolute tags
        map straight to a command; delta tags integrate the increment onto the
        MEASURED state read this tick, in the base frame (see module docstring)."""
        if "joint" in action:
            v = np.asarray(action["joint"], dtype=float)
            return JointPositions(v[:7]), (v[7] if v.shape[0] > 7 else None)
        if "joint_delta" in action:
            v = np.asarray(action["joint_delta"], dtype=float)
            q_ref = self.robot.read_once().q                    # measured, this tick
            return JointPositions(q_ref + v[:7]), (v[7] if v.shape[0] > 7 else None)
        if "cartesian" in action:
            c = action["cartesian"]
            cmd = CartesianPose(np.asarray(c["pose"], dtype=float).ravel())
            return cmd, c.get("gripper")
        if "cartesian_delta" in action:
            c = action["cartesian_delta"]
            p_ref, R_ref = vec_to_pose(self.robot.read_once().O_T_EE)  # measured, this tick
            p = p_ref + np.asarray(c.get("dpos", (0.0, 0.0, 0.0)), dtype=float).ravel()
            R = _delta_R(c["drot"]) @ R_ref if "drot" in c else R_ref  # base-frame compose
            return CartesianPose.from_matrix(p, R), c.get("gripper")
        raise ValueError("action needs one of joint/joint_delta/cartesian/"
                         f"cartesian_delta, got {list(action)}")

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
