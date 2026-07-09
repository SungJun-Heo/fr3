"""Damped Least Squares (DLS) inverse kinematics -- the "gentle brake" solver.

Given a target end-effector pose, find the joint angles that reach it. The core
of one iteration is the differential-IK step with a damping term:

    dq = Jᵀ (J Jᵀ + λ²I)⁻¹ · e          (e = 6D pose error, J = 6×n EE Jacobian)

The λ² ("damping") is the brake: far from singularities it does almost nothing
(accurate), but when the naive inverse would demand huge joint motion (near a
singular arm pose) it caps the response so the arm moves gently instead of
whipping. See docs/cartesian_ik_control_notes.md.

This solver keeps its OWN MjData scratch, so iterating never disturbs the live
sim state. It is used both for one-shot "solve to this pose" and, one step at a
time, for per-tick Cartesian streaming.
"""

import numpy as np
import mujoco


class DLSIKSolver:
    def __init__(self, model, site_name, joint_names, damping=0.05,
                 max_step=0.3):
        self.model = model
        self.data = mujoco.MjData(model)          # private FK/Jacobian scratch
        self.site = model.site(site_name).id
        self.qadr = np.array([model.joint(n).qposadr[0] for n in joint_names])
        self.dofs = np.array([model.joint(n).dofadr[0] for n in joint_names])
        self.n = len(joint_names)
        self.damping = damping                    # λ, the brake strength
        self.max_step = max_step                  # per-iteration |dq| clamp (safety)

    # -- one differential-IK step --------------------------------------

    def _fk(self, q):
        """Put the scratch arm at q and refresh the kinematics it needs."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)   # needed for site Jacobian

    def _pose_error(self, data, target_pos, target_mat):
        """6D error [Δposition(3), Δrotation(3)] from the pose in ``data``."""
        err_pos = np.asarray(target_pos) - data.site_xpos[self.site]
        # orientation error: rotation that takes current -> target, as a rotvec
        quat_cur = np.zeros(4)
        mujoco.mju_mat2Quat(quat_cur, data.site_xmat[self.site])
        quat_tgt = np.zeros(4)
        mujoco.mju_mat2Quat(quat_tgt, np.asarray(target_mat, float).flatten())
        quat_cur_inv = np.zeros(4)
        mujoco.mju_negQuat(quat_cur_inv, quat_cur)      # conjugate = inverse (unit quat)
        err_quat = np.zeros(4)
        mujoco.mju_mulQuat(err_quat, quat_tgt, quat_cur_inv)
        err_rot = np.zeros(3)
        mujoco.mju_quat2Vel(err_rot, err_quat, 1.0)     # quaternion -> rotvec
        return np.concatenate([err_pos, err_rot])

    def _jacobian(self, data):
        """6×n end-effector Jacobian (arm columns only) from ``data``."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site)
        return np.vstack([jacp[:, self.dofs], jacr[:, self.dofs]])

    def velocity_step(self, data, target_pos, target_mat):
        """One DLS step toward the target using the pose already in ``data``.

        The caller must have current kinematics for ``data`` (true right after
        ``mj_step``/``mj_forward``), so this does no FK -- it is the per-tick
        streaming path. Returns ``(dq, info)`` with info =
        ``{pos_err, rot_err, manipulability}``."""
        err = self._pose_error(data, target_pos, target_mat)
        J = self._jacobian(data)
        dq = self.dls_step(err, J)
        w = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))  # Yoshikawa manip.
        info = dict(pos_err=float(np.linalg.norm(err[:3])),
                    rot_err=float(np.linalg.norm(err[3:])),
                    manipulability=w)
        return dq, info

    def dls_step(self, err, J):
        """The DLS increment: dq = Jᵀ (JJᵀ + λ²I)⁻¹ e (with a |dq| clamp)."""
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + self.damping**2 * np.eye(6), err)
        norm = np.linalg.norm(dq)
        if norm > self.max_step:                  # clamp big steps for stability
            dq *= self.max_step / norm
        return dq

    # -- one-shot solve ------------------------------------------------

    def solve(self, target_pos, target_mat, q_init, max_iters=200,
              pos_tol=1e-4, rot_tol=1e-3):
        """Iterate the DLS step from q_init to reach the target pose.

        Returns (q, info) with info = {iters, pos_err, rot_err, converged}."""
        q = np.array(q_init, dtype=float)
        for i in range(max_iters):
            self._fk(q)
            err = self._pose_error(self.data, target_pos, target_mat)
            pos_err = np.linalg.norm(err[:3])
            rot_err = np.linalg.norm(err[3:])
            if pos_err < pos_tol and rot_err < rot_tol:
                return q, dict(iters=i, pos_err=pos_err, rot_err=rot_err,
                               converged=True)
            q = q + self.dls_step(err, self._jacobian(self.data))
        return q, dict(iters=max_iters, pos_err=pos_err, rot_err=rot_err,
                       converged=False)
