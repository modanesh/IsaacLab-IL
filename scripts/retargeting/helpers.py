from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.managers import SceneEntityCfg

from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # type: ignore


def get_robot_cfg(robot_name: str) -> ArticulationCfg:
    """Simple validation method for checks"""
    if robot_name != "anymal_d":
        raise ValueError(f"Unsupported robot: {robot_name!r}")
    
    robot_cfg: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    return robot_cfg


ROBOT_META = {
    "anymal_d": {
        "foot_bodies": ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
        "leg_roots": ["LF_HIP", "RF_HIP", "LH_HIP", "RH_HIP"],
        "joint_names": [
            "LF_HAA", "LF_HFE", "LF_KFE",
            "RF_HAA", "RF_HFE", "RF_KFE",
            "LH_HAA", "LH_HFE", "LH_KFE",
            "RH_HAA", "RH_HFE", "RH_KFE",
        ],
        "default_joint_pos": {
            "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
            "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
            "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
            "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
        },
        "leg_length": 0.55,
        "base_height": 0.55,
        "stance_length": 0.70,
        "stance_width": 0.38,
    }
}


@dataclass
class ReferenceMotion:
    root_pos: np.ndarray
    root_rot: np.ndarray  # wxyz
    foot_pos: np.ndarray
    root_lin_vel: np.ndarray | None
    root_ang_vel: np.ndarray | None
    dt: float
    thigh_pos: np.ndarray | None = None  # (T,4,3) FL,FR,BL,BR — used by posture mode
    torso_pitch: np.ndarray | None = None  # (T,) per-frame spine pitch (rad) — posture mode


def _normalize_quaternions_np(quats: np.ndarray) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float32).copy()
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats /= np.clip(norms, 1e-8, None)
    return quats


def ensure_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    quats = _normalize_quaternions_np(quats)
    for t in range(1, quats.shape[0]):
        if np.dot(quats[t - 1], quats[t]) < 0.0:
            quats[t] *= -1.0
    return quats


def _quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return out.astype(np.float32)


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).copy()
    q[..., 1:] *= -1.0
    return q


def quat_apply_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _normalize_quaternions_np(q)
    v_as_quat = np.concatenate([np.zeros(v.shape[:-1] + (1,), dtype=np.float32), v.astype(np.float32)], axis=-1)
    rotated = _quat_mul_np(_quat_mul_np(q, v_as_quat), quat_conjugate_np(q))
    return rotated[..., 1:]


def world_to_local_points(root_pos: np.ndarray, root_rot: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    rel = pts_world - root_pos[..., None, :] if root_pos.ndim == 2 else pts_world - root_pos
    q_inv = quat_conjugate_np(root_rot)
    if rel.ndim == 3:
        q_inv = np.repeat(q_inv[:, None, :], rel.shape[1], axis=1)
    return quat_apply_np(q_inv, rel)


def local_to_world_points(root_pos: np.ndarray, root_rot: np.ndarray, pts_local: np.ndarray) -> np.ndarray:
    q = root_rot
    if pts_local.ndim == 3:
        q = np.repeat(q[:, None, :], pts_local.shape[1], axis=1)
    return quat_apply_np(q, pts_local) + (root_pos[..., None, :] if root_pos.ndim == 2 else root_pos)


def _quat_to_angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
    quats = ensure_quaternion_continuity(quats)
    T = quats.shape[0]
    omega = np.zeros((T, 3), dtype=np.float32)
    q0 = quats[:-1]
    q1 = quats[1:]
    qr = _quat_mul_np(q1, quat_conjugate_np(q0))
    omega[1:] = 2.0 * qr[:, 1:] / dt
    if T > 1:
        omega[0] = omega[1]
    return omega


def load_reference_motion(path: str, fps: float) -> ReferenceMotion:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reference motion not found: {path}")

    try:
        data = dict(np.load(path, allow_pickle=True))
    except Exception as exc:
        raise ValueError(f"Unsupported reference motion format: {p.suffix.lower()!r}") from exc

    def _get(key: str, required: bool = True) -> np.ndarray | None:
        val = data.get(key)
        if val is None and required:
            raise KeyError(
                f"Required key '{key}' missing from reference motion file. "
                f"Available keys: {list(data.keys())}"
            )
        return np.asarray(val, dtype=np.float32) if val is not None else None

    root_pos = _get("root_pos")
    root_rot = ensure_quaternion_continuity(_get("root_rot"))
    foot_pos = _get("foot_pos")
    thigh_pos = _get("thigh_pos", required=False)
    torso_pitch = _get("torso_pitch", required=False)
    root_lin_vel = _get("root_lin_vel", required=False)
    root_ang_vel = _get("root_ang_vel", required=False)

    if root_pos is None or root_rot is None or foot_pos is None:
        raise ValueError("Reference motion is missing required arrays.")

    T = root_pos.shape[0]
    assert root_rot.shape == (T, 4), f"root_rot must be (T,4), got {root_rot.shape}"
    assert foot_pos.shape == (T, 4, 3), f"foot_pos must be (T,4,3), got {foot_pos.shape}"

    dt = 1.0 / fps
    if root_lin_vel is None:
        root_lin_vel = np.gradient(root_pos, dt, axis=0).astype(np.float32)
    if root_ang_vel is None:
        root_ang_vel = _quat_to_angular_velocity(root_rot, dt)

    print(f"[Loader] Loaded '{p.name}' — {T} frames @ {fps} Hz ({T / fps:.1f} s).")
    return ReferenceMotion(
        root_pos=root_pos,
        root_rot=root_rot,
        foot_pos=foot_pos,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        dt=dt,
        thigh_pos=thigh_pos,
        torso_pitch=torso_pitch,
    )


def compute_scale_factors(ref_motion: ReferenceMotion, meta: dict) -> dict:
    foot_local = world_to_local_points(ref_motion.root_pos, ref_motion.root_rot, ref_motion.foot_pos)

    lat_animal = np.mean(np.abs(foot_local[:, 0, 1] - foot_local[:, 1, 1]))
    lon_animal = np.mean(np.abs(foot_local[:, 0, 0] - foot_local[:, 2, 0]))

    mean_foot_h = ref_motion.foot_pos[:, :, 2].mean(axis=1)
    base_h_animal = float(np.median(ref_motion.root_pos[:, 2] - mean_foot_h))

    robot_width = float(meta["stance_width"])
    robot_length = float(meta["stance_length"])
    robot_base_h = float(meta["base_height"])

    sx = robot_length / max(lon_animal, 1e-4)
    sy = robot_width / max(lat_animal, 1e-4)
    sz = robot_base_h / max(base_h_animal, 1e-4)

    scale = {
        "x": float(sx),
        "y": float(sy),
        "z": float(sz),
        "base_h_animal": float(base_h_animal),
        "robot_base_h": robot_base_h,
        "animal_stance_length": float(lon_animal),
        "animal_stance_width": float(lat_animal),
        "robot_stance_length": robot_length,
        "robot_stance_width": robot_width,
    }
    print(
        "[Scale] "
        f"animal length≈{lon_animal:.3f}m width≈{lat_animal:.3f}m base_h≈{base_h_animal:.3f}m | "
        f"robot length≈{robot_length:.3f}m width≈{robot_width:.3f}m base_h≈{robot_base_h:.3f}m | "
        f"scale xyz=({sx:.3f}, {sy:.3f}, {sz:.3f})"
    )
    return scale


class QuadrupedRetargeter:
    def __init__(
        self,
        robot: Articulation,
        scene: InteractiveScene,
        sim,
        meta: dict,
        device: str,
        ik_iterations: int,
        ik_damping: float,
        physics_dt: float,
        num_envs: int,
        render: bool = False,
        relax_front_knees: bool = False,
    ):
        self.robot = robot
        self.scene = scene
        self.sim = sim
        self.meta = meta
        self.device = device
        self.ik_iters = ik_iterations
        self.physics_dt = physics_dt
        self.num_envs = num_envs
        # Render every inner IK iteration only when a viewer is attached;
        # in headless export runs this would just burn ~20 renders per frame.
        self.render = render
        self.is_fixed_base = bool(getattr(robot, "is_fixed_base", False))

        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": ik_damping},
        )
        self.ik_controllers = [
            DifferentialIKController(ik_cfg, num_envs=num_envs, device=device) for _ in range(4)
        ]

        all_joint_names = list(robot.joint_names)
        all_body_names = list(robot.body_names)
        self.leg_joint_ids: list[list[int]] = []
        for leg_i in range(4):
            leg_names = meta["joint_names"][leg_i * 3: leg_i * 3 + 3]
            self.leg_joint_ids.append([all_joint_names.index(name) for name in leg_names])

        self.foot_body_ids = [all_body_names.index(name) for name in meta["foot_bodies"]]

        default_map: dict = meta["default_joint_pos"]

        # self note: qpos means generalized position vector
        # naming scheme from mujoco
        self.default_qpos = self.robot.data.default_joint_pos.clone()
        for joint_name, value in default_map.items():
            joint_id = all_joint_names.index(joint_name)
            self.default_qpos[:, joint_id] = float(value)

        # --- Knee-branch constraint -------------------------------------------
        # A 2-link leg has TWO IK solutions for any reachable foot position
        # (knee bent one way or the other). The ANYmal-D's URDF KFE soft limits
        # are absurdly wide (~+-8.95 rad = +-512 deg), so the DLS solver is free
        # to slip into the WRONG (outward) branch — or wrap the angle past +-pi —
        # whenever the foot target is noisy. The warm start then carries that bad
        # branch forward for the rest of the clip. This was seen on the low-res
        # tigdog clips: front KFE ran to -4.5 rad (wrapped) / +2.3 rad (flipped)
        # instead of staying negative like every clean clip (~ -1.0..-1.8).
        #
        # ANYmal-D walks in the X-configuration: front knees flex NEGATIVE
        # (default -0.8), hind knees POSITIVE (default +0.8). We build per-joint
        # effective limits that keep each KFE on the sign of its default pose,
        # with a magnitude cap comfortably above the real gait range (clean clips
        # peaked ~ +-2.06 rad), so legitimate motion is untouched but the outward
        # branch and angle-wrapping are forbidden.
        KNEE_EPS = 0.05   # keep a hair off the straight-leg singularity
        KNEE_MAX = 2.80   # > observed gait max (~2.06), < the loose URDF limit
        soft = self.robot.data.soft_joint_pos_limits.clone()  # (E, J, 2)
        self.eff_lo = soft[..., 0].clone()
        self.eff_hi = soft[..., 1].clone()
        for jid, jname in enumerate(all_joint_names):
            if not jname.endswith("KFE"):
                continue
            if float(self.default_qpos[0, jid]) < 0.0:      # front knee: normally negative (X-config)
                self.eff_lo[:, jid] = torch.clamp(self.eff_lo[:, jid], min=-KNEE_MAX)
                # Posture mode: also allow the front knee to flex POSITIVE, so the front legs
                # can extend down-forward into a bow. The X-config negative-only branch keeps
                # the foot behind the knee and tucks the legs.
                self.eff_hi[:, jid] = KNEE_MAX if relax_front_knees else -KNEE_EPS
            else:                                            # hind knee: positive
                self.eff_lo[:, jid] = KNEE_EPS
                self.eff_hi[:, jid] = torch.clamp(self.eff_hi[:, jid], max=KNEE_MAX)

        # Fixed base pose for "pinned in the air" retargeting.
        # Robot root is locked at (0, 0, 0.5) world with identity orientation (wxyz).
        self.fixed_root_pos = torch.tensor(
            [[0.0, 0.0, 0.5]], dtype=torch.float32, device=device
        ).expand(num_envs, -1).contiguous()
        self.fixed_root_rot = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device
        ).expand(num_envs, -1).contiguous()

        # Warm start: previous frame's IK solution. Restarting every frame from
        # the default pose lets DLS hop between IK branches and shows up as
        # frame-to-frame jitter in the exported clip.
        self._warm_jpos: torch.Tensor | None = None

    def set_fixed_base(self, pos: np.ndarray, rot: np.ndarray) -> None:
        """Override the pinned base pose. Posture/sit retargeting uses this to tilt
        (and lower) the base instead of the default level (0,0,0.5) + identity."""
        self.fixed_root_pos = torch.tensor(
            np.asarray(pos, dtype=np.float32).reshape(1, 3), device=self.device
        ).expand(self.num_envs, -1).contiguous()
        self.fixed_root_rot = torch.tensor(
            np.asarray(rot, dtype=np.float32).reshape(1, 4), device=self.device
        ).expand(self.num_envs, -1).contiguous()

    def _flush_state(self, joint_pos, joint_vel=None):
        if joint_vel is None:
            joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)        # physics-only during inner IK
        self.scene.update(self.physics_dt)
        self.robot.update(self.physics_dt)

        if self.render:
            self.sim.render()

    def _teleport_base(self, root_pos_t: torch.Tensor, root_rot_t: torch.Tensor) -> None:
        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] = root_pos_t
        root_state[:, 3:7] = root_rot_t
        root_state[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])

    def solve_frame(self, root_pos: np.ndarray, root_rot: np.ndarray, foot_pos_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dev = self.device
        E = self.num_envs

        # Teleport the base to the requested per-frame pose. For locomotion this is the
        # constant level (0,0,0.5)+identity (behaviour unchanged); for posture mode it is
        # the per-frame tilted/lowered base, so stand<->bow transitions come through.
        # `foot_pos_local` is expressed in that same base frame.
        rp = torch.as_tensor(root_pos, dtype=torch.float32, device=dev).reshape(1, 3).expand(E, -1).contiguous()
        rr = torch.as_tensor(root_rot, dtype=torch.float32, device=dev).reshape(1, 4).expand(E, -1).contiguous()

        foot_local = torch.tensor(foot_pos_local, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1, -1)

        # Hard-pin the base every frame so physics / Jacobians see the fixed pose.
        self._teleport_base(rp, rr)

        # Continue from the previous frame's solution (first frame: default pose).
        jpos = self._warm_jpos.clone() if self._warm_jpos is not None else self.default_qpos.clone()
        jvel = torch.zeros_like(jpos)
        self._flush_state(jpos, jvel)

        for _ in range(self.ik_iters):
            jacobian_full = self.robot.root_physx_view.get_jacobians()

            for leg_i in range(4):
                ctrl = self.ik_controllers[leg_i]
                leg_ids = self.leg_joint_ids[leg_i]
                foot_id = self.foot_body_ids[leg_i]
                body_idx = foot_id - 1 if self.is_fixed_base else foot_id

                # get the end-effector pose in the world frame -> base frame
                ee_pos_w = self.robot.data.body_pos_w[:, foot_id, :]
                ee_quat_w = self.robot.data.body_quat_w[:, foot_id, :]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(rp, rr, ee_pos_w, ee_quat_w)

                # print(f"[INFO] Leg: {self.meta['joint_names'][leg_i]}, {foot_local[:, leg_i, :]}")
            
                target_pos_b = foot_local[:, leg_i, :]
                ctrl.set_command(target_pos_b, ee_quat=ee_quat_b)

                leg_jpos = jpos[:, leg_ids]
                # PhysX jacobians of FLOATING-base articulations carry 6 leading
                # base-DOF columns, so joint columns are offset by +6 (same
                # convention as isaaclab/envs/mdp/actions/task_space_actions.py).
                # Indexing with the raw joint ids would silently slice base-DOF
                # columns and the wrong joints' columns.
                jac_joint_ids = leg_ids if self.is_fixed_base else [i + 6 for i in leg_ids]
                leg_dof_ids = torch.tensor(jac_joint_ids, device=dev, dtype=torch.long)
                J = jacobian_full[:, body_idx, :, :][:, :, leg_dof_ids]

                jpos[:, leg_ids] = ctrl.compute(ee_pos_b, ee_quat_b, J, leg_jpos)

                # Clamp to the sign-consistent effective limits (knees pinned to
                # their X-configuration branch; other joints keep the URDF soft
                # limits). Prevents the outward-knee branch flip / angle wrap.
                lo = self.eff_lo[:, leg_ids]
                hi = self.eff_hi[:, leg_ids]
                jpos[:, leg_ids] = torch.clamp(jpos[:, leg_ids], lo, hi)

            self._teleport_base(rp, rr)
            self._flush_state(jpos, jvel)

        self.robot.update(self.physics_dt)

        self._warm_jpos = jpos.clone()

        return (
            jpos[0].detach().cpu().numpy().astype(np.float32),
            self.robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32),
        )
