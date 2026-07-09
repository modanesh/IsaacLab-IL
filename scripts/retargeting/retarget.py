import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import torch

ISAAC_TO_PYB = np.array([0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11], dtype=np.int64)
SIGN_FLIP = np.ones(12, dtype=np.float32)
SIGN_FLIP[2] = -1.0
SIGN_FLIP[3] = -1.0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-motion", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--physics_dt", type=float, default=1.0 / 120.0)
    p.add_argument("--ik_damping", type=float, default=0.05)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--robot", type=str, default="anymal_d")
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--height_offset", type=float, default=0.0)
    p.add_argument("--ik_iterations", type=int, default=20)
    p.add_argument("--visualize", action="store_true", default=False)
    p.add_argument("--plot_skeleton", action="store_true", default=False)
    p.add_argument(
        "--lateral-mode", type=str, default="pin", choices=["pin", "raw"],
        help=(
            "Lateral (body-frame y) foot targets. 'pin' (default) replaces the "
            "depth-derived lateral coordinate with a constant +-stance_width/2 "
            "per foot — for a straight walk the monocular-depth lateral signal "
            "is mostly noise (same magnitude as the true separation) and after "
            "scaling it swings foot targets across the body midline. "
            "'raw' keeps the depth-derived lateral motion."
        ),
    )
    p.add_argument(
        "--posture", action="store_true", default=False,
        help=(
            "Posture/sit mode for NON-locomotion clips (e.g. a dog sit + shake). "
            "Instead of pinning the base level at 0.5 m, tilt it nose-up by the "
            "animal's torso pitch (auto from the front-vs-hind thigh heights) and "
            "lower it, so the IK bends the hind knees and tilts the base rear-down "
            "to approximate the sit. Skip root recovery downstream."
        ),
    )
    p.add_argument("--posture-pitch", type=float, default=None,
                   help="Override the auto torso pitch (degrees nose-up). Default: auto from thighs.")
    p.add_argument("--posture-height", type=float, default=0.40,
                   help="Lowered base height (m) for posture mode (default 0.40 vs the 0.5 stand).")
    p.add_argument("--posture-reach", type=float, default=0.50,
                   help="Clamp each posture foot target to within this radius (m) of the base — "
                        "the bunched-sit front legs track noisily and over-reach the workspace.")
    p.add_argument("--posture-front-stretch", type=float, default=0.0,
                   help="Bow only: extend the FRONT feet forward (m, at full bow) and lower them to "
                        "the ground, scaled by bow depth, so the front legs reach out in front "
                        "instead of tucking. 0 = off.")
    p.add_argument("--posture-max-noseup", type=float, default=90.0,
                   help="Clamp the per-frame nose-up tilt (deg). Kills spurious lean-back when the "
                        "animal lifts its head (the spine reads a false nose-up). e.g. 3 for a bow.")
    p.add_argument("--posture-max-nosedown", type=float, default=90.0,
                   help="Cap the per-frame nose-DOWN tilt (deg). A steep bow tilt puts the front feet "
                        "out of reach; capping it (e.g. 12) keeps a moderate tilt so the front legs "
                        "can still extend forward-down with --posture-front-stretch.")
    p.add_argument("--posture-pad-standing", type=float, default=0.0,
                   help="Prepend/append this many seconds of the first/last (standing) frame, so the "
                        "policy sees a steady stand before and after the motion.")
    # --- Synthesized bow with an explicit front-leg forward step -------------------------
    # Builds a stand->bow->stand motion from the robot's own standing stance (read from a
    # level frame of the ref) rather than the held-bow clip: the hind feet stay planted in
    # the world while the FRONT feet lift and step forward as the base tilts down, then
    # reverse. Because we specify foot targets and solve IK, joint angles and foot targets
    # stay consistent (unlike a joint-space lift).
    p.add_argument("--synth-bow", action="store_true",
                   help="synthesize a stand->bow->stand motion with a front-leg forward step; "
                        "uses the ref only for dt and the robot's standing stance.")
    p.add_argument("--synth-fwd", type=float, default=0.15,
                   help="how far the front feet step forward (m) into the bow.")
    p.add_argument("--synth-lift", type=float, default=0.08,
                   help="peak lift of the front feet during the step (m).")
    p.add_argument("--synth-tilt", type=float, default=12.0,
                   help="bow base pitch (deg, nose-down).")
    p.add_argument("--synth-bz", type=float, default=0.45,
                   help="base height held during the bow (m); standing feet rest on the ground.")
    p.add_argument("--synth-stand", type=float, default=2.0, help="seconds standing at each end.")
    p.add_argument("--synth-trans", type=float, default=1.5,
                   help="seconds for each stand<->bow transition.")
    p.add_argument("--synth-hold", type=float, default=2.0, help="seconds held in the bow.")
    # --- Gait "collection": shrink an over-athletic reference into the robot's envelope ----
    # Scales the body-frame foot targets so the legs lift/tuck less (clearance) and step
    # shorter (stride), keeping stance feet grounded. Used to bring a greyhound gallop (huge
    # ~0.4m foot lift, knee ptp 2.5+ rad) down toward trot-like, deployable joint ranges.
    p.add_argument("--collect-z", type=float, default=1.0,
                   help="scale swing CLEARANCE (foot vertical excursion above stance); "
                        "<1 lowers the leg lift/tuck -> smaller knee range. e.g. 0.4.")
    p.add_argument("--collect-x", type=float, default=1.0,
                   help="scale STRIDE (foot fore-aft excursion about its mean); "
                        "<1 shortens steps -> smaller hip range / more collected. e.g. 0.7.")
    p.add_argument("--x-shift-front", type=float, default=0.0,
                   help="fore-aft shift [m] added to the FRONT foot targets (body frame), after "
                        "collection; + is forward. Re-centers a gait that loads the front feet "
                        "behind/under the chest (gallop stroke-back) onto the shoulders.")
    p.add_argument("--x-shift-hind", type=float, default=0.0,
                   help="fore-aft shift [m] added to the HIND foot targets, after collection.")
    p.add_argument("--smooth-targets-win", type=int, default=0,
                   help="odd Savitzky-Golay window (frames) low-passing the body-frame foot "
                        "targets before collection; 0 = off. Kills single-frame target spikes "
                        "that IK otherwise turns into joint-velocity spikes.")
    p.add_argument("--snap-z", type=float, default=0.0,
                   help="snap foot-target z within this margin [m] above the per-foot stance "
                        "level down onto it (after collection) — consolidates chattery contacts.")
    return p


parser = _build_arg_parser()

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = not args.visualize
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402f

from helpers import (  # noqa: E402
    ROBOT_META,
    QuadrupedRetargeter,
    compute_scale_factors,
    ensure_quaternion_continuity,
    get_robot_cfg,
    load_reference_motion,
    local_to_world_points,
    quat_apply_np,
    quat_conjugate_np,
    world_to_local_points,
)

class LiveSkeletonPlotter:
    """Live 3D skeleton plot in the robot's body frame (normalized coords).

    Root sits at the origin; feet are drawn at their body-frame positions.
    Axis limits are fixed from the full precomputed trajectory so the view
    doesn't jitter during playback.
    """

    FOOT_EDGES = [0, 1, 2, 3]  # root -> each of the four feet

    def __init__(self, foot_pos_local_all: np.ndarray, foot_names: list[str], title: str = "Retarget skeleton (body frame)"):
        plt.ion()
        self._fig = plt.figure(figsize=(7, 7))
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.view_init(elev=20.0, azim=-100.0)
        self._title = title
        self._foot_names = foot_names

        # Precompute fixed axis bounds from the whole trajectory.
        # Include the origin (root) so the frame is always visible.
        all_pts = np.concatenate(
            [np.zeros((1, 3), dtype=np.float32), foot_pos_local_all.reshape(-1, 3)],
            axis=0,
        )
        finite = np.isfinite(all_pts).all(axis=1)
        pts = all_pts[finite]
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        center = 0.5 * (mins + maxs)
        half = 0.5 * float(np.max(maxs - mins))
        half = max(half, 1e-3) * 1.1  # 10% padding
        self._bounds = {
            "x": (float(center[0] - half), float(center[0] + half)),
            "y": (float(center[1] - half), float(center[1] + half)),
            "z": (float(center[2] - half), float(center[2] + half)),
        }

    def update(self, root_world_t: np.ndarray, foot_local_t: np.ndarray, frame_idx: int, t_sec: float) -> None:
        """Redraw one frame. foot_local_t has shape (4, 3) in body frame."""
        ax = self._ax
        ax.clear()

        if root_world_t is None:
            root = np.zeros(3, dtype=np.float32)
        else: 
            root = root_world_t 

        # Scatter root and feet
        ax.scatter(root[0], root[1], root[2], s=90, label="root")
        ax.scatter(foot_local_t[:, 0], foot_local_t[:, 1], foot_local_t[:, 2], s=50, label="feet")

        # Edges from root to each foot
        for foot_i in self.FOOT_EDGES:
            fp = foot_local_t[foot_i]
            ax.plot([root[0], fp[0]], [root[1], fp[1]], [root[2], fp[2]], linewidth=2)

        # Labels
        for i, name in enumerate(self._foot_names):
            ax.text(foot_local_t[i, 0], foot_local_t[i, 1], foot_local_t[i, 2], name, fontsize=7)

        # Fixed formatting
        ax.set_xlabel("x (body)")
        ax.set_ylabel("y (body)")
        ax.set_zlabel("z (body)")
        ax.set_xlim(self._bounds["x"])
        ax.set_ylim(self._bounds["y"])
        ax.set_zlim(self._bounds["z"])
        xr = self._bounds["x"][1] - self._bounds["x"][0]
        yr = self._bounds["y"][1] - self._bounds["y"][0]
        zr = self._bounds["z"][1] - self._bounds["z"][0]
        try:
            ax.set_box_aspect((xr, yr, zr))
        except Exception:
            pass

        ax.set_title(f"{self._title} | frame={frame_idx} | t={t_sec:.2f}s")

        # Non-blocking redraw
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        plt.ioff()
        try:
            plt.close(self._fig)
        except Exception:
            pass


@configclass
class RetargetSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot: ArticulationCfg = None


def smooth_signal(x: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1 or x.shape[0] < 3:
        return x.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    xpad = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.empty_like(x, dtype=np.float32)
    for idx in np.ndindex(x.shape[1:]):
        out[(slice(None),) + idx] = np.convolve(xpad[(slice(None),) + idx], kernel, mode="valid")
    return out.astype(np.float32)

def build_scaled_targets(root_pos: np.ndarray,
                         root_rot: np.ndarray,
                         foot_pos: np.ndarray,
                         meta: dict,
                         scale: dict,
                         height_offset: float,
                         lateral_mode: str = "pin") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (scaled_root_pos, root_rot, scaled_foot_pos_world, foot_local_scaled).

    `foot_local_scaled` is expressed in the *original* reference root frame
    (from the un-modified root_pos / root_rot) so that feet motion stays
    relative to the dog's body even when the robot base is pinned.

    lateral_mode:
        "pin" — body-frame y of each foot target is a constant
                +-stance_width/2 (FL,BL: +y / FR,BR: -y). The fore-aft (x)
                and vertical (z) gait signal comes from clean pixel axes;
                the lateral axis comes from monocular depth whose noise is
                comparable to the true left-right separation (~2 sigma), so
                after scaling it swings targets across the body midline.
        "raw" — keep the depth-derived lateral motion (scaled).
    """
    root_rot = ensure_quaternion_continuity(root_rot)

    # Feet relative to the ORIGINAL reference root frame.
    foot_local_ref = world_to_local_points(root_pos, root_rot, foot_pos)

    foot_local_scaled = foot_local_ref.copy()
    foot_local_scaled[..., 0] *= scale["x"]
    foot_local_scaled[..., 1] *= scale["y"]
    foot_local_scaled[..., 2] *= scale["z"]
    if lateral_mode == "pin":
        # Foot order is [FL, FR, BL, BR] (helpers.ROBOT_META foot_bodies order).
        lateral_sign = np.array([+1.0, -1.0, +1.0, -1.0], dtype=np.float32)
        foot_local_scaled[..., 1] = lateral_sign[None, :] * (float(meta["stance_width"]) / 2.0)
        print(f"[Targets] lateral-mode=pin: foot y fixed at +-{meta['stance_width'] / 2.0:.3f} m")
    else:
        print("[Targets] lateral-mode=raw: keeping depth-derived lateral foot motion")
    # Keep targets within leg reach so DLS does not slam the joint limits.
    foot_local_scaled[..., 2] = np.clip(foot_local_scaled[..., 2], -meta["leg_length"] * 1.05, -0.02)

    # Kept for visualization / logging: world-frame feet under the (original)
    # reference root, lifted by the robot's base height + offset.
    scaled_root_pos = root_pos.astype(np.float32).copy()

    scaled_foot_pos_world = local_to_world_points(scaled_root_pos, root_rot, foot_local_scaled)

    return (
        scaled_root_pos.astype(np.float32),
        root_rot.astype(np.float32),
        scaled_foot_pos_world.astype(np.float32),
        foot_local_scaled.astype(np.float32),
    )

def compute_kinematics_from_positions(joint_pos: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    joint_pos_s = smooth_signal(joint_pos, window=7)
    joint_vel = np.gradient(joint_pos_s, dt, axis=0).astype(np.float32)
    joint_vel = smooth_signal(joint_vel, window=7)
    joint_acc = np.gradient(joint_vel, dt, axis=0).astype(np.float32)
    joint_acc = smooth_signal(joint_acc, window=7)
    return joint_vel, joint_acc


def to_training_joint_format(joint_pos_isaac: np.ndarray, isaac_default: np.ndarray) -> np.ndarray:
    raw_isaac_export = (joint_pos_isaac - isaac_default[None, :]) / SIGN_FLIP[None, :]
    return raw_isaac_export[:, ISAAC_TO_PYB].astype(np.float32)


def run_targeting(args) -> None:
    device = args.device
    num_envs = args.num_envs
    if num_envs != 1:
        raise ValueError("Retarget export currently expects --num_envs 1.")

    print("\n[Retarget] Initializing...")
    ref = load_reference_motion(args.ref_motion, args.fps)
    meta = ROBOT_META[args.robot]
    scale = compute_scale_factors(ref, meta)
    
    scaled_root_pos, scaled_root_rot, scaled_foot_pos_world, scaled_foot_pos_local = build_scaled_targets(
        ref.root_pos, ref.root_rot, ref.foot_pos, meta, scale, args.height_offset,
        lateral_mode=args.lateral_mode,
    )
    # NOTE: scaled_foot_pos_local is in the ORIGINAL reference root frame,
    # so the feet move relative to the (pinned) robot body exactly as they
    # moved relative to the dog's body.

    # --- Posture mode (non-locomotion clips: bow, sit, ...): tilt + lower the base
    # PER-FRAME by the torso pitch, re-expressing the foot targets in each frame's
    # tilted frame, so stand<->bow<->stand transitions come through and the legs
    # configure correctly. Default (no --posture) = constant level base = locomotion. ---
    Tn = scaled_foot_pos_local.shape[0]
    posture_base_pos = np.tile(np.array([0.0, 0.0, 0.5], dtype=np.float32), (Tn, 1))
    posture_base_rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (Tn, 1))
    if args.posture:
        if args.posture_pitch is not None:                        # constant override (deg)
            pitch_rad = np.full(Tn, np.radians(float(args.posture_pitch)), dtype=np.float32)
        elif ref.torso_pitch is not None:                         # per-frame spine pitch (preferred)
            pitch_rad = np.asarray(ref.torso_pitch, dtype=np.float32).reshape(-1)[:Tn]
        elif ref.thigh_pos is not None:                           # legacy constant thigh fallback
            tp = np.asarray(ref.thigh_pos); fr = tp[:, :2, :].mean(1); hd = tp[:, 2:, :].mean(1)
            pitch_rad = np.full(Tn, float(np.arctan2(
                (fr[:, 2] - hd[:, 2]).mean(), np.abs(fr[:, 0] - hd[:, 0]).mean())), dtype=np.float32)
        else:
            raise ValueError("--posture needs torso_pitch (or thigh_pos) in the ref, or --posture-pitch")
        # clamp spurious nose-up (head-lift) and cap nose-down (steep bow -> unreachable front feet)
        pitch_rad = np.minimum(pitch_rad, np.radians(float(args.posture_max_noseup)))
        pitch_rad = np.maximum(pitch_rad, -np.radians(float(args.posture_max_nosedown))).astype(np.float32)
        # per-frame tilt about the body lateral (y) axis: + = nose-up, - = nose-down (bow)
        half = pitch_rad / 2.0
        posture_base_rot = np.stack(
            [np.cos(half), np.zeros(Tn), -np.sin(half), np.zeros(Tn)], axis=1).astype(np.float32)
        posture_base_pos = np.tile(np.array([0.0, 0.0, float(args.posture_height)], dtype=np.float32), (Tn, 1))
        # WORLD foot targets (rotation cancels: world = base + level-frame foot). Optionally
        # stretch the FRONT feet forward + down to the ground during the bow so the front legs
        # reach out in front instead of tucking. Then re-express in each frame's tilted base.
        world_foot = posture_base_pos[:, None, :] + scaled_foot_pos_local       # (T,4,3)
        if args.posture_front_stretch > 0.0:
            bow = np.clip(-pitch_rad / np.radians(12.0), 0.0, 1.0)              # (T,) 0 stand -> 1 at full (capped) bow
            for f in (0, 1):                                                    # LF, RF
                world_foot[:, f, 0] += bow * float(args.posture_front_stretch)  # extend forward (world)
                world_foot[:, f, 2] = (1.0 - bow) * world_foot[:, f, 2] + bow * 0.02   # lower to ground
            world_foot[:, :2, 2] = np.maximum(world_foot[:, :2, 2], 0.02)   # keep FRONT feet >= ground (don't touch hind)
        rel = world_foot - posture_base_pos[:, None, :]
        q_inv = quat_conjugate_np(posture_base_rot)               # (T,4)
        for f in range(4):
            scaled_foot_pos_local[:, f, :] = quat_apply_np(q_inv, rel[:, f, :])
        # denoise (moving average) + clamp each target into a reachable shell so IK doesn't thrash
        sw = 9
        k = np.ones(sw) / sw
        sm = scaled_foot_pos_local.copy()
        for fi in range(4):
            for ax in range(3):
                col = scaled_foot_pos_local[:, fi, ax]
                sm[:, fi, ax] = np.convolve(np.pad(col, (sw // 2, sw // 2), "edge"), k, "valid")[:len(col)]
        reach = np.linalg.norm(sm, axis=2, keepdims=True)
        scaled_foot_pos_local = (sm * np.minimum(1.0, args.posture_reach / np.maximum(reach, 1e-6))).astype(np.float32)
        _pd = np.degrees(pitch_rad)
        print(f"[Posture] per-frame tilt {_pd.min():+.1f}..{_pd.max():+.1f} deg, base {args.posture_height:.2f} m, "
              f"reach {args.posture_reach:.2f} m, front-stretch {args.posture_front_stretch:.2f} m")

    sim_cfg = sim_utils.SimulationCfg(
        dt=args.physics_dt,
        render_interval=1 if args.visualize else 4,
        device=device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    if args.visualize:
        sim.set_camera_view(eye=[2.5, 2.0, 1.6], target=[0.0, 0.0, 0.45])
        matplotlib.use("TkAgg")

    robot_cfg = get_robot_cfg(args.robot)
    scene_cfg = RetargetSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.robot = robot_cfg
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print(f"\n[INFO] Setup Complete...")

    robot: Articulation = scene["robot"]
    scene.update(args.physics_dt)
    robot.update(args.physics_dt)

    if args.visualize:
        print(f"[INFO] Set up foot markers for visualization")
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        foot_marker_cfg = frame_marker_cfg.replace(prim_path="/Visuals/foot_targets")
        foot_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        foot_markers = VisualizationMarkers(foot_marker_cfg)
    else:
        foot_markers = None

    retargeter = QuadrupedRetargeter(
        robot=robot, scene=scene, sim=sim, meta=meta, device=device,
        ik_iterations=args.ik_iterations, ik_damping=args.ik_damping,
        physics_dt=args.physics_dt, num_envs=num_envs,
        render=args.visualize,
        relax_front_knees=False,   # relaxing didn't switch the IK branch; X-config is fine
    )

    T = ref.root_pos.shape[0]
    isaac_joint_names = list(robot.joint_names)
    isaac_default = retargeter.default_qpos[0].detach().cpu().numpy().astype(np.float32)

    # Export the FIXED base pose, broadcast across all T frames, so downstream
    # consumers see a consistent "pinned in air" trajectory matching simulation.
    # posture_base_* are already per-frame (T, ...); for locomotion they are the tiled
    # constant level base. Pass them through as the per-frame base trajectory.
    out_root_pos = posture_base_pos.astype(np.float32)
    out_root_rot = posture_base_rot.astype(np.float32)
    
    out_joint_pos = np.zeros((T, len(isaac_joint_names)), dtype=np.float32)
    out_foot_pos = np.zeros((T, 4, 3), dtype=np.float32)
    # Per-frame IK tracking error: achieved body-frame foot minus target.
    out_foot_err = np.zeros((T, 4, 3), dtype=np.float32)

    # Foot-target conditioning, in order: smooth -> collect -> snap -> shift. All operate on
    # the body-frame targets so the IK below keeps joints/feet consistent.
    if args.smooth_targets_win >= 5:
        from scipy.signal import savgol_filter
        scaled_foot_pos_local = savgol_filter(
            scaled_foot_pos_local, args.smooth_targets_win, 3, axis=0).astype(np.float32)
        print(f"[Smooth] savgol window {args.smooth_targets_win} on foot targets")

    # Gait collection: shrink the foot-target excursion into the robot's reachable/feasible
    # envelope (lower clearance, shorter stride) while keeping stance feet grounded.
    zmin = np.percentile(scaled_foot_pos_local[:, :, 2], 10, axis=0)  # per-foot stance level
    if args.collect_z < 1.0 or args.collect_x < 1.0:
        base = scaled_foot_pos_local
        xmean = base[:, :, 0].mean(axis=0)                # per-foot mean fore-aft
        sfl = base.copy()
        sfl[:, :, 2] = zmin[None] + args.collect_z * (base[:, :, 2] - zmin[None])
        sfl[:, :, 0] = xmean[None] + args.collect_x * (base[:, :, 0] - xmean[None])
        scaled_foot_pos_local = sfl.astype(np.float32)
        print(f"[Collect] clearance x{args.collect_z}, stride x{args.collect_x}")

    if args.snap_z > 0.0:  # collection keeps zmin fixed, so the pre-collect stance level holds
        z = scaled_foot_pos_local[:, :, 2]
        near = z < zmin[None] + args.snap_z
        scaled_foot_pos_local[:, :, 2] = np.where(near, zmin[None], z)
        print(f"[SnapZ] {near.mean() * 100:.0f}% of foot-target frames snapped to stance level")

    if args.x_shift_front != 0.0 or args.x_shift_hind != 0.0:
        scaled_foot_pos_local = scaled_foot_pos_local.copy()
        scaled_foot_pos_local[:, 0:2, 0] += args.x_shift_front   # feet order: LF RF LH RH
        scaled_foot_pos_local[:, 2:4, 0] += args.x_shift_hind
        print(f"[XShift] front {args.x_shift_front:+.3f}m, hind {args.x_shift_hind:+.3f}m")

    if args.synth_bow:
        # Rebuild the trajectory as a synthesized stand->bow->stand with a front-leg step,
        # solved by the same IK loop below so joints and foot targets stay consistent.
        fps = 1.0 / ref.dt
        rr_ref = np.asarray(ref.root_rot)
        pr = np.arcsin(np.clip(2.0 * (rr_ref[:, 0] * rr_ref[:, 2] - rr_ref[:, 3] * rr_ref[:, 1]), -1.0, 1.0))
        sidx = int(np.argmin(np.abs(pr)))                       # most-level frame = standing stance
        nominal_local = scaled_foot_pos_local[sidx].copy()      # (4,3) standing feet in the base frame
        for li, ri in [(0, 1), (2, 3)]:                         # symmetrize L/R (front pair, hind pair)
            mx = 0.5 * (nominal_local[li, 0] + nominal_local[ri, 0])
            mz = 0.5 * (nominal_local[li, 2] + nominal_local[ri, 2])
            my = 0.5 * (abs(nominal_local[li, 1]) + abs(nominal_local[ri, 1]))
            for j in (li, ri):
                nominal_local[j, 0] = mx
                nominal_local[j, 2] = mz
                nominal_local[j, 1] = np.sign(nominal_local[j, 1]) * my
        bz = float(args.synth_bz)
        nominal_local[:, 2] = -bz     # rest all standing feet on the ground (world z = 0) for base height bz
        ss = lambda tt: tt * tt * (3.0 - 2.0 * tt)
        Ns, Nt, Nh = [max(1, int(round(v * fps))) for v in (args.synth_stand, args.synth_trans, args.synth_hold)]
        a = np.concatenate([np.zeros(Ns), ss(np.linspace(0, 1, Nt)), np.ones(Nh),
                            ss(np.linspace(1, 0, Nt)), np.zeros(Ns)]).astype(np.float32)
        T = len(a)
        bump = (4.0 * a * (1.0 - a)).astype(np.float32)         # 0 at the holds, 1 mid-transition
        # World foot targets in a level frame: hind planted, front feet step forward + lift.
        wf = np.tile((nominal_local + np.array([0.0, 0.0, bz], np.float32))[None], (T, 1, 1)).astype(np.float32)
        for f in (0, 1):                                        # LF, RF: step forward with the tilt, lift mid-step
            wf[:, f, 0] += a * float(args.synth_fwd)
            wf[:, f, 2] += bump * float(args.synth_lift)
        half = (-a * np.radians(float(args.synth_tilt))) / 2.0  # nose-down base tilt schedule
        out_root_rot = np.stack([np.cos(half), np.zeros(T), -np.sin(half), np.zeros(T)], axis=1).astype(np.float32)
        out_root_pos = np.tile(np.array([0.0, 0.0, bz], np.float32), (T, 1))
        scaled_foot_pos_local = world_to_local_points(out_root_pos, out_root_rot, wf).astype(np.float32)
        out_joint_pos = np.zeros((T, len(isaac_joint_names)), dtype=np.float32)
        out_foot_pos = np.zeros((T, 4, 3), dtype=np.float32)
        out_foot_err = np.zeros((T, 4, 3), dtype=np.float32)
        print(f"[SynthBow] {T} frames ({T / fps:.1f}s) | stand {args.synth_stand}/trans {args.synth_trans}/"
              f"hold {args.synth_hold}s | front step +{args.synth_fwd}m lift {args.synth_lift}m "
              f"tilt {args.synth_tilt}deg base_z {bz} (ref stand frame {sidx})")

    skeleton_plotter = None
    if args.plot_skeleton:
        skeleton_plotter = LiveSkeletonPlotter(
            foot_pos_local_all=scaled_foot_pos_local,
            foot_names=meta["foot_bodies"],
        )

    
    print(f"[Retarget] Processing {T} frames on {device}...")
    start = time.time()
    frames_done = 0
    try:
        for t in range(T):
            jpos, _ = retargeter.solve_frame(
                root_pos=out_root_pos[t],
                root_rot=out_root_rot[t],
                foot_pos_local=scaled_foot_pos_local[t],
            )
            out_joint_pos[t] = jpos
            robot.update(args.physics_dt)
            for leg_i, foot_id in enumerate(retargeter.foot_body_ids):
                achieved_w = robot.data.body_pos_w[0, foot_id].detach().cpu().numpy()
                out_foot_pos[t, leg_i] = achieved_w
                # Foot error in the (possibly tilted) base frame:
                # R^-1 (achieved_world - base_pos) - target.
                ee_local = world_to_local_points(out_root_pos[t], out_root_rot[t], achieved_w)
                out_foot_err[t, leg_i] = ee_local - scaled_foot_pos_local[t, leg_i]

            if foot_markers is not None:
                # Targets in world frame under the pinned base: translate by
                # fixed_root_pos and rotate by identity -> just add the offset.
                world_targets = scaled_foot_pos_local[t] + np.array([0.0, 0.0, 0.5], dtype=np.float32)
                foot_markers.visualize(
                    translations=torch.tensor(world_targets, device=device, dtype=torch.float32)
                )
                sim.render()

            if skeleton_plotter is not None:
                skeleton_plotter.update(
                    root_world_t=out_root_pos[t],
                    foot_local_t=scaled_foot_pos_local[t],
                    frame_idx=t,
                    t_sec=t * ref.dt,
                )

            frames_done = t + 1
            if frames_done % 25 == 0 or t == T - 1:
                elapsed = max(time.time() - start, 1e-6)
                rate = frames_done / elapsed
                print(f"  frame {frames_done:5d}/{T} | {rate:.1f} frames/s")
    except KeyboardInterrupt:
        print(f"\n[Retarget] Interrupted at frame {frames_done}/{T} — saving the frames solved so far.")
    finally:
        if skeleton_plotter is not None:
            skeleton_plotter.close()

    if frames_done == 0:
        raise RuntimeError("[Retarget] No frames were solved; nothing to save.")
    if frames_done < T:
        T = frames_done
        out_root_pos = out_root_pos[:T]
        out_root_rot = out_root_rot[:T]
        out_joint_pos = out_joint_pos[:T]
        out_foot_pos = out_foot_pos[:T]
        out_foot_err = out_foot_err[:T]
        scaled_foot_pos_local = scaled_foot_pos_local[:T]

    # Pad the ends with the first/last (standing) frame so the policy sees a steady stand
    # before and after the motion (posture clips that start/end standing).
    if args.posture and args.posture_pad_standing > 0.0:
        npad = int(round(args.posture_pad_standing / ref.dt))
        def _pad(a):
            return np.concatenate([np.repeat(a[:1], npad, 0), a, np.repeat(a[-1:], npad, 0)], axis=0)
        out_root_pos = _pad(out_root_pos); out_root_rot = _pad(out_root_rot)
        out_joint_pos = _pad(out_joint_pos); out_foot_pos = _pad(out_foot_pos)
        out_foot_err = _pad(out_foot_err); scaled_foot_pos_local = _pad(scaled_foot_pos_local)
        T = out_joint_pos.shape[0]
        print(f"[Posture] padded {npad} frames ({args.posture_pad_standing:.1f}s) standing each end -> {T} frames")

    # IK quality summary: per-leg foot tracking error and joint jitter.
    err_norm = np.linalg.norm(out_foot_err, axis=-1)  # (T, 4)
    print("[Retarget] foot tracking error per leg [cm]: "
          + ", ".join(f"{n}: mean {err_norm[:, i].mean() * 100:.1f} / max {err_norm[:, i].max() * 100:.1f}"
                      for i, n in enumerate(meta["foot_bodies"])))
    dq = np.abs(np.diff(out_joint_pos, axis=0))
    worst = int(np.argmax(dq.max(axis=0)))
    print(f"[Retarget] per-frame joint jitter: mean {dq.mean():.4f} rad, "
          f"worst joint {isaac_joint_names[worst]} max {dq[:, worst].max():.4f} rad "
          f"(run diagnose_retarget.py on the output for full plots)")

    out_joint_vel, out_joint_acc = compute_kinematics_from_positions(out_joint_pos, ref.dt)
    joint_pos_for_training = to_training_joint_format(out_joint_pos, isaac_default)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        root_pos=out_root_pos,
        root_rot=out_root_rot,
        joint_pos_isaac=out_joint_pos,
        joint_vel=out_joint_vel,
        joint_acc=out_joint_acc,
        foot_pos=out_foot_pos,
        foot_pos_local=scaled_foot_pos_local.astype(np.float32),
        foot_err=out_foot_err,
        joint_pos_limits=robot.data.soft_joint_pos_limits[0].detach().cpu().numpy().astype(np.float32),
        frame_duration=np.array(ref.dt, dtype=np.float32),
        joint_pos=joint_pos_for_training,
        fps=np.array(args.fps, dtype=np.float32),
        robot=np.array([args.robot]),
        joint_names=np.array(isaac_joint_names),
        foot_names=np.array(meta["foot_bodies"]),
        dt=np.array(ref.dt, dtype=np.float32),
        isaac_default_joint_pos=isaac_default,
        scale_x=np.array(scale["x"], dtype=np.float32),
        scale_y=np.array(scale["y"], dtype=np.float32),
        scale_z=np.array(scale["z"], dtype=np.float32),
    )
    print(f"[Retarget] Saved retargeted trajectory -> '{out_path}'")


if __name__ == "__main__":
    code = 0
    try:
        run_targeting(args)
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    # NOTE: we deliberately do NOT call simulation_app.close() here. Isaac Sim's
    # app teardown HANGS the interpreter (close() never returns) *after* the
    # trajectory npz is already written inside run_targeting — that hang is
    # exactly what blocks a batch driver waiting on this process. The output is
    # fully saved and the OS reclaims the GPU/file handles on exit, so a graceful
    # close buys nothing. Flush and force-terminate with the real exit code so
    # callers (batch_retarget.py) advance to the next clip yet still see failures.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)