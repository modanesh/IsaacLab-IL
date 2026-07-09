#!/usr/bin/env python3
"""Joint-space polish for a retargeted motion: SMOOTH the jerk + optionally grow
an under-swinging joint's amplitude, then recompute the feet through the robot's
exact forward kinematics so joints and foot_pos_local stay consistent.

Why joint space (not foot targets / retarget --collect): scaling a foot TARGET
runs the inverse kinematics into ANYmal's reach wall (the knee snaps -> jerk, as
the greyhound-gallop v3 rebuild showed). Scaling a JOINT can't violate reach —
the foot simply follows FK wherever the leg reaches — and it directly shapes what
you see and what the imitation reward tracks (joint_pos + foot_pos_local).

Used to turn the collected greyhound gallop into a smooth, full hip-sweep gait:
its knee jerk was ~2x the trot's and the hind hips barely swung (HFE ptp
0.5-0.75 vs the trot's 1.2). This smooths to trot-like jerk and lets the hind
hips swing further, without leaving ANYmal's envelope.

Pipeline: retarget -> root recovery -> polish_motion.py -> root recovery again.

Usage (from IsaacLab/):
  python scripts/retargeting/polish_motion.py --in <motion>.npz --out <out>.npz \
      --smooth-joints-win 9 --boost-hind-hfe 1.3 [--boost-front-hfe 1.0]
"""
import argparse

import numpy as np

from isaaclab.app import AppLauncher

_p = argparse.ArgumentParser()
_p.add_argument("--in", dest="inp", required=True)
_p.add_argument("--out", required=True)
_p.add_argument("--robot", default="anymal_d")
_p.add_argument("--physics_dt", type=float, default=1.0 / 120.0)
_p.add_argument("--smooth-joints-win", type=int, default=9,
                help="odd Savitzky-Golay window (frames) low-passing the joint angles; 0 = off.")
_p.add_argument("--boost-hind-hfe", type=float, default=1.0,
                help="scale hind HFE (hip) amplitude about its per-joint mean; >1 = bigger hip sweep.")
_p.add_argument("--boost-front-hfe", type=float, default=1.0, help="scale front HFE amplitude.")
_p.add_argument("--boost-hind-kfe", type=float, default=1.0, help="scale hind KFE (knee) amplitude.")
AppLauncher.add_app_launcher_args(_p)
args = _p.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
import torch  # noqa: E402

from helpers import get_robot_cfg, ROBOT_META  # noqa: E402

BASE_Z = 0.5  # level locomotion base height used by the retarget FK convention


@configclass
class PolishSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)))
    dome_light = AssetBaseCfg(prim_path="/World/Light",
                              spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot: ArticulationCfg = None


def main():
    d = dict(np.load(args.inp, allow_pickle=True))
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    jp = d["joint_pos_isaac"].astype(np.float64).copy()          # (T,12)
    names = [str(n) for n in d["joint_names"]]
    T = jp.shape[0]

    # ---- joint-space edits: amplitude boost about per-joint mean, then smooth ----
    mean = jp.mean(0, keepdims=True)
    for i, n in enumerate(names):
        f_or_h = "F_" in n[:3] or n.startswith(("LF", "RF"))       # front vs hind by name
        is_front = n.startswith(("LF", "RF"))
        if "HFE" in n:
            g = args.boost_front_hfe if is_front else args.boost_hind_hfe
        elif "KFE" in n and not is_front:
            g = args.boost_hind_kfe
        else:
            g = 1.0
        if g != 1.0:
            jp[:, i] = mean[0, i] + g * (jp[:, i] - mean[0, i])

    if args.smooth_joints_win >= 5 and T >= args.smooth_joints_win:
        w = args.smooth_joints_win + (args.smooth_joints_win % 2 == 0)
        jp = savgol_filter(jp, w, 3, axis=0)
        print(f"[Polish] savgol window {w} on joint angles")

    # clamp to the robot's soft joint limits stored in the clip
    if "joint_pos_limits" in d:
        lim = d["joint_pos_limits"]                               # (12,2)
        jp = np.clip(jp, lim[:, 0][None], lim[:, 1][None])

    # ---- Isaac FK: write the (edited) joints under a level base, read the feet ----
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.physics_dt, device=args.device))
    meta = ROBOT_META[args.robot]
    scene_cfg = PolishSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = get_robot_cfg(args.robot)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot: Articulation = scene["robot"]
    dev = robot.device

    # map the clip's joint order -> the articulation's joint order
    art_names = list(robot.joint_names)
    clip_to_art = [art_names.index(n) for n in names]
    foot_ids = [list(robot.body_names).index(n) for n in meta["foot_bodies"]]  # LF RF LH RH

    root = robot.data.default_root_state.clone()
    root[:, :3] = torch.tensor([0.0, 0.0, BASE_Z], device=dev)
    root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    root[:, 7:] = 0.0

    foot_world = np.zeros((T, 4, 3), np.float32)
    q_art = np.zeros((T, len(art_names)), np.float32)
    q_art[:, clip_to_art] = jp
    for t in range(T):
        qj = torch.tensor(q_art[t:t + 1], dtype=torch.float32, device=dev)
        robot.write_root_pose_to_sim(root[:, :7]); robot.write_root_velocity_to_sim(root[:, 7:])
        robot.write_joint_state_to_sim(qj, torch.zeros_like(qj))
        scene.write_data_to_sim(); sim.step(render=False)
        scene.update(args.physics_dt); robot.update(args.physics_dt)
        foot_world[t] = robot.data.body_pos_w[0, foot_ids, :].cpu().numpy()

    foot_local = foot_world - np.array([0.0, 0.0, BASE_Z], np.float32)  # level base -> local = world - base

    # ---- re-derive kinematics and save ----
    jp32 = jp.astype(np.float32)
    d["joint_pos_isaac"] = jp32
    d["joint_pos"] = jp32
    d["joint_vel"] = (np.gradient(jp32, axis=0) * fps).astype(np.float32)
    d["joint_acc"] = (np.gradient(d["joint_vel"], axis=0) * fps).astype(np.float32)
    d["foot_pos"] = foot_world.astype(np.float32)
    d["foot_pos_local"] = foot_local.astype(np.float32)
    np.savez(args.out, **d)

    # report
    def ptp(sub): return {names[i]: round(float(np.ptp(jp[:, i])), 2) for i in range(12) if sub in names[i]}
    jerk = np.sqrt((np.diff(jp32, axis=0, n=2) ** 2).mean(0))
    print(f"[Polish] saved -> {args.out}")
    print(f"[Polish] HFE ptp (hip sweep): {ptp('HFE')}")
    print(f"[Polish] KFE ptp (knee):      {ptp('KFE')}")
    print(f"[Polish] joint jerkRMS mean {jerk.mean():.3f} (worst {jerk.max():.3f} @ {names[int(jerk.argmax())]})")
    print(f"[Polish] foot_local z range per foot: "
          f"{[(round(float(foot_local[:, i, 2].min()), 2), round(float(foot_local[:, i, 2].max()), 2)) for i in range(4)]}")


if __name__ == "__main__":
    main()
    simulation_app.close()
