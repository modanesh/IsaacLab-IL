import argparse
from isaaclab.app import AppLauncher

import matplotlib.pyplot as plt
import os
import sys
import uuid
import yaml
from tqdm import tqdm, trange


class _Tee:
    """Duplicate a text stream to a file so console output is also saved to disk.
    Used to capture the full training / eval log next to the model checkpoint."""
    def __init__(self, stream, fileobj):
        self.stream, self.file = stream, fileobj

    def write(self, data):
        self.stream.write(data)
        try:
            self.file.write(data)
            self.file.flush()
        except Exception:
            pass

    def flush(self):
        self.stream.flush()
        try:
            self.file.flush()
        except Exception:
            pass


def tee_stdout_to(log_path):
    """Redirect stdout (and stderr) so everything printed is also appended to log_path."""
    f = open(log_path, "w")
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    print(f"[log] capturing console output to {log_path}")


def _args_to_dict(args):
    """argparse Namespace -> a YAML-serialisable dict (basic types only)."""
    return {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
            for k, v in vars(args).items()}


def save_run_yaml(path, config: dict, results: dict | None = None):
    """Write a run's config (and optional results) to a YAML file for the record."""
    out = {"config": config}
    if results is not None:
        out["results"] = results
    with open(path, "w") as f:
        yaml.safe_dump(out, f, default_flow_style=False, sort_keys=True)
    print(f"[cfg] wrote {path}")

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Anymal-D-v0")
parser.add_argument("--motion", type=str, default=None,
                    help="single retargeted .npz (bc/rl/eval, or single-clip amp)")
parser.add_argument("--motion-list", type=str, default=None,
                    help="text file of .npz paths (one per line) AMP trains on the whole "
                         "library (the discriminator samples transitions from all clips). "
                         "Required for multi-clip AMP; for bc/rl the first clip is used.")
parser.add_argument("--mode", type=str, default="bc", choices=["bc", "rl", "amp", "eval", "launch_eval"],
                    help="bc=behavioral cloning, rl=PPO joint tracking, "
                         "amp=Adversarial Motion Prior (pure imitation, no hand-designed reward), "
                         "eval=evaluate checkpoint, launch_eval=evaluate the STANDSTILL launch "
                         "(init at rest, spin the gait up)")
parser.add_argument("--num_envs", type=int, default=64,
                    help="Parallel envs for data collection / RL training. Use 1 for eval.")
parser.add_argument("--seed", type=int, default=-1,
                    help="RNG seed for reproducible multi-seed studies (sets numpy + torch). "
                         "-1 (default) leaves seeding to the OS, matching how the released policies "
                         "were trained; a >=0 value only affects new runs.")

# BC args
parser.add_argument("--bc_epochs", type=int, default=200)
parser.add_argument("--bc_batch", type=int, default=1024)
parser.add_argument("--bc_lr", type=float, default=3e-4)
parser.add_argument("--bc_collect_s", type=float, default=60.0,
                    help="Seconds of motion to collect per env for BC dataset.")

# AMP args (Adversarial Motion Prior pure imitation, learned style reward)
parser.add_argument("--amp_iters", type=int, default=3000)
parser.add_argument("--amp_steps", type=int, default=24, help="PPO rollout horizon for AMP.")
parser.add_argument("--amp_lr", type=float, default=1e-4, help="Policy/value learning rate.")
parser.add_argument("--amp_disc_lr", type=float, default=1e-4, help="Discriminator learning rate.")
parser.add_argument("--amp_disc_epochs", type=int, default=1,
                    help="Discriminator update epochs per PPO iteration (1 = least likely to saturate).")
parser.add_argument("--amp_grad_penalty", type=float, default=10.0,
                    help="Gradient penalty on the discriminator's real-data inputs. "
                         "Raise (e.g. 20) if disc_acc pins at 1.0/1.0.")
parser.add_argument("--amp_disc_wd", type=float, default=1e-3,
                    help="Discriminator weight decay (AdamW). Raise to fight saturation.")
parser.add_argument("--amp_logit_reg", type=float, default=0.01,
                    help="Penalty on discriminator output magnitude (anti-saturation).")
parser.add_argument("--amp_reward_scale", type=float, default=2.0,
                    help="Scale on the learned style reward.")
parser.add_argument("--amp_action_rate_w", type=float, default=0.0,
                    help="Weight of the action-rate SMOOTHNESS penalty (per-step "
                         "mean (a_t - a_{t-1})^2 subtracted from the style reward). "
                         "Curbs the jitter-in-place AMP local optimum. Try 1-5; "
                         "raise if jittery, lower if the robot freezes / style reward craters.")
parser.add_argument("--amp_window", type=int, default=2,
                    help="Number of consecutive frames the discriminator sees per sample "
                         "(2 = a plain transition). A longer window (e.g. 6-10, ~0.1-0.2s at "
                         "50Hz) makes MOTION OVER TIME visible, so a standing-still policy is "
                         "obviously fake -> stops the 'smooth standing' degenerate solution.")
parser.add_argument("--amp_task_vel_w", type=float, default=0.0,
                    help="Weight of a positive velocity-command TASK reward added to the AMP "
                         "style reward (0=pure imitation). Specifies the GOAL (go at commanded "
                         "speed/heading), NOT the gait the gait stays learned from video. "
                         "Needed for a COMMANDABLE forward walker. Try 0.5-1.0.")

# RL args
parser.add_argument("--rl_iters", type=int, default=2000)
parser.add_argument("--rl_steps", type=int, default=24,
                    help="Rollout steps per PPO iteration (horizon).")
parser.add_argument("--rl_lr", type=float, default=1e-4)
parser.add_argument("--rl_sigma", type=float, default=0.25,
                    help="Sigma for joint-tracking reward: exp(-||dq||^2 / sigma^2).")
parser.add_argument("--rl_vel_weight", type=float, default=0.5,
                    help="Weight for forward velocity reward component.")

# DeepMimic-style multi-term tracking (RL mode): joint + foot + root.
# The joint term alone reproduces angles but permits a low-clearance SHUFFLE;
# the foot term forces body-frame foot clearance+stride to match the reference,
# and the root term tracks recovered forward velocity + base height (kills the
# in-place / crouch optima). Set --rl_foot_w 0 --rl_root_w 0 to recover the old
# joint-only behaviour. Weights ideally sum to ~1 (each term is in (0,1]).
parser.add_argument("--rl_joint_w", type=float, default=0.5,
                    help="Weight of the joint-angle tracking term.")
parser.add_argument("--rl_foot_w", type=float, default=0.3,
                    help="Weight of the body-frame FOOT-position tracking term "
                         "exp(-||p_foot - p_foot_ref||^2 / foot_sigma^2). This is what "
                         "forbids the low-clearance shuffle. 0 disables (joint-only).")
parser.add_argument("--rl_root_w", type=float, default=0.2,
                    help="Weight of the ROOT term: fore-aft/lateral body-frame velocity "
                         "(from recover_root_translation.py) + base height. Drives real "
                         "forward locomotion and forbids crouch. 0 disables.")
parser.add_argument("--rl_foot_sigma", type=float, default=0.20,
                    help="Sigma (m) for the foot-position tracking term.")
parser.add_argument("--rl_foot_z_weight", type=float, default=1.0,
                    help="Weight on the VERTICAL (clearance) component of the foot error. "
                         ">1 stops the fore-aft stride from dominating the foot term, which "
                         "otherwise lets the policy drag/over-lift feet while matching stride. "
                         "Try 3-5 if the fidelity diagnostic shows feet dragging.")
parser.add_argument("--rl_root_sigma", type=float, default=0.25,
                    help="Sigma for the root velocity+height term.")
parser.add_argument("--rl_term_joint", type=float, default=2.0,
                    help="DeepMimic early termination: end the episode when ||q-q_ref|| "
                         "(rad) exceeds this, so crouch/shuffle/stand can't be ridden out. "
                         "0 disables.")
parser.add_argument("--rl_term_foot", type=float, default=0.5,
                    help="Early termination threshold on ||p_foot-p_foot_ref|| (m). 0 disables.")
parser.add_argument("--rl_entropy_coef", type=float, default=0.0,
                    help="PPO entropy bonus weight. 0 (default) for a TRACKING task: a "
                         "positive value pushes the action log-std up unboundedly "
                         "(entropy explosion -> jittery actions). Was 0.01.")
parser.add_argument("--rl_action_rate_w", type=float, default=0.0,
                    help="Smoothness prior: subtract w * mean((a_t - a_{t-1})^2) from the "
                         "reward to penalise jittery action changes. Try 0.2-0.5. Keep modest "
                         "(a real swing needs large smooth action changes; too high freezes "
                         "the legs).")
parser.add_argument("--rl_log_std_min", type=float, default=-4.0,
                    help="Lower bound on the policy action log-std. The default -4.0 lets the std "
                         "collapse to ~0.018; on a hard/OOD task (the from-rest gallop) that makes "
                         "PPO ratios explode (kl -> 1e8, blow-up). Raise to -2.5 (std ~0.08) or "
                         "-2.0 (~0.14) to floor exploration and keep training stable.")

# --- Launch-from-rest curriculum (for deployable standstill starts) --------------
# A dynamic gait (the gallop) is normally reset already moving at the reference
# speed, so the policy never learns to start from a standstill and collapses when
# switched in on the robot. These envs instead reset at REST at a stable stance
# phase, and a per-env ramp spins the gait up: the phase clock advances at
# rate ramp*nominal and the reference velocity target is scaled by ramp, both
# going 0->1 over --rl_launch_s. The deployment controller mirrors this ramp.
parser.add_argument("--rl_launch_frac", type=float, default=0.0,
                    help="Fraction of env resets that start from REST (zero velocity at the "
                         "stance phase) and spin the gait up. 0 (default) = classic RSI only. "
                         "Try 0.3-0.5 to teach a standstill launch while retaining the gait.")
parser.add_argument("--rl_launch_s", type=float, default=1.5,
                    help="Duration (s) of the launch ramp: phase rate and the reference "
                         "velocity target ramp 0->1 over this window. Must match the "
                         "deployment launch_ramp_s.")
parser.add_argument("--rl_launch_phase", type=float, default=-1.0,
                    help="Cycle fraction [0,1) to launch from (the stance the robot settles "
                         "onto before spinning up). -1 (default) = auto: the most-grounded "
                         "all-feet-down frame of the clip. Should match the deployment phase_start.")
parser.add_argument("--rl_dr", action="store_true",
                    help="Enable sim2real balance domain randomization on top of the always-on "
                         "startup mass/COM DR: (1) friction SPREAD (Isaac's flat default is a "
                         "degenerate fixed 0.8/0.6, i.e. no spread), (2) mild external PUSHES with "
                         "a short interval so they fire inside short gesture episodes, and (3) a "
                         "per-env ACTUATOR effort-limit scale so the policy stops overfitting the "
                         "nominal ANYdrive strength and learns poses within a weaker torque budget. "
                         "Off by default so every existing per-clip recipe trains identically.")
parser.add_argument("--rl_dr_effort", type=float, nargs=2, default=[0.75, 1.0],
                    metavar=("LO", "HI"),
                    help="Per-env actuator effort-limit scale ~ U(LO,HI) when --rl_dr is set. "
                         "Weaken-biased (<=1) because the observed real failure is torque "
                         "saturation: training with less usable torque yields deployable poses.")
parser.add_argument("--rl_vel_cmd", type=float, default=-999.0,
                    help="NON-IMITATION baseline: pin the env base-velocity command to this constant "
                         "forward speed (m/s) for train AND eval, instead of Isaac's random command "
                         "curriculum. Combine with imitation weights 0 (--rl_joint_w 0 --rl_foot_w 0 "
                         "--rl_root_w 0), --task-reward-weight>0, and early-term off (--rl_term_joint 0 "
                         "--rl_term_foot 0) to train a velocity-tracking controller with no imitation. "
                         "Default -999 = off (normal imitation training).")

# Shared
parser.add_argument("--ckpt", type=str, default=None,
                    help="Checkpoint to load for eval, or BC checkpoint to init RL from.")
parser.add_argument("--bc_ckpt", type=str, default=None,
                    help="BC checkpoint to use as RL initialization.")
parser.add_argument("--save_dir", type=str, default="./checkpoints")
parser.add_argument("--stiffness", type=float, default=200.0)
parser.add_argument("--damping", type=float, default=5.0)
parser.add_argument("--task-reward-weight", type=float, default=0.0,
                    help="Weight of a positive velocity-command tracking reward (exp form) "
                         "added to the imitation reward in RL mode. 0 = pure imitation "
                         "(maximised by stepping in place when the reference clip has no "
                         "root translation). Try 1.0 for forward locomotion in the "
                         "reference style.")
parser.add_argument("--phase-obs", action="store_true", default=False,
                    help="Append [sin, cos] of the gait phase to the observation (obs 48->50). "
                         "The reference action is a function of clip phase, which pure "
                         "proprioception cannot disambiguate strongly recommended for both "
                         "BC and RL. Must be consistent between BC and RL warm-start; eval "
                         "auto-detects it from the checkpoint.")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from collections import deque
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.actuators import ActuatorNetLSTMCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply_inverse

from isaaclab.assets.asset_base import AssetBase

os.makedirs(args_cli.save_dir, exist_ok=True)

ISAAC_DEFAULT = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.4, -0.4, 0.4, -0.4,
    -0.8, 0.8, -0.8, 0.8,
], dtype=np.float32)

PYB_TO_ISAAC = np.array([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11], dtype=int)

SIGN_FLIP = np.ones(12, dtype=np.float32)
SIGN_FLIP[2] = -1.0
SIGN_FLIP[3] = -1.0

OBS_DIM = 48
ACTION_DIM = 12
ACTION_SCALE = 0.5

# --- Knee X-configuration hard clamp -------------------------------------------
# ANYmal-D front knees (KFE) must bend NEGATIVE, hind POSITIVE. With a low/zero
# joint-tracking weight the RL policy has nothing holding the knee branch, so it
# can drive a knee to the wrong (outward) branch physically wrong and unstable
# (the front-right-knee-bends-outward failure). The IK retargeter clamps this in
# helpers.py; the policy had no equivalent. We clamp the KFE *action* so the
# commanded target cannot cross the branch boundary a hard guarantee, reward-
# independent. Joint order (Isaac BFS): KFE = idx 8(LF),9(LH),10(RF),11(RH);
# default KFE front -0.8 / hind +0.8; target = default + ACTION_SCALE*action.
ACTION_CLAMP = 6.0         # |action| bound: keeps joint targets within ~+-3 rad of default
KNEE_EPS = 0.05
KFE_FRONT_IDX = [8, 10]    # LF_KFE, RF_KFE target must stay <= -KNEE_EPS
KFE_HIND_IDX = [9, 11]     # LH_KFE, RH_KFE target must stay >= +KNEE_EPS
_KFE_FRONT_ACT_MAX = (-KNEE_EPS - (-0.8)) / ACTION_SCALE   # = 1.5 at scale 0.5
_KFE_HIND_ACT_MIN = (KNEE_EPS - 0.8) / ACTION_SCALE        # = -1.5 at scale 0.5


def clamp_knee_actions(acts):
    """Bound every action and force the KFE joints onto the X-config branch.

    The +-ACTION_CLAMP bound is a stability failsafe: without it a single
    destabilising PPO update can drive actions to absurd values (~470 was observed),
    which makes the action-rate penalty ~ -44000/step, collapses the return to -2e5,
    and diverges the value function (vf -> 1e23) a catastrophic blow-up after an
    otherwise excellent run. Bounding the action keeps targets sane and the reward
    finite so training stays recoverable."""
    acts = acts.clone().clamp(-ACTION_CLAMP, ACTION_CLAMP)
    acts[:, KFE_FRONT_IDX] = acts[:, KFE_FRONT_IDX].clamp(max=_KFE_FRONT_ACT_MAX)
    acts[:, KFE_HIND_IDX] = acts[:, KFE_HIND_IDX].clamp(min=_KFE_HIND_ACT_MIN)
    return acts


def _clip_joint_pos_isaac(data) -> np.ndarray:
    """Absolute Isaac-order joint positions (T, 12) from a motion clip npz.

    Preferred path (IsaacLab-native): the 'joint_pos_isaac' key exported by
    retarget.py already absolute joint angles in articulation order, used
    as-is (no permutation, no sign flips, no re-centring).

    Legacy fallback: the training-format 'joint_pos' key (PyBullet order,
    SIGN_FLIP convention, e.g. clips converted from external mocap JSON).
    Those carry unknown absolute offsets, so the clip is re-centred on the
    Isaac default pose, as before.
    """
    if "joint_pos_isaac" in data:
        q = np.asarray(data["joint_pos_isaac"], dtype=np.float32)
        print("[Motion] using Isaac-native 'joint_pos_isaac' (no convention conversion)")
        return q
    raw_pyb = np.asarray(data["joint_pos"], dtype=np.float32)
    raw_isaac = raw_pyb[:, PYB_TO_ISAAC]
    clip_mean = raw_isaac.mean(axis=0)
    delta = (raw_isaac - clip_mean) * SIGN_FLIP[None, :]
    print("[Motion] legacy pyb-format 'joint_pos': converted to Isaac order and re-centred on default pose")
    return delta + ISAAC_DEFAULT[None, :]


def load_motion(npz_path: str) -> dict:
    data = np.load(npz_path)
    frame_dt = float(data["frame_duration"])
    joint_pos = _clip_joint_pos_isaac(data)
    T = joint_pos.shape[0]
    duration = T * frame_dt
    print(f"[Motion] {T} frames  dt={frame_dt:.4f}s  duration={duration:.2f}s")
    return {"joint_pos": joint_pos, "frame_dt": frame_dt, "T": T, "duration": duration}


def load_motion_interpolated(npz_path: str) -> dict:
    data = np.load(npz_path)
    frame_dt = float(data["frame_duration"])
    joint_pos = _clip_joint_pos_isaac(data)

    T = joint_pos.shape[0]
    duration = T * frame_dt

    # 1. Append the first frame to the very end to create a closed loop
    joint_pos_looped = np.vstack((joint_pos, joint_pos[0]))
    
    # 2. Create the time array for the keyframes (length T + 1)
    t_frames = np.linspace(0, duration, T + 1)
    
    # 3. Fit the periodic cubic spline across all joints (axis=0)
    cs = CubicSpline(t_frames, joint_pos_looped, axis=0, bc_type='periodic')

    # 3b. DeepMimic targets: body-frame foot positions + root (fore-aft/lateral
    #     velocity, base height), splined the same periodic way. Optional a clip
    #     may predate foot_pos_local / recover_root_translation.py, in which case
    #     the RL tracker falls back to joint-only tracking for the missing term(s).
    clip_extra = {}
    if "foot_pos_local" in data.files:
        fl = np.asarray(data["foot_pos_local"], np.float32).reshape(T, -1)   # (T,12)
        clip_extra["cs_foot"] = CubicSpline(
            t_frames, np.vstack((fl, fl[0])), axis=0, bc_type='periodic')
    if "root_lin_vel" in data.files:
        rlv = np.asarray(data["root_lin_vel"], np.float32).reshape(T, -1)[:, :2]  # body vx,vy
        if "root_pos" in data.files:
            bh = np.asarray(data["root_pos"], np.float32).reshape(T, -1)[:, 2:3]
        else:
            bh = np.full((T, 1), 0.5, np.float32)
        root_tgt = np.concatenate([rlv, bh], axis=1)                          # (T,3)
        clip_extra["cs_root"] = CubicSpline(
            t_frames, np.vstack((root_tgt, root_tgt[0])), axis=0, bc_type='periodic')

    print(f"[Motion] {T} frames  dt={frame_dt:.4f}s  duration={duration:.2f}s"
          f"  | DeepMimic targets: foot={'Y' if 'cs_foot' in clip_extra else 'N'} "
          f"root={'Y' if 'cs_root' in clip_extra else 'N'}")

    # Store the spline object directly in the clip dictionary
    return {
        "joint_pos": joint_pos,
        "frame_dt": frame_dt,
        "T": T,
        "duration": duration,
        "cs": cs,
        **clip_extra,
    }


# def get_joint_target(clip: dict, t: float) -> np.ndarray:
#     jp = clip["joint_pos"]
#     dt = clip["frame_dt"]
#     T = clip["T"]
#     u = (t / dt) % T
#     i0 = int(np.floor(u))
#     i1 = (i0 + 1) % T
#     a = float(u - i0)
#     return (1.0 - a) * jp[i0] + a * jp[i1]


# def get_joint_target_batch(clip: dict, t_arr: np.ndarray) -> np.ndarray:
#     """Vectorized version for N envs.  t_arr: (N,)  Returns: (N, 12)"""
#     jp = clip["joint_pos"]
#     dt = clip["frame_dt"]
#     T = clip["T"]
#     u = (t_arr / dt) % T  # (N,)
#     i0 = np.floor(u).astype(int)  # (N,)
#     i1 = (i0 + 1) % T
#     a = (u - i0)[:, None]  # (N,1)
#     return (1.0 - a) * jp[i0] + a * jp[i1]  # (N,12)


# def get_ref_joint_velocities_batch(clip: dict, t_arr: np.ndarray) -> np.ndarray:
#     jv = np.gradient(clip["joint_pos"], axis=0)
#     dt = clip["frame_dt"]
#     T = clip["T"]
#     u = (t_arr / dt) % T
#     i0 = np.floor(u).astype(int)
#     i1 = (i0 + 1) % T
#     a = (u - i0)[:, None]  # (N,1)
#     return (1.0 - a) * jv[i0] + a * jv[i1]


def get_joint_target_batch(clip: dict, t_arr: np.ndarray) -> np.ndarray:
    """Evaluate the cubic spline for smooth joint positions."""
    # Wrap the time array to stay within the clip duration
    t_wrapped = t_arr % clip["duration"]
    
    # Evaluate the spline at the queried times
    return clip["cs"](t_wrapped).astype(np.float32)

def get_ref_joint_velocities_batch(clip: dict, t_arr: np.ndarray) -> np.ndarray:
    """Evaluate the first derivative of the cubic spline for accurate joint velocities."""
    t_wrapped = t_arr % clip["duration"]

    # The '1' argument tells CubicSpline to return the 1st derivative
    return clip["cs"](t_wrapped, 1).astype(np.float32)


def get_foot_target_batch(clip: dict, t_arr: np.ndarray):
    """Reference body-frame foot positions (N,4,3) at times t_arr, or None if the
    clip has no foot_pos_local (then the RL tracker drops the foot term)."""
    if "cs_foot" not in clip:
        return None
    t = t_arr % clip["duration"]
    return clip["cs_foot"](t).astype(np.float32).reshape(-1, 4, 3)


def get_root_target_batch(clip: dict, t_arr: np.ndarray):
    """Reference root target (N,3) = [vx_body, vy_body, base_height] at t_arr, or
    None if the clip has no recovered root translation."""
    if "cs_root" not in clip:
        return None
    t = t_arr % clip["duration"]
    return clip["cs_root"](t).astype(np.float32)


def phase_features(phase_t: np.ndarray, duration: float, device: str) -> torch.Tensor:
    """[sin, cos] gait-clock features for each env, shape (N, 2)."""
    ph = 2.0 * np.pi * ((phase_t % duration) / duration)
    feat = np.stack([np.sin(ph), np.cos(ph)], axis=-1).astype(np.float32)
    return torch.tensor(feat, dtype=torch.float32, device=device)


def build_policy_obs(obs, phase_t: np.ndarray, clip: dict, device: str, use_phase: bool) -> torch.Tensor:
    """Slice the env observation and optionally append the gait clock."""
    if isinstance(obs, dict):
        obs = obs["policy"]
    obs_t = obs[:, :OBS_DIM].to(device)
    if use_phase:
        obs_t = torch.cat([obs_t, phase_features(phase_t, clip["duration"], device)], dim=-1)
    # Guard: a sim instability can emit a NaN/Inf observation; passing it to the
    # policy produces a NaN action mean and crashes the Normal(). Sanitise so one
    # bad env-step can't kill a multi-hour run (the env's fall-termination will
    # reset that env shortly anyway).
    # Sanitise AND bound: a transient sim blow-up gives huge-but-finite obs (e.g. a
    # 1e35 joint velocity from an exploded env) that nan_to_num does NOT catch, but
    # which still produces a huge/inf policy mean and crashes Normal(). Clamp far
    # beyond any real observation (real |obs| < ~30) so one bad env can't kill a
    # multi-hour run; that env's fall-termination resets it the next step.
    return torch.nan_to_num(obs_t, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-100.0, 100.0)


# Bounds on the policy's action log-std. The upper bound caps exploration noise
# at the init level (sigma=exp(-1.0)=0.37): without it, the entropy bonus pushes
# log_std up unboundedly (observed ent 5->12 => sigma 0.37->0.68 => ~0.34 rad of
# random offset per joint per step with ACTION_SCALE=0.5 == the jittery actions).
LOG_STD_MIN, LOG_STD_MAX = args_cli.rl_log_std_min, -1.0


class MLP(nn.Module):
    """
    Simple MLP policy: obs → action.
    Used for both BC (deterministic) and RL (stochastic via log_std).
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden=(512, 256, 128), stochastic: bool = False):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(in_dim, action_dim)
        self.stochastic = stochastic
        if stochastic:
            self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mean = self.mean(h)
        if self.stochastic:
            std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
            return torch.distributions.Normal(mean, std)
        return mean

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        if self.stochastic:
            dist = self.forward(obs)
            return dist.mean if deterministic else dist.rsample()
        return self.forward(obs)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden=(256, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def make_env(num_envs: int, stiffness: float, damping: float):
    env_cfg = parse_env_cfg(
        task_name=args_cli.task, device="cuda:0", num_envs=num_envs)
    dc_cfg = ActuatorNetLSTMCfg(
            joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
            network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
            saturation_effort=120.0,
            effort_limit=80.0,
            velocity_limit=7.5,
        )
    env_cfg.scene.robot = env_cfg.scene.robot.replace(actuators={"legs": dc_cfg})

    if getattr(args_cli, "rl_dr", False):
        # (1) friction SPREAD -- the flat Anymal-D cfg ships a degenerate fixed
        #     0.8/0.6 (no randomization); widen it so feet see a range of grip.
        pm = env_cfg.events.physics_material.params
        pm["static_friction_range"] = (0.6, 1.0)
        pm["dynamic_friction_range"] = (0.4, 0.8)
        # (2) mild external PUSHES -- flat_env_cfg sets push_robot=None. Rebuild it
        #     with a SHORT interval (so it actually fires inside a ~4 s gesture) and
        #     a gentle magnitude (a hard shove on a 3-leg stance is unrecoverable).
        from isaaclab.managers import EventTermCfg as _EventTerm
        import isaaclab_tasks.manager_based.locomotion.velocity.mdp as _velmdp
        env_cfg.events.push_robot = _EventTerm(
            func=_velmdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.5, 3.0),
            params={"velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
        )

    vc = getattr(args_cli, "rl_vel_cmd", -999.0)
    if vc > -900.0:
        # NON-IMITATION velocity-control baseline: pin the env's base-velocity command
        # to a constant forward vc (straight, no turn), so the task-reward policy is
        # commanded at the clip's recovered speed instead of Isaac's random command
        # curriculum. Range collapsed to a point -> every resample yields (vc,0,0).
        cr = env_cfg.commands.base_velocity
        cr.heading_command = False                 # use ang_vel_z directly (=0), not heading tracking
        cr.ranges.lin_vel_x = (vc, vc)
        cr.ranges.lin_vel_y = (0.0, 0.0)
        cr.ranges.ang_vel_z = (0.0, 0.0)
        if hasattr(cr.ranges, "heading"):
            cr.ranges.heading = (0.0, 0.0)
    env = gym.make(args_cli.task, cfg=env_cfg)
    return env


def apply_actuator_dr(env, num_envs: int, lo: float, hi: float, device: str):
    """Per-env actuator effort-limit randomization (call once, after env.reset()).

    effort_limit is a (num_envs, num_joints) tensor the LSTM ANYdrive net clips its
    output against. Scaling it per env by ~U(lo,hi) makes some envs weaker, so the
    policy must find poses achievable within a smaller torque budget instead of
    overfitting the nominal actuator strength -- the direct fix for the real-robot
    torque saturation. No built-in event exists for the LSTM actuator, hence the
    manual write; guarded so a stock (implicit-actuator) build degrades gracefully.
    """
    robot = env.unwrapped.scene["robot"]
    act = robot.actuators.get("legs")
    if act is None or not hasattr(act, "effort_limit"):
        print("[DR] actuator effort randomization skipped (no 'legs' effort_limit).")
        return
    scale = torch.empty(num_envs, 1, device=device).uniform_(lo, hi)   # (N,1) broadcast over joints
    act.effort_limit = act.effort_limit * scale
    if hasattr(act, "effort_limit_sim") and act.effort_limit_sim is not None:
        act.effort_limit_sim = act.effort_limit_sim * scale
    print(f"[DR] per-env actuator effort-limit scaled ~U({lo},{hi}); "
          f"mean now {act.effort_limit.mean().item():.1f} N·m.")


def init_joints_to_motion(env, clip: dict, phase_t_arr: np.ndarray, device: str, env_ids=None):
    """Initialize envs' joints to their respective motion phases.

    env_ids: indices of the envs to (re)initialize; None = all envs.
    IMPORTANT: pass only the envs that actually reset writing all envs on
    every individual reset teleports the *surviving* envs back onto the
    reference pose, erasing their accumulated state and corrupting both the
    BC dataset and the RL rollouts.
    """
    robot: AssetBase = env.unwrapped.scene["robot"]
    am = env.unwrapped.action_manager
    jids = am._terms["joint_pos"]._joint_ids

    if env_ids is None:
        q_np = get_joint_target_batch(clip, phase_t_arr)          # (N,12)
        ids_t = None
    else:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        q_np = get_joint_target_batch(clip, phase_t_arr[env_ids])  # (len(ids),12)
        ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=device)

    q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
    v_t = torch.zeros_like(q_t)
    robot.write_joint_state_to_sim(q_t, v_t, joint_ids=jids, env_ids=ids_t)

    # TODO: Reset the root states as well
    # s_t = robot.data.default_root_state.clone()  # Shape is (len(env_ids), 13). Refer to articulation.py base class
    # robot.write_root_state_to_sim(root_state=s_t, env_ids=None) # all environments are used

    env.unwrapped.scene.write_data_to_sim()


def reset_envs_to_reference(env, clip: dict, phase_t_arr: np.ndarray, device: str,
                            env_ids=None, seed_velocity: bool = True):
    """Full DeepMimic-style RSI reset for the given envs: set joints AND the base
    (height + reference forward/lateral velocity) to the clip at each env's phase.

    Unlike init_joints_to_motion (joints only see its TODO), this also reseats
    the base, which is REQUIRED for the DeepMimic early-termination path: an env
    terminated for drifting off-reference did NOT fall, so the env's own autoreset
    never fired and its base would otherwise stay in the drifted (tilted/low)
    state. Seeding the reference base velocity (default on) is standard RSI and
    lets forward locomotion start from the reference's own +v, which the root
    reward then has to sustain.
    """
    robot = env.unwrapped.scene["robot"]
    am = env.unwrapped.action_manager
    jids = am._terms["joint_pos"]._joint_ids

    if env_ids is None:
        sel = np.arange(len(phase_t_arr))
        ids_t = None
    else:
        sel = np.asarray(env_ids, dtype=np.int64)
        ids_t = torch.as_tensor(sel, dtype=torch.long, device=device)

    # joints -> reference pose, zero joint velocity
    q_t = torch.tensor(get_joint_target_batch(clip, phase_t_arr[sel]),
                       dtype=torch.float32, device=device)
    robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t), joint_ids=jids, env_ids=ids_t)

    # base -> default root state (correct per-env origin + identity orientation),
    # with reference base height and (optionally) reference body-frame velocity.
    # The reference root orientation is identity (no turn), so body ~ world here.
    root = robot.data.default_root_state.clone()
    root = root if ids_t is None else root[ids_t]
    rt = get_root_target_batch(clip, phase_t_arr[sel])
    if rt is not None:
        rt_t = torch.as_tensor(rt, dtype=torch.float32, device=device)
        root[:, 2] = rt_t[:, 2]                      # base height
        if seed_velocity:
            root[:, 7:9] = rt_t[:, :2]               # world lin vel (x,y) ~ body (identity yaw)
    robot.write_root_state_to_sim(root, env_ids=ids_t)
    env.unwrapped.scene.write_data_to_sim()


def reference_action(clip: dict, t_arr: np.ndarray, default_pos: np.ndarray, device: str) -> torch.Tensor:
    """Compute reference actions for all envs."""
    q_np = get_joint_target_batch(clip, t_arr)  # (N,12)
    a_np = (q_np - default_pos[None, :]) / ACTION_SCALE
    return torch.tensor(a_np, dtype=torch.float32, device=device)


def collect_bc_data(env, clip: dict, device: str, collect_s: float = 60.0,
                    use_phase: bool = False) -> tuple:
    """
    Roll out the reference motion for `collect_s` seconds per env.
    Returns (obs_buf, act_buf) as tensors on CPU.
    """
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    n_steps = int(collect_s / sim_dt)

    obs_list = []
    act_list = []

    # Random starting phases for diversity
    phase_t = np.random.uniform(0, clip["duration"], size=num_envs)
    obs, _ = env.reset()
    obs = obs['policy']
    init_joints_to_motion(env, clip, phase_t, device)

    print(f"[BC collect] {n_steps} steps × {num_envs} envs = "
          f"{n_steps * num_envs:,} samples")

    n_resets = 0
    for step in range(n_steps):
        acts = reference_action(clip, phase_t, default_pos, device)  # (N,12)
        obs_list.append(build_policy_obs(obs, phase_t, clip, device, use_phase).cpu())
        act_list.append(acts.cpu())

        obs, _, terminated, truncated, _ = env.step(acts)
        obs = obs['policy']
        phase_t += sim_dt

        # On reset: re-initialize ONLY the terminated envs to new random phases
        done = terminated | truncated
        if done.any():
            done_np = done.cpu().numpy()
            n_done = int(done_np.sum())
            n_resets += n_done
            phase_t[done_np] = np.random.uniform(0, clip["duration"], size=n_done)
            init_joints_to_motion(env, clip, phase_t, device,
                                  env_ids=np.nonzero(done_np)[0])

        if step % 500 == 0:
            print(f"  step {step}/{n_steps}  "
                  f"collected {len(obs_list) * num_envs:,} samples")

    obs_buf = torch.cat(obs_list, dim=0)  # (N_total, obs_dim)
    act_buf = torch.cat(act_list, dim=0)  # (N_total, 12)

    # Data-quality report: if the open-loop reference replay falls quickly,
    # most samples come from stumbling/fallen states and the time-indexed
    # labels are meaningless there.
    mean_ep_s = (n_steps * num_envs * sim_dt) / max(n_resets + num_envs, 1)
    print(f"[BC collect] terminations: {n_resets}  |  mean episode length ≈ {mean_ep_s:.1f}s "
          f"(clip is {clip['duration']:.1f}s)")
    if mean_ep_s < 2.0:
        print("[BC collect] WARNING: open-loop replay falls almost immediately the dataset "
              "is dominated by falling states with feedforward labels. BC cannot work on "
              "this; the clip/gains need fixing or go straight to RL fine-tuning.")
    print(f"[BC collect] Done. Dataset: {obs_buf.shape}")
    return obs_buf, act_buf


def train_bc(obs_buf: torch.Tensor, act_buf: torch.Tensor, policy: MLP, device: str, epochs: int = 200, batch: int = 1024, lr: float = 3e-4, save_path: str = "policy_bc.pt"):
    """Train policy via supervised regression on (obs, action) pairs."""
    policy = policy.to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    N = obs_buf.shape[0]

    obs_d = obs_buf.to(device)
    act_d = act_buf.to(device)

    best_loss = float("inf")
    losses = deque(maxlen=50)
    loss_history = []   

    print(f"\n[BC train] {epochs} epochs  batch={batch}  lr={lr}")
    for epoch in range(epochs):
        # Shuffle
        idx = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch):
            b = idx[start:start + batch]
            pred = policy(obs_d[b])
            loss = nn.functional.mse_loss(pred, act_d[b])
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"epoch": epoch, "loss": best_loss,
                        "state_dict": policy.state_dict()}, save_path)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}/{epochs}  loss={avg_loss:.5f}  "
                  f"best={best_loss:.5f}  lr={sched.get_last_lr()[0]:.2e}")

    print(f"[BC train] Best loss: {best_loss:.5f}  saved → {save_path}")
    return policy, loss_history


class PPOBuffer:
    """Minimal rollout buffer for PPO."""

    def __init__(self, num_envs: int, horizon: int, obs_dim: int, action_dim: int, device: str):
        self.obs = torch.zeros(horizon, num_envs, obs_dim, device=device)
        self.acts = torch.zeros(horizon, num_envs, action_dim, device=device)
        self.logps = torch.zeros(horizon, num_envs, device=device)
        self.rewards = torch.zeros(horizon, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device)
        self.values = torch.zeros(horizon, num_envs, device=device)
        self.ptr = 0
        self.horizon = horizon

    def store(self, obs, acts, logps, rewards, dones, values):
        t = self.ptr
        self.obs[t] = obs
        self.acts[t] = acts
        self.logps[t] = logps
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self.ptr = (self.ptr + 1) % self.horizon

    def compute_returns(self, last_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95):
        """GAE-Lambda returns."""
        H, N = self.horizon, self.obs.shape[1]
        adv = torch.zeros(H, N, device=self.obs.device)
        gae = torch.zeros(N, device=self.obs.device)
        for t in reversed(range(H)):
            next_val = last_value if t == H - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            adv[t] = gae
        returns = adv + self.values
        return adv.reshape(-1), returns.reshape(-1)

    def get_flat(self):
        return (self.obs.reshape(-1, self.obs.shape[-1]),
                self.acts.reshape(-1, self.acts.shape[-1]),
                self.logps.reshape(-1),
                self.values.reshape(-1))

# TODO: Define custom rewards that combine multiple different rewards instead of only MSE shadow to actual state
def motion_tracking_reward(joint_pos: torch.Tensor, 
                           q_ref: torch.Tensor, 
                           sigma: float = 0.25, 
                           vel_reward: torch.Tensor = None, 
                           vel_weight: float = 0.5, 
                           joint_vel_reward: torch.Tensor = None,
                           joint_vel_weight:float = 0.05) -> torch.Tensor:
    """
    Reward = exp(-||q - q_ref||^2 / sigma^2)
           + vel_weight * vel_reward (optional forward velocity term)

    joint_pos: (N, 12)  current joint positions
    q_ref:     (N, 12)  reference joint positions from motion clip
    """
    diff = joint_pos - q_ref  # (N, 12)
    sq_dist = (diff ** 2).sum(dim=-1)  # (N,)
    track_rew = torch.exp(-sq_dist / (sigma ** 2))  # (N,)

    if vel_reward is not None:
        track_rew += vel_weight * vel_reward.clamp(0) 

    if joint_vel_reward is not None:
        track_rew += joint_vel_weight * joint_vel_reward.clamp(0)

    return track_rew


def deepmimic_reward(q, q_ref, foot_local, foot_ref, root_now, root_ref,
                     joint_w=0.5, foot_w=0.3, root_w=0.2,
                     joint_sigma=0.25, foot_sigma=0.20, root_sigma=0.25,
                     foot_z_weight=1.0):
    """DeepMimic-style weighted-sum imitation reward, each term in (0, 1].

        r = joint_w * exp(-||q - q_ref||^2        / joint_sigma^2)   # joint angles
          + foot_w  * exp(-(dx^2 + dy^2 + foot_z_weight*dz^2) / foot_sigma^2)  # body-frame feet
          + root_w  * exp(-||root - root_ref||^2  / root_sigma^2)    # [vx, vy, height]

    The foot term (body-frame foot position) is what forbids the low-clearance
    SHUFFLE. But the plain L2 over (x,y,z) is DOMINATED by the large fore-aft
    stride sweep (~0.5-0.7 m), so the small vertical CLEARANCE swing (~0.1 m) is
    under-rewarded and the policy drags/over-lifts feet while still matching
    stride. `foot_z_weight` (>1) up-weights the vertical error so matching each
    foot's lift height matters as much as its fore-aft placement.

    The root term forbids stepping-in-place and crouch. foot/root are OPTIONAL: if
    a clip lacks foot_pos_local or recovered root translation, that term's argument
    is None, the term is dropped, and its weight is folded into the joint term so
    the total still spans ~(0, 1].

    Returns (reward (N,), joint_err (N,), foot_err (N,)); the *_err are *unweighted*
    L2 norms used for early termination and logging (foot_err is 0 if no foot target).
    """
    je = q - q_ref                                          # (N,12)
    joint_err = je.norm(dim=-1)                             # (N,)
    r_joint = torch.exp(-(je ** 2).sum(-1) / joint_sigma ** 2)

    eff_joint_w = joint_w
    r = torch.zeros_like(r_joint)
    foot_err = torch.zeros_like(r_joint)

    if foot_local is not None and foot_ref is not None:
        fe = foot_local - foot_ref                              # (N,4,3) body frame
        foot_err = fe.reshape(fe.shape[0], -1).norm(dim=-1)     # unweighted L2 (early-term/log)
        # vertical (clearance) error weighted up so it is not swamped by stride.
        wsq = (fe[..., 0] ** 2 + fe[..., 1] ** 2
               + foot_z_weight * fe[..., 2] ** 2).sum(dim=-1)   # (N,)
        r = r + foot_w * torch.exp(-wsq / foot_sigma ** 2)
    else:
        eff_joint_w += foot_w

    if root_now is not None and root_ref is not None:
        re = root_now - root_ref                            # (N,3)
        r = r + root_w * torch.exp(-(re ** 2).sum(-1) / root_sigma ** 2)
    else:
        eff_joint_w += root_w

    r = r + eff_joint_w * r_joint
    return r, joint_err, foot_err


# =====================================================================
# Adversarial Motion Prior (AMP)
# ---------------------------------------------------------------------
# Pure imitation with NO hand-designed reward. A discriminator is trained to
# tell apart short state-transitions sampled from the reference clip ("real")
# from transitions produced by the policy ("fake"); the policy's reward is how
# "real" its transitions look to the discriminator. There is no phase, no
# velocity-command term, no survival bonus to game the only objective is
# "move like the reference data" (Peng et al. 2021, "AMP").
#
# AMP features are a per-frame state descriptor computed IDENTICALLY from the
# reference clip and from the live sim, so the discriminator compares like with
# like:  [ joint_pos(12), foot_pos_local(12) ].  The discriminator sees a PAIR
# of consecutive frames, so velocity is encoded implicitly (and identically for
# both sides) rather than as an explicit feature. Two systematic "tells" that
# would otherwise let the discriminator separate trivially (and which the policy
# could never close) are removed:
#   1. Frame-rate: the reference is resampled to the SIM CONTROL rate in
#      build_reference_amp, so a reference transition spans the same dt as one
#      env.step (otherwise 30 fps ref vs 50 Hz control = 1.67x larger deltas).
#   2. Velocity source: explicit joint_vel is dropped. The clip's joint_vel is a
#      smoothed finite-difference; the sim's is noisy instantaneous velocity, a
#      free tell. The implicit pos_t -> pos_t+1 delta is the same for both.
# - foot_pos_local is each foot expressed in the BASE frame, so its z component
#   already encodes how high the body sits above its feet a sit/crouch shows
#   up here consistently in both reference and sim, with no absolute-frame
#   convention to mismatch.
# - Absolute root height is deliberately EXCLUDED: it is a zero-variance
#   constant in the pinned reference (0.5 m), and that pin height is not the
#   robot's natural standing height, so a height feature would both blow up
#   under normalization and flag normal standing as "fake".
# - Root linear/angular velocity are also omitted for now: the dog clip is
#   pinned/in-place so they are identically zero in the reference. They become
#   the locomotion signal once translating clips (e.g. led-walk horse videos)
#   flow through a root-translation-preserving retarget.
# =====================================================================

AMP_FOOT_BODIES = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
# joint_pos(12) + foot_pos_local(12) + root_lin_vel_body(3). The root linear
# velocity (recovered into the references by recover_root_translation.py) makes
# FORWARD MOTION part of the style the discriminator rewards so walking forward
# becomes the high-reward behaviour instead of in-place stepping / crouching.
# Constant components (e.g. lateral/vertical root vel ~0) are dropped by keep_mask.
AMP_FEAT_DIM = 12 + 12 + 3  # = 27


def assemble_amp_features(joint_pos: torch.Tensor, foot_local: torch.Tensor,
                          root_lin_vel: torch.Tensor) -> torch.Tensor:
    """Pack the per-frame AMP state descriptor. Inputs are (N, ...) tensors.

    joint_pos (N,12), foot_local (N,4,3) or (N,12), root_lin_vel (N,3) body frame.
    Returns (N, AMP_FEAT_DIM).
    """
    n = joint_pos.shape[0]
    return torch.cat([joint_pos.reshape(n, -1), foot_local.reshape(n, -1),
                      root_lin_vel.reshape(n, -1)], dim=-1)


def compute_amp_obs_sim(robot, foot_ids: list[int], device: str) -> torch.Tensor:
    """AMP features for every env from the live sim, matching the reference layout."""
    jp = robot.data.joint_pos[:, :12]
    root_pos = robot.data.root_pos_w                       # (N,3)
    root_quat = robot.data.root_quat_w                     # (N,4) wxyz
    foot_w = robot.data.body_pos_w[:, foot_ids, :]         # (N,4,3)
    rel = foot_w - root_pos[:, None, :]                    # (N,4,3)
    # rotate each foot offset into the base frame
    n, k, _ = rel.shape
    foot_local = quat_apply_inverse(
        root_quat[:, None, :].expand(n, k, 4).reshape(-1, 4),
        rel.reshape(-1, 3),
    ).reshape(n, k, 3)
    root_lin_vel = robot.data.root_lin_vel_b               # (N,3) body-frame
    return assemble_amp_features(jp, foot_local, root_lin_vel)


def compute_foot_local_sim(robot, foot_ids: list, device: str) -> torch.Tensor:
    """Live body-frame foot positions (N,4,3) for the DeepMimic foot-tracking term.
    Same transform as compute_amp_obs_sim, kept separate so the AMP path is
    untouched. foot_ids must be in AMP_FOOT_BODIES order [LF,RF,LH,RH] to match
    the clip's foot_pos_local."""
    root_pos = robot.data.root_pos_w                       # (N,3)
    root_quat = robot.data.root_quat_w                     # (N,4) wxyz
    foot_w = robot.data.body_pos_w[:, foot_ids, :]         # (N,4,3)
    rel = foot_w - root_pos[:, None, :]
    n, k, _ = rel.shape
    return quat_apply_inverse(
        root_quat[:, None, :].expand(n, k, 4).reshape(-1, 4),
        rel.reshape(-1, 3),
    ).reshape(n, k, 3)


def compute_foot_track_sim(robot, foot_ids: list, device: str) -> torch.Tensor:
    """Foot representation for the DeepMimic foot term, BASE-ROLL-INVARIANT in z.

    x,y stay body-frame (footfall pattern relative to heading), but the vertical
    component becomes the WORLD clearance (foot_z - base_z), which base roll/pitch
    CANNOT change. The plain body-frame z = R(base)^-1·(foot-base) let the policy
    fake clearance by ROLLING the base instead of lifting the foot (the diagonal
    base wobble seen when foot_z_weight was raised to 8). Using the world vertical
    kills that exploit, so up-weighting z now rewards REAL foot lift only.

    It stays comparable to the reference foot_pos_local z because the reference
    base is at identity orientation, where body-frame z == foot_z - base_z.
    """
    fl = compute_foot_local_sim(robot, foot_ids, device).clone()    # (N,4,3) body-frame
    foot_world_z = robot.data.body_pos_w[:, foot_ids, 2]            # (N,4) world height
    base_z = robot.data.root_pos_w[:, 2:3]                         # (N,1)
    fl[:, :, 2] = foot_world_z - base_z                            # roll-invariant clearance
    return fl


def build_reference_amp(clip_npz: dict, device: str, control_dt: float
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build normalized reference AMP features + (s, s') transitions from the clip.

    Returns (transitions (M, 2*Fk), feat_mean (Fk,), feat_std (Fk,), keep_mask (F,)).
    The clip is a closed loop, so the last->first transition is included.

    The reference is resampled from its own frame rate to `control_dt` (the sim
    control step) so that a reference transition spans the SAME time as one
    env.step. Otherwise the frame-rate gap (e.g. 30 fps clip vs 50 Hz control)
    makes reference transitions advance more motion than any policy transition,
    a tell the discriminator exploits and the policy can never close.

    Features that are CONSTANT in the reference (e.g. the lateral foot positions
    pinned by `retarget --lateral-mode pin`, or any non-moving joint) are dropped
    via `keep_mask`: a constant feature carries no motion-distribution signal, and
    keeping it (normalized) would let the discriminator demand an exact
    convention-specific value the policy's natural state may not match.
    """
    jp = np.asarray(clip_npz["joint_pos_isaac"], dtype=np.float32)         # (T,12)
    foot_local = np.asarray(clip_npz["foot_pos_local"], dtype=np.float32)  # (T,4,3)
    T = jp.shape[0]
    raw = np.concatenate([jp.reshape(T, -1), foot_local.reshape(T, -1)], axis=-1)  # (T,24)

    # Resample to the control rate (periodic linear interp around the loop).
    ref_dt = float(clip_npz.get("frame_duration", clip_npz.get("dt", 1.0 / 30.0)))
    duration = T * ref_dt
    T2 = max(2, int(round(duration / control_dt)))
    t_src = np.arange(T + 1) * ref_dt                                  # include wrap point
    raw_loop = np.concatenate([raw, raw[:1]], axis=0)                  # (T+1, 24)
    t_dst = np.arange(T2) * control_dt
    raw_rs = np.stack([np.interp(t_dst, t_src, raw_loop[:, c]) for c in range(raw.shape[1])], axis=-1)

    feats = torch.tensor(raw_rs, dtype=torch.float32, device=device)   # (T2, F)
    raw_std = feats.std(dim=0)
    keep_mask = raw_std > 1e-3                                          # drop constant features
    feats = feats[:, keep_mask]
    mean = feats.mean(dim=0)
    std = feats.std(dim=0).clamp_min(1e-2)
    feats_n = (feats - mean) / std

    nxt = torch.roll(feats_n, shifts=-1, dims=0)                        # wrap (loop)
    transitions = torch.cat([feats_n, nxt], dim=-1)                    # (T2, 2*Fk)
    print(f"[AMP] reference: {T} frames @ {1/ref_dt:.0f}fps -> resampled to {T2} @ "
          f"{1/control_dt:.0f}Hz -> {transitions.shape[0]} transitions; "
          f"kept {int(keep_mask.sum())}/{keep_mask.numel()} features "
          f"(dropped {int((~keep_mask).sum())} constant)")
    return transitions, mean, std, keep_mask


def _resample_clip_raw(clip_npz: dict, control_dt: float) -> np.ndarray:
    """Resample one clip's raw AMP features (joint_pos, foot_pos_local,
    root_lin_vel) to the sim control rate, with a periodic wrap point.
    Returns (T2, AMP_FEAT_DIM=27). root_lin_vel defaults to 0 if a clip predates
    recover_root_translation.py (then it contributes no forward signal)."""
    jp = np.asarray(clip_npz["joint_pos_isaac"], dtype=np.float32)
    foot_local = np.asarray(clip_npz["foot_pos_local"], dtype=np.float32)
    T = jp.shape[0]
    rlv = (np.asarray(clip_npz["root_lin_vel"], dtype=np.float32)
           if "root_lin_vel" in clip_npz else np.zeros((T, 3), np.float32))
    rlv = rlv.reshape(T, -1)[:, :3]
    raw = np.concatenate([jp.reshape(T, -1), foot_local.reshape(T, -1), rlv], axis=-1)
    ref_dt = float(clip_npz.get("frame_duration", clip_npz.get("dt", 1.0 / 30.0)))
    T2 = max(2, int(round((T * ref_dt) / control_dt)))
    t_src = np.arange(T + 1) * ref_dt
    raw_loop = np.concatenate([raw, raw[:1]], axis=0)
    t_dst = np.arange(T2) * control_dt
    return np.stack([np.interp(t_dst, t_src, raw_loop[:, c]) for c in range(raw.shape[1])],
                    axis=-1).astype(np.float32)


def build_reference_amp_multi(clips: list[dict], device: str, control_dt: float, window: int = 2
                              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool AMP reference WINDOWS across a CLIP LIBRARY with a SHARED keep_mask
    and normalization. Each window is `window` consecutive frames; with window=2
    this is the usual (s,s') transition. A longer window lets the discriminator
    see MOTION OVER TIME, so a standing-still policy (K near-identical frames) is
    obviously fake against reference windows that all contain stepping this is
    what stops the "smooth standing" degenerate solution of pure-imitation AMP.

    Each clip wraps within ITSELF (loop), never across clips (no fake A->B seam).
    A single short clip lets the discriminator memorize it and collapse the style
    reward; many clips give a richer distribution it cannot trivially separate.
    """
    raws = [_resample_clip_raw(c, control_dt) for c in clips]      # list of (T2_i, 24)
    pool = torch.tensor(np.concatenate(raws, axis=0), dtype=torch.float32, device=device)
    keep_mask = pool.std(dim=0) > 1e-3                              # drop globally-constant feats
    mean = pool[:, keep_mask].mean(dim=0)
    std = pool[:, keep_mask].std(dim=0).clamp_min(1e-2)
    all_win = []
    for raw_rs in raws:
        feats_n = (torch.tensor(raw_rs, dtype=torch.float32, device=device)[:, keep_mask] - mean) / std
        # window[i] = concat(feats_n[i], feats_n[i+1], ..., feats_n[i+window-1]) mod T2
        all_win.append(torch.cat([torch.roll(feats_n, shifts=-k, dims=0) for k in range(window)], dim=-1))
    windows = torch.cat(all_win, dim=0)
    print(f"[AMP] reference library: {len(clips)} clips -> {windows.shape[0]} windows of "
          f"{window} frames; kept {int(keep_mask.sum())}/{keep_mask.numel()} features "
          f"(dropped {int((~keep_mask).sum())} constant), shared normalization")
    return windows, mean, std, keep_mask


class Discriminator(nn.Module):
    """LSGAN discriminator over AMP state-transitions (real ~ +1, fake ~ -1).

    Deliberately small: the reference is a single short clip (~hundreds of
    transitions), so a large discriminator memorizes it in a few steps and
    saturates (real/fake acc -> 1.0), collapsing the style reward to zero.
    """

    def __init__(self, feat_dim: int, window: int = 2, hidden=(256, 128)):
        super().__init__()
        layers = []
        in_dim = window * feat_dim    # K-frame window (K=2 == a plain transition)
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).squeeze(-1)

    def style_reward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        # AMP reward: r = max(0, 1 - 0.25 (D - 1)^2), in [0, 1]; scaled.
        with torch.no_grad():
            d = self.forward(x)
            r = torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)
        return scale * r


def train_ppo(env, clip: dict, policy: MLP, value_net: ValueNet,
              device: str, n_iters: int = 2000, horizon: int = 24,
              lr: float = 1e-4, sigma: float = 0.25, vel_weight: float = 0.5,
              save_path: str = "policy_rl.pt",
              clip_eps: float = 0.2, entropy_coef: float = 0.0,
              vf_coef: float = 0.5, grad_clip: float = 1.0,
              n_epochs_per_iter: int = 4, minibatch: int = 512,
              use_phase: bool = False, task_reward_weight: float = 0.0,
              joint_w: float = 0.5, foot_w: float = 0.3, root_w: float = 0.2,
              foot_sigma: float = 0.20, root_sigma: float = 0.25,
              term_joint: float = 2.0, term_foot: float = 0.5,
              action_rate_w: float = 0.0, foot_z_weight: float = 1.0,
              launch_frac: float = 0.0, launch_s: float = 1.5, launch_phase: float = -1.0):
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    default_t = torch.tensor(default_pos[:12], dtype=torch.float32, device=device)

    # DeepMimic foot/root tracking targets (fall back to joint-only if absent).
    foot_ids = [list(robot.body_names).index(n) for n in AMP_FOOT_BODIES]
    has_foot = "cs_foot" in clip
    has_root = "cs_root" in clip

    pobs_dim = OBS_DIM + (2 if use_phase else 0)
    all_params = list(policy.parameters()) + list(value_net.parameters())
    optim = torch.optim.Adam(all_params, lr=lr)
    cur_lr, desired_kl = lr, 0.01     # adaptive-LR state (rsl_rl-style trust region)
    buf = PPOBuffer(num_envs, horizon, pobs_dim, ACTION_DIM, device)

    duration = clip["duration"]

    # --- Launch-from-rest curriculum -----------------------------------------
    # launch_ramp_t[i] = seconds since env i last did a REST reset; np.inf marks a
    # classic RSI env (reset already moving). For a launch env the phase advances
    # at ramp*sim_dt and the reference velocity target is scaled by ramp, both
    # going 0->1 over launch_s, so the policy learns to spin the gait up from a
    # standstill instead of only ever tracking a steady gallop. launch_frac=0
    # (default) disables this and recovers the classic behaviour exactly.
    launch_ramp_t = np.full(num_envs, np.inf)

    def _auto_launch_phase():
        """Cycle fraction of the most-grounded (all-feet-down) stance, from cs_foot."""
        if "cs_foot" not in clip:
            return 0.0
        ts = np.linspace(0, duration, 200, endpoint=False)
        z = np.asarray(clip["cs_foot"](ts)).reshape(len(ts), 4, 3)[:, :, 2]   # (200,4) foot heights
        g = z.min()
        ndown = (z < g + 0.05).sum(1)
        spread = z.max(1) - z.min(1)
        best = min(range(len(ts)), key=lambda i: (-int(ndown[i]), float(spread[i])))
        return float(ts[best] / duration)

    lp = (launch_phase % 1.0) if launch_phase >= 0.0 else (_auto_launch_phase() if launch_frac > 0.0 else 0.0)
    launch_phase_t = lp * duration
    if launch_frac > 0.0:
        print(f"[PPO] launch-from-rest: frac={launch_frac}  ramp={launch_s}s  from phase {lp:.3f} ({launch_phase_t:.2f}s)")

    def _smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    phase_t = np.random.uniform(0, duration, size=num_envs)

    def _reset(ids):
        """Reset the given envs. A launch_frac share start from REST at the stance
        phase (ramp from 0); the rest use classic RSI (random phase + ref velocity)."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return
        is_launch = np.random.rand(ids.size) < launch_frac
        normal, launch = ids[~is_launch], ids[is_launch]
        if normal.size:
            phase_t[normal] = np.random.uniform(0, duration, size=normal.size)
            launch_ramp_t[normal] = np.inf
            reset_envs_to_reference(env, clip, phase_t, device, env_ids=normal, seed_velocity=True)
        if launch.size:
            phase_t[launch] = launch_phase_t
            launch_ramp_t[launch] = 0.0
            reset_envs_to_reference(env, clip, phase_t, device, env_ids=launch, seed_velocity=False)

    obs, _ = env.reset()
    obs = obs["policy"]
    if getattr(args_cli, "rl_dr", False):
        apply_actuator_dr(env, num_envs, args_cli.rl_dr_effort[0], args_cli.rl_dr_effort[1], device)
    _reset(np.arange(num_envs))

    best_mean_return = -float("inf")
    ep_returns = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)          # survival time [s]
    ep_return = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, device=device)
    prev_acts = torch.zeros(num_envs, ACTION_DIM, device=device)   # for action-rate penalty
    curve = []                              # (iteration, mean_return, survival_s) per iter -> curve.csv

    print(f"\n[PPO] {n_iters} iters horizon={horizon}  lr={lr}  "
          f"task_reward_weight={task_reward_weight}")
    print(f"[PPO] DeepMimic terms: joint_w={joint_w} "
          f"foot_w={foot_w if has_foot else 0.0}{'' if has_foot else ' (clip has no foot_pos_local)'} "
          f"root_w={root_w if has_root else 0.0}{'' if has_root else ' (clip has no recovered root)'}; "
          f"early-term joint>{term_joint or '-'} foot>{term_foot if (term_foot and has_foot) else '-'}")
    print(f"[PPO] anti-jitter: entropy_coef={entropy_coef} (log_std clamped <= {LOG_STD_MAX}), "
          f"action_rate_w={action_rate_w}  | sigmas joint={sigma} foot={foot_sigma} root={root_sigma}"
          f" | foot_z_weight={foot_z_weight}")
    if task_reward_weight > 0.0:
        print("[PPO] adding a positive velocity-command tracking reward "
              "(imitation acts as the style term; total reward stays positive "
              "so surviving is always worthwhile)")

    for iteration in trange(n_iters, file=sys.__stdout__):
        policy.eval()
        value_net.eval()

        with torch.no_grad():
            for _ in range(horizon):

                # TODO: Find another way of converting the types because inefficient
                if isinstance(obs, dict):
                    obs = obs['policy']

                obs_t = build_policy_obs(obs, phase_t, clip, device, use_phase)
                dist = policy(obs_t)
                acts = dist.rsample()
                acts = clamp_knee_actions(acts)        # hard X-config: knees can't flip outward
                logps = dist.log_prob(acts).sum(-1)
                values = value_net(obs_t)

                next_obs, _, terminated, truncated, _ = env.step(acts)
                done = (terminated | truncated).float() #TODO: add more terminal conditions
                # Launch envs advance the phase clock at ramp*sim_dt (ramp 0->1 over
                # launch_s); classic envs (launch_ramp_t = inf) advance at full rate.
                ramp = np.where(np.isinf(launch_ramp_t), 1.0, _smoothstep(launch_ramp_t / launch_s))
                phase_t += ramp * sim_dt
                launch_ramp_t += sim_dt

                # --- DeepMimic-style multi-term tracking reward ---
                # joint angles + body-frame foot positions (clearance/stride) +
                # root (fore-aft/lateral velocity + base height). The foot term is
                # what forbids the low-clearance shuffle; the root term forbids
                # stepping-in-place and crouch. q_ref is phase-indexed via the
                # cubic spline; foot/root targets likewise (None if clip lacks them).
                q_cur = robot.data.joint_pos[:, :12]
                q_ref = torch.as_tensor(get_joint_target_batch(clip, phase_t),
                                        dtype=torch.float32, device=device)

                if has_foot:
                    foot_cur = compute_foot_track_sim(robot, foot_ids, device)
                    foot_ref = torch.as_tensor(get_foot_target_batch(clip, phase_t),
                                               dtype=torch.float32, device=device)
                else:
                    foot_cur = foot_ref = None

                if has_root:
                    root_now = torch.cat([robot.data.root_lin_vel_b[:, :2],
                                          robot.data.root_pos_w[:, 2:3]], dim=-1)  # (N,3)
                    rr = get_root_target_batch(clip, phase_t)          # (N,3): vx, vy, base height
                    if launch_frac > 0.0:
                        rr = rr.copy()
                        rr[:, :2] *= ramp[:, None]                     # ramp the velocity target 0->nominal (height kept)
                    root_ref = torch.as_tensor(rr, dtype=torch.float32, device=device)
                else:
                    root_now = root_ref = None

                reward, joint_err, foot_err = deepmimic_reward(
                    q_cur, q_ref, foot_cur, foot_ref, root_now, root_ref,
                    joint_w=joint_w, foot_w=foot_w, root_w=root_w,
                    joint_sigma=sigma, foot_sigma=foot_sigma, root_sigma=root_sigma,
                    foot_z_weight=foot_z_weight)

                # Task term: velocity-command tracking, POSITIVE by construction
                # (exp form, in (0, 1]), mirroring the env's own tracking terms
                # but without its penalty terms. Never blend the env's *net*
                # reward here: its penalties make the per-step total negative
                # while the robot flails, and a negative per-step reward teaches
                # the policy to terminate as fast as possible ("RL suicide" —
                # observed as survival collapsing to ~0.7 s).
                if task_reward_weight > 0.0:
                    cmd = obs_t[:, 9:12]                       # (vx, vy, wz) commands
                    v_xy = robot.data.root_lin_vel_b[:, :2]    # true body-frame velocity
                    w_z = robot.data.root_ang_vel_b[:, 2]
                    lin_err = ((cmd[:, :2] - v_xy) ** 2).sum(dim=-1)
                    ang_err = (cmd[:, 2] - w_z) ** 2
                    r_task = 0.75 * torch.exp(-lin_err / 0.25) + 0.25 * torch.exp(-ang_err / 0.25)
                    reward = reward + task_reward_weight * r_task

                # Action-rate smoothness penalty (anti-jitter): penalise large
                # step-to-step action changes. Operates on the sampled action, so
                # it also nudges the exploration std down. Kept modest so a real
                # leg SWING (a large but smooth target change) survives while
                # high-frequency jitter is damped.
                if action_rate_w > 0.0:
                    reward = reward - action_rate_w * ((acts - prev_acts) ** 2).mean(dim=-1)
                prev_acts = acts.detach()
                reward = reward.clamp(min=-5.0)   # bound per-step reward: a penalty spike can't blow up returns/value

                # DeepMimic early termination: end episodes that drift off the
                # reference manifold so crouch / shuffle / stand can't be ridden
                # out for survival reward. The reset path below reseats these envs
                # on-reference (joints + base + reference velocity).
                # GRACE: a launch env starts at REST but the reference is already a
                # full gallop pose, so joint/foot error is unavoidably large while it
                # accelerates. Suppress DRIFT termination for a SHORT grace after a rest
                # reset (the env's own fall termination still applies) so the policy is
                # not killed on step 1 while it settles. The grace is deliberately short
                # (LAUNCH_GRACE_S): if it were the whole ramp, a launch env that cannot
                # gallop just CROUCHES and survives the grace, and that survival-crouch
                # gets rewarded and corrupts the shared policy (return/base-height decay).
                # After the grace, drift-termination resumes and punishes the crouch, so
                # only a genuine launch (tracking the slowly-advancing stance) survives.
                LAUNCH_GRACE_S = 0.25
                in_launch = torch.as_tensor(launch_ramp_t < LAUNCH_GRACE_S, device=device)
                drift = torch.zeros(num_envs, device=device)
                if term_joint and term_joint > 0:
                    drift = torch.maximum(drift, (joint_err > term_joint).float())
                if term_foot and term_foot > 0 and has_foot:
                    drift = torch.maximum(drift, (foot_err > term_foot).float())
                done = torch.maximum(done, drift * (~in_launch).float())

                # Directly the MDP into the IsaacLab Buffer, ignoring the Manager-based configurations
                buf.store(obs_t, acts, logps, reward, done, values)
                ep_return += reward
                ep_len += 1

                # Handle resets only the terminated envs (a launch_frac share of
                # them restart from rest; the rest use classic RSI).
                reset_mask = done.bool().cpu().numpy()
                if reset_mask.any():
                    _reset(np.nonzero(reset_mask)[0])
                    prev_acts[done.bool()] = 0.0   # no action-rate penalty across the reset seam
                    for r, L in zip(ep_return[done.bool()], ep_len[done.bool()]):
                        ep_returns.append(r.item())
                        ep_lengths.append(L.item() * sim_dt)
                    ep_return[done.bool()] = 0.0
                    ep_len[done.bool()] = 0.0

                obs = next_obs

            # TODO: Find a more suitable way of converting obs instead of converting it from dict
            # Bootstrap last value
            if isinstance(obs, dict):
                obs = obs['policy']

            last_val = value_net(build_policy_obs(obs, phase_t, clip, device, use_phase))
            adv, ret = buf.compute_returns(last_val)
            # Stability failsafes (no-ops in the normal range): bound the value targets
            # so a reward/value outlier can't explode the value loss (vf was seen hitting
            # 1e23) the cause of the training collapses we hit at high env counts.
            adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
            ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 500.0)

        policy.train()
        value_net.train()

        obs_flat, act_flat, old_logp_flat, _ = buf.get_flat()
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        # Bound normalised advantages: a single value-mispredict outlier otherwise lurches
        # the policy into an irrecoverable collapse. Normal advantages are within ~±4, so
        # this only clips genuine outliers.
        adv_norm = adv_norm.clamp(-10.0, 10.0)

        total_pg_loss = total_vf_loss = total_ent = 0.0
        n_updates = 0
        kl_sum, kl_cnt = 0.0, 0

        # Runaway guard: stop this iteration's epochs once a minibatch's approx-KL
        # blows past 10x the target (0.01). PPO's clipping is one-sided (a NEGATIVE
        # advantage with a grown ratio is NOT clipped), so the bimodal advantages
        # from the launch envs (steady gallop vs. failed launch) otherwise let the
        # policy drift within the 4 epochs until ratios explode (pg/kl -> 1e6).
        # Normal updates never approach this, so it is a no-op for the stable gaits.
        KL_STOP = 0.1
        stop_epochs = False
        for _ in range(n_epochs_per_iter):
            if stop_epochs:
                break
            idx = torch.randperm(obs_flat.shape[0], device=device)
            for start in range(0, idx.shape[0], minibatch):
                b = idx[start:start + minibatch]
                dist = policy(obs_flat[b])
                new_lp = dist.log_prob(act_flat[b]).sum(-1)
                # Clamp the log-ratio BEFORE exp (avoid +inf -> NaN on a spike).
                logratio = (new_lp - old_logp_flat[b]).clamp(-20.0, 20.0)
                ratio = logratio.exp()
                with torch.no_grad():           # track approx-KL for the adaptive LR + guard
                    approx_kl = float(((ratio - 1.0) - logratio).mean())
                if approx_kl > KL_STOP:          # drifted too far -> skip this update and stop epochs
                    stop_epochs = True           # (not counted in kl_sum: it's a skipped step)
                    break
                kl_sum += approx_kl; kl_cnt += 1

                pg_loss = -torch.min(
                    ratio * adv_norm[b],
                    ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_norm[b]
                ).mean()
                vf_loss = nn.functional.mse_loss(value_net(obs_flat[b]), ret[b])
                ent = dist.entropy().sum(-1).mean()

                loss = pg_loss + vf_coef * vf_loss - entropy_coef * ent
                if not torch.isfinite(loss):
                    # A rare spike still slipped through skip this step rather than
                    # let NaN/Inf gradients poison the weights (which kills all future
                    # forwards). Training continues from the last good weights.
                    continue
                optim.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(all_params, grad_clip)
                if not torch.isfinite(gnorm):
                    optim.zero_grad()        # inf/nan gradient would corrupt the weights -> skip
                    continue
                optim.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent += ent.item()
                n_updates += 1

        # Adaptive LR (rsl_rl trust region): if the policy moved too far this iteration
        # (the ratio-saturation -> pg=1e8 -> collapse that killed the horse clip), shrink
        # the LR; if it barely moved, grow it. Bounds the step size WITHOUT the hard
        # KL-stop that previously froze training.
        if kl_cnt > 0:
            mean_kl = kl_sum / kl_cnt
            if mean_kl > 2.0 * desired_kl:
                cur_lr = max(cur_lr / 1.5, 1e-5)
            elif 0.0 < mean_kl < 0.5 * desired_kl:
                cur_lr = min(cur_lr * 1.5, 1e-3)
            for g in optim.param_groups:
                g["lr"] = cur_lr

        mean_return = np.mean(ep_returns) if ep_returns else 0.0
        curve.append((iteration, mean_return, np.mean(ep_lengths) if ep_lengths else 0.0))

        if mean_return > best_mean_return and ep_returns:
            best_mean_return = mean_return
            torch.save({"iter": iteration, "return": float(best_mean_return),
                        "policy": policy.state_dict(),
                        "value": value_net.state_dict()}, save_path)

        if iteration % 200 == 0 or iteration == n_iters - 1:
            mean_len = np.mean(ep_lengths) if ep_lengths else 0.0
            tqdm.write(f"  iter {iteration:5d}/{n_iters}  "
                  f"return={mean_return:.3f}  best={best_mean_return:.3f}  "
                  f"survival={mean_len:.1f}s  "
                  f"pg={total_pg_loss / max(n_updates, 1):.4f}  "
                  f"vf={total_vf_loss / max(n_updates, 1):.4f}  "
                  f"ent={total_ent / max(n_updates, 1):.3f}  "
                  f"kl={kl_sum / max(kl_cnt, 1):.4f}  lr={cur_lr:.1e}")

            # Physical/tracking snapshot from the current sim state (cheap, every
            # 50 iters): the numbers that actually say whether it STEPS and MOVES.
            with torch.no_grad():
                rd = robot.data
                q_c = rd.joint_pos[:, :12]
                q_r = torch.as_tensor(get_joint_target_batch(clip, phase_t),
                                      dtype=torch.float32, device=device)
                r_j = torch.exp(-((q_c - q_r) ** 2).sum(-1) / sigma ** 2).mean().item()
                r_f = 0.0
                if has_foot:
                    f_c = compute_foot_track_sim(robot, foot_ids, device)
                    f_r = torch.as_tensor(get_foot_target_batch(clip, phase_t),
                                          dtype=torch.float32, device=device)
                    r_f = torch.exp(-((f_c - f_r).reshape(f_c.shape[0], -1) ** 2)
                                    .sum(-1) / foot_sigma ** 2).mean().item()
                feet_air = (rd.body_pos_w[:, foot_ids, 2] > 0.04).float().mean().item() * 100
                fwd = rd.root_lin_vel_b[:, 0].mean().item()
                bh = rd.root_pos_w[:, 2].mean().item()
            tqdm.write(f"     [track] r_joint={r_j:.2f} r_foot={r_f:.2f}  "
                  f"feet_air={feet_air:.0f}%  fwd_vel={fwd:+.2f}m/s  base_h={bh:.2f}m"
                  f"  (ref fwd≈+0.30, base≈0.50)")

    print(f"[PPO] Best mean return: {best_mean_return:.3f}  saved → {save_path}")
    try:
        np.savetxt(os.path.join(os.path.dirname(save_path), "curve.csv"),
                   np.asarray(curve, dtype=np.float32), delimiter=",",
                   header="iter,return,survival_s", comments="")
    except Exception as e:
        print(f"[PPO] curve.csv not written: {e}")


def _disc_grad_penalty(disc: Discriminator, real: torch.Tensor) -> torch.Tensor:
    """E[ ||grad_x D(x)||^2 ] on real samples stabilises adversarial training."""
    real = real.detach().clone().requires_grad_(True)
    d = disc(real)
    grad = torch.autograd.grad(d.sum(), real, create_graph=True)[0]
    return grad.pow(2).sum(dim=-1).mean()


def train_amp(env, clips: "list[dict] | dict", policy: MLP, value_net: ValueNet,
              device: str, n_iters: int = 3000, horizon: int = 24, lr: float = 1e-4,
              disc_lr: float = 1e-4, disc_epochs: int = 1, grad_penalty: float = 10.0,
              disc_wd: float = 1e-3, logit_reg: float = 0.01,
              reward_scale: float = 2.0, save_path: str = "policy_amp.pt",
              clip_eps: float = 0.2, entropy_coef: float = 0.0, vf_coef: float = 0.5,
              grad_clip: float = 1.0, n_epochs_per_iter: int = 4, minibatch: int = 512,
              replay_size: int = 200000, action_rate_w: float = 0.0, window: int = 2,
              task_vel_w: float = 0.0):
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    foot_ids = [list(robot.body_names).index(n) for n in AMP_FOOT_BODIES]

    if isinstance(clips, dict):           # allow a single clip too
        clips = [clips]
    window = max(2, int(window))
    clip_durations = np.array([clip_data_duration(c) for c in clips])
    ref_trans, feat_mean, feat_std, feat_mask = build_reference_amp_multi(clips, device, sim_dt, window=window)
    M = ref_trans.shape[0]
    masked_dim = int(feat_mask.sum())
    # Discriminator sees a `window`-frame segment of the kept (non-constant) feats,
    # so motion-over-time is visible and standing-still is obviously fake.
    disc = Discriminator(masked_dim, window=window).to(device)

    all_params = list(policy.parameters()) + list(value_net.parameters())
    optim = torch.optim.Adam(all_params, lr=lr)
    # AdamW weight decay + a logit penalty below keep the discriminator from
    # memorizing the tiny reference set and saturating.
    disc_optim = torch.optim.AdamW(disc.parameters(), lr=disc_lr, weight_decay=disc_wd)
    buf = PPOBuffer(num_envs, horizon, OBS_DIM, ACTION_DIM, device)

    # Replay buffer of past policy ("fake") transitions: training the
    # discriminator only against the latest rollout (all from one near-identical
    # policy) lets it separate trivially. A rolling history of fakes keeps the
    # negative distribution diverse and slows saturation.
    replay = torch.zeros(0, window * masked_dim, device=device)

    # Reference-state initialisation (RSI): start each episode at a random phase of
    # a RANDOMLY-ASSIGNED clip from the library, so the policy must learn the whole
    # distribution of gaits rather than one cycle.
    env_clip_idx = np.random.randint(0, len(clips), size=num_envs)
    phase_t = np.array([np.random.uniform(0, clip_durations[env_clip_idx[i]]) for i in range(num_envs)])
    obs, _ = env.reset()
    obs = obs["policy"]
    init_joints_to_motion_amp_multi(env, clips, env_clip_idx, phase_t, device)

    def cur_feats():
        raw = compute_amp_obs_sim(robot, foot_ids, device)[:, feat_mask]
        return (raw - feat_mean) / feat_std

    # Rolling buffer of the last `window` per-frame feature vectors (per env),
    # initialised to the start pose. The sim-side AMP "window" mirrors the
    # reference window layout (oldest -> newest).
    feat_buf = cur_feats().unsqueeze(0).repeat(window, 1, 1)   # (window, N, Fk)
    prev_acts = None          # for the action-rate smoothness penalty
    last_arate = 0.0          # running mean penalty, for logging
    last_vtrack = 0.0         # running mean velocity-tracking task reward

    best_mean_return = -float("inf")
    ep_returns, ep_lengths = deque(maxlen=100), deque(maxlen=100)
    ep_return = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, device=device)

    print(f"\n[AMP] {n_iters} iters horizon={horizon}  lr={lr}  disc_lr={disc_lr}  "
          f"reward_scale={reward_scale}  action_rate_w={action_rate_w}  (style + smoothness regularizer)")

    last_vx = last_air = last_bh = last_cerr = 0.0   # physical diagnostics (per rollout)
    for iteration in trange(n_iters, file=sys.__stdout__):
        policy.eval(); value_net.eval()
        rollout_trans = []
        ph_vx = ph_air = ph_bh = ph_cerr = 0.0; ph_n = 0
        with torch.no_grad():
            for _ in range(horizon):
                if isinstance(obs, dict):
                    obs = obs["policy"]
                obs_t = obs[:, :OBS_DIM].to(device)
                dist = policy(obs_t)
                acts = dist.rsample()
                acts = clamp_knee_actions(acts)        # hard X-config: knees can't flip outward
                logps = dist.log_prob(acts).sum(-1)
                values = value_net(obs_t)

                next_obs, _, terminated, truncated, _ = env.step(acts)
                done = (terminated | truncated).float()

                # physical diagnostics: what the robot actually does this step
                _rd = robot.data
                _cmd = obs_t[:, 9:12]
                _vb, _wb = _rd.root_lin_vel_b, _rd.root_ang_vel_b
                _footz = _rd.body_pos_w[:, foot_ids, 2]
                ph_vx += _vb[:, 0].mean().item()
                ph_air += (_footz > 0.04).float().mean().item()
                ph_bh += _rd.root_pos_w[:, 2].mean().item()
                ph_cerr += (((_vb[:, 0] - _cmd[:, 0]) ** 2 + (_vb[:, 1] - _cmd[:, 1]) ** 2
                             + (_wb[:, 2] - _cmd[:, 2]) ** 2).sqrt()).mean().item()
                ph_n += 1

                cur_feat = cur_feats()
                feat_buf = torch.roll(feat_buf, shifts=-1, dims=0)
                feat_buf[-1] = cur_feat
                trans = feat_buf.permute(1, 0, 2).reshape(num_envs, -1)  # (N, window*Fk)
                reward = disc.style_reward(trans, reward_scale)          # (N,)

                # Action-rate smoothness penalty: discourage frame-to-frame target
                # jitter (the discriminator judges single transitions and is blind
                # to high-freq wiggle, so the policy otherwise jitters in place).
                # This is a smoothness PRIOR, not a task/gait/velocity objective.
                if action_rate_w > 0.0 and prev_acts is not None:
                    arate = ((acts - prev_acts) ** 2).mean(dim=-1)       # (N,)
                    reward = reward - action_rate_w * arate
                    last_arate = float(arate.mean().item())
                prev_acts = acts.detach()

                # Optional velocity-command TASK reward (positive, in [0,1]). This
                # is the only hand-specified term and it specifies a GOAL (go at the
                # commanded speed/heading), NOT a gait the gait still comes purely
                # from the style discriminator. The discriminator is root-motion-
                # blind (body-frame features), so translating forward does NOT fight
                # the style reward. obs layout: lin_vel0:3, ang_vel3:6, grav6:9, cmd9:12.
                if task_vel_w > 0.0:
                    cmd = obs_t[:, 9:12]
                    vb = robot.data.root_lin_vel_b
                    wb = robot.data.root_ang_vel_b
                    lin_err = (vb[:, 0] - cmd[:, 0]) ** 2 + (vb[:, 1] - cmd[:, 1]) ** 2
                    ang_err = (wb[:, 2] - cmd[:, 2]) ** 2
                    vtrack = 0.75 * torch.exp(-lin_err / 0.25) + 0.25 * torch.exp(-ang_err / 0.25)
                    reward = reward + task_vel_w * vtrack
                    last_vtrack = float(vtrack.mean().item())

                buf.store(obs_t, acts, logps, reward, done, values)
                rollout_trans.append(trans)
                ep_return += reward
                ep_len += 1

                reset_mask = done.bool().cpu().numpy()
                if reset_mask.any():
                    ridx = np.nonzero(reset_mask)[0]
                    env_clip_idx[ridx] = np.random.randint(0, len(clips), size=ridx.size)
                    phase_t[ridx] = np.array([np.random.uniform(0, clip_durations[e]) for e in env_clip_idx[ridx]])
                    init_joints_to_motion_amp_multi(env, clips, env_clip_idx, phase_t, device, env_ids=ridx)
                    # Refill the just-reset envs' window with their post-teleport
                    # pose so the next windows don't span the RSI discontinuity.
                    fresh = cur_feats()
                    feat_buf[:, ridx, :] = fresh[ridx].unsqueeze(0)
                    for r, L in zip(ep_return[done.bool()], ep_len[done.bool()]):
                        ep_returns.append(r.item())
                        ep_lengths.append(L.item() * sim_dt)
                    ep_return[done.bool()] = 0.0
                    ep_len[done.bool()] = 0.0

                obs = next_obs

            if isinstance(obs, dict):
                obs = obs["policy"]
            last_val = value_net(obs[:, :OBS_DIM].to(device))
            adv, ret = buf.compute_returns(last_val)

        n = max(ph_n, 1)
        last_vx, last_air, last_bh, last_cerr = ph_vx / n, ph_air / n, ph_bh / n, ph_cerr / n

        # -------- discriminator update (LSGAN + gradient penalty + regularization) --------
        new_fakes = torch.cat(rollout_trans, dim=0).detach()             # (H*N, 2F)
        replay = torch.cat([replay, new_fakes], dim=0)[-replay_size:]    # rolling history
        disc.train()
        d_real_acc = d_fake_acc = d_loss_val = 0.0
        n_dupd = 0
        # Fix the number of discriminator minibatch steps to the amount of FRESH
        # data per iter (not the replay size), so a large replay buffer does not
        # silently train the discriminator harder and re-trigger saturation.
        n_disc_updates = disc_epochs * max(1, new_fakes.shape[0] // minibatch)
        for _ in range(n_disc_updates):
            pb = replay[torch.randint(0, replay.shape[0], (minibatch,), device=device)]
            rb = ref_trans[torch.randint(0, M, (minibatch,), device=device)]
            d_real = disc(rb)
            d_fake = disc(pb)
            # LSGAN + gradient penalty on real + logit (output) regularization,
            # the latter directly discouraging the huge confident logits that
            # cause saturation.
            loss_d = (0.5 * ((d_real - 1.0) ** 2).mean()
                      + 0.5 * ((d_fake + 1.0) ** 2).mean()
                      + 0.5 * grad_penalty * _disc_grad_penalty(disc, rb)
                      + logit_reg * (d_real ** 2).mean())
            disc_optim.zero_grad()
            loss_d.backward()
            disc_optim.step()
            d_real_acc += (d_real > 0).float().mean().item()
            d_fake_acc += (d_fake < 0).float().mean().item()
            d_loss_val += loss_d.item()
            n_dupd += 1

        # -------- PPO update on the learned style reward --------
        policy.train(); value_net.train()
        obs_flat, act_flat, old_logp_flat, _ = buf.get_flat()
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(n_epochs_per_iter):
            idx = torch.randperm(obs_flat.shape[0], device=device)
            for start in range(0, idx.shape[0], minibatch):
                b = idx[start:start + minibatch]
                dist = policy(obs_flat[b])
                new_lp = dist.log_prob(act_flat[b]).sum(-1)
                ratio = (new_lp - old_logp_flat[b]).exp()
                pg_loss = -torch.min(
                    ratio * adv_norm[b],
                    ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_norm[b],
                ).mean()
                vf_loss = nn.functional.mse_loss(value_net(obs_flat[b]), ret[b])
                ent = dist.entropy().sum(-1).mean()
                loss = pg_loss + vf_coef * vf_loss - entropy_coef * ent
                optim.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(all_params, grad_clip)
                if not torch.isfinite(gnorm):
                    optim.zero_grad()        # inf/nan gradient would corrupt the weights -> skip
                    continue
                optim.step()

        mean_return = np.mean(ep_returns) if ep_returns else 0.0
        mean_len = np.mean(ep_lengths) if ep_lengths else 0.0
        # Per-step style reward this rollout the metric that actually says
        # whether the discriminator is giving the policy a usable signal.
        # (Sign accuracy being ~1.0 is normal for a healthy LSGAN and does NOT
        # mean the reward has vanished.)
        mean_step_reward = buf.rewards.mean().item()
        d_loss_avg = d_loss_val / max(n_dupd, 1)
        if mean_return > best_mean_return and ep_returns:
            best_mean_return = mean_return
            torch.save({"iter": iteration, "return": float(best_mean_return),
                        "policy": policy.state_dict(), "value": value_net.state_dict(),
                        "disc": disc.state_dict(),
                        "amp_feat_mean": feat_mean.cpu().numpy(),
                        "amp_feat_std": feat_std.cpu().numpy(),
                        "amp_feat_mask": feat_mask.cpu().numpy()}, save_path)

        if iteration % 50 == 0 or iteration == n_iters - 1:
            tqdm.write(f"  iter {iteration:5d}/{n_iters}  style_return={mean_return:7.2f}  "
                  f"best={best_mean_return:7.2f}  survival={mean_len:4.1f}s  "
                  f"step_reward={mean_step_reward:.3f}/{reward_scale:.1f}  "
                  f"d_loss={d_loss_avg:.3f}  "
                  f"disc_acc(r/f)={d_real_acc / max(n_dupd,1):.2f}/{d_fake_acc / max(n_dupd,1):.2f}"
                  + (f"  arate_pen={last_arate:.4f}" if action_rate_w > 0 else "")
                  + (f"  vtrack={last_vtrack:.3f}" if task_vel_w > 0 else ""))
            tqdm.write(f"           [phys] fwd_vel={last_vx:+.3f} m/s  cmd_err={last_cerr:.3f}  "
                       f"feet_air={last_air*100:.0f}%  base_h={last_bh:.2f}m  "
                       f"(walking if fwd_vel tracks cmd & feet_air>~20%)")
            # Real collapse = the reward actually died AND the discriminator
            # stopped learning (loss ~0). Sign accuracy alone is not the signal.
            if mean_step_reward < 0.05 * reward_scale and d_loss_avg < 0.01:
                tqdm.write("    [AMP] WARNING: style reward collapsed (discriminator won outright). "
                      "Raise --amp_grad_penalty / --amp_disc_wd, or lower --amp_disc_lr.")

    print(f"[AMP] Best mean style return: {best_mean_return:.3f}  saved → {save_path}")


def clip_data_duration(clip_data: dict) -> float:
    T = int(np.asarray(clip_data["joint_pos_isaac"]).shape[0])
    dt = float(clip_data.get("frame_duration", clip_data.get("dt", 1.0 / 30.0)))
    return T * dt


def init_joints_to_motion_amp(env, clip_data: dict, phase_t_arr: np.ndarray, device: str, env_ids=None):
    """RSI for AMP: write reference joint poses (sampled by phase) to the given envs.

    Uses the raw clip frames directly (no spline), so it does not depend on the
    DeepMimic-style `clip` dict that the bc/rl paths build.
    """
    robot = env.unwrapped.scene["robot"]
    am = env.unwrapped.action_manager
    jids = am._terms["joint_pos"]._joint_ids
    q_all = np.asarray(clip_data["joint_pos_isaac"], dtype=np.float32)
    T = q_all.shape[0]
    dt = float(clip_data.get("frame_duration", clip_data.get("dt", 1.0 / 30.0)))

    sel = np.arange(len(phase_t_arr)) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
    frames = (np.round(phase_t_arr[sel] / dt).astype(int)) % T
    q_np = q_all[frames]
    q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
    v_t = torch.zeros_like(q_t)
    ids_t = None if env_ids is None else torch.as_tensor(sel, dtype=torch.long, device=device)
    robot.write_joint_state_to_sim(q_t, v_t, joint_ids=jids, env_ids=ids_t)
    env.unwrapped.scene.write_data_to_sim()


def init_joints_to_motion_amp_multi(env, clips: list[dict], env_clip_idx: np.ndarray,
                                    phase_t_arr: np.ndarray, device: str, env_ids=None):
    """RSI across a clip LIBRARY: each env is initialised from its assigned clip
    (env_clip_idx[e]) at its phase. Generalises init_joints_to_motion_amp to many
    clips of differing length/dt."""
    robot = env.unwrapped.scene["robot"]
    am = env.unwrapped.action_manager
    jids = am._terms["joint_pos"]._joint_ids
    sel = np.arange(len(phase_t_arr)) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
    q_np = np.zeros((len(sel), 12), dtype=np.float32)
    for k, e in enumerate(sel):
        cd = clips[int(env_clip_idx[e])]
        q_all = np.asarray(cd["joint_pos_isaac"], dtype=np.float32)
        T = q_all.shape[0]
        dt = float(cd.get("frame_duration", cd.get("dt", 1.0 / 30.0)))
        q_np[k] = q_all[int(round(phase_t_arr[e] / dt)) % T]
    q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
    v_t = torch.zeros_like(q_t)
    ids_t = None if env_ids is None else torch.as_tensor(sel, dtype=torch.long, device=device)
    robot.write_joint_state_to_sim(q_t, v_t, joint_ids=jids, env_ids=ids_t)
    env.unwrapped.scene.write_data_to_sim()


def evaluate(env, clip: dict, policy: MLP, device: str, n_episodes: int = 5,
             use_phase: bool = False, dump_rollout: str | None = None):
    robot = env.unwrapped.scene["robot"]
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    num_envs = env.unwrapped.num_envs

    phase_t = np.zeros(num_envs)
    obs, _ = env.reset()
    obs = obs['policy']
    init_joints_to_motion(env, clip, phase_t, device)

    policy.eval()
    ep_returns = []      # per-step tracking reward, averaged per episode
    ep_lengths = []      # episode survival time [s]
    ep_return = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, device=device)

    # ---- physical-behaviour diagnostics: tell standing / in-place-stepping /
    # walking apart, and whether the policy responds to velocity commands ----
    foot_ids = [list(robot.body_names).index(n) for n in AMP_FOOT_BODIES]
    # contact sensor -> TRUE per-foot contact force (for reliable gait/duty; foot-height
    # contact detection is confounded by base bounce). None-safe if the sensor is absent.
    _cs = env.unwrapped.scene.sensors.get("contact_forces") if hasattr(env.unwrapped.scene, "sensors") else None
    cs_foot_ids = ([list(_cs.body_names).index(n) for n in AMP_FOOT_BODIES]
                   if _cs is not None else None)
    phys = {k: [] for k in ("vx", "vy", "wz", "cmd_vx", "cmd_vy", "cmd_wz",
                            "base_h", "jvel", "foot_air", "cmd_err")}

    # ---- reference-fidelity diagnostics: does the DETERMINISTIC policy actually
    # reproduce the clip's gait? The training [track] line is STOCHASTIC and is
    # the average over a noisy/exploring policy; THIS measures the deployable
    # (mean) policy against the reference per-joint and per-foot. ----
    FOOT_LBL = ["LF", "RF", "LH", "RH"]
    _ts = np.linspace(0, clip["duration"], 400)
    _fr = get_foot_target_batch(clip, _ts)                       # (400,4,3) body-frame or None
    has_foot_fid = _fr is not None
    ref_clear = np.ptp(_fr[:, :, 2], axis=0) if has_foot_fid else np.zeros(4)   # per-foot z swing
    ref_stride = np.ptp(_fr[:, :, 0], axis=0) if has_foot_fid else np.zeros(4)  # per-foot x sweep
    jerr_sum = np.zeros(12)            # accumulated mean |q - q_ref| per joint
    jsq_sum, jsq_n = 0.0, 0            # accumulated mean squared joint error -> RMS e_q (deg)
    ferr_hist = []                     # per-step mean per-foot Cartesian error (m) -> e_p (cm)
    foot_hist = []                     # per-step body-frame foot pos (env 0)
    rj_hist, rf_hist = [], []          # per-step component rewards (sigmas 0.5/0.3)
    n_fid = 0

    # ---- rollout capture: env-0 WORLD foot + joint trajectory for the LONGEST
    # single episode (no mid-episode resets), so scripts/gait_signature.py can
    # verify the deployed policy reproduces the reference footfall. Keeping the
    # longest (not the first) avoids an unlucky short first episode -> a noisy/NaN
    # footfall reading. ----
    roll_capture = dump_rollout is not None
    roll_cur_f, roll_cur_j = [], []          # current in-progress episode
    roll_cur_fb, roll_cur_b = [], []         # body-frame feet + base pose (gait / support-polygon)
    roll_cur_jv, roll_cur_tau, roll_cur_a = [], [], []   # joint vel, applied torque, action (naturalness)
    roll_cur_c = []                          # per-foot contact-force magnitude (true gait/duty)
    roll_foot, roll_joint = [], []           # best (longest) episode so far
    roll_footb, roll_base = [], []
    roll_jv, roll_tau, roll_a, roll_c = [], [], [], []

    def log_phys():
        rd = robot.data
        cmd = obs[:, 9:12] if obs.shape[1] >= 12 else torch.zeros((num_envs, 3), device=device)
        vb = rd.root_lin_vel_b                                   # body-frame lin vel
        wb = rd.root_ang_vel_b
        foot_h = rd.body_pos_w[:, foot_ids, 2]                   # (N,4)
        phys["vx"].append(vb[:, 0].mean().item());  phys["vy"].append(vb[:, 1].mean().item())
        phys["wz"].append(wb[:, 2].mean().item())
        phys["cmd_vx"].append(cmd[:, 0].mean().item()); phys["cmd_vy"].append(cmd[:, 1].mean().item())
        phys["cmd_wz"].append(cmd[:, 2].mean().item())
        phys["base_h"].append(rd.root_pos_w[:, 2].mean().item())
        phys["jvel"].append(rd.joint_vel[:, :12].abs().mean().item())
        phys["foot_air"].append((foot_h > foot_h.amin() + 0.04).float().mean().item())
        cerr = ((vb[:, 0] - cmd[:, 0]) ** 2 + (vb[:, 1] - cmd[:, 1]) ** 2
                + (wb[:, 2] - cmd[:, 2]) ** 2).sqrt().mean().item()
        phys["cmd_err"].append(cerr)

    print(f"\n[Eval] Running {n_episodes} episodes...")

    while len(ep_returns) < n_episodes * num_envs:
        with torch.no_grad():
            if isinstance(obs, dict):
                obs = obs['policy']
            obs_t = build_policy_obs(obs, phase_t, clip, device, use_phase)
            acts = policy.act(obs_t, deterministic=True)
            acts = clamp_knee_actions(acts)            # hard X-config: knees can't flip outward

        obs, _, terminated, truncated, _ = env.step(acts)
        if isinstance(obs, dict):
            obs = obs['policy']
        phase_t += sim_dt
        log_phys()

        if roll_capture:                                   # env-0 world foot + joints, this step
            roll_cur_f.append(robot.data.body_pos_w[0, foot_ids, :].cpu().numpy())
            roll_cur_j.append(robot.data.joint_pos[0, :12].cpu().numpy())
            roll_cur_fb.append(compute_foot_track_sim(robot, foot_ids, device)[0].cpu().numpy())   # (4,3) body frame
            roll_cur_b.append(torch.cat([robot.data.root_pos_w[0], robot.data.root_quat_w[0]]).cpu().numpy())  # (7,) pos+quat
            roll_cur_jv.append(robot.data.joint_vel[0, :12].cpu().numpy())          # (12,) rad/s  (naturalness)
            _tau = getattr(robot.data, "applied_torque", None)                      # (12,) N*m applied at joints
            roll_cur_tau.append((_tau[0, :12] if _tau is not None else torch.zeros(12, device=device)).cpu().numpy())
            roll_cur_a.append(acts[0].detach().cpu().numpy())                       # (12,) policy action
            if cs_foot_ids is not None:                                             # (4,) N contact force per foot
                roll_cur_c.append(torch.linalg.norm(
                    _cs.data.net_forces_w[0, cs_foot_ids, :], dim=-1).cpu().numpy())
            else:
                roll_cur_c.append(np.zeros(4, np.float32))

        q_cur = robot.data.joint_pos[:, :12]
        q_ref_np = get_joint_target_batch(clip, phase_t)
        q_ref = torch.tensor(q_ref_np, dtype=torch.float32, device=device)
        reward = motion_tracking_reward(q_cur, q_ref)

        # ---- fidelity accumulation (deterministic policy vs reference) ----
        jerr_sum += (q_cur - q_ref).abs().mean(dim=0).cpu().numpy()
        jsq_sum += ((q_cur - q_ref) ** 2).mean().item(); jsq_n += 1        # RMS over joints+envs
        rj_hist.append(torch.exp(-((q_cur - q_ref) ** 2).sum(-1) / 0.5 ** 2).mean().item())
        if has_foot_fid:
            f_c = compute_foot_track_sim(robot, foot_ids, device)          # (N,4,3), z=roll-invariant clearance
            f_r = torch.as_tensor(get_foot_target_batch(clip, phase_t),
                                  dtype=torch.float32, device=device)
            rf_hist.append(torch.exp(-((f_c - f_r).reshape(f_c.shape[0], -1) ** 2)
                                     .sum(-1) / 0.3 ** 2).mean().item())
            ferr_hist.append(torch.linalg.norm(f_c - f_r, dim=-1).mean().item())  # mean per-foot ||.|| (m)
            foot_hist.append(f_c[0].cpu().numpy())                          # env 0
        n_fid += 1

        ep_return += reward
        ep_len += 1
        done = (terminated | truncated).bool()
        if done.any():
            # Per-EPISODE normalization (a global step counter would deflate
            # every episode after the first and make runs incomparable).
            for r, L in zip(ep_return[done], ep_len[done]):
                ep_returns.append((r / L.clamp(min=1)).item())
                ep_lengths.append(L.item() * sim_dt)
            ep_return[done] = 0.0
            ep_len[done] = 0.0
            done_np = done.cpu().numpy()
            if roll_capture and done_np[0]:                # env 0 finished an episode: keep the longest
                if len(roll_cur_f) > len(roll_foot):
                    roll_foot, roll_joint, roll_footb, roll_base = roll_cur_f, roll_cur_j, roll_cur_fb, roll_cur_b
                    roll_jv, roll_tau, roll_a, roll_c = roll_cur_jv, roll_cur_tau, roll_cur_a, roll_cur_c
                roll_cur_f, roll_cur_j, roll_cur_fb, roll_cur_b = [], [], [], []
                roll_cur_jv, roll_cur_tau, roll_cur_a, roll_cur_c = [], [], [], []
            phase_t[done_np] = 0.0
            init_joints_to_motion(env, clip, phase_t, device,
                                  env_ids=np.nonzero(done_np)[0])

    if roll_capture and len(roll_cur_f) > len(roll_foot):  # a still-running final episode may be longest
        roll_foot, roll_joint, roll_footb, roll_base = roll_cur_f, roll_cur_j, roll_cur_fb, roll_cur_b
        roll_jv, roll_tau, roll_a, roll_c = roll_cur_jv, roll_cur_tau, roll_cur_a, roll_cur_c
    if dump_rollout is not None and len(roll_foot) >= 20:
        np.savez(dump_rollout,
                 foot_pos=np.asarray(roll_foot, np.float32),          # (T,4,3) WORLD, order LF RF LH RH
                 foot_pos_body=np.asarray(roll_footb, np.float32),    # (T,4,3) BODY frame (contact/gait)
                 base_pose=np.asarray(roll_base, np.float32),         # (T,7) root pos(3)+quat(4) WORLD
                 joint_pos_isaac=np.asarray(roll_joint, np.float32),  # (T,12)
                 joint_vel=np.asarray(roll_jv, np.float32),           # (T,12) rad/s  (naturalness: saturation)
                 torque=np.asarray(roll_tau, np.float32),             # (T,12) N*m    (naturalness: peak/RMS torque, CoT)
                 action=np.asarray(roll_a, np.float32),               # (T,12)        (naturalness: action-rate RMS)
                 contact_force=np.asarray(roll_c, np.float32),        # (T,4) N       (TRUE contact -> duty/gait)
                 fps=np.float32(1.0 / sim_dt),
                 foot_names=np.array(AMP_FOOT_BODIES))
        print(f"[Eval] Saved first-episode rollout ({len(roll_foot)} steps @ {1.0/sim_dt:.0f}Hz) "
              f"-> {dump_rollout}  (run scripts/gait_signature.py on it to check footfall order)")

    mean_r = np.mean(ep_returns)
    std_r = np.std(ep_returns)
    print(f"[Eval] Mean per-step tracking reward: {mean_r:.4f} ± {std_r:.4f}  (1.0 = perfect)")
    print(f"[Eval] Mean episode length (survival): {np.mean(ep_lengths):.2f}s ± {np.std(ep_lengths):.2f}s  "
          f"(clip duration {clip['duration']:.1f}s; episodes also end on timeout)")

    m = {k: float(np.mean(v)) for k, v in phys.items()}
    print("\n[Eval] ---- physical behaviour (what the robot actually does) ----")
    print(f"  forward vel  vx = {m['vx']:+.3f} m/s   (commanded {m['cmd_vx']:+.3f})")
    print(f"  lateral vel  vy = {m['vy']:+.3f} m/s   (commanded {m['cmd_vy']:+.3f})")
    print(f"  yaw rate     wz = {m['wz']:+.3f} rad/s (commanded {m['cmd_wz']:+.3f})")
    print(f"  velocity-command tracking error = {m['cmd_err']:.3f}  (0 = perfect; ~|cmd| = ignoring commands)")
    print(f"  base height = {m['base_h']:.3f} m   | mean |joint vel| = {m['jvel']:.3f} rad/s")
    print(f"  feet airborne (stepping) = {m['foot_air']*100:.0f}% of foot-steps")
    # quick verdict
    moving = abs(m['vx']) > 0.1 or abs(m['vy']) > 0.1 or abs(m['wz']) > 0.2
    stepping = m['foot_air'] > 0.05 and m['jvel'] > 0.5
    tracks = m['cmd_err'] < 0.5 * (abs(m['cmd_vx']) + abs(m['cmd_vy']) + abs(m['cmd_wz']) + 1e-6)
    print(f"  => {'WALKS/MOVES' if moving else 'IN-PLACE (no net locomotion)'}; "
          f"{'STEPPING' if stepping else 'near-FROZEN legs'}; "
          f"{'follows commands' if tracks else 'IGNORES velocity commands (expected: no task reward + in-place refs)'}")

    # ---- reference fidelity: WHERE does the deployable gait deviate from the clip? ----
    results = {
        "survival_s": round(float(np.mean(ep_lengths)) if ep_lengths else 0.0, 2),
        "mean_tracking_reward": round(float(mean_r), 4),
        "forward_vel_mps": round(m["vx"], 3),
        "base_height_m": round(m["base_h"], 3),
        "feet_air_pct": round(m["foot_air"] * 100, 1),
        "cmd_track_err": round(m["cmd_err"], 3),
    }
    if n_fid > 0:
        jerr = jerr_sum / n_fid                                  # (12,) mean |q-q_ref| per joint
        # joint groups (Isaac BFS: 0-3 HAA, 4-7 HFE, 8-11 KFE; per group LF,LH,RF,RH).
        # front = LF,RF (group idx 0,2); hind = LH,RH (group idx 1,3).
        fh = lambda base: (float(np.mean([jerr[base], jerr[base + 2]])),
                           float(np.mean([jerr[base + 1], jerr[base + 3]])))
        haaF, haaH = fh(0); hfeF, hfeH = fh(4); kfeF, kfeH = fh(8)
        results["r_joint"] = round(float(np.mean(rj_hist)), 3)
        results["r_foot"] = round(float(np.mean(rf_hist)), 3) if rf_hist else None
        # PHYSICAL (sigma-independent) errors for the paper: RMS joint angle (deg) and
        # mean per-foot Cartesian error (cm).
        results["e_q_deg"] = round(float(np.rad2deg(np.sqrt(jsq_sum / max(jsq_n, 1)))), 1)
        results["e_p_cm"] = round(float(np.mean(ferr_hist) * 100.0), 1) if ferr_hist else None
        print("\n[Eval] ---- reference fidelity (DETERMINISTIC policy vs clip gait) ----")
        print(f"  PHYSICAL error: e_q = {results['e_q_deg']} deg (RMS joint)  |  "
              f"e_p = {results['e_p_cm']} cm (mean per-foot)")
        print(f"  per-step (deterministic):  r_joint={results['r_joint']:.2f}  "
              f"r_foot={results['r_foot'] if results['r_foot'] is not None else float('nan'):.2f}   (1.0=perfect; sigmas 0.5/0.3)")
        print(f"  joint err |q-q_ref| (rad)  front / hind:  "
              f"HAA {haaF:.2f}/{haaH:.2f}   HFE {hfeF:.2f}/{hfeH:.2f}   KFE {kfeF:.2f}/{kfeH:.2f}")
        if has_foot_fid and foot_hist:
            fa = np.asarray(foot_hist)                           # (S,4,3) body-frame
            ach_clear = np.percentile(fa[:, :, 2], 95, axis=0) - np.percentile(fa[:, :, 2], 5, axis=0)
            ach_stride = np.percentile(fa[:, :, 0], 95, axis=0) - np.percentile(fa[:, :, 0], 5, axis=0)
            results["clearance_pct"], results["stride_pct"] = {}, {}
            print("  foot CLEARANCE (body-frame z swing, m)  ref -> policy:")
            for i, l in enumerate(FOOT_LBL):
                frac = ach_clear[i] / ref_clear[i] if ref_clear[i] > 1e-3 else 1.0
                results["clearance_pct"][l] = round(float(frac * 100))
                flag = "  <-- DRAGS (≪ref)" if (ref_clear[i] > 0.06 and frac < 0.5) else ""
                print(f"     {l}: {ref_clear[i]:.2f} -> {ach_clear[i]:.2f}  ({frac*100:3.0f}% of ref){flag}")
            print("  foot STRIDE (body-frame x sweep, m)      ref -> policy:")
            for i, l in enumerate(FOOT_LBL):
                frac = ach_stride[i] / ref_stride[i] if ref_stride[i] > 1e-3 else 1.0
                results["stride_pct"][l] = round(float(frac * 100))
                print(f"     {l}: {ref_stride[i]:.2f} -> {ach_stride[i]:.2f}  ({frac*100:3.0f}% of ref)")
    return results

def plot_joint_motion(clip: dict, joint_idx: int = 0, num_cycles: int = 3, save_dir: str = "./checkpoints"):
    """
    Simulates high-resolution queries to the motion clip over multiple cycles
    to visualize the wrapping behavior and mathematical derivatives.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a high-resolution time array to see between the frames
    eval_dt = 0.005 # 5ms physics steps
    total_time = clip["duration"] * num_cycles
    t_eval = np.arange(0, total_time, eval_dt)
    
    # Query the spline for position and velocity using the wrapped time
    t_wrapped = (t_eval % clip["duration"])
    pos_eval = np.array(clip["joint_pos"])[t_wrapped.astype(int)]          # Position
    # vel_eval = clip["cs"](t_wrapped, 1)       # Velocity (1st derivative)
    vel_eval = get_ref_joint_velocities_batch(clip, t_eval)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Plot Position
    ax1.plot(t_eval, pos_eval[:, joint_idx], label=f"Spline Position (Joint {joint_idx})", color='blue')
    
    # Overlay the original mocap keyframes for reference
    # We plot them repeatedly for each cycle
    t_frames = np.linspace(0, clip["duration"], clip["T"] + 1)
    for i in range(num_cycles):
        ax1.plot(t_frames + (i * clip["duration"]), clip["joint_pos"][t_frames.astype(int)][:, joint_idx],
                 'ro', markersize=4, label="Keyframes" if i == 0 else "")
                 
    # 2. Plot Velocity
    ax2.plot(t_eval, vel_eval[:, joint_idx], label=f"Spline Velocity (Joint {joint_idx})", color='green')
    
    # Draw vertical lines at the wrap boundaries
    for i in range(num_cycles + 1):
        for ax in [ax1, ax2]:
            ax.axvline(i * clip["duration"], color='gray', linestyle='--', alpha=0.5)
            
    ax1.set_ylabel("Position (rad)")
    ax1.set_title(f"Target Position over {num_cycles} Cycles (Checking Wrap Behavior)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel("Velocity (rad/s)")
    ax2.set_xlabel("Time (Seconds)")
    ax2.set_title("Target Velocity (Checking for Jitter/Spikes)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"joint_{joint_idx}_motion.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"[Diagnostic] Saved motion graph to {plot_path}")


def plot_joint_motion_cubic(clip: dict, joint_idx: int = 0, num_cycles: int = 3, save_dir: str = "./checkpoints"):
    """
    Simulates high-resolution queries to the motion clip over multiple cycles
    to visualize the wrapping behavior and mathematical derivatives.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a high-resolution time array to see between the frames
    eval_dt = 0.005 # 5ms physics steps
    total_time = clip["duration"] * num_cycles
    t_eval = np.arange(0, total_time, eval_dt)
    
    # Query the spline for position and velocity using the wrapped time
    t_wrapped = (t_eval % clip["duration"])
    pos_eval = clip["cs"](t_wrapped)          # Position
    # vel_eval = clip["cs"](t_wrapped, 1)       # Velocity (1st derivative)
    vel_eval = get_ref_joint_velocities_batch(clip, t_eval)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Plot Position
    ax1.plot(t_eval, pos_eval[:, joint_idx], label=f"Spline Position (Joint {joint_idx})", color='blue')
    
    # Overlay the original mocap keyframes for reference
    # We plot them repeatedly for each cycle
    t_frames = np.linspace(0, clip["duration"], clip["T"] + 1)
    for i in range(num_cycles):
        ax1.plot(t_frames + (i * clip["duration"]), clip["cs"](t_frames.astype(int))[:, joint_idx],
                 'ro', markersize=4, label="Keyframes" if i == 0 else "")
                 
    # 2. Plot Velocity
    ax2.plot(t_eval, vel_eval[:, joint_idx], label=f"Spline Velocity (Joint {joint_idx})", color='green')
    
    # Draw vertical lines at the wrap boundaries
    for i in range(num_cycles + 1):
        for ax in [ax1, ax2]:
            ax.axvline(i * clip["duration"], color='gray', linestyle='--', alpha=0.5)
            
    ax1.set_ylabel("Position (rad)")
    ax1.set_title(f"Target Position over {num_cycles} Cycles (Checking Wrap Behavior)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel("Velocity (rad/s)")
    ax2.set_xlabel("Time (Seconds)")
    ax2.set_title("Target Velocity (Checking for Jitter/Spikes)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"joint_{joint_idx}_motion.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"[Diagnostic] Saved motion graph to {plot_path}")


def _resolve_motion_paths() -> list[str]:
    """Motion paths from --motion-list (one per line) or a single --motion."""
    paths = []
    if args_cli.motion_list:
        paths = [ln.strip() for ln in open(args_cli.motion_list) if ln.strip() and not ln.startswith("#")]
    elif args_cli.motion:
        paths = [args_cli.motion]
    if not paths:
        raise ValueError("provide --motion <npz> or --motion-list <file>")
    return paths


def auto_launch_phase(clip: dict) -> float:
    """Cycle fraction of the most-grounded (all-feet-down) stance, from cs_foot; 0 if absent."""
    if "cs_foot" not in clip:
        return 0.0
    ts = np.linspace(0, clip["duration"], 200, endpoint=False)
    z = np.asarray(clip["cs_foot"](ts)).reshape(len(ts), 4, 3)[:, :, 2]
    nd = (z < z.min() + 0.05).sum(1)
    sp = z.max(1) - z.min(1)
    best = min(range(len(ts)), key=lambda i: (-int(nd[i]), float(sp[i])))
    return float(ts[best] / clip["duration"])


def evaluate_from_rest(env, clip: dict, policy: MLP, device: str, launch_s: float = 1.5,
                       launch_phase: float = 0.276, n_episodes: int = 10, use_phase: bool = True,
                       max_s: float = 5.0) -> dict:
    """Test the STANDSTILL launch: init the robot at REST at the stance phase, spin the
    gait up with the launch ramp (phase rate 0->nominal over launch_s), and report how
    long it survives and the forward speed it reaches. This is what the classic eval does
    NOT test (it starts mid-gait, already moving)."""
    robot = env.unwrapped.scene["robot"]
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    num_envs = env.unwrapped.num_envs
    duration = clip["duration"]
    lp_t = (launch_phase % 1.0) * duration

    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    ref_v = float(np.abs(get_root_target_batch(clip, np.linspace(0, duration, 100))[:, 0]).mean()) \
        if "cs_root" in clip else 0.7

    phase_t = np.full(num_envs, lp_t)
    launch_t = np.zeros(num_envs)
    obs, _ = env.reset()
    obs = obs["policy"]
    reset_envs_to_reference(env, clip, phase_t, device, seed_velocity=False)
    policy.eval()

    surv, vmax = [], []
    ep_len = torch.zeros(num_envs, device=device)
    ep_vmax = torch.zeros(num_envs, device=device)
    max_steps = int(max_s / sim_dt)
    print(f"\n[LaunchEval] from REST at phase {launch_phase:.3f}, ramp {launch_s}s, cap {max_s}s "
          f"(ref fwd speed ~{ref_v:.2f} m/s)")

    while len(surv) < n_episodes * num_envs:
        with torch.no_grad():
            if isinstance(obs, dict):
                obs = obs["policy"]
            obs_t = build_policy_obs(obs, phase_t, clip, device, use_phase)
            acts = clamp_knee_actions(policy.act(obs_t, deterministic=True))
        obs, _, terminated, truncated, _ = env.step(acts)
        ramp = smoothstep(launch_t / launch_s)
        phase_t += ramp * sim_dt
        launch_t += sim_dt
        ep_vmax = torch.maximum(ep_vmax, robot.data.root_lin_vel_b[:, 0])
        ep_len += 1
        done = (terminated | truncated).bool() | (ep_len >= max_steps)
        if done.any():
            for L, v in zip(ep_len[done], ep_vmax[done]):
                surv.append(L.item() * sim_dt)
                vmax.append(v.item())
            ep_len[done] = 0.0
            ep_vmax[done] = 0.0
            dn = done.cpu().numpy()
            phase_t[dn] = lp_t
            launch_t[dn] = 0.0
            reset_envs_to_reference(env, clip, phase_t, device, env_ids=np.nonzero(dn)[0], seed_velocity=False)

    surv = np.asarray(surv)
    vmax = np.asarray(vmax)
    # "launched" = stayed up most of the window AND actually accelerated forward
    launched = (surv > 0.9 * max_s) & (vmax > max(0.4, 0.5 * ref_v))
    print(f"[LaunchEval] survival {surv.mean():.2f}+-{surv.std():.2f}s | max fwd vel "
          f"{vmax.mean():.2f}+-{vmax.std():.2f} m/s | launch success {launched.mean() * 100:.0f}% "
          f"(up >{0.9 * max_s:.1f}s AND >{max(0.4, 0.5 * ref_v):.2f} m/s)")
    return {"launch_survival_s": round(float(surv.mean()), 2),
            "launch_vmax_mps": round(float(vmax.mean()), 2),
            "launch_success_pct": round(float(launched.mean() * 100), 1)}


def main():
    device = "cuda:0"

    # Optional reproducible seeding for multi-seed studies (no-op for the default -1).
    if args_cli.seed is not None and args_cli.seed >= 0:
        np.random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)
        torch.cuda.manual_seed_all(args_cli.seed)
        print(f"[Seed] numpy/torch seeded with {args_cli.seed}")

    # Generate random 6 character hash for this run
    run_hash = uuid.uuid4().hex[:6]

    # In eval, the clip to compare against is the one the policy was TRAINED on. Read it
    # from the checkpoint's train_config.yaml so the eval can't be run against the wrong
    # clip (which silently produces meaningless fidelity numbers), and so --motion need
    # not be passed. An explicit --motion still overrides.
    if args_cli.mode == "eval" and args_cli.ckpt and args_cli.motion is None:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(args_cli.ckpt)), "train_config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                tcfg = yaml.safe_load(f) or {}
            tmotion = (tcfg.get("config") or {}).get("motion")
            if tmotion:
                args_cli.motion = tmotion
                print(f"[Eval] motion read from train_config.yaml: {tmotion}")
        if args_cli.motion is None:
            print("[Eval] WARNING: no train_config.yaml (or no motion in it) next to the ckpt; "
                  "falling back to the default clip fidelity numbers may be meaningless.")

    motion_paths = _resolve_motion_paths()
    if args_cli.motion is None:
        args_cli.motion = motion_paths[0]   # for the bc/rl/eval spline `clip` + plotting below
    # clip = load_motion(args_cli.motion)
    clip = load_motion_interpolated(args_cli.motion)

    plot_joint_motion_cubic(clip, joint_idx=0, num_cycles=3)

    # Single env for eval, multiple for training
    num_envs = 1 if args_cli.mode == "eval" else args_cli.num_envs
    env = make_env(num_envs, args_cli.stiffness, args_cli.damping)

    pobs_dim = OBS_DIM + (2 if args_cli.phase_obs else 0)
    if args_cli.phase_obs:
        print(f"[Main] phase observation enabled: obs dim {OBS_DIM} -> {pobs_dim}")

    if args_cli.mode == "bc":
        policy = MLP(pobs_dim, ACTION_DIM, stochastic=False)
        # Injected the random hash here
        save_path = os.path.join(args_cli.save_dir, f"policy_bc.pt")

        obs_buf, act_buf = collect_bc_data(
            env, clip, device, collect_s=args_cli.bc_collect_s,
            use_phase=args_cli.phase_obs)

        policy, loss_history = train_bc(
            obs_buf, act_buf, policy, device,
            epochs=args_cli.bc_epochs,
            batch=args_cli.bc_batch,
            lr=args_cli.bc_lr,
            save_path=save_path,
        )

        print(f"\n[BC] Training complete.  Evaluating trained policy...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_history) + 1), loss_history, label='MSE Loss', color='b', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean Squared Error', fontsize=12)
        plt.title('Behavioral Cloning Training Loss', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(args_cli.save_dir, "bc_loss_curve.png")
        plt.savefig(plot_path, dpi=300)
        plt.close() # Close the figure to free up memory
        print(f"[BC] Saved loss curve plot to {plot_path}")

        print(f"\n[BC] Training complete.  Evaluating trained policy...")
        evaluate(env, clip, policy.to(device), device, use_phase=args_cli.phase_obs)
        print(f"------------------- RUN HASH: {run_hash} -------------------")


    elif args_cli.mode == "rl":
        policy = MLP(pobs_dim, ACTION_DIM, stochastic=True)
        value_net = ValueNet(pobs_dim)
        # Each run gets its own folder, grouped per source clip:
        #   <save_dir>/<clip_stem>/policy_rl_<hash>/model.pt (+ train_config.yaml, logs).
        clip_stem = os.path.splitext(os.path.basename(args_cli.motion))[0]
        run_dir = os.path.join(args_cli.save_dir, clip_stem, f"policy_rl_{run_hash}")
        os.makedirs(run_dir, exist_ok=True)
        tee_stdout_to(os.path.join(run_dir, "train.log"))   # capture the full training log
        save_path = os.path.join(run_dir, "model.pt")
        save_run_yaml(os.path.join(run_dir, "train_config.yaml"), _args_to_dict(args_cli))
        print(f"[RL] run dir: {run_dir}")

        # Optionally warm-start from BC checkpoint
        bc_ckpt = args_cli.bc_ckpt or args_cli.ckpt
        if bc_ckpt and os.path.exists(bc_ckpt):
            # weights_only=False: our own checkpoints contain numpy scalars
            ck = torch.load(bc_ckpt, map_location=device, weights_only=False)
            sd = ck.get("state_dict") or ck.get("policy")
            ckpt_obs_dim = sd["net.0.weight"].shape[1]
            if ckpt_obs_dim != pobs_dim:
                raise ValueError(
                    f"BC checkpoint expects obs dim {ckpt_obs_dim} but this run uses {pobs_dim}. "
                    f"Pass {'--phase-obs' if ckpt_obs_dim == OBS_DIM + 2 else 'no --phase-obs'} "
                    f"to match how the BC policy was trained.")
            policy.load_state_dict(sd, strict=False)
            # Warm-start the VALUE net too when the checkpoint carries it. Loading only
            # the policy leaves a fresh value net predicting ~0 while the warm-started
            # policy already returns ~7-8, so every advantage is huge (~return) and the
            # first PPO updates explode (pg/kl -> 1e8, policy destroyed, survival -> 0).
            vsd = ck.get("value")
            if vsd is not None and vsd.get("net.0.weight", torch.empty(0, pobs_dim)).shape[-1] == pobs_dim:
                value_net.load_state_dict(vsd, strict=False)
                print(f"[RL] Loaded BC init from {bc_ckpt} (policy + value)")
            else:
                print(f"[RL] Loaded policy from {bc_ckpt}; value net is FRESH -- a warm-started "
                      f"run may be unstable (advantage blow-up). Prefer training from scratch.")
        elif bc_ckpt:
            print(f"[RL] Warning: checkpoint {bc_ckpt} not found, training from scratch.")

        train_ppo(
            env, clip, policy.to(device), value_net.to(device), device,
            n_iters=args_cli.rl_iters,
            horizon=args_cli.rl_steps,
            lr=args_cli.rl_lr,
            sigma=args_cli.rl_sigma,
            vel_weight=args_cli.rl_vel_weight,
            save_path=save_path,
            use_phase=args_cli.phase_obs,
            task_reward_weight=args_cli.task_reward_weight,
            joint_w=args_cli.rl_joint_w,
            foot_w=args_cli.rl_foot_w,
            root_w=args_cli.rl_root_w,
            foot_sigma=args_cli.rl_foot_sigma,
            root_sigma=args_cli.rl_root_sigma,
            term_joint=args_cli.rl_term_joint,
            term_foot=args_cli.rl_term_foot,
            entropy_coef=args_cli.rl_entropy_coef,
            action_rate_w=args_cli.rl_action_rate_w,
            foot_z_weight=args_cli.rl_foot_z_weight,
            launch_frac=args_cli.rl_launch_frac,
            launch_s=args_cli.rl_launch_s,
            launch_phase=args_cli.rl_launch_phase,
        )
        print(f"------------------- RUN HASH: {run_hash} -------------------")

        # Auto launch-eval on the BEST saved model: the standstill launch is the goal
        # of a launch-curriculum run and the classic eval does not test it. Writes
        # launch_eval_config.yaml into the run dir so a sweep can rank configs.
        if args_cli.rl_launch_frac > 0.0 and os.path.exists(save_path):
            best = torch.load(save_path, map_location=device, weights_only=False)
            eval_policy = MLP(pobs_dim, ACTION_DIM, stochastic=True).to(device)
            eval_policy.load_state_dict(best.get("policy") or best.get("state_dict"))
            lp = args_cli.rl_launch_phase if args_cli.rl_launch_phase >= 0 else auto_launch_phase(clip)
            res = evaluate_from_rest(env, clip, eval_policy, device, launch_s=args_cli.rl_launch_s,
                                     launch_phase=lp, use_phase=args_cli.phase_obs, n_episodes=3)
            save_run_yaml(os.path.join(run_dir, "launch_eval_config.yaml"), _args_to_dict(args_cli), res)


    elif args_cli.mode == "amp":
        if args_cli.phase_obs:
            raise ValueError("AMP is phaseless; do not pass --phase-obs.")
        # AMP needs the raw reference arrays (joint_pos_isaac, foot_pos_local), not
        # the spline-based `clip` dict. Load the WHOLE library for the discriminator.
        clips = []
        for p in motion_paths:
            cd = dict(np.load(p, allow_pickle=True))
            if "joint_pos_isaac" not in cd:
                print(f"[AMP] skipping {p}: no 'joint_pos_isaac'")
                continue
            clips.append(cd)
        if not clips:
            raise ValueError("AMP requires retarget.py clips with 'joint_pos_isaac' (and foot_pos_local).")
        print(f"[AMP] loaded {len(clips)} clip(s) for the reference library")
        policy = MLP(OBS_DIM, ACTION_DIM, stochastic=True)
        value_net = ValueNet(OBS_DIM)
        # Injected the random hash here
        save_path = os.path.join(args_cli.save_dir, f"policy_amp_{run_hash}.pt")

        # Optional BC warm-start: seed the policy with the reference gait (which
        # actually lifts feet / steps), so PPO refines a real stepping gait for
        # stability instead of discovering it from a shaky drift. The BC ckpt must
        # be phaseless (obs dim == OBS_DIM) to match the AMP policy.
        bc_ckpt = args_cli.bc_ckpt or args_cli.ckpt
        if bc_ckpt and os.path.exists(bc_ckpt):
            ck = torch.load(bc_ckpt, map_location=device, weights_only=False)
            sd = ck.get("state_dict") or ck.get("policy")
            ckpt_obs_dim = sd["net.0.weight"].shape[1]
            if ckpt_obs_dim != OBS_DIM:
                raise ValueError(f"BC ckpt obs dim {ckpt_obs_dim} != AMP obs dim {OBS_DIM}; "
                                 f"train BC WITHOUT --phase-obs for AMP warm-start.")
            policy.load_state_dict(sd, strict=False)   # strict=False: skips log_std
            print(f"[AMP] warm-started policy from BC checkpoint {bc_ckpt}")
        elif bc_ckpt:
            print(f"[AMP] Warning: BC ckpt {bc_ckpt} not found; training from scratch.")

        train_amp(
            env, clips, policy.to(device), value_net.to(device), device,
            n_iters=args_cli.amp_iters,
            horizon=args_cli.amp_steps,
            lr=args_cli.amp_lr,
            disc_lr=args_cli.amp_disc_lr,
            disc_epochs=args_cli.amp_disc_epochs,
            grad_penalty=args_cli.amp_grad_penalty,
            disc_wd=args_cli.amp_disc_wd,
            logit_reg=args_cli.amp_logit_reg,
            reward_scale=args_cli.amp_reward_scale,
            action_rate_w=args_cli.amp_action_rate_w,
            window=args_cli.amp_window,
            task_vel_w=args_cli.amp_task_vel_w,
            save_path=save_path,
        )
        print(f"------------------- RUN HASH: {run_hash} -------------------")

    elif args_cli.mode == "eval":
        if not args_cli.ckpt:
            raise ValueError("--ckpt required for eval mode")

        ckpt_dir = os.path.dirname(os.path.abspath(args_cli.ckpt))
        tee_stdout_to(os.path.join(ckpt_dir, "eval.log"))   # capture the full eval log

        # weights_only=False: our own checkpoints contain numpy scalars
        ck = torch.load(args_cli.ckpt, map_location=device, weights_only=False)
        # Auto-detect BC vs RL checkpoint
        is_rl = "policy" in ck
        sd = ck.get("policy") or ck.get("state_dict")
        # Auto-detect whether the checkpoint was trained with the phase clock
        # from its input-layer width no flag matching needed.
        ckpt_obs_dim = sd["net.0.weight"].shape[1]
        use_phase = ckpt_obs_dim == OBS_DIM + 2
        policy = MLP(ckpt_obs_dim, ACTION_DIM, stochastic=is_rl)
        policy.load_state_dict(sd)
        policy = policy.to(device)
        print(f"[Eval] Loaded {'RL' if is_rl else 'BC'} policy from {args_cli.ckpt} "
              f"(obs dim {ckpt_obs_dim}{', phase clock' if use_phase else ''})")

        eval_results = evaluate(env, clip, policy, device, n_episodes=10, use_phase=use_phase,
                                dump_rollout=os.path.join(ckpt_dir, "rollout.npz"))
        # Record eval config + results next to the model (ckpt_dir set above) so each
        # policy_rl_<hash>/ folder ends up with train_config.yaml + eval_config.yaml + logs.
        save_run_yaml(os.path.join(ckpt_dir, "eval_config.yaml"), _args_to_dict(args_cli), eval_results)

    elif args_cli.mode == "launch_eval":
        if not args_cli.ckpt:
            raise ValueError("--ckpt required for launch_eval mode")
        ckpt_dir = os.path.dirname(os.path.abspath(args_cli.ckpt))
        tee_stdout_to(os.path.join(ckpt_dir, "launch_eval.log"))
        ck = torch.load(args_cli.ckpt, map_location=device, weights_only=False)
        sd = ck.get("policy") or ck.get("state_dict")
        ckpt_obs_dim = sd["net.0.weight"].shape[1]
        use_phase = ckpt_obs_dim == OBS_DIM + 2
        policy = MLP(ckpt_obs_dim, ACTION_DIM, stochastic="policy" in ck)
        policy.load_state_dict(sd)
        policy = policy.to(device)
        # Launch phase: explicit --rl_launch_phase, else the clip's most-grounded stance.
        lp = args_cli.rl_launch_phase if args_cli.rl_launch_phase >= 0 else auto_launch_phase(clip)
        res = evaluate_from_rest(env, clip, policy, device, launch_s=args_cli.rl_launch_s,
                                 launch_phase=lp, use_phase=use_phase)
        save_run_yaml(os.path.join(ckpt_dir, "launch_eval_config.yaml"), _args_to_dict(args_cli), res)


    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
