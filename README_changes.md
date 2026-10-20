# 🤖 IsaacLab-IL: Motion Imitation for ANYmal-D

This is a **customized [Isaac Lab](https://github.com/isaac-sim/IsaacLab) fork** for the
[**CopyQuad**](https://github.com/modanesh/CopyQuad) project. It holds the second half of
the pipeline: turning a lifted 3D reference into an ANYmal-D policy that reproduces the
animal's gait.

> 📥 The 3D keypoint CSV that feeds step 3 comes from the top-level
> [CopyQuad](https://github.com/modanesh/CopyQuad) repo (DeepLabCut -> make_3d_csv.py; no depth model).
> This fork ships as its `IsaacLab/` submodule.

```
 📄 3D keypoint CSV  (from CopyQuad)
      |  3-  scripts/retargeting/preprocess_csv.py                   ->  cleaned reference skeleton (.npz)
      |  4-  scripts/retargeting/retarget.py                         ->  ANYmal-D joint trajectory
      |  5-  scripts/environments/data/recover_root_translation.py   ->  global body motion
      |  6-  scripts/environments/train_motion_imitation.py          ->  tracking policy
 🤖 ANYmal-D policy
```

---

## 🚀 Pipeline

### 3- 🧹 Preprocess - CSV to reference skeleton
```bash
python scripts/retargeting/preprocess_csv.py --csv <clip>_with_depth_npz.csv --output scripts/environments/data/motions/_refs/<clip>_ref.npz --fps <fps>
```
Confidence-weighted keypoint smoothing, camera to robot transform, hip regularization.

### 4- 🦿 Retarget - anisotropy-aware IK to ANYmal-D
```bash
python scripts/retargeting/retarget.py --ref-motion scripts/environments/data/motions/_refs/<clip>_ref.npz --output scripts/environments/data/motions/<clip>.npz --fps <fps> --lateral-mode pin
```
Monocular depth is *anisotropically* noisy, so the lateral axis is **pinned** while the
clean image-plane axes drive the IK, plus a knee-branch constraint and hip regularization.
`--collect-z/--collect-x` shrink an over-athletic reference into the robot's reachable
envelope. The two **gestures** (bow, give-paw) then take one extra keyframe-synthesis
pass, see step 4b.

### 4b- 🙇 Gesture synthesis (bow and give-paw only)
```bash
python scripts/retargeting/synth_gesture.py --gesture bow     --input <bow_retarget>.npz     --output scripts/environments/data/motions/dog_bow_2.npz
python scripts/retargeting/synth_gesture.py --gesture givepaw --input <givepaw_retarget>.npz --output scripts/environments/data/motions/dog_givepaw_gentle.npz --sched 20 25 20 25 20 --alpha-max 0.55 --brace 0.12
```
Neither gesture clip contains one clean repetition: the bow is mostly a *held* pose with
almost no transition, and the give-paw clip has two paw-lifts plus jitter in the raised
leg. This pass takes two **real retargeted poses** from the clip (a neutral stand and the
extreme of the gesture) and smoothstep-eases between them, giving one clean repetition with
exact loop closure. The poses come from the video; only the timing between them is
synthesized. The three gaits do **not** use this step.

### 5- 🧭 Recover global root translation
```bash
python scripts/environments/data/recover_root_translation.py --motion scripts/environments/data/motions/<clip>.npz
```
Fixed-base IK discards world translation; this restores it from stance-foot cadence & stride.

>️ **Optional for clip conditioning.** `scripts/retargeting/polish_motion.py` smooths joints
> and reshapes amplitude in *joint space* to fit ANYmal's actuator limits (for athletic
> references). Slow-motion removal lives in the CopyQuad repo's `scripts/`.

### 6- 🎯 Train - phase-indexed DeepMimic policy
```bash
python scripts/environments/train_motion_imitation.py --motion scripts/environments/data/motions/<clip>.npz --mode rl --phase-obs --num_envs 512 --rl_iters 4000 --headless
```
- `--mode rl`: 🎯 phase-indexed DeepMimic tracking (PPO). **This is the main mode.**
- `--mode eval --ckpt <...>/model.pt`: 📊 evaluate a saved policy (tracking reward, forward speed, survival, footfall).
- `--mode bc`: a supervised behavioral-cloning warm-start (legacy).

Key reward knobs: `--rl_root_w` (forward-speed demand), `--rl_foot_z_weight` (clearance),
`--rl_joint_w`, `--rl_sigma`, `--rl_action_rate_w`. Checkpoints land in
`scripts/environments/checkpoints/<clip>/policy_rl_<hash>/`.

---

## 🎥 Playback

```bash
python scripts/environments/anymal_motion_playback.py --motion scripts/environments/data/motions/horses_trot_v2.npz
```
- 🟢 default **kinematic** (base pinned, joints written directly, retarget ground truth)
- 🔵 `--physics` (free base + gravity + PD targets, dynamic feasibility)
- 📸 `--hold-frame N` (freeze one frame, feet grounded, for figures)

---

## 🐕 Shipped assets

Five reference motions and one trained policy each, under
`scripts/environments/data/motions/` and `scripts/environments/checkpoints/`:

| Motion | Behavior | Policy |
|---|---|---|
| `dog1_slow` | 🐕 dog walk | `policy_rl_8c5eb2` |
| `horses_walk` | 🐴 horse walk | `policy_rl_ca7822` |
| `horses_trot_v2` | 🐎 horse trot | `policy_rl_846766` |
| `dog_bow_2` | 🙇 dog bow | `policy_rl_97e40f` |
| `dog_givepaw_gentle` | 🐾 dog give-paw | `policy_rl_4f2367` |

These are the five behaviors of the paper, and each policy is the one deployed on the
physical ANYmal-D.

> 🗄️ All experiments, diagnostics, and intermediate artifacts are preserved on the
> `archive/pre-release` branch.
