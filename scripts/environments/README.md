# 🎯 scripts/environments — Policy Training & Playback

Motion-imitation policy training for ANYmal-D.

- **`train_motion_imitation.py`** — the trainer. `--mode rl` (phase-indexed DeepMimic, the
  main mode), `--mode eval` (score a checkpoint), `--mode bc` (legacy supervised warm-start).
  Two-stage BC → PPO flow; checkpoints in `checkpoints/<clip>/policy_rl_<hash>/`.
- **`anymal_motion_playback.py`** — render a reference clip on ANYmal-D
  (kinematic / `--physics` / `--hold-frame N`).
- **`data/`** — reference motions (`motions/`) and the root-recovery step
  (`recover_root_translation.py`).

See [`../../README_changes.md`](../../README_changes.md) for the full pipeline and commands.
