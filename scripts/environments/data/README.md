# 📁 scripts/environments/data

Reference-motion assets and the root-recovery step.

- **`motions/`** — retargeted ANYmal-D reference trajectories (`<clip>.npz`), one per gait.
  `_refs/` holds the intermediate preprocessed skeletons (git-ignored).
- **`recover_root_translation.py`** — restores global body translation on a retargeted
  `.npz` from stance-foot cadence and stride (pipeline step ⑤).
