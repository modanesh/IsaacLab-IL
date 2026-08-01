"""Synthesize a clean, single give-paw gesture from a (jittery) retargeted clip.

The video-retargeted give-paw has two paw-lifts and residual jitter in the raised
front-right leg. For deployment we want ONE smooth motion: stand -> RF paw up ->
stand. This takes the two *poses* from the retarget (so it still looks like a real
give-paw and the 3-legged balance is valid) and drives a smoothstep-eased
trajectory between them -- inherently jitter-free, perfect loop closure, stationary
root. Pure NumPy, no Isaac / GPU.

Feet are interpolated the same way as the joints (both keyframes are FK-consistent
in the source clip, so the interpolation stays close to consistent); joint
vel/acc are recomputed from the smooth trajectory; the root is held fixed
(stationary gesture).

Usage:
  synth_givepaw.py --input .../dog_givepaw.npz --output .../dog_givepaw.npz
"""
import argparse
import numpy as np


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="a retargeted give-paw npz (source of the poses)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--stand-win", type=int, nargs=2, default=[0, 40], help="frames averaged for the STAND pose")
    ap.add_argument("--pawup-win", type=int, nargs=2, default=[90, 140], help="frames averaged for the PAW-UP (held) pose")
    # schedule (frames @ source fps): stand, raise, hold, lower, stand
    ap.add_argument("--sched", type=int, nargs=5, default=[30, 50, 80, 50, 30])
    ap.add_argument("--alpha-max", type=float, default=1.0,
                    help="Cap the blend toward the paw-up pose (1.0 = full reach). <1 makes a "
                         "GENTLER gesture: RF lift AND the weight-shift onto the other 3 legs both "
                         "scale with alpha, so 0.55 halves the CoM excursion -> far more stable, "
                         "still clearly a give-paw. The direct fix for the real-robot tip-over.")
    ap.add_argument("--brace", type=float, default=0.0,
                    help="Optional support-base WIDENING (radians) applied to the 3 stance legs' HAA "
                         "in the paw-up keyframe (splays them outward as the paw lifts, un-splays on "
                         "return). Signs verified from the clip: LF/LH more negative, RH more "
                         "positive (RF is the lifted leg). 0 = off (leaves stance geometry untouched).")
    args = ap.parse_args()

    d = dict(np.load(args.input, allow_pickle=True))
    (s0, s1), (p0, p1) = args.stand_win, args.pawup_win

    # alpha in [0,1]: 0 = stand pose, 1 = paw-up pose. smoothstep on the raise/lower
    # segments gives zero velocity at every segment boundary -> C1-smooth, no jitter.
    nst, nrs, nhd, nlw, nst2 = args.sched
    alpha = args.alpha_max * np.concatenate([
        np.zeros(nst),
        smoothstep(np.linspace(0.0, 1.0, nrs)),
        np.ones(nhd),
        smoothstep(np.linspace(1.0, 0.0, nlw)),
        np.zeros(nst2),
    ]).astype(np.float64)
    T = len(alpha)

    # Isaac joint order: [LF,LH,RF,RH]_HAA (idx 0..3), _HFE (4..7), _KFE (8..11). RF (idx 2)
    # is the lifted leg. Widen the stance triangle by abducting the 3 support HAAs outward.
    BRACE_HAA = {0: -1.0, 1: -1.0, 3: +1.0}   # {joint_idx: outward sign}; verified from stand pose

    def blend(arr, pawup_mod=None):
        """Interpolate a (Tsrc, ...) time series between its stand and paw-up keyframes.

        pawup_mod(pawup) may return a modified paw-up keyframe (used for --brace)."""
        stand = arr[s0:s1].mean(0)
        pawup = arr[p0:p1].mean(0)
        if pawup_mod is not None:
            pawup = pawup_mod(pawup.copy())
        a = alpha.reshape((T,) + (1,) * (arr.ndim - 1))
        return ((1.0 - a) * stand + a * pawup).astype(arr.dtype)

    def brace_pawup(pawup):
        for j, sgn in BRACE_HAA.items():
            pawup[j] += sgn * args.brace
        return pawup

    dt = float(np.asarray(d["frame_duration"]))
    q = blend(np.asarray(d["joint_pos_isaac"]), pawup_mod=brace_pawup if args.brace else None)
    out = dict(d)  # keep all metadata (limits, fps, names, scales, ...)
    out["joint_pos_isaac"] = q
    if "joint_pos" in d:
        out["joint_pos"] = blend(np.asarray(d["joint_pos"]))
    for k in ("foot_pos", "foot_pos_local"):
        if k in d:
            out[k] = blend(np.asarray(d[k]))
    if "foot_err" in d:
        out["foot_err"] = np.zeros((T,) + np.asarray(d["foot_err"]).shape[1:], np.float32)

    # smooth vel/acc from the eased joint trajectory
    v = np.gradient(q.astype(np.float64), axis=0) / dt
    out["joint_vel"] = v.astype(np.float32)
    out["joint_acc"] = (np.gradient(v, axis=0) / dt).astype(np.float32)

    # stationary root: hold the stand pose's base, zero velocity
    rp = np.asarray(d["root_pos"]); rr = np.asarray(d["root_rot"])
    rp0 = rp[s0:s1].mean(0); rr0 = rr[s0:s1].mean(0); rr0 = rr0 / np.linalg.norm(rr0)
    out["root_pos"] = np.tile(rp0, (T, 1)).astype(np.float32)
    out["root_rot"] = np.tile(rr0, (T, 1)).astype(np.float32)
    if "root_lin_vel" in d:
        out["root_lin_vel"] = np.zeros((T, 3), np.float32)
    if "root_translation_recovered" in d:
        out["root_translation_recovered"] = np.asarray(0.0, np.float32)

    np.savez(args.output, **out)
    print(f"[SynthGivePaw] {T} frames ({T*dt:.2f}s @ {1/dt:.0f}fps): "
          f"stand {nst} -> raise {nrs} -> hold {nhd} -> lower {nlw} -> stand {nst2} "
          f"| alpha_max={args.alpha_max} brace={args.brace:.3f}rad")
    print(f"[SynthGivePaw] loop closure |q0-qLast| max = {np.abs(q[0]-q[-1]).max():.3f} rad")
    print(f"[SynthGivePaw] wrote {args.output}")


if __name__ == "__main__":
    main()
