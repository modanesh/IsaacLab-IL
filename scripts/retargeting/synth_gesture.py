"""synth_gesture.py -- keyframe synthesis for the two non-locomotion gestures.

Both gestures need one extra pass after retargeting, for the same reason: the
source clip does not contain a clean single repetition of the gesture.

  * bow      -- the clip is mostly a *held* bow with almost no transition, so a
                policy trained on it just holds the extended pose.
  * givepaw  -- the clip contains two paw-lifts plus residual jitter in the
                raised front-right leg.

This pass does not invent a pose. It takes two *real retargeted poses* from the
clip (a neutral stand and the extreme of the gesture) and drives a smoothstep-
eased trajectory between them, giving one clean repetition with zero velocity at
every segment boundary and exact loop closure. The poses stay grounded in the
video; only the timing between them is synthesized.

Pure NumPy/SciPy, no Isaac, no GPU.

    # bow: stand -> bow -> stand, keyframes auto-detected from base pitch
    python scripts/retargeting/synth_gesture.py --gesture bow \
        --input  scripts/environments/data/motions/dog_bow_2_base.npz \
        --output scripts/environments/data/motions/dog_bow_2.npz

    # give-paw: stand -> RF paw up -> stand, keyframes averaged over frame windows
    python scripts/retargeting/synth_gesture.py --gesture givepaw \
        --input  <retargeted give-paw npz> \
        --output scripts/environments/data/motions/dog_givepaw_gentle.npz \
        --sched 20 25 20 25 20 --alpha-max 0.55 --brace 0.12

See the "Gesture clips need one extra pass" section of the top-level README for
which flags produced the two released references.
"""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


def smoothstep(t):
    """C1 ease: zero derivative at both ends, so segment joins are jerk-free."""
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
# bow
# --------------------------------------------------------------------------- #
def synth_bow(d, args):
    """stand <-> bow <-> stand, keyframes picked from the base-pitch trace.

    Every per-frame field is interpolated between the two keyframes, with the
    base orientation slerped so the body tilt is a proper rotation.
    """
    dt = float(d["dt"])
    fps = 1.0 / dt
    rr = np.asarray(d["root_rot"])
    T0 = rr.shape[0]

    # pitch per frame -> stand = most level, bow = most nose-down
    w, x, y, z = rr[:, 0], rr[:, 1], rr[:, 2], rr[:, 3]
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    # stand keyframe: most-retracted front feet among level frames (=> bigger forward step);
    # bow keyframe: most tilted.
    fx = np.asarray(d["foot_pos"])[:, :2, 0].mean(axis=1)     # mean front-foot x per frame
    level = np.abs(np.degrees(pitch)) < 3.0
    cand = np.where(level)[0]
    s = int(cand[np.argmin(fx[cand])]) if len(cand) else int(np.argmin(np.abs(pitch)))
    b = int(np.argmax(np.abs(pitch)))
    print(f"keyframes: stand=frame {s} (pitch {np.degrees(pitch[s]):+.1f}), "
          f"bow=frame {b} (pitch {np.degrees(pitch[b]):+.1f})")

    # hold 0 -> ease 0->1 -> hold 1 -> ease 1->0 -> hold 0
    Ns, Nt, Nb = (int(round(v * fps)) for v in (args.stand, args.trans, args.bow))
    alpha = np.concatenate([
        np.zeros(Ns), smoothstep(np.linspace(0, 1, Nt)), np.ones(Nb),
        smoothstep(np.linspace(1, 0, Nt)), np.zeros(Ns)
    ]).astype(np.float32)
    Tn = len(alpha)

    out = {}
    for k, v in d.items():
        a = np.asarray(v)
        if a.ndim >= 1 and a.shape[0] == T0:                  # per-frame field -> interpolate
            if k == "root_rot":
                keys = Rot.from_quat(np.stack([rr[s][[1, 2, 3, 0]], rr[b][[1, 2, 3, 0]]]))  # wxyz->xyzw
                q = Slerp([0.0, 1.0], keys)(alpha).as_quat()                                # (Tn,4) xyzw
                out[k] = np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]]).astype(np.float32)
            else:
                ps, pb = a[s], a[b]
                sh = (slice(None),) + (None,) * (a.ndim - 1)
                out[k] = (ps[None] + alpha[sh] * (pb - ps)[None]).astype(np.float32)
        else:
            out[k] = v                                        # metadata unchanged

    # front-knee lift bump mid-transition so the front legs LIFT and step (not slide) forward
    if args.lift > 0.0:
        jn = [str(x) for x in d["joint_names"]]
        fk = [jn.index("LF_KFE"), jn.index("RF_KFE")]
        bump = (args.lift * 4.0 * alpha * (1.0 - alpha)).astype(np.float32)  # peaks mid-transition
        out["joint_pos_isaac"][:, fk] -= bump[:, None]                       # more flexed -> foot lifts

    jp = out["joint_pos_isaac"]
    out["joint_vel"] = np.gradient(jp, dt, axis=0).astype(np.float32)
    out["joint_acc"] = np.gradient(out["joint_vel"], dt, axis=0).astype(np.float32)

    print(f"synthesized {Tn} frames ({Tn / fps:.1f}s): stand {args.stand}s -> [trans {args.trans}s] -> "
          f"bow {args.bow}s -> [trans {args.trans}s] -> stand {args.stand}s")
    return out


# --------------------------------------------------------------------------- #
# give-paw
# --------------------------------------------------------------------------- #
# Isaac joint order: [LF,LH,RF,RH]_HAA (idx 0..3), _HFE (4..7), _KFE (8..11). RF (idx 2)
# is the lifted leg. Widen the stance triangle by abducting the 3 support HAAs outward.
BRACE_HAA = {0: -1.0, 1: -1.0, 3: +1.0}   # {joint_idx: outward sign}; verified from stand pose


def synth_givepaw(d, args):
    """stand <-> RF paw up <-> stand, keyframes averaged over two frame windows.

    The root is held at the stand pose (stationary gesture), and only the joint /
    foot fields are interpolated.
    """
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

    print(f"[GivePaw] {T} frames ({T * dt:.2f}s @ {1 / dt:.0f}fps): "
          f"stand {nst} -> raise {nrs} -> hold {nhd} -> lower {nlw} -> stand {nst2} "
          f"| alpha_max={args.alpha_max} brace={args.brace:.3f}rad")
    print(f"[GivePaw] loop closure |q0-qLast| max = {np.abs(q[0] - q[-1]).max():.3f} rad")
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gesture", required=True, choices=["bow", "givepaw"],
                    help="which gesture to synthesize")
    ap.add_argument("--input", required=True, help="a retargeted gesture npz (source of the poses)")
    ap.add_argument("--output", required=True)

    g = ap.add_argument_group("bow")
    g.add_argument("--stand", type=float, default=2.0, help="seconds held standing at each end")
    g.add_argument("--trans", type=float, default=1.5, help="seconds for each stand<->bow transition")
    g.add_argument("--bow", type=float, default=2.0, help="seconds held in the bow")
    g.add_argument("--lift", type=float, default=0.0,
                   help="front-knee flex (rad) added mid-transition so the front legs LIFT and step "
                        "forward (peaks mid-transition, 0 at the holds). e.g. 0.3.")

    g = ap.add_argument_group("givepaw")
    g.add_argument("--stand-win", type=int, nargs=2, default=[0, 40],
                   help="frames averaged for the STAND pose")
    g.add_argument("--pawup-win", type=int, nargs=2, default=[90, 140],
                   help="frames averaged for the PAW-UP (held) pose")
    g.add_argument("--sched", type=int, nargs=5, default=[30, 50, 80, 50, 30],
                   help="frames @ source fps: stand, raise, hold, lower, stand")
    g.add_argument("--alpha-max", type=float, default=1.0,
                   help="Cap the blend toward the paw-up pose (1.0 = full reach). <1 makes a "
                        "GENTLER gesture: RF lift AND the weight-shift onto the other 3 legs both "
                        "scale with alpha, so 0.55 halves the CoM excursion -> far more stable, "
                        "still clearly a give-paw. The direct fix for the real-robot tip-over.")
    g.add_argument("--brace", type=float, default=0.0,
                   help="Optional support-base WIDENING (radians) applied to the 3 stance legs' HAA "
                        "in the paw-up keyframe (splays them outward as the paw lifts, un-splays on "
                        "return). Signs verified from the clip: LF/LH more negative, RH more "
                        "positive (RF is the lifted leg). 0 = off (leaves stance geometry untouched).")
    args = ap.parse_args()

    d = dict(np.load(args.input, allow_pickle=True))
    out = synth_bow(d, args) if args.gesture == "bow" else synth_givepaw(d, args)
    np.savez(args.output, **out)
    print(f"[{args.gesture}] wrote {args.output}")


if __name__ == "__main__":
    main()
