#!/usr/bin/env python3
"""recover_root_translation.py — put forward motion back into the references.

retarget.py pins the base, so every clip is "in place": root_pos constant, root
velocity ~0. AMP/RL then has no reference forward motion to reward and the policy
settles into in-place stepping. The forward speed is recoverable from the feet.

THE OLD METHOD (--legacy) estimated a per-frame body velocity as
    v_body(t) = - mean_over_stance_feet( d/dt foot_pos_local_xy )
with a loose stance threshold (any foot within 3 cm of its own minimum). That let
SWING feet — which move FORWARD in the body frame — leak into the average, which
diluted the speed on clean clips (dog: stands still at times) and REVERSED it on
others (dlc_horse_007: recovered -0.04 m/s while the feet imply +0.3 forward). That
broken per-frame signal is why playback bases wobble / stall despite a steady video.

THE FIX (default, "steady"): the video walks are steady fixed-speed, and the feet
(faithfully retargeted) pin the speed two robust ways:
  • stride x cadence  — stride = median foot fore-aft sweep (m/cycle); cadence =
    footfalls/sec (counted, robust for few cycles). Validated: dog1_slow 0.29,
    dog1 0.40 — matching the known trained speeds (0.30 / 0.43).
  • median-over-feet  — at each frame the >=2-3 planted feet share one backward
    velocity while the swing foot moves forward, so the per-frame MEDIAN over the 4
    feet = -v (rejects swing contamination). Used as a fallback when the clip is too
    short (<2 gait cycles) for a reliable cadence.
We take a single robust forward speed v and write a CONSTANT v (vy=0) — a steady
forward reference that matches the video, removing the per-frame stalls/reversals.

Writes `root_lin_vel` (T,3 body frame; training reads [:, :2]) and `root_pos`
(T,3; training reads [:, 2]=height, playback reads [:, 0:2]) into the clip npz.

Usage:
    python3 scripts/environments/data/recover_root_translation.py --batch
    python3 scripts/environments/data/recover_root_translation.py --motion <npz> --dry-run
    python3 scripts/environments/data/recover_root_translation.py --motion <npz> --legacy
"""
import argparse
from pathlib import Path

import numpy as np

MOTION_DIR = Path(__file__).resolve().parent / "motions"
STANCE_TOL = 0.03          # m above per-foot min still counts as contact (legacy only)
SMOOTH_WIN = 9             # frames, moving-average on the velocity estimate (legacy only)
CONF_SPEED = 0.08          # m/s forward speed to call a clip "forward-confident"
MIN_CYCLES = 2.0           # need >= this many gait cycles for a trustworthy cadence
CSV3D_DIR = Path(__file__).resolve().parents[4] / "data" / "3d_keypoints"
TRUNK_KP = ["neck_base", "back_middle", "back_end"]
FRONT_KP = ["front_left_paw", "front_right_paw"]
HIND_KP = ["back_left_paw", "back_right_paw"]
TRUNK_MIN_TRAVEL = 0.5     # body-lengths the torso must translate to trust it (static camera)


def trunk_speed_from_video(stem: str, fl: np.ndarray, dt: float):
    """Forward speed (m/s) from the TORSO's pixel displacement in the source video,
    self-calibrated px->m via the front-to-hind distance measured in BOTH frames (same
    body, two rulers). Returns (v, travel_bodylengths, r2) or None if the 3D csv is
    missing. Only meaningful with a STATIC camera (torso translates); a tracking camera
    keeps the torso centred -> travel ~0 -> caller falls back to the foot-based speed."""
    csv = CSV3D_DIR / f"{stem}_with_depth_npz.csv"
    if not csv.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv, header=[0, 1])
        def cen(names, a):
            return np.mean([df[(n, a)].to_numpy(float) for n in names], axis=0)
        tx, ty = cen(TRUNK_KP, "x"), cen(TRUNK_KP, "y")
        n = len(tx); dur = n * dt
        net = np.array([tx[-1] - tx[0], ty[-1] - ty[0]]); nrm = float(np.linalg.norm(net))
        if nrm < 1e-6 or dur <= 0:
            return None
        fdir = net / nrm
        fpos = tx * fdir[0] + ty * fdir[1]                      # torso position along its travel axis (px)
        wb_px = float(np.median(np.abs(cen(FRONT_KP, "x") - cen(HIND_KP, "x"))))  # front-hind span (px)
        wb_m = float(np.median(np.abs(fl[:, 0:2, 0].mean(1) - fl[:, 2:4, 0].mean(1))))  # same span (robot m)
        if wb_px < 1e-6:
            return None
        v = float((fpos[-1] - fpos[0]) / dur * wb_m / wb_px)
        travel_bl = float((fpos[-1] - fpos[0]) / wb_px)
        tt = np.arange(n) * dt; fit = np.polyval(np.polyfit(tt, fpos, 1), tt)
        r2 = float(1 - np.sum((fpos - fit) ** 2) / (np.sum((fpos - fpos.mean()) ** 2) + 1e-9))
        return v, travel_bl, r2
    except Exception:
        return None


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w < 3 or len(x) < w:
        return x
    ker = np.ones(w) / w
    pad = np.r_[x[:w // 2][::-1], x, x[-(w // 2):][::-1]]
    return np.convolve(pad, ker, "valid")[:len(x)]


def _count_footfalls(z: np.ndarray, dt: float) -> int:
    """Footfalls = local minima of foot height (one per gait cycle). Counted directly
    (not via FFT) so it stays meaningful for short clips with only 1-2 cycles."""
    rng = float(z.max() - z.min())
    if rng < 1e-4:
        return 0
    try:                                    # prominence/spacing reject noise dips
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(-z, prominence=0.25 * rng, distance=max(2, int(0.2 / dt)))
        return len(peaks)
    except Exception:                       # fallback: mid-height down-crossings
        below = z < (z.min() + 0.5 * rng)
        return int(np.sum((~below[:-1]) & below[1:]))


def leg_velocity_profile(fl: np.ndarray, dt: float) -> np.ndarray:
    """Per-frame NON-SLIP speed (T,): -median over the planted feet (low z AND moving
    backward, which rejects the forward-swinging foot) of fore-aft velocity. This is the
    speed the base must move so feet don't slide in Isaac; its SHAPE captures the gait's
    accel/decel and, by construction, moves WITH the legs."""
    T = fl.shape[0]
    vx = np.gradient(fl[:, :, 0], dt, axis=0)                       # (T,4) body-frame foot vel
    z = fl[:, :, 2]
    lo = z < (z.min(axis=0)[None, :] + 0.40 * np.ptp(z, axis=0)[None, :])
    out = np.full(T, np.nan, np.float32)
    for t in range(T):
        s = lo[t] & (vx[t] < 0)                                    # planted AND sweeping backward
        if s.any():
            out[t] = -np.median(vx[t, s])
    ok = ~np.isnan(out)
    if ok.sum() > 1:
        out[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), out[ok])
    elif ok.sum() == 0:
        out[:] = 0.0
    return np.clip(_smooth(out, SMOOTH_WIN), 0.0, None).astype(np.float32)


def estimate_speed(fl: np.ndarray, dt: float, stem: str = None):
    """Per-frame forward-velocity PROFILE (T,). The SHAPE comes from the legs (non-slip,
    captures accel/decel, moves with the feet); the MAGNITUDE is the best of: stride x
    cadence (validated when >=2 cycles), or — for a static-camera clip — the geometric
    mean of the torso's video speed and the leg-sweep rate (reconciles 'how far the body
    actually went' with 'how fast the feet sweep' so the base never outruns the legs)."""
    T = fl.shape[0]
    dur = T * dt
    stride = float(np.median(np.ptp(fl[:, :, 0], axis=0)))          # m / cycle
    falls = [_count_footfalls(fl[:, i, 2], dt) for i in range(4)]
    cadence = float(np.median(falls)) / dur if dur > 0 else 0.0     # cycles / s
    n_cycles = cadence * dur

    legp = leg_velocity_profile(fl, dt)                            # (T,) non-slip shape + rate
    leg_mean = max(float(legp.mean()), 1e-6)

    if n_cycles >= MIN_CYCLES and cadence > 0:
        M, method = stride * cadence, "stride*cadence"             # validated non-slip magnitude
    else:
        M, method = leg_mean, "leg-sweep"
    # Static-camera reconciliation: combine the torso's video distance with the non-slip
    # leg rate (geometric mean) — pulls an over-read torso speed back toward what the legs
    # can actually support (kills the 'rushed' look) while honouring the video distance.
    if stem is not None:
        tr = trunk_speed_from_video(stem, fl, dt)
        if tr is not None:
            v_tr, travel_bl, r2 = tr
            if travel_bl > TRUNK_MIN_TRAVEL and v_tr > 0 and r2 > 0.8:
                M = float(np.sqrt(v_tr * leg_mean))
                method = f"torso~leg({v_tr:.2f}~{leg_mean:.2f})"

    # apply the leg-derived shape at the chosen magnitude; smooth heavily (≈quarter-clip)
    # so only the MACRO accel/decel survives, not within-stride wobble -> a steady gait
    # stays effectively constant while a decelerating clip (049/070) is tracked.
    shape = _smooth(legp, max(SMOOTH_WIN, T // 4))
    shape = shape / max(float(shape.mean()), 1e-6)
    profile = np.maximum(M * shape, 0.25 * M).astype(np.float32)   # decelerate, but never stall
    return profile, method, stride, cadence, n_cycles


def recover(npz: dict, steady: bool = True, stem: str = None) -> dict:
    fl = np.asarray(npz["foot_pos_local"], np.float32)             # (T,4,3) body frame, x=fwd
    dt = float(np.asarray(npz.get("frame_duration", npz.get("dt", 1.0 / 30.0))).ravel()[0])
    T = fl.shape[0]
    base_z = float(np.asarray(npz["root_pos"])[0, 2]) if "root_pos" in npz else 0.5
    old_vx = float(np.asarray(npz["root_lin_vel"], np.float32).reshape(T, -1)[:, 0].mean()) \
        if "root_lin_vel" in npz else float("nan")

    if steady:
        profile, method, stride, cadence, n_cycles = estimate_speed(fl, dt, stem)
        vbody = np.zeros((T, 2), np.float32)
        vbody[:, 0] = profile                                      # per-frame forward; lateral pinned -> 0
        mean_vx = float(profile.mean())
        frac_fwd = float((profile > 0).mean())
        confident = mean_vx > CONF_SPEED
    else:  # --- legacy per-frame estimate (kept for comparison) ---
        method, stride, cadence, n_cycles = "legacy", float("nan"), float("nan"), float("nan")
        fz = fl[:, :, 2]
        stance = fz <= (fz.min(axis=0)[None, :] + STANCE_TOL)
        vfoot = np.gradient(fl[:, :, :2], dt, axis=0)
        vbody = np.full((T, 2), np.nan, np.float32)
        for t in range(T):
            s = stance[t]
            if s.any():
                vbody[t] = -vfoot[t, s, :].mean(0)
        for k in range(2):
            col = vbody[:, k]; ok = ~np.isnan(col)
            col[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], col[ok]) if ok.sum() > 1 else 0.0
            vbody[:, k] = _smooth(col, SMOOTH_WIN)
        mean_vx = float(vbody[:, 0].mean())
        frac_fwd = float((vbody[:, 0] > 0).mean())
        confident = (mean_vx > CONF_SPEED) and (frac_fwd > 0.65)

    # integrate body velocity -> world root trajectory (yaw≈0 -> world == body horizontally)
    root_lin_vel = np.zeros((T, 3), np.float32)
    root_lin_vel[:, :2] = vbody
    root_pos = np.zeros((T, 3), np.float32)
    root_pos[:, 0] = np.cumsum(vbody[:, 0]) * dt
    root_pos[:, 1] = np.cumsum(vbody[:, 1]) * dt
    root_pos[:, 2] = base_z
    return {"root_lin_vel": root_lin_vel, "root_pos": root_pos,
            "mean_vx": mean_vx, "old_vx": old_vx, "frac_fwd": frac_fwd, "confident": confident,
            "method": method, "stride": stride, "cadence": cadence, "n_cycles": n_cycles, "dt": dt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", type=str)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--motion-dir", type=str, default=str(MOTION_DIR))
    ap.add_argument("--legacy", action="store_true", help="use the old per-frame estimator (for comparison)")
    ap.add_argument("--dry-run", action="store_true", help="report only; don't write npz")
    args = ap.parse_args()

    if args.batch:
        files = sorted(p for p in Path(args.motion_dir).glob("*.npz") if "_refs" not in str(p))
    elif args.motion:
        files = [Path(args.motion)]
    else:
        ap.error("provide --motion or --batch")

    conf = []
    print(f"{'clip':22s} {'old_vx':>7s} {'new_v':>7s} {'method':>16s} {'cyc':>5s}  confident")
    print("-" * 70)
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        if "foot_pos_local" not in d:
            continue
        r = recover(d, steady=not args.legacy, stem=f.stem)
        flag = "YES" if r["confident"] else ""
        print(f"{f.stem:22s} {r['old_vx']:+7.2f} {r['mean_vx']:+7.2f} {r['method']:>16s} "
              f"{r['n_cycles']:>5.1f}  {flag}")
        if r["confident"]:
            conf.append(str(f))
        if not args.dry_run:
            d["root_lin_vel"] = r["root_lin_vel"]
            d["root_pos"] = r["root_pos"]
            d["root_translation_recovered"] = np.array([1.0 if r["confident"] else 0.0], np.float32)
            np.savez(f, **d)

    if args.batch:
        out = Path(args.motion_dir) / "forward_confident_clips.txt"
        if not args.dry_run:
            out.write_text("\n".join(conf) + "\n")
        print(f"\n[Recover] {len(files)} clips | forward-confident: {len(conf)}"
              + (f" -> {out}" if not args.dry_run else " (dry-run)"))


if __name__ == "__main__":
    main()
