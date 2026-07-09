"""
preprocess_csv.py Convert a video-keypoint CSV (DLC + depth) to a
retargeting-ready .npz

Coordinate convention
---------------------
The source CSV comes from copydog's video pipeline (DeepLabCut 2D keypoints +
Video-Depth-Anything depth, see copydog/scripts/get_depth.py) and is in a
right-handed CAMERA/IMAGE frame:

    x_src: image x, pixels, positive right
    y_src: image y, pixels, positive DOWN
    z_src: root-relative depth offset, positive AWAY from the camera
           (uncalibrated units derived from disparity much smaller in
           magnitude than the pixel axes; see get_depth.py)

IsaacLab uses a z-up right-handed frame (x, y, z)_iso. We convert with a
proper rotation of -90° about the x-axis:

                       | 1  0   0 |
    R_x(-90°)  =       | 0  0   1 |
                       | 0 -1   0 |

    (x, y, z)_iso = (x_src, z_src, -y_src)   # x preserved, depth -> +y, up(-y_img) -> +z

This is a determinant-+1 rotation (handedness preserved), unlike a naive
axis swap which would invert chirality and silently corrupt downstream
rotations (e.g. swap the left/right legs). For a subject walking toward
+x_img this puts its anatomical left at +y_iso, as required. The transform
is applied once, immediately after CSV parsing, so every subsequent
operation (smoothing, root-frame estimation, foot extraction) runs in
IsaacLab coordinates and requires no further changes.

NOTE: the three axes are NOT in the same units (x, z in pixels; y in depth
units). Per-axis scale factors in helpers.compute_scale_factors absorb this
to first order, but the y axis carries far less signal than x/z.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


FOOT_NAMES = [
    "front_left_paw",
    "front_right_paw",
    "back_left_paw",
    "back_right_paw",
]

THIGH_NAMES = [
    "front_left_thai",
    "front_right_thai",
    "back_left_thai",
    "back_right_thai",
]

TRUNK_NAMES = ["neck_base", "back_middle", "back_end"]

def _camera_to_isaac(xyz_src: np.ndarray) -> np.ndarray:
    """Rotate camera-frame keypoints (x right, y down, z away-from-camera)
    into IsaacLab's z-up frame.

    Applied rotation R_x(-90 deg): (x, y, z)_src -> (x, z, -y)_iso

    Args:
        xyz_src: Array with shape (..., 3) holding source (x, y, z) positions.

    Returns:
        Array of the same shape in IsaacLab coordinates.
    """
    out = np.empty_like(xyz_src)
    out[..., 0] = xyz_src[..., 0]
    out[..., 1] = xyz_src[..., 2]
    out[..., 2] = -xyz_src[..., 1]
    return out


def parse_keypoint_csv(csv_path: str) -> dict[str, np.ndarray]:
    """Parse a two-row-header video-keypoint CSV into a keypoint dictionary.

    The CSV is expected to have a part name row followed by a field name row
    (e.g. "x", "y", "z", "likelihood"), with one data row per frame thereafter.
    Positions are transformed from the camera frame into the IsaacLab
    z-up coordinate frame via ``_camera_to_isaac``.

    Args:
        csv_path: Path to the keypoint CSV file (output of copydog's get_depth.py).

    Returns:
        A dict mapping each body-part name to a float32 array of shape
        ``(T, 4)`` with columns ``[x, y, likelihood, z]`` in IsaacLab frame,
        where ``T`` is the number of frames.
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Parse using primitive string parsing
    # eg. header_parts:     front_left_paw
    # eg. header_fields:    x
    header_parts = lines[0].strip().split(",")
    header_fields = lines[1].strip().split(",")

    # Build a nested lookup: col_map[part][field] -> column index in `raw`.
    # The two header rows are "wide" each part name spans multiple columns
    # (one per field: x, y, z, likelihood). Zipping them gives us the
    # (part, field) pair for each column position, which we invert into a
    # dict-of-dicts so downstream code can do col_map["front_left_paw"]["x"]
    # instead of scanning the header arrays on every access.
    col_map: dict[str, dict[str, int]] = {}
    for col_i, (part, field) in enumerate(zip(header_parts, header_fields)):
        part = part.strip()
        field = field.strip()
        col_map.setdefault(part, {})[field] = col_i
        
    data_lines = lines[2:]      # strip the headers (fixed at 2 lines)
    raw = np.zeros((len(data_lines), len(header_parts)), dtype=np.float64)  # create a large block of buffer

    # For every line convert string to float and strip, adding it to the raw buffer
    for t, line in enumerate(data_lines):
        raw[t] = [float(v.strip()) for v in line.strip().split(",")]


    # Construct the dictionary, then iterate through each frame
    keypoints: dict[str, np.ndarray] = {}
    for part, fields in col_map.items():
        # Gather source-frame (x, y, z) then rotate into IsaacLab (z-up) frame.
        src_xyz = np.stack(
            [raw[:, fields["x"]], raw[:, fields["y"]], raw[:, fields["z"]]],
            axis=-1,
        )
        iso_xyz = _camera_to_isaac(src_xyz)
        likelihood = raw[:, fields["likelihood"]]

        # Preserve the [x, y, likelihood, z] column layout expected by the rest
        # of the pipeline (see xyz() helper below). Values are now z-up.
        keypoints[part] = np.stack(
            [iso_xyz[:, 0], iso_xyz[:, 1], likelihood, iso_xyz[:, 2]],
            axis=-1,
        ).astype(np.float32)

    print(f"[CSV] Parsed {raw.shape[0]} frames, {len(keypoints)} keypoints (IsaacLab z-up)")
    return keypoints


def xyz(kp: np.ndarray) -> np.ndarray:
    return kp[:, [0, 1, 3]]


def smooth_series(x: np.ndarray, window: int = 7) -> np.ndarray:
    """Smooths a time-series using a specified windows size. The operation is similar to
    convoluting over a window-ed size kernel."""
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


def smooth_low_confidence(keypoints: dict[str, np.ndarray], threshold: float = 0.5) -> dict[str, np.ndarray]:
    for name, kp in keypoints.items():
        lh = kp[:, 2]
        low = lh < threshold
        if not np.any(low):
            continue
        good_idx = np.where(~low)[0]
        bad_idx = np.where(low)[0]
        if len(good_idx) < 2:
            continue
        print(f"  [Smooth] {name}: interpolating {len(bad_idx)}/{len(kp)} low-confidence frames")
        for col in [0, 1, 3]:
            kp[bad_idx, col] = np.interp(bad_idx, good_idx, kp[good_idx, col])
    return keypoints


def temporal_smooth_keypoints(keypoints: dict[str, np.ndarray], window: int = 7) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, kp in keypoints.items():
        kp_sm = kp.copy()
        kp_sm[:, [0, 1, 3]] = smooth_series(kp[:, [0, 1, 3]], window=window)
        out[name] = kp_sm.astype(np.float32)
    return out


def estimate_root_pos(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    """Root position in IsaacLab z-up frame.

    The (x, y) components are the trunk centroid (mean of neck_base,
    back_middle, back_end). The z component is overridden with the mean
    height of the four thigh keypoints, i.e.

        root_z(t) = (1/4) * sum_{i in thighs} p^(i)_z(t),

    so the root tracks the hip plane rather than sitting on the spine.
    This keeps the root at a retargeting-friendly height (close to the
    robot's base link) while preserving the horizontal trajectory from
    the trunk (more similar to morphology of Anymal-D).
    """
    trunk_xyz = np.stack([xyz(keypoints[n]) for n in TRUNK_NAMES], axis=1)
    root = trunk_xyz.mean(axis=1)                                   # (T, 3)

    thigh_xyz = np.stack([xyz(keypoints[n]) for n in THIGH_NAMES], axis=1)
    root[:, 2] = thigh_xyz[:, :, 2].mean(axis=1)                    # override z

    return root.astype(np.float32)


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)


def _rotmat_to_quat_batch(R: np.ndarray) -> np.ndarray:
    """Converts a batch of rotation matrices into quaternions via numpy broadcasting"""
    B = R.shape[0]
    q = np.zeros((B, 4), dtype=np.float64)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    d0, d1, d2 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]

    m1 = tr > 0
    s1 = np.sqrt(tr[m1] + 1.0) * 2.0
    q[m1, 0] = 0.25 * s1
    q[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s1
    q[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s1
    q[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s1

    m2 = (~m1) & (d0 >= d1) & (d0 >= d2)
    s2 = np.sqrt(1.0 + d0[m2] - d1[m2] - d2[m2]) * 2.0
    q[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s2
    q[m2, 1] = 0.25 * s2
    q[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s2
    q[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s2

    m3 = (~m1) & (~m2) & (d1 >= d2)
    s3 = np.sqrt(1.0 + d1[m3] - d0[m3] - d2[m3]) * 2.0
    q[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s3
    q[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s3
    q[m3, 2] = 0.25 * s3
    q[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s3

    m4 = (~m1) & (~m2) & (~m3)
    s4 = np.sqrt(1.0 + d2[m4] - d0[m4] - d1[m4]) * 2.0
    q[m4, 0] = (R[m4, 1, 0] - R[m4, 0, 1]) / s4
    q[m4, 1] = (R[m4, 0, 2] + R[m4, 2, 0]) / s4
    q[m4, 2] = (R[m4, 1, 2] + R[m4, 2, 1]) / s4
    q[m4, 3] = 0.25 * s4

    q /= np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    return q.astype(np.float32)


def ensure_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    """Preservecs the continuity of the quaternions by inverting the coefficient values."""
    q = quats.copy().astype(np.float32)
    q /= np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    for t in range(1, q.shape[0]):
        if np.dot(q[t - 1], q[t]) < 0.0:
            q[t] *= -1.0
    return q


def mean_quaternion(quats: np.ndarray) -> np.ndarray:
    """Compute the mean orientation of a set of unit quaternions.

    Uses the eigenvalue method of Markley et al. (2007). Given unit
    quaternions q_1, ..., q_T (rows of `quats`), forms the 4x4 symmetric
    accumulator. This was referenced in the motion imitation paper.

        M = (1/T) * sum_t q_t q_t^T,

    and returns the unit eigenvector of M corresponding to the largest
    eigenvalue. This is the rotation that minimises the sum of squared
    chordal distances on S^3 and is invariant to the sign of each q_t
    (since q q^T == (-q)(-q)^T), sidestepping the hemisphere ambiguity.

    Args:
        quats: Array of shape (T, 4) with quaternions in (w, x, y, z) order.

    Returns:
        Array of shape (4,), unit-norm, representing the mean rotation.
    """
    assert quats.ndim == 2 and quats.shape[1] == 4, quats.shape
    q = quats.astype(np.float64)
    q /= np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    M = (q[:, :, None] * q[:, None, :]).mean(axis=0)  # (4, 4)
    eigvals, eigvecs = np.linalg.eigh(M)              # ascending order
    mean_q = eigvecs[:, -1]                           # largest eigenvalue
    mean_q = mean_q / np.clip(np.linalg.norm(mean_q), 1e-8, None)
    # Canonicalise sign so w >= 0 (purely cosmetic; q and -q are equivalent).
    if mean_q[0] < 0.0:
        mean_q = -mean_q
    return mean_q.astype(np.float32)


def estimate_root_rot(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    """Constant root orientation from the regularized thigh configuration.

    Straight-walk prior: the dog's heading is constant, so we discard
    per-frame rotation estimates and instead extract a single heading from
    the time-averaged thigh geometry.

    For each thigh i in {FL, FR, BL, BR} we compute its mean world-frame
    offset from the root:

        d_bar_i = (1/T) * sum_t (p_thigh_i(t) - p_root(t))   in R^3.

    Because the clip is a straight walk with (by assumption) constant
    heading, these mean offsets span the canonical body-frame hip rectangle
    rotated by the unknown heading R. Forward and lateral axes follow from
    the rectangle:

        f_hat = normalize( mean(d_FL, d_FR) - mean(d_BL, d_BR) )   # +x body
        l_hat = normalize( mean(d_FL, d_BL) - mean(d_FR, d_BR) )   # +y body

    We project (f_hat, l_hat) onto the horizontal plane before extracting
    z_hat = f_hat x l_hat, enforcing the straight-walk assumption that the
    heading is a pure yaw (no pitch/roll bias from thigh-swing noise).
    The in-plane axes are then re-orthogonalised.

    This is strictly better-conditioned than neck<->back-end: the four hip
    joints form a roughly rigid rectangle, whereas the spine deforms during
    locomotion and injects a heading bias.
    """
    root = estimate_root_pos(keypoints)  # (T, 3), already in IsaacLab z-up
    thighs = np.stack([xyz(keypoints[n]) for n in THIGH_NAMES], axis=1)  # (T, 4, 3)

    # Time-averaged thigh offsets from root (FL, FR, BL, BR).
    d_bar = (thighs - root[:, None, :]).mean(axis=0)  # (4, 3)
    d_fl, d_fr, d_bl, d_br = d_bar[0], d_bar[1], d_bar[2], d_bar[3]

    fwd = 0.5 * (d_fl + d_fr) - 0.5 * (d_bl + d_br)   # front - back  (+x body)
    # left - right (+y body). Must point LEFT so that z = fwd x lat points UP;
    # the reversed sign builds an upside-down (180deg-roll) body frame.
    lat = 0.5 * (d_fl + d_bl) - 0.5 * (d_fr + d_br)

    # Straight-walk prior: heading is a pure yaw. Kill any z component in
    # the in-plane axes before building the frame so noise in thigh height
    # cannot tilt the body (no pitch/roll bias).
    fwd[2] = 0.0
    lat[2] = 0.0
    fwd = fwd / np.clip(np.linalg.norm(fwd), 1e-8, None)
    lat = lat / np.clip(np.linalg.norm(lat), 1e-8, None)

    z_axis = np.cross(fwd, lat)
    z_axis = z_axis / np.clip(np.linalg.norm(z_axis), 1e-8, None)   # +z world
    y_axis = np.cross(z_axis, fwd)                                   # re-orthog. left
    y_axis = y_axis / np.clip(np.linalg.norm(y_axis), 1e-8, None)

    R_single = np.stack([fwd, y_axis, z_axis], axis=-1)[None, :, :]  # (1, 3, 3)
    q_single = _rotmat_to_quat_batch(R_single)[0]                    # (4,)
    if q_single[0] < 0.0:
        q_single = -q_single                                          # canonical sign

    T = root.shape[0]
    return np.broadcast_to(q_single[None, :], (T, 4)).copy().astype(np.float32)


def extract_foot_positions(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    """Obtain the foot positions from the keypoints dictionary into a numpy array"""
    return np.stack([xyz(keypoints[n]) for n in FOOT_NAMES], axis=1).astype(np.float32)

def extract_thigh_positions(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    """Obtain the thigh positions from the keypoints dictionary into a numpy array"""
    return np.stack([xyz(keypoints[n]) for n in THIGH_NAMES], axis=1).astype(np.float32)


def _quat_apply_batch(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v by unit quaternions q (wxyz) using the closed-form

        v' = v + 2 * w * (u x v) + 2 * (u x (u x v)),

    where q = (w, u) and u in R^3. Vectorised over any leading batch shape as
    long as q and v broadcast together on their leading dims.

    Args:
        q: (..., 4) unit quaternions, (w, x, y, z).
        v: (..., 3) vectors.

    Returns:
        (..., 3) rotated vectors.
    """
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    w = q[..., 0:1]
    u = q[..., 1:4]
    # uxv and ux(uxv) via numpy cross with broadcasting
    uxv = np.cross(u, v)
    uxuxv = np.cross(u, uxv)
    return (v + 2.0 * w * uxv + 2.0 * uxuxv).astype(np.float32)


def _world_to_body(pts_world: np.ndarray, root_pos: np.ndarray, root_rot: np.ndarray) -> np.ndarray:
    """World-frame points -> body-frame points.

    Args:
        pts_world: (T, K, 3) world-frame points (K keypoints per frame).
        root_pos:  (T, 3) root positions in world.
        root_rot:  (T, 4) root orientations (wxyz, world<-body).

    Returns:
        (T, K, 3) body-frame points.
    """
    rel = pts_world - root_pos[:, None, :]                       # (T, K, 3)
    q_inv = root_rot.copy()
    q_inv[..., 1:] *= -1.0                                       # conjugate
    q_inv_b = np.broadcast_to(q_inv[:, None, :], rel.shape[:2] + (4,))
    return _quat_apply_batch(q_inv_b, rel)


def _body_to_world(pts_body: np.ndarray, root_pos: np.ndarray, root_rot: np.ndarray) -> np.ndarray:
    """Body-frame points -> world-frame points (inverse of `_world_to_body`)."""
    q_b = np.broadcast_to(root_rot[:, None, :], pts_body.shape[:2] + (4,))
    rotated = _quat_apply_batch(q_b, pts_body)
    return rotated + root_pos[:, None, :]


def regularize_hips_to_anchor(
    keypoints: dict[str, np.ndarray],
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    lambda_reg: float,
) -> dict[str, np.ndarray]:
    """L2-regularise hip (thigh) keypoints toward their per-hip body-frame mean.

    Motivation
    ----------
    On the ANYmal-D, the four hip joints (leg roots) are rigidly attached to
    the base: their positions in the body frame are *constants*. Raw video keypoints of
    the dog's thighs, however, drifts frame-to-frame due to sensor noise and
    soft-tissue artefacts, sometimes to the point of a left hip appearing on
    the right side of the body. Because the thighs act as leg anchors for
    downstream retargeting (scale factors, IK geometry), this contaminates
    every later stage.

    Formulation
    -----------
    Let p_i(t) in R^3 be the body-frame position of hip i at frame t, obtained
    by transforming the raw world-frame thigh keypoint into the (nearly
    constant) root frame. Let p_bar_i = (1/T) sum_t p_i(t) be the fixed
    anchor for hip i. We solve, independently per (i, t),

        p_tilde_i(t) = argmin_q  || q - p_i(t) ||^2 + lambda * || q - p_bar_i ||^2,

    which admits the closed-form ridge solution

        p_tilde_i(t) = ( p_i(t) + lambda * p_bar_i ) / (1 + lambda).

    Limits:
      lambda = 0       -> no correction (original keypoints preserved).
      lambda -> inf    -> hips fully pinned to the per-hip mean (rigid).
      lambda ~ O(1-10) -> progressive shrinkage toward the anchor, eliminating
                          pathological cross-body drift while preserving
                          small, physical fluctuations.

    The correction is applied in the **body frame** (where "fixed on the
    robot" is the meaningful notion of fixed), then rotated back into world
    coordinates before being written to `keypoints` so that downstream stages
    (`estimate_root_pos` re-estimation notwithstanding, `extract_thigh_positions`,
    `compute_scale_factors`, `retarget.py`) see a layout-compatible signal.

    Note that the **foot** keypoints are deliberately left untouched: only the
    leg roots are being corrected. Foot motion fidelity is the point of the
    retarget; hip motion is nuisance noise.

    Args:
        keypoints:  Dict mapping keypoint name -> (T, 4) array in the
                    [x, y, likelihood, z] column layout used by this pipeline.
                    Modified in place for the four thigh entries.
        root_pos:   (T, 3) body-origin trajectory in world coordinates.
        root_rot:   (T, 4) body orientation (wxyz) at each frame. May be
                    per-frame or broadcast-constant; both are handled.
        lambda_reg: Non-negative regularisation strength. If <= 0, the input
                    is returned unchanged.

    Returns:
        The same `keypoints` dict with regularised thighs written back.
    """
    if lambda_reg <= 0.0:
        print("[HipReg] lambda <= 0, skipping hip regularisation.")
        return keypoints

    # (T, 4_hips, 3) world-frame thighs in pipeline order
    # [FL, FR, BL, BR], matching THIGH_NAMES.
    thighs_world = np.stack([xyz(keypoints[n]) for n in THIGH_NAMES], axis=1).astype(np.float32)

    # Body-frame thighs.
    thighs_body = _world_to_body(thighs_world, root_pos.astype(np.float32),
                                 root_rot.astype(np.float32))                # (T, 4, 3)

    # Per-hip anchor: mean over time in body frame.
    anchors = thighs_body.mean(axis=0, keepdims=True)                        # (1, 4, 3)

    # Closed-form ridge shrinkage toward the anchor.
    thighs_body_reg = (thighs_body + lambda_reg * anchors) / (1.0 + lambda_reg)

    # Diagnostics: how much did each hip move on average?
    pre_dev = np.linalg.norm(thighs_body - anchors, axis=-1).mean(axis=0)    # (4,)
    post_dev = np.linalg.norm(thighs_body_reg - anchors, axis=-1).mean(axis=0)
    print(
        f"[HipReg] lambda={lambda_reg:.2f} | mean |p - p_bar| per hip (pre -> post) [m]: "
        + ", ".join(
            f"{name}: {pre:.4f}->{post:.4f}"
            for name, pre, post in zip(THIGH_NAMES, pre_dev, post_dev)
        )
    )

    # Push back to world.
    thighs_world_reg = _body_to_world(thighs_body_reg, root_pos.astype(np.float32),
                                      root_rot.astype(np.float32))           # (T, 4, 3)

    # Write corrected x, y, z back into `keypoints`, preserving likelihood col.
    for hip_i, name in enumerate(THIGH_NAMES):
        kp = keypoints[name]                                                 # (T, 4) [x,y,lh,z]
        kp[:, 0] = thighs_world_reg[:, hip_i, 0]
        kp[:, 1] = thighs_world_reg[:, hip_i, 1]
        kp[:, 3] = thighs_world_reg[:, hip_i, 2]
        keypoints[name] = kp

    return keypoints


def csv_to_npz(
    csv_path: str,
    output_path: str,
    fps: float,
    lh_threshold: float = 0.5,
    hip_lambda: float = 5.0,
) -> None:
    """Convert a video-keypoint CSV to retarget-ready npz  (Acts as the main function for this script).

    Args:
        csv_path:    Input keypoint CSV (DeepLabCut + depth, from copydog).
        output_path: Destination .npz.
        fps:         Source frame rate (Hz).
        lh_threshold: Likelihood threshold below which frames are interpolated.
        hip_lambda:  L2 regularisation strength pulling each hip (thigh)
                     keypoint toward its body-frame temporal mean. Pass 0.0 to
                     disable. On the ANYmal-D, hips are rigidly mounted on the
                     base and *cannot* translate; this regulariser forces the
                     extracted hips to honour that constraint without discarding
                     their (small) per-frame variation entirely.
    """
    
    dt = 1.0 / fps  # the dt becomes important for retargeting as Isaac as a diff rate than the clip
    # Pipelines keypoints through three functions, you can optionally take some of them out
    keypoints = parse_keypoint_csv(csv_path)
    keypoints = smooth_low_confidence(keypoints, threshold=lh_threshold)
    keypoints = temporal_smooth_keypoints(keypoints, window=7)

    # --- First-pass root estimation (uses raw-but-smoothed thighs). --------
    # The root frame is needed to define "body frame", in which the hip
    # anchor is meaningful. We tolerate noisy thighs here because the root
    # estimators average over 3-4 keypoints, which attenuates per-hip noise.
    root_pos = smooth_series(estimate_root_pos(keypoints), window=5)
    root_rot = estimate_root_rot(keypoints)
    root_rot = smooth_series(root_rot, window=5)

    # --- Hip (thigh) L2 regularisation toward the body-frame per-hip mean.
    keypoints = regularize_hips_to_anchor(
        keypoints=keypoints,
        root_pos=root_pos,
        root_rot=root_rot,
        lambda_reg=hip_lambda,
    )

    # --- Second-pass root re-estimation from the corrected thighs. --------
    # `estimate_root_pos` pegs root-z to the mean thigh height, and
    # `estimate_root_rot` uses thigh x,y to build the body axes, so both
    # benefit from the de-noised hips.
    root_pos = smooth_series(estimate_root_pos(keypoints), window=5)
    root_rot = estimate_root_rot(keypoints)
    # root_rot is already (near-)constant across T; smoothing a constant is a
    # no-op but we retain the call for pipeline symmetry.
    root_rot = smooth_series(root_rot, window=5)

    foot_pos = smooth_series(extract_foot_positions(keypoints), window=5)
    thigh_pos = smooth_series(extract_thigh_positions(keypoints), window=5)

    # Per-frame torso PITCH (radians) from the spine (neck_base -> back_end). The thighs are
    # hip-regularized (rigid => constant tilt), so they can't drive a posture transition; the
    # spine is NOT regularized, so it captures the real stand<->bow<->stand body tilt per frame.
    # pitch = elevation of the (neck - tail) vector from horizontal (heading-invariant):
    # ~0 standing, negative when the front dips (bow), positive when the front rises (sit/beg).
    _nb = xyz(keypoints["neck_base"]); _be = xyz(keypoints["back_end"])
    _spine = _nb - _be
    _pitch = np.arctan2(_spine[:, 2], np.maximum(np.hypot(_spine[:, 0], _spine[:, 1]), 1e-6))
    _w = 5; _k = np.ones(_w) / _w
    torso_pitch = np.convolve(np.pad(_pitch, (_w // 2, _w // 2), "edge"), _k, "valid")[:len(_pitch)].astype(np.float32)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save to output location via compressed format
    # Adjust here if you need more for retarget.py
    np.savez_compressed(
        str(out_path),
        root_pos=root_pos.astype(np.float32),
        root_rot=root_rot.astype(np.float32),
        foot_pos=foot_pos.astype(np.float32),
        thigh_pos=thigh_pos.astype(np.float32),
        torso_pitch=torso_pitch,
        fps=np.array([fps], dtype=np.float32),
        dt=np.array([dt], dtype=np.float32),
        up_axis=np.array(["z"]),  # IsaacLab convention marker
    )
    print(f"[Preprocess] Saved {root_pos.shape[0]} frames @ {fps} Hz -> '{out_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert animal video-keypoint CSV (DLC + depth) to retargeting .npz")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--lh-threshold", type=float, default=0.5)
    parser.add_argument(
        "--hip-lambda",
        type=float,
        default=5.0,
        help=(
            "L2 regularisation strength pulling each hip (thigh) keypoint "
            "toward its body-frame temporal mean. 0 disables; ~5-10 enforces "
            "near-rigid hip anchors as on the ANYmal-D."
        ),
    )
    args = parser.parse_args()
    # Runs the main function
    csv_to_npz(args.csv, args.output, args.fps, args.lh_threshold, hip_lambda=args.hip_lambda)