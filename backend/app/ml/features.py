"""Keypoint layout + normalisation shared by training and live inference.

Each frame sent by the browser is a flat list of FRAME_DIM floats (x, y, z per point) in this order:
  pose (33) | left hand (21) | right hand (21) | face subset (len(FACE_IDX))
Missing parts (e.g. a hand not visible) are sent as zeros.
The frontend mirror of this layout lives in frontend/src/lib/keypoints.ts — keep them in sync.
"""

import numpy as np

N_POSE, N_HAND = 33, 21
# MediaPipe FaceMesh indices: lips, inner lips, eyebrows, nose tip, chin (non-manual markers)
FACE_IDX = [61, 291, 0, 17, 13, 14, 78, 308, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 1, 152]
N_POINTS = N_POSE + 2 * N_HAND + len(FACE_IDX)  # 95
FRAME_DIM = N_POINTS * 3  # 285
SEQ_LEN = 64  # frames after resampling
FEAT_DIM = FRAME_DIM * 2  # positions + velocities

L_SHOULDER, R_SHOULDER = 11, 12
LH = slice(N_POSE, N_POSE + N_HAND)
RH = slice(N_POSE + N_HAND, N_POSE + 2 * N_HAND)


def hands_present(frame: np.ndarray) -> bool:
    pts = frame.reshape(N_POINTS, 3)
    return bool(np.abs(pts[LH]).sum() > 0 or np.abs(pts[RH]).sum() > 0)


def resample(seq: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Linearly resample a (T, D) sequence to (length, D)."""
    t = seq.shape[0]
    if t == length:
        return seq
    if t == 1:
        return np.repeat(seq, length, axis=0)
    src = np.linspace(0, t - 1, length)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo)[:, None]
    return seq[lo] * (1 - w) + seq[hi] * w


def normalize(seq: np.ndarray) -> np.ndarray:
    """(T, FRAME_DIM) raw -> (T, FRAME_DIM) body-centred, shoulder-width scaled. Zeros (missing) stay zero."""
    pts = seq.reshape(seq.shape[0], N_POINTS, 3).astype(np.float32).copy()
    missing = np.all(pts == 0, axis=2)  # (T, N)
    ls, rs = pts[:, L_SHOULDER], pts[:, R_SHOULDER]
    center = (ls + rs) / 2  # (T, 3)
    scale = np.linalg.norm((ls - rs)[:, :2], axis=1)  # (T,)
    valid = scale > 1e-4
    # frames without a visible body: reuse the mean of frames that have one
    if valid.any():
        center[~valid] = center[valid].mean(axis=0)
        scale[~valid] = scale[valid].mean()
    else:
        center[:] = 0.5
        scale[:] = 0.3
    pts = (pts - center[:, None, :]) / scale[:, None, None]
    pts[missing] = 0.0
    return pts.reshape(seq.shape[0], FRAME_DIM)


def to_features(raw: np.ndarray) -> np.ndarray:
    """(T, FRAME_DIM) raw keypoints -> (SEQ_LEN, FEAT_DIM) model input."""
    x = resample(normalize(raw))
    vel = np.diff(x, axis=0, prepend=x[:1])
    return np.concatenate([x, vel], axis=1).astype(np.float32)
