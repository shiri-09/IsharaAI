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


def fill_hand_gaps(pts: np.ndarray) -> np.ndarray:
    """Linearly interpolate short interior dropouts of each hand (tracking loss) in a (T, N_POINTS, 3) array."""
    t = pts.shape[0]
    frames = np.arange(t)
    for sl in (LH, RH):
        seen = np.abs(pts[:, sl]).sum(axis=(1, 2)) > 0
        if seen.sum() < 2 or seen.all():
            continue
        first, last = np.argmax(seen), t - 1 - np.argmax(seen[::-1])
        gap = (~seen) & (frames > first) & (frames < last)
        if not gap.any():
            continue
        block = pts[:, sl].reshape(t, -1)
        for c in range(block.shape[1]):
            block[gap, c] = np.interp(frames[gap], frames[seen], block[seen, c])
        pts[:, sl] = block.reshape(t, N_HAND, 3)
    return pts


# upper-body pose (0-22: face, shoulders, arms, hands), both hands, face subset; legs/hips are rarely in frame
KEEP = np.r_[0:23, N_POSE : N_POINTS]
N_KEEP = len(KEEP)
FEAT_DIM = N_KEEP * 2 * 2 + 2 * N_HAND * 3  # xy positions + xy velocities + hand-local xyz shapes


def featurize(norm: np.ndarray) -> np.ndarray:
    """Normalised (T, FRAME_DIM) -> (SEQ_LEN, FEAT_DIM) model input."""
    pts = fill_hand_gaps(norm.reshape(norm.shape[0], N_POINTS, 3).copy())
    pts = resample(pts.reshape(pts.shape[0], -1)).reshape(SEQ_LEN, N_POINTS, 3)
    xy = pts[:, KEEP, :2].reshape(SEQ_LEN, -1)
    vel = np.diff(xy, axis=0, prepend=xy[:1])
    shapes = []
    for sl in (LH, RH):
        h = pts[:, sl]  # (T, 21, 3)
        present = np.abs(h).sum(axis=(1, 2)) > 0
        local = h - h[:, :1]  # relative to wrist
        size = np.linalg.norm(local[:, 9, :2], axis=1, keepdims=True)  # wrist -> middle-finger knuckle
        local = local / np.maximum(size, 1e-3)[:, :, None]
        local[~present] = 0.0
        shapes.append(local.reshape(SEQ_LEN, -1))
    return np.concatenate([xy, vel, *shapes], axis=1).astype(np.float32)


def to_features(raw: np.ndarray) -> np.ndarray:
    """(T, FRAME_DIM) raw keypoints -> (SEQ_LEN, FEAT_DIM) model input."""
    return featurize(normalize(raw))
