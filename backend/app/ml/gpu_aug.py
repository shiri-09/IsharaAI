"""Fast training data pipeline on the GPU.

The CPU augmentation in train.py runs Python per sample per epoch, which makes experiments slow. Here every clip is
prepared once (normalised, hand gaps filled, resampled to PREP_LEN frames) and stored as one tensor; augmentation
and feature extraction then run batched on the GPU. `features_torch` produces exactly what `features.featurize`
produces for the same resampled points, so a model trained here runs unchanged on the live (numpy) path.
"""

import numpy as np
import torch

from .features import FACE_IDX, KEEP, LH, N_HAND, N_POINTS, N_POSE, RH, SEQ_LEN, fill_hand_gaps, normalize, resample

PREP_LEN = 128

_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
               (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
_FACE_PAIRS = [(61, 291), (78, 308), (70, 300), (63, 293), (105, 334), (66, 296), (107, 336)]
PERM = np.arange(N_POINTS)
for _a, _b in _POSE_PAIRS:
    PERM[_a], PERM[_b] = _b, _a
for _j in range(N_HAND):
    PERM[N_POSE + _j], PERM[N_POSE + N_HAND + _j] = N_POSE + N_HAND + _j, N_POSE + _j
_FOFF = N_POSE + 2 * N_HAND
for _a, _b in _FACE_PAIRS:
    _ia, _ib = FACE_IDX.index(_a), FACE_IDX.index(_b)
    PERM[_FOFF + _ia], PERM[_FOFF + _ib] = _FOFF + _ib, _FOFF + _ia


def prep(raw: np.ndarray, canon=None) -> np.ndarray:
    """Raw (T, FRAME_DIM) -> (PREP_LEN, N_POINTS, 3): normalised (optionally hand-canonicalised), gaps filled, resampled."""
    n = normalize(raw)
    if canon is not None:
        n = canon(n)
    pts = fill_hand_gaps(n.reshape(len(raw), N_POINTS, 3).copy())
    return resample(pts.reshape(len(pts), -1), PREP_LEN).reshape(PREP_LEN, N_POINTS, 3).astype(np.float32)


def features_torch(pts: torch.Tensor) -> torch.Tensor:
    """(B, SEQ_LEN, N_POINTS, 3) resampled points -> (B, SEQ_LEN, FEAT_DIM), same as features.featurize."""
    b, t = pts.shape[:2]
    keep = torch.as_tensor(KEEP, device=pts.device)
    xy = pts[:, :, keep, :2].reshape(b, t, -1)
    vel = torch.diff(xy, dim=1, prepend=xy[:, :1])
    shapes = []
    for sl in (LH, RH):
        h = pts[:, :, sl]
        present = h.abs().sum(dim=(2, 3)) > 0
        local = h - h[:, :, :1]
        size = local[:, :, 9, :2].norm(dim=-1, keepdim=True).clamp_min(1e-3)
        local = local / size[..., None]
        local = local * present[..., None, None]
        shapes.append(local.reshape(b, t, -1))
    return torch.cat([xy, vel, *shapes], dim=-1)


def _interp_time(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x: (B, L, P, 3); pos: (B, T) fractional frame positions -> (B, T, P, 3)."""
    L = x.shape[1]
    pos = pos.clamp(0, L - 1)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=L - 1)
    w = (pos - lo.float())[..., None, None]
    idx = lambda i: i[..., None, None].expand(-1, -1, x.shape[2], x.shape[3])  # noqa: E731
    return torch.gather(x, 1, idx(lo)) * (1 - w) + torch.gather(x, 1, idx(hi)) * w


class GpuAugment:
    def __init__(self, device: str, mirror_p=0.3, rot_deg=15.0, scale=0.15, shift=0.15, noise=0.015,
                 crop=(0.8, 1.0), warp=0.3, hand_drop_p=0.2):
        self.device = device
        self.perm = torch.as_tensor(PERM, device=device)
        self.mirror_p, self.rot, self.scale, self.shift, self.noise = mirror_p, rot_deg, scale, shift, noise
        self.crop, self.warp, self.hand_drop_p = crop, warp, hand_drop_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, PREP_LEN, N_POINTS, 3) prepared clips -> (B, SEQ_LEN, FEAT_DIM) augmented features."""
        dev = x.device
        b, L = x.shape[:2]
        # temporal crop + non-uniform speed (random positive increments)
        frac = torch.empty(b, device=dev).uniform_(*self.crop)
        start = torch.rand(b, device=dev) * (1 - frac) * (L - 1)
        inc = 1 + self.warp * (torch.rand(b, SEQ_LEN, device=dev) * 2 - 1)
        cum = torch.cumsum(inc, 1)
        cum = (cum - cum[:, :1]) / (cum[:, -1:] - cum[:, :1])
        pos = start[:, None] + cum * frac[:, None] * (L - 1)
        pts = _interp_time(x, pos)
        missing = pts.abs().sum(-1) == 0  # (B, T, P)
        # mirror
        m = torch.rand(b, device=dev) < self.mirror_p
        if m.any():
            mp = pts[m][:, :, self.perm].clone()
            mp[..., 0] *= -1
            pts[m] = mp
            missing[m] = missing[m][:, :, self.perm]
        # rotation, scale, shift, jitter
        a = torch.deg2rad(torch.empty(b, device=dev).uniform_(-self.rot, self.rot))
        c, s = torch.cos(a), torch.sin(a)
        xy = pts[..., :2]
        rx = xy[..., 0] * c[:, None, None] - xy[..., 1] * s[:, None, None]
        ry = xy[..., 0] * s[:, None, None] + xy[..., 1] * c[:, None, None]
        sx = torch.empty(b, device=dev).uniform_(1 - self.scale, 1 + self.scale)[:, None, None]
        sy = torch.empty(b, device=dev).uniform_(1 - self.scale, 1 + self.scale)[:, None, None]
        tx = torch.empty(b, device=dev).uniform_(-self.shift, self.shift)[:, None, None]
        ty = torch.empty(b, device=dev).uniform_(-self.shift, self.shift)[:, None, None]
        pts = torch.stack([rx * sx + tx, ry * sy + ty, pts[..., 2]], -1)
        pts = pts + torch.randn_like(pts) * self.noise
        # simulated tracking loss: one hand gone for a few frames
        drop = torch.rand(b, device=dev) < self.hand_drop_p
        if drop.any():
            t0 = torch.randint(0, SEQ_LEN, (b,), device=dev)
            ln = torch.randint(1, 6, (b,), device=dev)
            tt = torch.arange(SEQ_LEN, device=dev)[None]
            span = (tt >= t0[:, None]) & (tt < (t0 + ln)[:, None]) & drop[:, None]
            left = torch.rand(b, device=dev) < 0.5
            hand = torch.zeros(b, N_POINTS, dtype=torch.bool, device=dev)
            hand[left, LH] = True
            hand[~left, RH] = True
            missing = missing | (span[:, :, None] & hand[:, None, :])
        pts = pts.masked_fill(missing[..., None], 0.0)
        return features_torch(pts)
