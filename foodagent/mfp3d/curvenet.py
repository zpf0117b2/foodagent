"""Point-cloud encoders for MFP3D's 3D branch (delta^P).

`CurveNetEncoder` is an independent implementation of the CurveNet design (Xiang et al., "Walk in the Cloud",
ICCV'21) that MFP3D selects as its 3D backbone:
  * LPFA  - local point feature aggregation over k nearest neighbours with relative-position encoding;
  * CIC   - curve inception convolution blocks (bottleneck -> optional curve grouping/aggregation -> LPFA ->
            expansion, residual), with farthest-point down-sampling + radius-masked max pooling between stages;
  * curve grouping - learned walks of fixed length from top-scoring start points; each step picks a neighbour
            with a straight-through Gumbel softmax and suppresses walks that turn back on themselves;
  * curve aggregation - point features attend to intra-curve and inter-curve descriptors.
The head maps max+avg pooled features to a C=512 vector, matching MFP3D's image feature size.

`PointNetEncoder` is a light alternative (Qi et al., CVPR'17) for quick experiments / CPU debugging.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- point utilities
def pairwise_sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a (B,3,M), b (B,3,N) -> (B,M,N) squared distances."""
    return ((a.unsqueeze(-1) - b.unsqueeze(-2)) ** 2).sum(1)


def knn(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """xyz (B,3,N) -> (B,N,k) neighbour indices (self included)."""
    return pairwise_sqdist(xyz, xyz).topk(k, dim=-1, largest=False).indices


def gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B,C,N), idx (B,M,k) -> (B,C,M,k)."""
    b, c, _ = x.shape
    _, m, k = idx.shape
    return x.gather(2, idx.reshape(b, 1, m * k).expand(b, c, m * k)).reshape(b, c, m, k)


def gather_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B,C,N), idx (B,M) -> (B,C,M)."""
    return x.gather(2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))


def farthest_point_sample(xyz: torch.Tensor, m: int, random_start: bool = False) -> torch.Tensor:
    """xyz (B,3,N) -> (B,m) indices of iteratively farthest points."""
    b, _, n = xyz.shape
    idx = torch.zeros(b, m, dtype=torch.long, device=xyz.device)
    dist = torch.full((b, n), float("inf"), device=xyz.device)
    far = (torch.randint(0, n, (b,), device=xyz.device) if random_start
           else torch.zeros(b, dtype=torch.long, device=xyz.device))
    batch = torch.arange(b, device=xyz.device)
    for i in range(m):
        idx[:, i] = far
        centre = xyz[batch, :, far].unsqueeze(-1)  # (B,3,1)
        dist = torch.minimum(dist, ((xyz - centre) ** 2).sum(1))
        far = dist.argmax(-1)
    return idx


def conv_bn(cin: int, cout: int, dim: int = 1, act: bool = True) -> nn.Sequential:
    conv, bn = (nn.Conv1d, nn.BatchNorm1d) if dim == 1 else (nn.Conv2d, nn.BatchNorm2d)
    layers: List[nn.Module] = [conv(cin, cout, 1, bias=False), bn(cout)]
    if act:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- building blocks
class LPFA(nn.Module):
    """Local point feature aggregation."""

    def __init__(self, in_ch: int, out_ch: int, k: int, initial: bool = False):
        super().__init__()
        self.k, self.initial = k, initial
        if not initial:
            self.pos = conv_bn(9, in_ch, dim=2, act=False)
        self.mlp = conv_bn(9 if initial else in_ch, out_ch, dim=2)

    def forward(self, x: torch.Tensor, xyz: torch.Tensor, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        idx = knn(xyz, self.k) if idx is None else idx
        nb_xyz = gather_neighbors(xyz, idx)  # (B,3,N,k)
        ctr = xyz.unsqueeze(-1).expand_as(nb_xyz)
        geo = torch.cat([ctr, nb_xyz, nb_xyz - ctr], dim=1)  # (B,9,N,k)
        if self.initial:
            return self.mlp(geo).max(-1).values
        f = gather_neighbors(x, idx) - x.unsqueeze(-1) + self.pos(geo)
        return self.mlp(f).mean(-1)


class CurveGrouping(nn.Module):
    """Learned walks: `curve_num` curves of `curve_length` points each -> (B,C,M,L) curve features."""

    def __init__(self, ch: int, k: int, curve_num: int, curve_length: int, tau: float = 1.0):
        super().__init__()
        self.k, self.curve_num, self.curve_length, self.tau = k, curve_num, curve_length, tau
        self.start_score = nn.Conv1d(ch, 1, 1)
        self.step_score = nn.Sequential(conv_bn(2 * ch, ch // 2 or 1, dim=2), nn.Conv2d(ch // 2 or 1, 1, 1))
        self.turn_penalty = nn.Parameter(torch.tensor(1.0))

    def _choose(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)

    def forward(self, x: torch.Tensor, xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        m = min(self.curve_num, n)
        score = torch.sigmoid(self.start_score(x))  # (B,1,N)
        cur = score.squeeze(1).topk(m, dim=-1).indices  # (B,M)
        feat = gather_points(x * score, cur)  # gated -> gradient reaches the start scorer
        cur_xyz, prev_dir = gather_points(xyz, cur), None
        curve = [feat]
        for _ in range(self.curve_length - 1):
            nb_idx = idx.gather(1, cur.unsqueeze(-1).expand(-1, -1, idx.shape[-1]))  # (B,M,k)
            nb_feat = gather_neighbors(x, nb_idx)  # (B,C,M,k)
            nb_xyz = gather_neighbors(xyz, nb_idx)  # (B,3,M,k)
            logits = self.step_score(torch.cat([feat.unsqueeze(-1).expand_as(nb_feat), nb_feat], 1)).squeeze(1)
            step = nb_xyz - cur_xyz.unsqueeze(-1)
            logits = logits.masked_fill(nb_idx == cur.unsqueeze(-1), -1e4)  # no standing still
            if prev_dir is not None:  # crossover suppression: discourage turning back
                cos = F.cosine_similarity(step, prev_dir.unsqueeze(-1), dim=1, eps=1e-8)
                logits = logits - F.softplus(self.turn_penalty) * F.relu(-cos) * 10.0
            w = self._choose(logits)  # (B,M,k) one-hot (straight-through while training)
            feat = (nb_feat * w.unsqueeze(1)).sum(-1)
            new_xyz = (nb_xyz * w.unsqueeze(1)).sum(-1)
            prev_dir, cur_xyz = new_xyz - cur_xyz, new_xyz
            cur = nb_idx.gather(-1, w.argmax(-1, keepdim=True)).squeeze(-1)
            curve.append(feat)
        return torch.stack(curve, dim=-1)  # (B,C,M,L)


class CurveAggregation(nn.Module):
    """Points attend to intra-curve (one descriptor per curve) and inter-curve (per walk step) descriptors."""

    def __init__(self, ch: int):
        super().__init__()
        mid = max(ch // 2, 1)
        self.query = nn.Conv1d(ch, mid, 1, bias=False)
        self.key_intra, self.key_inter = nn.Conv1d(ch, mid, 1, bias=False), nn.Conv1d(ch, mid, 1, bias=False)
        self.val_intra, self.val_inter = nn.Conv1d(ch, ch, 1, bias=False), nn.Conv1d(ch, ch, 1, bias=False)
        self.fuse = conv_bn(2 * ch, ch, act=False)
        self.scale = mid ** -0.5

    def _attend(self, q: torch.Tensor, desc: torch.Tensor, key: nn.Module, val: nn.Module) -> torch.Tensor:
        att = torch.softmax(torch.einsum("bcn,bcm->bnm", q, key(desc)) * self.scale, dim=-1)
        return torch.einsum("bcm,bnm->bcn", val(desc), att)

    def forward(self, x: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        intra = self._attend(q, curves.mean(-1), self.key_intra, self.val_intra)  # descriptors per curve
        inter = self._attend(q, curves.mean(-2), self.key_inter, self.val_inter)  # descriptors per step
        return F.leaky_relu(x + self.fuse(torch.cat([intra, inter], 1)), 0.2)


class CIC(nn.Module):
    """Curve inception convolution block with optional down-sampling."""

    def __init__(self, npoint: int, radius: float, k: int, in_ch: int, out_ch: int, bottleneck_ratio: int = 2,
                 curve: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.npoint, self.radius, self.k = npoint, radius, k
        mid = max(in_ch // bottleneck_ratio, 1)
        self.reduce = conv_bn(in_ch, mid)
        self.group = CurveGrouping(mid, k, *curve) if curve else None
        self.aggregate = CurveAggregation(mid) if curve else None
        self.lpfa = LPFA(mid, mid, k)
        self.expand = conv_bn(mid, out_ch, act=False)
        self.shortcut = conv_bn(in_ch, out_ch, act=False) if in_ch != out_ch else nn.Identity()

    def downsample(self, xyz: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sel = farthest_point_sample(xyz, self.npoint, random_start=self.training)
        new_xyz = gather_points(xyz, sel)
        k = min(self.k, xyz.shape[-1])
        dist, nb = pairwise_sqdist(new_xyz, xyz).topk(k, dim=-1, largest=False)  # (B,M,k)
        outside = dist > self.radius ** 2
        outside[..., 0] = False  # always keep the nearest point
        feats = gather_neighbors(x, nb).masked_fill(outside.unsqueeze(1), float("-inf"))
        return new_xyz, feats.max(-1).values

    def forward(self, xyz: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if xyz.shape[-1] > self.npoint:
            xyz, x = self.downsample(xyz, x)
        idx = knn(xyz, min(self.k, xyz.shape[-1]))
        h = self.reduce(x)
        if self.group is not None:
            h = self.aggregate(h, self.group(h, xyz, idx))
        h = self.expand(self.lpfa(h, xyz, idx))
        return xyz, F.leaky_relu(h + self.shortcut(x), 0.2)


CURVE_SETTINGS = {
    # (curve_num, curve_length) for the four stages; None = no curves at that stage
    "default": [(100, 5), (100, 5), None, None],
    "long": [(10, 30), None, None, None],
    "none": [None, None, None, None],
}


class CurveNetEncoder(nn.Module):
    """CurveNet backbone -> C-dim global feature."""

    def __init__(self, out_dim: int = 512, k: int = 20, num_points: int = 1024, setting: str = "default"):
        super().__init__()
        curves = CURVE_SETTINGS[setting]
        n1, n2, n3 = num_points, max(num_points // 4, 8), max(num_points // 16, 4)
        self.lpfa = LPFA(9, 32, k, initial=True)
        self.k = k
        spec = [  # (npoint, radius, in, out, bottleneck, curve)
            (n1, 0.05, 32, 64, 2, curves[0]), (n1, 0.05, 64, 64, 4, curves[0]),
            (n1, 0.05, 64, 128, 2, curves[1]), (n1, 0.1, 128, 128, 4, curves[1]),
            (n2, 0.1, 128, 256, 2, curves[2]), (n2, 0.2, 256, 256, 4, curves[2]),
            (n3, 0.2, 256, 512, 2, curves[3]), (n3, 0.4, 512, 512, 4, curves[3]),
        ]
        self.blocks = nn.ModuleList(CIC(npt, r, k, ci, co, br, cv) for npt, r, ci, co, br, cv in spec)
        self.conv = conv_bn(512, 1024)
        self.fc = nn.Sequential(nn.Linear(2048, out_dim, bias=False), nn.BatchNorm1d(out_dim),
                                nn.ReLU(inplace=True), nn.Dropout(0.5))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points.transpose(1, 2).contiguous() if points.shape[-1] == 3 else points  # (B,3,N)
        x = self.lpfa(None, xyz, knn(xyz, min(self.k, xyz.shape[-1])))
        for blk in self.blocks:
            xyz, x = blk(xyz, x)
        x = self.conv(x)
        return self.fc(torch.cat([x.max(-1).values, x.mean(-1)], dim=1))


class PointNetEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(conv_bn(3, 64), conv_bn(64, 128), conv_bn(128, 1024))
        self.fc = nn.Sequential(nn.Linear(1024, out_dim, bias=False), nn.BatchNorm1d(out_dim),
                                nn.ReLU(inplace=True), nn.Dropout(0.3))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points.transpose(1, 2).contiguous() if points.shape[-1] == 3 else points
        return self.fc(self.mlp(xyz).max(-1).values)
