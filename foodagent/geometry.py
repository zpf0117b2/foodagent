"""Camera geometry shared by the volume expert and MFP3D point-cloud reconstruction (pure numpy)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

Box = Tuple[float, float, float, float]

# Nutrition5k overhead rig (Thames et al., CVPR'21): camera 35.9 cm above the table, 5.957e-3 cm^2 per pixel
N5K_CAMERA_DISTANCE_CM = 35.9
N5K_PIXEL_AREA_CM2 = 5.957e-3
N5K_FOCAL_PX = N5K_CAMERA_DISTANCE_CM / math.sqrt(N5K_PIXEL_AREA_CM2)  # ~465 px @ 640x480


@dataclass
class CameraIntrinsics:
    fx: float = N5K_FOCAL_PX
    fy: float = N5K_FOCAL_PX
    cx: Optional[float] = None  # defaults to image centre
    cy: Optional[float] = None
    ref_width: int = 640  # resolution the intrinsics refer to; rescaled for other image sizes

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "CameraIntrinsics":
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx=f, fy=f, ref_width=width)

    def for_size(self, width: int, height: int) -> Tuple[float, float, float, float]:
        s = width / self.ref_width
        cx = self.cx * s if self.cx is not None else (width - 1) / 2.0
        cy = self.cy * s if self.cy is not None else (height - 1) / 2.0
        return self.fx * s, self.fy * s, cx, cy


def backproject(depth_cm: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """(H, W) depth -> (H, W, 3) camera-frame points in cm."""
    h, w = depth_cm.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = depth_cm.astype(np.float64)
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)


def anchor_depth(depth_cm: np.ndarray, background: np.ndarray, target_cm: Optional[float]) -> Tuple[np.ndarray, float]:
    """Rescale monocular depth so that the median background depth equals `target_cm` (global scale fix)."""
    if not target_cm or not background.any():
        return depth_cm, 1.0
    med = float(np.median(depth_cm[background]))
    if med <= 0:
        return depth_cm, 1.0
    scale = target_cm / med
    return depth_cm * scale, scale


def fit_plane_ransac(points: np.ndarray, iters: int = 200, thresh_cm: float = 0.4,
                     seed: int = 0) -> Optional[Tuple[np.ndarray, float]]:
    """Fit n.p + d = 0 (|n| = 1) robustly; returns (n, d) or None. n is oriented towards the camera (n_z < 0)."""
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    best_inliers, best = 0, None
    for _ in range(iters):
        sample = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ sample[0]
        inliers = np.abs(points @ n + d) < thresh_cm
        count = int(inliers.sum())
        if count > best_inliers:
            best_inliers, best = count, inliers
    if best is None:
        return None
    pts = points[best]  # least-squares refinement on inliers
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    n = vt[-1]
    if n[2] > 0:  # camera looks along +z; "up" (towards camera) is -z
        n = -n
    return n, float(-n @ centroid)


def geometric_volume(depth_cm: np.ndarray, mask: np.ndarray, support_mask: np.ndarray,
                     fx: float, fy: float, cx: float, cy: float) -> Tuple[Optional[float], Dict[str, Any]]:
    """Integrate food height above a support plane fitted to `support_mask` pixels."""
    pts = backproject(depth_cm, fx, fy, cx, cy)
    support = pts[support_mask & np.isfinite(depth_cm) & (depth_cm > 0)]
    if len(support) > 20000:
        support = support[np.random.default_rng(0).choice(len(support), 20000, replace=False)]
    plane = fit_plane_ransac(support)
    if plane is None or not mask.any():
        return None, {"error": "no support plane" if plane is None else "empty mask"}
    n, d = plane
    food = pts[mask]
    heights = np.clip(food @ n + d, 0.0, None)  # signed distance towards camera
    # depth of the support plane along each food pixel's ray -> pixel footprint on the plane
    v_idx, u_idx = np.nonzero(mask)
    rays = np.stack([(u_idx - cx) / fx, (v_idx - cy) / fy, np.ones_like(u_idx, dtype=np.float64)], axis=-1)
    denom = rays @ n
    z_plane = np.where(np.abs(denom) > 1e-9, -d / denom, food[:, 2])
    footprint = (z_plane / fx) * (z_plane / fy)
    volume = float(np.sum(heights * footprint))
    return volume, {"plane_normal": n.round(4).tolist(), "plane_d": round(d, 3),
                    "mean_height_cm": float(heights.mean()), "max_height_cm": float(heights.max())}


def box_mask(shape: Tuple[int, int], box: Box) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), bool)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    return m


def expand_box(box: Box, shape: Tuple[int, int], ratio: float) -> Box:
    h, w = shape
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * ratio, (y2 - y1) * ratio
    return max(0.0, x1 - dx), max(0.0, y1 - dy), min(float(w), x2 + dx), min(float(h), y2 + dy)


def mask_bbox(mask: np.ndarray) -> Optional[Box]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
