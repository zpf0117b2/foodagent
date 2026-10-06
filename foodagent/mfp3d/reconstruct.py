"""MFP3D Stage 1 - 3D reconstruction module (Ma et al., ICPR'24, Sec. 3.1).

Depth point clouds: segment the food (SAM mask), estimate a depth map (ZoeDepth), and sample `num_points`
(1,024 in the paper) points from the food foreground. Two coordinate conventions are supported, and the one a
model was trained with is stored in its checkpoint so that inference reconstructs clouds identically:

  * "uvz"    - paper convention: keep the two image coordinates and use the estimated depth as the third
               dimension (u, v normalised by the longer image side; depth in metres).
  * "metric" - back-project with camera intrinsics into camera-frame XYZ (cm) / `coord_scale`. This keeps the
               physical size of the food independent of image resolution and field of view, which matters when a
               model trained on MetaFood3D is applied to Nutrition5k inside FoodAgent.

Both conventions centre the cloud on its centroid (translation carries no portion information).
The 2D branch receives the masked RGB image (background zeroed), cropped to the mask bounding box with a margin,
padded to a square and resized to `image_size`, then ImageNet-normalised.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ..geometry import backproject, mask_bbox

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class ReconConfig:
    mode: str = "metric"  # "metric" | "uvz"
    num_points: int = 1024
    coord_scale: float = 10.0  # metric mode: cm -> network units (10 => decimetres)
    center: bool = True
    image_size: int = 224
    crop_margin: float = 0.1
    crop_to_mask: bool = True  # False: whole masked frame resized (paper's single-object setting)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReconConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def sample_points(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((n, 3), np.float32)
    idx = rng.choice(len(points), n, replace=len(points) < n)
    return points[idx].astype(np.float32)


def depth_point_cloud(depth_cm: np.ndarray, mask: np.ndarray, cfg: ReconConfig,
                      intrinsics: Optional[Tuple[float, float, float, float]] = None,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """(num_points, 3) float32 cloud of the masked food region."""
    rng = rng or np.random.default_rng(0)
    mask = mask & np.isfinite(depth_cm) & (depth_cm > 0)
    if cfg.mode == "metric":
        if intrinsics is None:
            raise ValueError("metric reconstruction needs camera intrinsics")
        pts = backproject(depth_cm, *intrinsics)[mask] / cfg.coord_scale
    elif cfg.mode == "uvz":
        h, w = depth_cm.shape
        v, u = np.nonzero(mask)
        s = float(max(h, w))
        pts = np.stack([u / s, v / s, depth_cm[mask] / 100.0], axis=-1)
    else:
        raise ValueError(f"unknown reconstruction mode {cfg.mode}")
    cloud = sample_points(pts, cfg.num_points, rng)
    if cfg.center and len(pts):
        cloud = cloud - cloud.mean(axis=0, keepdims=True)
    return cloud.astype(np.float32)


def normalize_unit_cube(cloud: np.ndarray) -> np.ndarray:
    """'Normalized GTPC' ablation: rescale every axis to [0, 1], removing true scale."""
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    return ((cloud - lo) / np.maximum(hi - lo, 1e-9)).astype(np.float32)


def masked_crop(image: Image.Image, mask: np.ndarray, cfg: ReconConfig, normalize: bool = True) -> np.ndarray:
    """(3, S, S) float32 network input: masked food, cropped around the mask, ImageNet-normalised."""
    rgb = np.asarray(image.convert("RGB"), np.float32) / 255.0
    rgb = rgb * mask[..., None]
    box = mask_bbox(mask) if cfg.crop_to_mask else None
    if box is not None:
        x1, y1, x2, y2 = box
        side = max(x2 - x1, y2 - y1) * (1 + 2 * cfg.crop_margin)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
        size = max(1, int(round(side)))
        canvas = np.zeros((size, size, 3), np.float32)  # zero padding outside the image
        h, w = mask.shape
        sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(w, x1 + size), min(h, y1 + size)
        canvas[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = rgb[sy1:sy2, sx1:sx2]
        rgb = canvas
    pil = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((cfg.image_size, cfg.image_size),
                                                                              Image.BILINEAR)
    out = np.asarray(pil, np.float32) / 255.0
    if normalize:
        out = (out - IMAGENET_MEAN) / IMAGENET_STD
    return out.transpose(2, 0, 1).copy()
