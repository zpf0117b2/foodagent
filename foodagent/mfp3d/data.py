"""MFP3D training data: HDF5 storage and a PyTorch dataset with augmentation.

H5 schema (identical to the preprocessed MetaFood3D files released with MFP3D, so those can be used directly):
    data        (N, P, 3) float32   point clouds
    image       (N, 3, S, S) float32 network-ready RGB (ImageNet-normalised, or in [0, 1] - auto-detected)
    image_path  (N,) str            provenance
    weight / volume / energy / protein / fat / carb   (N,) float32   targets (any subset)
  optional, written by our builder:
    object_id   (N,) str            MetaFood3D object (several frames of one object share it; used for splits)
    attrs["recon"] JSON             ReconConfig used to build the clouds (copied into trained checkpoints)
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .model import TARGETS
from .reconstruct import IMAGENET_MEAN, IMAGENET_STD

# how each target scales with an isotropic geometric scale factor s (portion ~ volume ~ s^3)
SCALE_EXPONENT = {"volume": 3, "weight": 3, "energy": 3, "protein": 3, "fat": 3, "carb": 3}


def _decode(a) -> List[str]:
    return [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in a]


def write_h5(path: str, points: np.ndarray, images: np.ndarray, targets: Dict[str, np.ndarray],
             image_paths: Sequence[str], object_ids: Optional[Sequence[str]] = None,
             recon: Optional[dict] = None) -> None:
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=points.astype(np.float32), compression="gzip")
        f.create_dataset("image", data=images.astype(np.float32), compression="gzip")
        str_dt = h5py.string_dtype("utf-8")
        f.create_dataset("image_path", data=np.array(list(image_paths), dtype=object), dtype=str_dt)
        if object_ids is not None:
            f.create_dataset("object_id", data=np.array(list(object_ids), dtype=object), dtype=str_dt)
        for k, v in targets.items():
            f.create_dataset(k, data=np.asarray(v, np.float32))
        if recon is not None:
            f.attrs["recon"] = json.dumps(recon)


class H5Arrays:
    """Loads one split fully into memory (MetaFood3D-sized data fits easily)."""

    def __init__(self, path: str, target: str):
        import h5py

        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        with h5py.File(path, "r") as f:
            missing = [k for k in ("data", "image", target) if k not in f]
            if missing:
                raise KeyError(f"{path} lacks {missing}")
            self.points = f["data"][:].astype(np.float32)
            self.images = f["image"][:].astype(np.float32)
            self.targets = f[target][:].astype(np.float32)
            n = len(self.targets)
            self.image_paths = _decode(f["image_path"][:]) if "image_path" in f else [str(i) for i in range(n)]
            self.object_ids = _decode(f["object_id"][:]) if "object_id" in f else list(self.image_paths)
            self.recon = json.loads(f.attrs["recon"]) if "recon" in f.attrs else None
        if self.points.ndim == 3 and self.points.shape[1] == 3 and self.points.shape[2] != 3:
            self.points = self.points.transpose(0, 2, 1).copy()  # accept (N,3,P)
        keep = np.isfinite(self.targets) & (self.targets > 0)
        if not keep.all():
            self._subset(np.nonzero(keep)[0])
        # images stored in [0, 1] -> ImageNet-normalise (the network always sees normalised input)
        if self.images.size and self.images.min() >= 0.0 and self.images.max() <= 1.0:
            self.images = (self.images - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
        self.target = target

    def _subset(self, idx: np.ndarray) -> None:
        self.points, self.images, self.targets = self.points[idx], self.images[idx], self.targets[idx]
        self.image_paths = [self.image_paths[i] for i in idx]
        self.object_ids = [self.object_ids[i] for i in idx]

    def split_by_object(self, val_fraction: float, seed: int = 42) -> "tuple[np.ndarray, np.ndarray]":
        """Indices (train, val) with all frames of an object on the same side."""
        objs = sorted(set(self.object_ids))
        rng = np.random.default_rng(seed)
        rng.shuffle(objs)
        n_val = max(1, int(round(len(objs) * val_fraction))) if val_fraction > 0 else 0
        val_objs = set(objs[:n_val])
        val = np.array([i for i, o in enumerate(self.object_ids) if o in val_objs], dtype=int)
        train = np.array([i for i, o in enumerate(self.object_ids) if o not in val_objs], dtype=int)
        return train, val

    def __len__(self) -> int:
        return len(self.targets)


def augment_cloud(cloud: np.ndarray, target: float, target_name: str, rng: np.random.Generator,
                  up_axis: int = 2, scale_range: float = 0.0, jitter: float = 0.002) -> "tuple[np.ndarray, float]":
    """Volume-preserving augmentation: rotation about the up axis, jitter, shuffling.

    Optional isotropic scaling (`scale_range` > 0, e.g. 0.2 -> s in [0.8, 1.2]) rescales the target by s**3 so the
    label stays physically consistent - scaling clouds without changing the label would teach the network to ignore
    size, which the MFP3D ablation shows is critical information.
    """
    cloud = cloud.copy()
    theta = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(theta), math.sin(theta)
    a, b = [ax for ax in range(3) if ax != up_axis]
    xa, xb = cloud[:, a].copy(), cloud[:, b].copy()
    cloud[:, a], cloud[:, b] = c * xa - s * xb, s * xa + c * xb
    if scale_range > 0:
        k = rng.uniform(1 - scale_range, 1 + scale_range)
        cloud *= k
        target = target * k ** SCALE_EXPONENT.get(target_name, 3)
    if jitter > 0:
        cloud += np.clip(rng.normal(0, jitter, cloud.shape), -5 * jitter, 5 * jitter).astype(np.float32)
    rng.shuffle(cloud)
    return cloud.astype(np.float32), float(target)


class MFP3DTorchDataset:
    """torch Dataset view over H5Arrays (optionally a subset)."""

    def __init__(self, arrays: H5Arrays, indices: Optional[np.ndarray] = None, augment: bool = False,
                 up_axis: int = 2, scale_range: float = 0.0, seed: int = 0):
        self.a = arrays
        self.idx = np.arange(len(arrays)) if indices is None else np.asarray(indices)
        self.augment, self.up_axis, self.scale_range = augment, up_axis, scale_range
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        import torch

        j = int(self.idx[i])
        cloud, target = self.a.points[j], float(self.a.targets[j])
        image = self.a.images[j]
        if self.augment:
            cloud, target = augment_cloud(cloud, target, self.a.target, self.rng, self.up_axis, self.scale_range)
            if self.rng.random() < 0.5:  # horizontal flip of the image branch
                image = image[:, :, ::-1].copy()
        return torch.from_numpy(cloud), torch.from_numpy(np.ascontiguousarray(image)), torch.tensor(target)
