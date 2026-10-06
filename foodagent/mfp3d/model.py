"""MFP3D Stages 2-3: feature extraction (ResNet50 image branch + CurveNet point branch) and portion regression.

    f_I = delta^I(x_I)        ResNet50 (ImageNet) trunk, last two layers removed, pooled, FC -> 512
    f_P = delta^P(x_P)        CurveNet -> 512
    f   = f_I (+) f_P         concatenation -> 1024
    y_t = phi(f)              linear layer -> scalar portion attribute t (volume, weight, energy, ...)
Trained end-to-end with an L1 loss, one model per target attribute.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .curvenet import CurveNetEncoder, PointNetEncoder

TARGETS = ("volume", "weight", "energy", "protein", "fat", "carb")


@dataclass
class MFP3DConfig:
    point_encoder: str = "curvenet"  # "curvenet" | "pointnet" | "none" (image-only)
    use_image: bool = True  # False -> "point cloud only" rows of MFP3D Table 5
    feat_dim: int = 512
    k: int = 20
    curve_setting: str = "default"
    num_points: int = 1024
    image_pretrained: bool = True
    target: str = "volume"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MFP3DConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.trunk = nn.Sequential(*list(net.children())[:-2])  # drop avgpool + fc
        self.fc = nn.Linear(2048, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(self.trunk(images), 1).flatten(1)
        return F.relu(self.fc(x))


class MFP3D(nn.Module):
    def __init__(self, cfg: MFP3DConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.point_encoder == "curvenet":
            self.point_encoder: Optional[nn.Module] = CurveNetEncoder(cfg.feat_dim, cfg.k, cfg.num_points,
                                                                      cfg.curve_setting)
        elif cfg.point_encoder == "pointnet":
            self.point_encoder = PointNetEncoder(cfg.feat_dim)
        elif cfg.point_encoder == "none":
            self.point_encoder = None
        else:
            raise ValueError(f"unknown point encoder {cfg.point_encoder}")
        self.image_encoder = ImageEncoder(cfg.feat_dim, cfg.image_pretrained) if cfg.use_image else None
        n_feat = cfg.feat_dim * ((self.point_encoder is not None) + (self.image_encoder is not None))
        if n_feat == 0:
            raise ValueError("MFP3D needs at least one input modality")
        self.head = nn.Linear(n_feat, 1)

    def init_output_bias(self, value: float) -> None:
        """Start predictions at the training-set mean (speeds up L1 training on raw-unit targets)."""
        with torch.no_grad():
            self.head.bias.fill_(float(value))

    def forward(self, points: Optional[torch.Tensor], images: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = []
        if self.point_encoder is not None:
            feats.append(self.point_encoder(points))
        if self.image_encoder is not None:
            if images is None:
                raise ValueError("this MFP3D model expects images")
            feats.append(self.image_encoder(images))
        return self.head(torch.cat(feats, dim=1)).squeeze(1)
