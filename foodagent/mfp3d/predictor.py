"""Load a trained MFP3D checkpoint and run batched inference (used by FoodAgent's category-1 expert)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .model import MFP3D, MFP3DConfig
from .reconstruct import ReconConfig


def save_checkpoint(path: str, model: MFP3D, recon: ReconConfig, extra: Optional[Dict[str, Any]] = None) -> None:
    import torch

    torch.save({"model_state": model.state_dict(), "model_config": model.cfg.to_dict(),
                "recon_config": recon.to_dict(), **(extra or {})}, path)


class MFP3DPredictor:
    """Callable wrapper: clouds (P,3) + masked crops (3,S,S) -> portion values in the target's unit.

    target "volume" -> cm^3 (= ml), "weight" -> g, "energy" -> kcal, macros -> g.
    """

    def __init__(self, model: MFP3D, recon: ReconConfig, device: str = "cpu", meta: Optional[Dict] = None):
        self.model = model.eval().to(device)
        self.recon = recon
        self.device = device
        self.meta = meta or {}

    @property
    def target(self) -> str:
        return self.model.cfg.target

    @property
    def uses_image(self) -> bool:
        return self.model.image_encoder is not None

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "MFP3DPredictor":
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = MFP3DConfig.from_dict(ckpt["model_config"])
        cfg.image_pretrained = False  # weights come from the checkpoint
        model = MFP3D(cfg)
        model.load_state_dict(ckpt["model_state"])
        meta = {k: v for k, v in ckpt.items() if k not in ("model_state", "model_config", "recon_config")}
        return cls(model, ReconConfig.from_dict(ckpt["recon_config"]), device, meta)

    def predict(self, clouds: Sequence[np.ndarray], crops: Optional[Sequence[np.ndarray]] = None) -> List[float]:
        import torch

        if not len(clouds):
            return []
        pts = torch.from_numpy(np.stack(clouds).astype(np.float32)).to(self.device)
        imgs = None
        if self.uses_image:
            imgs = torch.from_numpy(np.stack(crops).astype(np.float32)).to(self.device)
        with torch.inference_mode():
            out = self.model(pts, imgs)
        return [max(0.0, float(v)) for v in out.cpu().numpy()]
