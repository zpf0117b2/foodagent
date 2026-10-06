"""Category-1 expert: MFP3D monocular portion estimation (FoodAgent Sec. II-A-1; Ma et al., ICPR'24).

Per image (RGB only):
  1. Segment Anything (SAM), prompted with each element's bbox from Qwen2.5-VL grounding, isolates the element.
  2. ZoeDepth predicts metric depth for the whole image; the global monocular scale is anchored with the known
     Nutrition5k rig distance (35.9 cm camera-to-table, `anchor_depth_cm`; None disables it).
  3. MFP3D (when a trained checkpoint is given): every element's depth point cloud and masked RGB crop are built
     with exactly the reconstruction settings stored in the checkpoint (foodagent/mfp3d/reconstruct.py) and the
     network regresses the portion - volume in cm^3 (FoodAgent then uses mass = volume x density) or, for a
     weight-trained checkpoint, mass in g directly.
  4. Without a checkpoint, a geometric estimate is used instead: RANSAC fit of the plate/table plane around the
     element and integration of food height above it.

Nutrition5k camera: 5.957e-3 cm^2 per pixel at 35.9 cm (Thames et al., CVPR'21) -> f ~= 465 px at 640x480.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from ..geometry import (Box, CameraIntrinsics, N5K_CAMERA_DISTANCE_CM, N5K_FOCAL_PX, N5K_PIXEL_AREA_CM2,  # noqa: F401
                        anchor_depth, backproject, box_mask, expand_box, fit_plane_ransac, geometric_volume)


@dataclass
class VolumeResult:
    volume_cm3: Optional[float]
    mask_area_px: int = 0
    method: str = ""
    info: Dict[str, Any] = field(default_factory=dict)
    mass_g: Optional[float] = None  # set when the MFP3D checkpoint regresses weight directly


class MFP3DVolumeEstimator:
    def __init__(self, sam_model: str = "facebook/sam-vit-huge", depth_model: str = "Intel/zoedepth-nyu-kitti",
                 device: Optional[str] = None, camera: Optional[CameraIntrinsics] = None,
                 anchor_depth_cm: Optional[float] = N5K_CAMERA_DISTANCE_CM, mfp3d=None,
                 support_ring_ratio: float = 0.35, min_mask_px: int = 10):
        """`mfp3d`: a foodagent.mfp3d.MFP3DPredictor (or None for the geometric estimate)."""
        self.sam_model_id, self.depth_model_id = sam_model, depth_model
        self.device = device
        self.camera = camera or CameraIntrinsics()
        self.anchor_depth_cm = anchor_depth_cm
        self.mfp3d = mfp3d
        self.support_ring_ratio = support_ring_ratio
        self.min_mask_px = min_mask_px
        self._sam = self._depth = None

    # ---- lazy model loading
    def _torch_device(self):
        import torch

        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self._sam is None:
            from transformers import AutoImageProcessor, SamModel, SamProcessor, ZoeDepthForDepthEstimation

            dev = self._torch_device()
            self._sam = (SamModel.from_pretrained(self.sam_model_id).to(dev).eval(),
                         SamProcessor.from_pretrained(self.sam_model_id))
            self._depth = (ZoeDepthForDepthEstimation.from_pretrained(self.depth_model_id).to(dev).eval(),
                           AutoImageProcessor.from_pretrained(self.depth_model_id))

    def predict_depth_cm(self, image: Image.Image) -> np.ndarray:
        import torch

        self._load()
        model, proc = self._depth
        # no padding so the prediction covers exactly the (aspect-preserving) resized image; we upsample it
        # ourselves, which avoids version-specific post-processing helpers
        inputs = proc(images=image, do_pad=False, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs)
        depth = torch.nn.functional.interpolate(out.predicted_depth.unsqueeze(1).float(),
                                                size=(image.height, image.width), mode="bicubic",
                                                align_corners=False)[0, 0].clamp_min(1e-3)
        return depth.cpu().numpy() * 100.0  # metres -> cm

    def segment(self, image: Image.Image, boxes: Sequence[Box]) -> List[np.ndarray]:
        import torch

        self._load()
        model, proc = self._sam
        inputs = proc(image, input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs, multimask_output=True)
        masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                         inputs["reshaped_input_sizes"].cpu())[0]  # (B, 3, H, W)
        best = out.iou_scores[0].argmax(dim=-1).cpu()  # (B,)
        return [masks[i, best[i]].numpy().astype(bool) & box_mask(masks.shape[-2:], b) for i, b in enumerate(boxes)]

    # ---- main entry
    def estimate(self, image: Image.Image, boxes: Sequence[Box]) -> List[VolumeResult]:
        if not boxes:
            return []
        image = image.convert("RGB")
        masks = self.segment(image, boxes)
        depth = self.predict_depth_cm(image)
        return self.estimate_from(image, depth, masks, boxes)

    def estimate_from(self, image: Image.Image, depth_cm: np.ndarray, masks: Sequence[np.ndarray],
                      boxes: Sequence[Box]) -> List[VolumeResult]:
        """Portion per element given a depth map and element masks (also usable with precomputed inputs)."""
        h, w = depth_cm.shape
        intr = self.camera.for_size(w, h)
        food_union = np.zeros((h, w), bool)
        for m in masks:
            food_union |= m
        background = ~food_union
        depth, scale = anchor_depth(depth_cm, background, self.anchor_depth_cm)

        results: List[Optional[VolumeResult]] = [None] * len(masks)
        valid = []
        for i, m in enumerate(masks):
            if int(m.sum()) < self.min_mask_px:
                results[i] = VolumeResult(None, int(m.sum()), "failed", {"error": "mask too small"})
            else:
                valid.append(i)
        if self.mfp3d is not None:
            self._run_mfp3d(image, depth, masks, valid, intr, scale, results)
        else:
            for i in valid:
                results[i] = self._geometric(depth, masks[i], boxes[i], background, intr, scale)
        return results  # type: ignore[return-value]

    def _run_mfp3d(self, image, depth, masks, valid, intr, scale, results) -> None:
        from ..mfp3d.reconstruct import depth_point_cloud, masked_crop

        recon = self.mfp3d.recon
        rng = np.random.default_rng(0)
        clouds = [depth_point_cloud(depth, masks[i], recon, intr, rng) for i in valid]
        crops = [masked_crop(image, masks[i], recon) for i in valid] if self.mfp3d.uses_image else None
        values = self.mfp3d.predict(clouds, crops)
        target = self.mfp3d.target
        for i, v in zip(valid, values):
            info = {"depth_scale": round(scale, 4), "mfp3d_target": target, "mfp3d_value": v}
            area = int(masks[i].sum())
            if target == "volume":
                results[i] = VolumeResult(v if v > 0 else None, area, "mfp3d" if v > 0 else "failed", info)
            elif target == "weight":
                results[i] = VolumeResult(None, area, "mfp3d" if v > 0 else "failed", info, mass_g=v if v > 0 else None)
            else:
                results[i] = VolumeResult(None, area, "failed",
                                          {**info, "error": f"MFP3D checkpoint target '{target}' is not a portion"})

    def _geometric(self, depth, mask, box, background, intr, scale) -> VolumeResult:
        h, w = depth.shape
        ring = box_mask((h, w), expand_box(box, (h, w), self.support_ring_ratio)) & background
        if ring.sum() < 50:
            ring = background
        vol, ginfo = geometric_volume(depth, mask, ring, *intr)
        ok = vol is not None and vol > 0
        return VolumeResult(vol if ok else None, int(mask.sum()), "geometric" if ok else "failed",
                            {"depth_scale": round(scale, 4), **ginfo})
