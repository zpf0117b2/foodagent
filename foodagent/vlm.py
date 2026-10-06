"""Qwen2.5-VL backend (local inference with HF transformers).

Messages use the Qwen chat format:
    [{"role": "user", "content": [{"type": "image", "image": PIL.Image}, {"type": "text", "text": "..."}]}]

TransformersQwenVL loads Qwen/Qwen2.5-VL-*-Instruct locally; CachedVLM stores responses on disk so repeated runs /
resumed evaluations do not re-query the model.

Grounding note: Qwen2.5-VL emits absolute pixel coordinates in the coordinate frame of the image
*after* its internal `smart_resize`. We therefore pre-resize every image to that frame ourselves
(`prepare_image`) so returned `bbox_2d` values can be mapped back to the original resolution.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from PIL import Image

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

Messages = List[Dict[str, Any]]


class VLM(Protocol):
    name: str

    def chat(self, messages: Messages, max_new_tokens: int = 1024) -> str: ...


# --------------------------------------------------------------------------- image helpers
def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
    """Target (h, w) used by the Qwen2.5-VL processor: multiples of `factor`, pixel count bounded."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def prepare_image(image: Image.Image, min_pixels: int = MIN_PIXELS,
                  max_pixels: int = MAX_PIXELS) -> Tuple[Image.Image, Tuple[float, float]]:
    """Resize to the model's grid. Returns (resized image, (sx, sy)) mapping model coords -> original."""
    image = image.convert("RGB")
    w, h = image.size
    rh, rw = smart_resize(h, w, min_pixels=min_pixels, max_pixels=max_pixels)
    resized = image if (rw, rh) == (w, h) else image.resize((rw, rh), Image.BICUBIC)
    return resized, (w / rw, h / rh)


def _image_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _load_image(obj: Any) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj
    return Image.open(obj).convert("RGB")


def _iter_images(messages: Messages):
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image":
                    yield _load_image(part["image"])


# --------------------------------------------------------------------------- local transformers backend
class TransformersQwenVL:
    """Local Qwen2.5-VL via HF transformers (>= 4.56, which accepts `dtype=`)."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device_map: str = "auto", dtype: str = "bfloat16",
                 quantization: Optional[str] = None, attn_implementation: Optional[str] = None,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs: Dict[str, Any] = {"device_map": device_map, "dtype": getattr(torch, dtype)}
        if quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit",
                bnb_4bit_compute_dtype=getattr(torch, dtype), bnb_4bit_quant_type="nf4",
            )
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.name = model_id
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        self._torch = torch

    def chat(self, messages: Messages, max_new_tokens: int = 1024) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = list(_iter_images(messages)) or None
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# --------------------------------------------------------------------------- cache wrapper
class CachedVLM:
    """Disk cache keyed on (model, messages incl. image hashes, max_new_tokens)."""

    def __init__(self, inner: VLM, cache_dir: Optional[str]):
        self.inner = inner
        self.name = inner.name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, messages: Messages, max_new_tokens: int) -> str:
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(str(max_new_tokens).encode())
        for m in messages:
            h.update(m["role"].encode())
            content = m["content"]
            for p in [{"type": "text", "text": content}] if isinstance(content, str) else content:
                if p["type"] == "image":
                    h.update(hashlib.sha256(_image_bytes(_load_image(p["image"]))).digest())
                else:
                    h.update(p["text"].encode())
        return h.hexdigest()

    def chat(self, messages: Messages, max_new_tokens: int = 1024) -> str:
        if not self.cache_dir:
            return self.inner.chat(messages, max_new_tokens)
        path = self.cache_dir / f"{self._key(messages, max_new_tokens)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        response = self.inner.chat(messages, max_new_tokens)
        path.write_text(json.dumps({"model": self.name, "response": response}), encoding="utf-8")
        return response


def build_vlm(model: str = DEFAULT_MODEL, cache_dir: Optional[str] = None, **kwargs) -> VLM:
    return CachedVLM(TransformersQwenVL(model, **kwargs), cache_dir)


def user_message(text: str, images: Optional[List[Image.Image]] = None) -> Messages:
    content: List[Dict[str, Any]] = [{"type": "image", "image": im} for im in (images or [])]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]
