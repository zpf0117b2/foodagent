"""Category-3 expert: direct LLM inference for minor / easy elements (one batched Qwen2.5-VL call)."""
from __future__ import annotations

from typing import Dict, List, Tuple

from PIL import Image

from .. import prompts
from ..jsonutil import clamp, extract_json, to_float
from ..types import FoodElement, Nutrients
from ..vlm import VLM, prepare_image, user_message


def internal_knowledge_estimate(e: FoodElement) -> Nutrients:
    """Agent's own estimate from the analysis step: estimated mass x per-100 g profile."""
    return Nutrients(calories=e.per_100g.calories, fat=e.per_100g.fat, carb=e.per_100g.carb,
                     protein=e.per_100g.protein).scale(e.estimated_mass_g / 100.0) + Nutrients(mass=e.estimated_mass_g)


class LLMDirectExpert:
    def __init__(self, vlm: VLM, max_new_tokens: int = 1024):
        self.vlm = vlm
        self.max_new_tokens = max_new_tokens

    def estimate(self, image: Image.Image, elements: List[FoodElement]) -> Tuple[Dict[int, Nutrients], str]:
        """Returns {position in `elements`: Nutrients} and the raw model output."""
        if not elements:
            return {}, ""
        listing = "\n".join(f"{i + 1}. {e.name}: {e.description}" for i, e in enumerate(elements))
        model_img, _ = prepare_image(image)
        raw = self.vlm.chat(user_message(prompts.LLM_DIRECT.format(element_list=listing), [model_img]),
                            self.max_new_tokens)
        parsed = extract_json(raw)
        items = parsed.get("elements", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
        out: Dict[int, Nutrients] = {}
        for pos, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            idx = int(to_float(it.get("index"), pos + 1)) - 1
            if not 0 <= idx < len(elements) or idx in out:
                continue
            out[idx] = Nutrients(calories=clamp(to_float(it.get("calories")), 0, 5000),
                                 mass=clamp(to_float(it.get("mass_g", it.get("mass"))), 0, 3000),
                                 fat=clamp(to_float(it.get("fat")), 0, 500), carb=clamp(to_float(it.get("carb")), 0, 1000),
                                 protein=clamp(to_float(it.get("protein")), 0, 500))
        for i, e in enumerate(elements):  # anything the model skipped falls back to the analysis estimate
            out.setdefault(i, internal_knowledge_estimate(e))
        return out, raw
