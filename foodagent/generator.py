"""Response generator (Sec. II-C): converts expert outputs into nutrition and aggregates the dish profile."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PIL import Image

from . import prompts
from .experts.llm_direct import internal_knowledge_estimate
from .experts.nutrition_db import NutritionDB
from .experts.rag import RAGExpert
from .jsonutil import clamp, extract_json, to_float
from .types import ElementEstimate, FoodElement, FoodReference, Nutrients
from .vlm import VLM, prepare_image, user_message


class ResponseGenerator:
    def __init__(self, vlm: VLM, max_new_tokens: int = 1024):
        self.vlm = vlm
        self.max_new_tokens = max_new_tokens

    # ---------------------------------------------------------------- category 1: mass = volume x density
    @staticmethod
    def from_volume(e: FoodElement, volume_cm3: Optional[float], db: Optional[NutritionDB],
                    mass_g: Optional[float] = None) -> ElementEstimate:
        """Eq. (1): mass = volume x density (or the mass regressed directly by a weight-trained MFP3D)."""
        mass = mass_g if mass_g is not None else volume_cm3 * e.density_g_per_cm3
        per_g, source = None, "agent"
        if db is not None:
            try:
                per_g = db.per_gram(e.name)
                source = db.name
            except Exception:  # API hiccup -> internal knowledge
                per_g = None
        if per_g is None or per_g.calories <= 0:
            per_g = Nutrients(e.per_100g.calories / 100, 1.0, e.per_100g.fat / 100, e.per_100g.carb / 100,
                              e.per_100g.protein / 100)
            source = "agent"
        n = per_g.scale(mass)
        n.mass = mass
        return ElementEstimate(e, n, {"volume_cm3": volume_cm3, "density": e.density_g_per_cm3, "mass_g": mass,
                                      "per_gram_source": source})

    # ---------------------------------------------------------------- category 2: units vs reference unit
    def estimate_rag_units(self, image: Image.Image, elements: List[FoodElement],
                           refs: Dict[int, List[FoodReference]], rag: Optional[RAGExpert] = None
                           ) -> Tuple[List[ElementEstimate], str]:
        if not elements:
            return [], ""
        model_img, _ = prepare_image(image)
        images = [model_img]
        blocks = []
        for i, e in enumerate(elements):
            lines = [f'Element {i + 1}: "{e.name}" - {e.description}']
            for j, r in enumerate(refs[i]):
                photo = rag.reference_image(r) if rag else None
                tag = ""
                if photo is not None:
                    photo.thumbnail((336, 336))
                    images.append(photo)
                    tag = f" [reference photo = image {len(images)}]"
                lines.append(f"  Reference {j}: {r.describe()}{tag}")
            blocks.append("\n".join(lines))
        photo_note = "; some references include a photo, given as additional images" if len(images) > 1 else ""
        prompt = prompts.RAG_UNITS.format(photo_note=photo_note, element_blocks="\n".join(blocks))
        raw = self.vlm.chat(user_message(prompt, images), self.max_new_tokens)
        parsed = extract_json(raw)
        items = parsed.get("elements", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])

        chosen: Dict[int, Tuple[int, float]] = {}
        for pos, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            i = int(to_float(it.get("element"), pos + 1)) - 1
            if 0 <= i < len(elements) and i not in chosen:
                j = int(to_float(it.get("reference"), 0))
                j = j if 0 <= j < len(refs[i]) else 0
                chosen[i] = (j, clamp(to_float(it.get("num_units")), 0.0, 50.0))

        out = []
        for i, e in enumerate(elements):
            if i in chosen:
                j, units = chosen[i]
                ref = refs[i][j]
                n = ref.nutrients.scale(units)
                n.mass = ref.serving_weight_g * units
                out.append(ElementEstimate(e, n, {"reference": ref.to_dict(), "num_units": units}))
            else:
                out.append(ElementEstimate(e, internal_knowledge_estimate(e),
                                           {"fallback": "model gave no unit estimate; used agent knowledge"}))
        return out, raw

    # ---------------------------------------------------------------- aggregation + user-facing answer
    @staticmethod
    def aggregate(estimates: List[ElementEstimate]) -> Nutrients:
        total = Nutrients()
        for est in estimates:
            total = total + est.nutrients
        return total

    @staticmethod
    def describe_amount(est: ElementEstimate) -> str:
        d = est.details
        if "num_units" in d:
            r = d["reference"]
            return f"around {d['num_units'] * r['serving_qty']:g} {r['serving_unit']} of {est.element.name}"
        return f"{est.nutrients.mass:.0f}g of {est.element.name}"

    @classmethod
    def render(cls, estimates: List[ElementEstimate], total: Nutrients, title: str = "Total estimation") -> str:
        parts = ", ".join(cls.describe_amount(e) for e in estimates)
        return (f"There is {parts}.\n{title}:\nCalories {total.calories:.2f}kcal, Fat {total.fat:.2f}g, "
                f"Carbohydrates {total.carb:.2f}g, Protein {total.protein:.2f}g")
