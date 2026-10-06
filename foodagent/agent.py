"""Nutrition reasoning agent: element analysis/decomposition and three-category expert routing (Sec. II-B)."""
from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from . import prompts
from .jsonutil import clamp, extract_json, to_float
from .types import FoodElement, Nutrients
from .vlm import VLM, prepare_image, user_message

CATEGORY_MAIN_CLEAR = 1  # -> MFP3D volume estimator
CATEGORY_MAIN_UNCLEAR = 2  # -> NutritionX RAG
CATEGORY_MINOR = 3  # -> direct LLM inference
DEFAULT_ROUTE = {CATEGORY_MAIN_CLEAR: "volume", CATEGORY_MAIN_UNCLEAR: "rag", CATEGORY_MINOR: "llm"}


def _as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "clear")
    return bool(v)


def normalize_name(name: str) -> str:
    return " ".join(name.lower().strip().split())


def match_name(query: str, candidates: Sequence[str], cutoff: float = 0.5) -> Optional[int]:
    """Index of the candidate best matching `query` (exact > substring > fuzzy), or None."""
    q = normalize_name(query)
    norm = [normalize_name(c) for c in candidates]
    if q in norm:
        return norm.index(q)
    for i, c in enumerate(norm):
        if q in c or (c and c in q):
            return i
    scores = [difflib.SequenceMatcher(None, q, c).ratio() for c in norm]
    if scores and max(scores) >= cutoff:
        return scores.index(max(scores))
    return None


class NutritionReasoningAgent:
    def __init__(self, vlm: VLM, max_new_tokens: int = 2048):
        self.vlm = vlm
        self.max_new_tokens = max_new_tokens

    # ---------------------------------------------------------------- 1) element analysis & decomposition
    def analyze(self, image: Image.Image, target_names: Optional[Sequence[str]] = None
                ) -> Tuple[List[FoodElement], Dict[str, Any]]:
        model_img, (sx, sy) = prepare_image(image)
        target_clause = (prompts.TARGET_CLAUSE.format(names=", ".join(f'"{n}"' for n in target_names))
                         if target_names else "")
        prompt = prompts.ELEMENT_ANALYSIS.format(target_clause=target_clause)
        raw = self.vlm.chat(user_message(prompt, [model_img]), self.max_new_tokens)
        parsed = extract_json(raw)
        items = parsed.get("elements", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])

        w, h = image.size
        elements = [e for e in (self._parse_element(it, sx, sy, w, h) for it in items if isinstance(it, dict)) if e]
        elements = self._merge_duplicates(elements)

        # referring task: guarantee every requested item exists as an element
        if target_names:
            for name in target_names:
                idx = match_name(name, [e.name for e in elements], cutoff=0.75)
                if idx is None:
                    elements.append(FoodElement(name=name, description="requested by user; not localized",
                                                importance="main", visually_clear=False,
                                                route_notes=["added: requested item missing from analysis"]))
                else:
                    elements[idx].name = name
        return elements, {"analysis_raw": raw, "model_image_size": list(model_img.size)}

    @staticmethod
    def _parse_element(it: Dict[str, Any], sx: float, sy: float, w: int, h: int) -> Optional[FoodElement]:
        name = str(it.get("name", "")).strip()
        if not name:
            return None
        bbox = None
        box = it.get("bbox_2d") or it.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, x2, y2 = (to_float(v) for v in box)
            x1, x2 = sorted((clamp(x1 * sx, 0, w), clamp(x2 * sx, 0, w)))
            y1, y2 = sorted((clamp(y1 * sy, 0, h), clamp(y2 * sy, 0, h)))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                bbox = (x1, y1, x2, y2)
        p = it.get("per_100g") or {}
        per_100g = Nutrients(calories=to_float(p.get("calories")), fat=to_float(p.get("fat")),
                             carb=to_float(p.get("carb", p.get("carbohydrates"))), protein=to_float(p.get("protein")),
                             mass=100.0)
        importance = "minor" if str(it.get("importance", "main")).lower().startswith("minor") else "main"
        return FoodElement(
            name=name,
            description=str(it.get("description", "")),
            bbox=bbox,
            importance=importance,
            visually_clear=_as_bool(it.get("visually_clear", False)),
            density_g_per_cm3=clamp(to_float(it.get("density_g_per_cm3"), 1.0) or 1.0, 0.05, 3.0),
            estimated_mass_g=clamp(to_float(it.get("estimated_mass_g")), 0.0, 3000.0),
            per_100g=per_100g,
        )

    @staticmethod
    def _merge_duplicates(elements: List[FoodElement]) -> List[FoodElement]:
        """Same-name elements (e.g. two bacon strips) are merged: union box, summed mass."""
        merged: Dict[str, FoodElement] = {}
        for e in elements:
            key = normalize_name(e.name)
            if key not in merged:
                merged[key] = e
                continue
            m = merged[key]
            m.estimated_mass_g += e.estimated_mass_g
            if m.bbox and e.bbox:
                m.bbox = (min(m.bbox[0], e.bbox[0]), min(m.bbox[1], e.bbox[1]),
                          max(m.bbox[2], e.bbox[2]), max(m.bbox[3], e.bbox[3]))
            m.visually_clear = m.visually_clear and e.visually_clear
            m.route_notes.append("merged duplicate element")
        return list(merged.values())

    # ---------------------------------------------------------------- 2) expert assignment
    @staticmethod
    def classify(e: FoodElement) -> int:
        if e.importance == "minor":
            return CATEGORY_MINOR
        if e.visually_clear and e.bbox is not None:
            return CATEGORY_MAIN_CLEAR
        return CATEGORY_MAIN_UNCLEAR

    def route(self, elements: List[FoodElement], available: Dict[str, bool],
              fallback: Dict[str, str]) -> List[FoodElement]:
        """Assign category and expert route. Unavailable experts (ablations) fall back per `fallback`."""
        for e in elements:
            e.category = self.classify(e)
            route = DEFAULT_ROUTE[e.category]
            seen = set()
            while not available.get(route, True) and route not in seen:
                seen.add(route)
                e.route_notes.append(f"expert '{route}' disabled -> '{fallback[route]}'")
                route = fallback[route]
            e.route = route
        return elements
