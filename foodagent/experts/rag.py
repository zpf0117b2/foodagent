"""Category-2 expert: retrieval-augmented estimation against the offline nutrition database.

Retrieval returns, for every element, reference servings (unit + per-unit nutrition, optionally a local reference
photo). The response generator then asks Qwen2.5-VL how many reference units are present in the dish
(see generator.ResponseGenerator.estimate_rag_units).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PIL import Image

from ..types import FoodElement, FoodReference
from .nutrition_db import NutritionDB


class RAGExpert:
    def __init__(self, db: NutritionDB, top_k: int = 5, use_reference_images: bool = False):
        self.db = db
        self.top_k = top_k
        self.use_reference_images = use_reference_images

    def retrieve(self, element: FoodElement) -> List[FoodReference]:
        refs = self.db.search(element.name, self.top_k)
        return [r for r in refs if r.serving_weight_g > 0 or r.nutrients.calories > 0]

    def reference_image(self, ref: FoodReference) -> Optional[Image.Image]:
        """Local reference photo of a retrieved entry, if the table provides one."""
        if not (self.use_reference_images and ref.photo):
            return None
        try:
            return Image.open(ref.photo).convert("RGB")
        except OSError:
            return None

    def retrieve_all(self, elements: List[FoodElement]) -> Dict[int, List[FoodReference]]:
        return {i: self.retrieve(e) for i, e in enumerate(elements)}
