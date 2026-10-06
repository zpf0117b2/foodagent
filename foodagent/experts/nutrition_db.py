"""Offline nutrition knowledge base for the RAG expert (the paper's "NutritionX" reference database).

A CSV table of reference servings, one row per (food, unit):
    name,serving_qty,serving_unit,serving_weight_g,calories,fat,carb,protein[,photo]
`photo` is an optional local image path of the reference food.

Build it once with scripts/build_local_db.py from the USDA FoodData Central CSV release (public domain, includes
household portions such as "1 cup" with gram weights). No network access or API key is needed at run time.
"""
from __future__ import annotations

import csv
import difflib
import re
from pathlib import Path
from typing import List, Optional, Protocol

from ..types import FoodReference, Nutrients


class NutritionDB(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> List[FoodReference]: ...

    def per_gram(self, query: str) -> Optional[Nutrients]: ...


def per_gram_from_reference(ref: FoodReference) -> Optional[Nutrients]:
    if ref.serving_weight_g <= 0:
        return None
    n = ref.nutrients.scale(1.0 / ref.serving_weight_g)
    n.mass = 1.0
    return n


_TOKEN = re.compile(r"[a-z]+")


def _tokens(s: str) -> set:
    return {t.rstrip("s") for t in _TOKEN.findall(s.lower())}


class LocalNutritionDB:
    FIELDS = ["name", "serving_qty", "serving_unit", "serving_weight_g", "calories", "fat", "carb", "protein", "photo"]

    def __init__(self, csv_path: str):
        self.name = f"local:{Path(csv_path).name}"
        self.refs: List[FoodReference] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                w = float(row["serving_weight_g"])
                self.refs.append(FoodReference(
                    row["name"], float(row.get("serving_qty") or 1), row.get("serving_unit") or "serving", w,
                    Nutrients(float(row["calories"]), w, float(row["fat"]), float(row["carb"]), float(row["protein"])),
                    row.get("photo") or None, self.name))
        self._tokens = [_tokens(r.name) for r in self.refs]

    def score(self, query: str, i: int) -> float:
        qt, nt = _tokens(query), self._tokens[i]
        jacc = len(qt & nt) / max(1, len(qt | nt))
        return max(jacc, difflib.SequenceMatcher(None, query.lower(), self.refs[i].name.lower()).ratio())

    def search(self, query: str, k: int = 5) -> List[FoodReference]:
        qt = _tokens(query)
        # cheap pre-filter on shared tokens keeps fuzzy matching fast on large tables
        cand = [i for i, t in enumerate(self._tokens) if t & qt] or range(len(self.refs))
        ranked = sorted(cand, key=lambda i: self.score(query, i), reverse=True)
        return [self.refs[i] for i in ranked[:k]]

    def per_gram(self, query: str) -> Optional[Nutrients]:
        hits = self.search(query, 1)
        return per_gram_from_reference(hits[0]) if hits else None


def build_nutrition_db(kind: str, local_csv: Optional[str] = None) -> Optional[NutritionDB]:
    if kind == "none":
        return None
    if kind == "local":
        if not local_csv or not Path(local_csv).exists():
            raise FileNotFoundError(f"nutrition table not found: {local_csv} (build it with scripts/build_local_db.py)")
        return LocalNutritionDB(local_csv)
    raise ValueError(f"unknown nutrition db: {kind}")
