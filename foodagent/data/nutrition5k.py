"""Nutrition5k loader (https://github.com/google-research-datasets/Nutrition5k).

Expected layout under `root` (mirrors gs://nutrition5k_dataset/nutrition5k_dataset/):
    metadata/dish_metadata_cafe1.csv, metadata/dish_metadata_cafe2.csv
    dish_ids/splits/{rgb,depth}_{train,test}_ids.txt
    imagery/realsense_overhead/<dish_id>/rgb.png

Dish metadata rows have no header:
    dish_id, total_calories, total_mass, total_fat, total_carb, total_protein,
    [num_ingrs,]  (ingr_id, ingr_name, grams, calories, fat, carb, protein) * n
(The README lists `num_ingrs`, but the released CSVs omit it; both forms are accepted.)

Which split to use: FoodAgent takes a single overhead RGB image. Only dishes in the `depth_*` splits are guaranteed
to have the overhead RealSense capture (`rgb.png`); the larger `rgb_*` splits also contain dishes that only have
side-angle videos. We therefore evaluate on `depth_test_ids.txt` using only its RGB image (no depth).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from PIL import Image

from ..types import Nutrients

GCS_ROOT = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"
METADATA_FILES = ("metadata/dish_metadata_cafe1.csv", "metadata/dish_metadata_cafe2.csv",
                  "metadata/ingredients_metadata.csv")
SPLIT_FILES = tuple(f"dish_ids/splits/{m}_{s}_ids.txt" for m in ("rgb", "depth") for s in ("train", "test"))


@dataclass
class Ingredient:
    id: str
    name: str
    grams: float
    calories: float
    fat: float
    carb: float
    protein: float

    @property
    def nutrients(self) -> Nutrients:
        return Nutrients(self.calories, self.grams, self.fat, self.carb, self.protein)


@dataclass
class Dish:
    dish_id: str
    total: Nutrients
    ingredients: List[Ingredient] = field(default_factory=list)
    image_path: Optional[Path] = None

    def load_image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")

    def top_caloric_ingredients(self, k: int = 2) -> List[Ingredient]:
        """Referring task targets: the 1st, 2nd, ... highest-calorie ingredients of the dish.

        Ingredients sharing a name are merged first so the target is unambiguous.
        """
        merged: Dict[str, Ingredient] = {}
        for ing in self.ingredients:
            key = ing.name.lower().strip()
            if key in merged:
                m = merged[key]
                merged[key] = Ingredient(m.id, m.name, m.grams + ing.grams, m.calories + ing.calories,
                                         m.fat + ing.fat, m.carb + ing.carb, m.protein + ing.protein)
            else:
                merged[key] = ing
        return sorted(merged.values(), key=lambda i: i.calories, reverse=True)[:k]


def parse_dish_row(row: List[str]) -> Dish:
    row = [c.strip() for c in row if c is not None]
    while row and row[-1] == "":
        row.pop()
    dish_id = row[0]
    total = Nutrients(*(float(v) for v in row[1:6]))  # calories, mass, fat, carb, protein
    start = 6
    if len(row) > 6 and not row[6].startswith("ingr_"):  # optional num_ingrs column
        start = 7
    ingredients = []
    for j in range(start, len(row) - 6, 7):
        iid, name, *vals = row[j:j + 7]
        g, cal, fat, carb, prot = (float(v) for v in vals)
        ingredients.append(Ingredient(iid, name, g, cal, fat, carb, prot))
    return Dish(dish_id, total, ingredients)


class Nutrition5k:
    def __init__(self, root: str):
        self.root = Path(root)
        self.dishes: Dict[str, Dish] = {}
        for name in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
            path = self.root / "metadata" / name
            if not path.exists():
                continue
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if row and row[0].startswith("dish_"):
                        d = parse_dish_row(row)
                        self.dishes[d.dish_id] = d
        if not self.dishes:
            raise FileNotFoundError(f"no dish metadata found under {self.root / 'metadata'}")

    def image_path(self, dish_id: str) -> Path:
        return self.root / "imagery" / "realsense_overhead" / dish_id / "rgb.png"

    def split_ids(self, split: str = "test", modality: str = "depth") -> List[str]:
        path = self.root / "dish_ids" / "splits" / f"{modality}_{split}_ids.txt"
        return [l.strip() for l in path.read_text().splitlines() if l.strip()]

    def iter_split(self, split: str = "test", modality: str = "depth", require_image: bool = True,
                   limit: Optional[int] = None) -> Iterator[Dish]:
        n = 0
        for dish_id in self.split_ids(split, modality):
            dish = self.dishes.get(dish_id)
            if dish is None:
                continue
            p = self.image_path(dish_id)
            if require_image and not p.exists():
                continue
            dish.image_path = p
            yield dish
            n += 1
            if limit and n >= limit:
                break
