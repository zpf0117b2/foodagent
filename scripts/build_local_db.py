"""Build the offline nutrition table used by the RAG expert (no API, no network at run time).

  --from_fdc DIR             USDA FoodData Central CSV release (public domain), e.g. the "SR Legacy" and/or
                             "FNDDS" CSV downloads from https://fdc.nal.usda.gov/download-datasets/ unpacked into DIR
                             (needs food.csv, nutrient.csv, food_nutrient.csv, food_portion.csv, measure_unit.csv).
                             Every food yields a "100 g" reference plus one reference per household portion
                             (e.g. "1 cup, chopped = 160 g") - the unit + per-unit profile the paper's RAG expert uses.
  --from_n5k_ingredients CSV Nutrition5k's ingredients_metadata.csv as "100 g" references. These are the per-gram
                             tables the Nutrition5k labels were computed from (privileged) - report such runs separately.

    python scripts/build_local_db.py --from_fdc data/fdc/FoodData_Central_sr_legacy_food_csv --out data/nutrition_db.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterator, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodagent.experts.nutrition_db import LocalNutritionDB  # noqa: E402
from foodagent.types import Nutrients  # noqa: E402

Row = Tuple[str, float, str, float, Nutrients, str]
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


def _find(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def _fdc_nutrient_ids(path: Path) -> Dict[str, set]:
    """Map our fields to FDC nutrient ids (resolved by name so any FDC release works)."""
    ids: Dict[str, set] = {"calories": set(), "protein": set(), "fat": set(), "carb": set()}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name, unit = r["name"].strip().lower(), r.get("unit_name", "").strip().upper()
            if name.startswith("energy") and unit == "KCAL":
                ids["calories"].add(r["id"])
            elif name == "protein":
                ids["protein"].add(r["id"])
            elif name == "total lipid (fat)":
                ids["fat"].add(r["id"])
            elif name.startswith("carbohydrate, by difference"):
                ids["carb"].add(r["id"])
    return ids


def from_fdc(root_dir: str, data_types: Tuple[str, ...]) -> Iterator[Row]:
    root = Path(root_dir)
    foods: Dict[str, str] = {}
    with open(_find(root, "food.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not data_types or r.get("data_type", "") in data_types:
                foods[r["fdc_id"]] = r["description"].strip().lower()
    nid = _fdc_nutrient_ids(_find(root, "nutrient.csv"))
    field_of = {i: k for k, s in nid.items() for i in s}
    per100: Dict[str, Dict[str, float]] = {}
    with open(_find(root, "food_nutrient.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = field_of.get(r["nutrient_id"])
            if k and r["fdc_id"] in foods and r.get("amount"):
                per100.setdefault(r["fdc_id"], {}).setdefault(k, float(r["amount"]))
    units = {}
    with open(_find(root, "measure_unit.csv"), newline="", encoding="utf-8") as f:
        units = {r["id"]: r["name"].strip() for r in csv.DictReader(f)}

    def profile(fid: str, grams: float) -> Nutrients:
        p = per100[fid]
        k = grams / 100.0
        return Nutrients(p.get("calories", 0.0) * k, grams, p.get("fat", 0.0) * k, p.get("carb", 0.0) * k,
                         p.get("protein", 0.0) * k)

    complete = {fid for fid, p in per100.items() if "calories" in p}
    for fid in sorted(complete, key=lambda x: foods[x]):
        yield foods[fid], 100.0, "g", 100.0, profile(fid, 100.0), ""
    with open(_find(root, "food_portion.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fid = r["fdc_id"]
            if fid not in complete or not r.get("gram_weight"):
                continue
            grams = float(r["gram_weight"])
            desc = (r.get("portion_description") or "").strip()
            if not desc or desc.lower().startswith("quantity not specified"):
                unit = units.get(r.get("measure_unit_id", ""), "")
                unit = "" if unit.lower() in ("undetermined", "") else unit
                desc = ", ".join(x for x in (unit, (r.get("modifier") or "").strip()) if x) or "serving"
            qty = float(r["amount"]) if r.get("amount") else 1.0
            if grams > 0:
                yield foods[fid], qty, desc, grams, profile(fid, grams), ""


def from_n5k_ingredients(path: str) -> Iterator[Row]:
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = 100.0
            yield (row["ingr"], 100.0, "g", 100.0,
                   Nutrients(float(row["cal/g"]) * k, 100.0, float(row["fat(g)"]) * k,
                             float(row["carb(g)"]) * k, float(row["protein(g)"]) * k), "")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--from_fdc")
    g.add_argument("--from_n5k_ingredients")
    p.add_argument("--fdc_data_types", nargs="*", default=["sr_legacy_food", "survey_fndds_food", "foundation_food"])
    args = p.parse_args()
    rows = (from_fdc(args.from_fdc, tuple(args.fdc_data_types)) if args.from_fdc
            else from_n5k_ingredients(args.from_n5k_ingredients))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LocalNutritionDB.FIELDS)
        for name, qty, unit, weight, nut, photo in rows:
            w.writerow([name, f"{qty:g}", unit, f"{weight:.2f}", f"{nut.calories:.3f}", f"{nut.fat:.3f}",
                        f"{nut.carb:.3f}", f"{nut.protein:.3f}", photo])
            n += 1
    print(f"wrote {n} references to {args.out}")


if __name__ == "__main__":
    main()
