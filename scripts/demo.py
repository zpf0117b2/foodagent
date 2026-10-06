"""Run FoodAgent on one image.

    python scripts/demo.py path/to/dish.png --local_db_csv data/nutrition_db.csv --mfp3d_ckpt best.pt
    python scripts/demo.py dish.png --refer "scrambled eggs" "bacon"
  (any option of scripts/run_eval.py can be appended)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from run_eval import build_method, parse_args  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--refer", nargs="*", default=None)
    p.add_argument("--json", action="store_true", help="print the full trace")
    args, rest = p.parse_known_args()
    eval_args = parse_args(["--data_root", "-"] + rest)
    est = build_method(eval_args).run(Image.open(args.image), args.refer)
    print(est.response)
    for e in est.elements:
        print(f"  [{e.element.category}->{e.element.route}] {e.element.name}: "
              f"{e.nutrients.calories:.0f} kcal, {e.nutrients.mass:.0f} g")
    if args.json:
        print(json.dumps(est.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
