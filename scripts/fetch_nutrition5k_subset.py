"""Download only what the evaluation needs from the public Nutrition5k bucket (instead of the 181 GB tarball):
metadata CSVs, split lists, and the overhead rgb.png of the dishes in the chosen split(s).

    python scripts/fetch_nutrition5k_subset.py --out data/nutrition5k --splits depth_test
    # ~500 images, ~200 MB.  Equivalent gsutil:  gsutil -m cp -r gs://nutrition5k_dataset/nutrition5k_dataset/metadata .
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodagent.data.nutrition5k import GCS_ROOT, METADATA_FILES, SPLIT_FILES  # noqa: E402


def fetch(rel: str, out: Path, overwrite: bool = False) -> str:
    dst = out / rel
    if dst.exists() and not overwrite:
        return "cached"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(f"{GCS_ROOT}/{rel}", dst)
        return "ok"
    except Exception as exc:
        if dst.exists():
            dst.unlink()
        return f"error {exc}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/nutrition5k")
    p.add_argument("--splits", nargs="+", default=["depth_test"],
                   help="any of depth_test depth_train rgb_test rgb_train")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    out = Path(args.out)

    for rel in METADATA_FILES + SPLIT_FILES:
        print(rel, fetch(rel, out))
    ids = []
    for s in args.splits:
        ids += [l.strip() for l in (out / f"dish_ids/splits/{s}_ids.txt").read_text().splitlines() if l.strip()]
    rels = [f"imagery/realsense_overhead/{d}/rgb.png" for d in dict.fromkeys(ids)]
    with ThreadPoolExecutor(args.workers) as ex:
        status = list(ex.map(lambda r: fetch(r, out), rels))
    missing = sum(s.startswith("error") for s in status)
    print(f"{len(rels) - missing}/{len(rels)} overhead images available ({missing} missing - "
          f"these dishes have no overhead capture and are skipped by the loader)")


if __name__ == "__main__":
    main()
