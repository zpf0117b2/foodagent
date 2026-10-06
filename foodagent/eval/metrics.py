"""Metrics, matching Nutrition5k's scripts/compute_eval_statistics.py:

    MAE_field   = mean_i |pred_i - gt_i|
    PMAE_field  = 100 * MAE_field / mean_i gt_i       (percentage of the ground-truth mean)
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ..types import NUTRIENT_FIELDS, Nutrients


def mae_pmae(preds: Sequence[Nutrients], gts: Sequence[Nutrients],
             fields: Iterable[str] = NUTRIENT_FIELDS) -> Dict[str, float]:
    if len(preds) != len(gts) or not preds:
        raise ValueError("need equal, non-empty prediction / ground-truth lists")
    out: Dict[str, float] = {"n": len(preds)}
    for f in fields:
        errs = [abs(getattr(p, f) - getattr(g, f)) for p, g in zip(preds, gts)]
        mae = statistics.mean(errs)
        gt_mean = statistics.mean(getattr(g, f) for g in gts)
        out[f"{f}_MAE"] = mae
        out[f"{f}_PMAE"] = 100.0 * mae / gt_mean if gt_mean else float("nan")
    return out


def write_official_csv(path: str, rows: List[Tuple[str, Nutrients]]) -> None:
    """dish_id,calories,mass,fat,carb,protein (no header) - the format compute_eval_statistics.py reads."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for dish_id, n in rows:
            w.writerow([dish_id] + [f"{getattr(n, k):.6f}" for k in NUTRIENT_FIELDS])


def format_table(metrics: Dict[str, float], fields: Iterable[str] = NUTRIENT_FIELDS) -> str:
    lines = [f"{'field':<10}{'MAE':>12}{'PMAE':>10}"]
    for f in fields:
        if f"{f}_MAE" in metrics:
            lines.append(f"{f:<10}{metrics[f'{f}_MAE']:>12.2f}{metrics[f'{f}_PMAE']:>9.1f}%")
    return "\n".join(lines)
