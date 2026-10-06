"""Evaluate FoodAgent / the Qwen2.5-VL direct baseline on Nutrition5k.

Tasks
  total : Nutrition Estimation - calories / mass / fat / carb / protein of the whole dish   (Table I)
  refer : Referring Nutrition Estimation - nutrition of the 1st and 2nd highest-calorie
          ingredients, named in the query                                                     (Table II)

Examples
  # Table I, FoodAgent (local Qwen2.5-VL, offline nutrition table, trained MFP3D)
  python scripts/run_eval.py --data_root data/nutrition5k --task total --method foodagent \
      --local_db_csv data/nutrition_db.csv --mfp3d_ckpt outputs/mfp3d/depth_metric/volume/best.pt
  # ablations
  ... --ablation no_mfp3d       /       ... --ablation no_nutritionx
  # Qwen2.5-VL holistic baseline
  ... --method qwen_direct
Results (resumable): <out_dir>/predictions.jsonl, metrics.json and official-format CSVs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodagent.data.nutrition5k import Nutrition5k  # noqa: E402
from foodagent.eval.metrics import format_table, mae_pmae, write_official_csv  # noqa: E402
from foodagent.types import Nutrients  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--modality", default="depth", choices=["depth", "rgb"],
                   help="split list to use; 'depth' lists are the dishes that have the overhead rgb.png")
    p.add_argument("--task", default="total", choices=["total", "refer"])
    p.add_argument("--method", default="foodagent", choices=["foodagent", "qwen_direct"])
    p.add_argument("--ablation", default="none", choices=["none", "no_mfp3d", "no_nutritionx"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out_dir", default=None)
    # VLM
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--quantization", default=None, choices=[None, "4bit", "8bit"])
    p.add_argument("--attn_implementation", default=None)
    p.add_argument("--cache_dir", default=".cache")
    # experts
    p.add_argument("--nutrition_db", default="local", choices=["local", "none"])
    p.add_argument("--local_db_csv", default="data/nutrition_db.csv", help="built by scripts/build_local_db.py")
    p.add_argument("--rag_top_k", type=int, default=5)
    p.add_argument("--rag_images", action="store_true", help="show retrieved reference photos to the VLM")
    p.add_argument("--sam_model", default="facebook/sam-vit-huge")
    p.add_argument("--depth_model", default="Intel/zoedepth-nyu-kitti")
    p.add_argument("--mfp3d_ckpt", default=None,
                   help="trained MFP3D checkpoint (scripts/mfp3d_train.py); without it a geometric estimate is used")
    p.add_argument("--no_depth_anchor", action="store_true", help="use raw ZoeDepth scale (no rig-distance anchor)")
    p.add_argument("--expert_device", default=None)
    p.add_argument("--volume_fallback", default="llm", choices=["llm", "rag"])
    return p.parse_args(argv)


def build_method(args):
    from foodagent.vlm import build_vlm

    vlm = build_vlm(args.model, cache_dir=str(Path(args.cache_dir) / "vlm"), quantization=args.quantization,
                    attn_implementation=args.attn_implementation)
    if args.method == "qwen_direct":
        from foodagent.pipeline import QwenDirectBaseline

        return QwenDirectBaseline(vlm)

    from foodagent.experts.nutrition_db import build_nutrition_db
    from foodagent.experts.rag import RAGExpert
    from foodagent.experts.volume import MFP3DVolumeEstimator
    from foodagent.pipeline import FoodAgent, FoodAgentConfig

    cfg = FoodAgentConfig(use_volume_expert=args.ablation != "no_mfp3d",
                          use_rag_expert=args.ablation != "no_nutritionx" and args.nutrition_db != "none",
                          volume_fallback=args.volume_fallback)
    volume = None
    if cfg.use_volume_expert:
        predictor = None
        if args.mfp3d_ckpt:
            from foodagent.mfp3d.predictor import MFP3DPredictor

            predictor = MFP3DPredictor.load(args.mfp3d_ckpt, args.expert_device)
        volume = MFP3DVolumeEstimator(args.sam_model, args.depth_model, device=args.expert_device,
                                      anchor_depth_cm=None if args.no_depth_anchor else 35.9,
                                      mfp3d=predictor)
    rag = None
    if cfg.use_rag_expert:
        rag = RAGExpert(build_nutrition_db(args.nutrition_db, args.local_db_csv), args.rag_top_k, args.rag_images)
    return FoodAgent(vlm, volume, rag, cfg)


def default_out_dir(args) -> Path:
    tag = args.method if args.method == "qwen_direct" else f"foodagent_{args.ablation}"
    return Path("results") / f"{args.task}_{tag}_{args.modality}_{args.split}"


def load_done(path: Path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["dish_id"]] = rec
    return done


def compute_metrics(records, task: str):
    if task == "total":
        preds = [Nutrients.from_dict(r["pred"]) for r in records]
        gts = [Nutrients.from_dict(r["gt"]) for r in records]
        return {"total": mae_pmae(preds, gts)}
    out = {}
    for rank in (0, 1):
        pairs = [(Nutrients.from_dict(r["pred_targets"][rank]), Nutrients.from_dict(r["gt_targets"][rank]))
                 for r in records if len(r["gt_targets"]) > rank]
        if pairs:
            out[f"refer_{rank + 1}"] = mae_pmae([p for p, _ in pairs], [g for _, g in pairs])
    return out


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    pred_path = out_dir / "predictions.jsonl"
    done = load_done(pred_path)

    ds = Nutrition5k(args.data_root)
    dishes = list(ds.iter_split(args.split, args.modality, limit=args.limit))
    print(f"{len(dishes)} dishes ({args.modality}_{args.split}); {len(done)} already done -> {out_dir}")
    method = None
    failures = []
    with open(pred_path, "a", encoding="utf-8") as fout:
        for k, dish in enumerate(dishes):
            if dish.dish_id in done:
                continue
            method = method or build_method(args)
            targets = dish.top_caloric_ingredients(2) if args.task == "refer" else []
            names = [t.name for t in targets] or None
            t0 = time.time()
            try:
                est = method.run(dish.load_image(), names)
            except Exception as exc:  # keep going; failed dishes are retried on the next run
                failures.append(dish.dish_id)
                print(f"[{k + 1}/{len(dishes)}] {dish.dish_id} FAILED: {exc!r}")
                traceback.print_exc()
                continue
            rec = {"dish_id": dish.dish_id, "pred": est.total.to_dict(), "gt": dish.total.to_dict(),
                   "seconds": round(time.time() - t0, 2), "estimate": est.to_dict()}
            if args.task == "refer":
                tgt = est.trace.get("targets", {})
                rec["target_names"] = names
                rec["pred_targets"] = [tgt.get(n, Nutrients().to_dict()) for n in names]
                rec["gt_targets"] = [t.nutrients.to_dict() for t in targets]
            done[dish.dish_id] = rec
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"[{k + 1}/{len(dishes)}] {dish.dish_id} pred {est.total.calories:.0f} kcal "
                  f"gt {dish.total.calories:.0f} kcal ({rec['seconds']}s)")

    ids = [d.dish_id for d in dishes if d.dish_id in done]
    records = [done[i] for i in ids]
    if not records:
        print("no predictions to score")
        return
    metrics = compute_metrics(records, args.task)
    metrics["failed_dishes"] = failures
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if args.task == "total":
        write_official_csv(str(out_dir / "predictions.csv"), [(r["dish_id"], Nutrients.from_dict(r["pred"])) for r in records])
        write_official_csv(str(out_dir / "groundtruth.csv"), [(r["dish_id"], Nutrients.from_dict(r["gt"])) for r in records])
    for name, m in metrics.items():
        if isinstance(m, dict):
            print(f"\n== {name} (n={m['n']}) ==\n{format_table(m)}")
    if failures:
        print(f"\n{len(failures)} dishes failed and were excluded (re-run to retry).")


if __name__ == "__main__":
    main()
