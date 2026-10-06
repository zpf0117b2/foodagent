"""Evaluate a trained MFP3D checkpoint on an H5 split (MAE / MAPE, as in MFP3D Tables 1-5).

    python scripts/mfp3d_test.py --checkpoint outputs/mfp3d/depth_metric/volume/best.pt \
        --h5 data/metafood3d/depth_metric/test.h5 --save_csv preds.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodagent.mfp3d.data import H5Arrays, MFP3DTorchDataset  # noqa: E402
from foodagent.mfp3d.predictor import MFP3DPredictor  # noqa: E402
from mfp3d_train import evaluate  # noqa: E402


def main(argv=None):
    from torch.utils.data import DataLoader

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--save_csv", default=None)
    a = p.parse_args(argv)
    pred = MFP3DPredictor.load(a.checkpoint, a.device)
    arr = H5Arrays(a.h5, pred.target)
    metrics, yhat, y = evaluate(pred.model, DataLoader(MFP3DTorchDataset(arr), batch_size=a.batch_size), pred.device)
    print(json.dumps({"target": pred.target, **metrics}, indent=2))
    if a.save_csv:
        with open(a.save_csv, "w", encoding="utf-8") as f:
            f.write("image_path,object_id,gt,pred\n")
            for path, oid, g, q in zip(arr.image_paths, arr.object_ids, y, yhat):
                f.write(f"{path},{oid},{g:.4f},{q:.4f}\n")
    return metrics


if __name__ == "__main__":
    main()
