"""Train MFP3D (one model per target) with L1 loss.

    python scripts/mfp3d_train.py --data_dir data/metafood3d/depth_metric --target volume
    python scripts/mfp3d_train.py --data_dir data/mfp3d_release/depth --target energy   # authors' preprocessed H5
Ablations (MFP3D Table 5): --no_image (point cloud only), --point_encoder none (image only).
Defaults follow the released configuration: AdamW, lr 1e-3, batch 16, 200 epochs, 1,024 points, seed 42.
Model selection uses a validation split carved from train (by object), not the test set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodagent.mfp3d.data import H5Arrays, MFP3DTorchDataset  # noqa: E402
from foodagent.mfp3d.model import MFP3D, MFP3DConfig, TARGETS  # noqa: E402
from foodagent.mfp3d.predictor import save_checkpoint  # noqa: E402
from foodagent.mfp3d.reconstruct import ReconConfig  # noqa: E402


def mae_mape(pred: np.ndarray, gt: np.ndarray) -> dict:
    err = np.abs(pred - gt)
    return {"MAE": float(err.mean()), "MAPE": float(100.0 * np.mean(err / np.maximum(gt, 1e-6))), "n": int(len(gt))}


def evaluate(model, loader, device) -> "tuple[dict, np.ndarray, np.ndarray]":
    import torch

    model.eval()
    preds, gts = [], []
    with torch.inference_mode():
        for pts, img, y in loader:
            out = model(pts.to(device), img.to(device) if model.image_encoder is not None else None)
            preds.append(out.clamp_min(0).cpu().numpy())
            gts.append(y.numpy())
    p, g = np.concatenate(preds), np.concatenate(gts)
    return mae_mape(p, g), p, g


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="folder with train.h5 and test.h5")
    p.add_argument("--target", default="volume", choices=TARGETS)
    p.add_argument("--point_encoder", default="curvenet", choices=["curvenet", "pointnet", "none"])
    p.add_argument("--no_image", action="store_true")
    p.add_argument("--curve_setting", default="default", choices=["default", "long", "none"])
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--cosine", action="store_true", help="cosine LR decay (off = constant LR)")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--up_axis", type=int, default=2, help="rotation axis for augmentation (2 = depth axis)")
    p.add_argument("--scale_aug", type=float, default=0.0, help="isotropic scale aug range, target scaled by s^3")
    p.add_argument("--no_pretrained", action="store_true", help="random-init ResNet50 instead of ImageNet weights")
    p.add_argument("--recon_mode", default=None, choices=[None, "metric", "uvz"],
                   help="reconstruction convention of the H5 clouds if the file does not record it")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out_dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    import torch
    from torch.utils.data import DataLoader

    a = parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(a.data_dir)
    out = Path(a.out_dir or f"outputs/mfp3d/{data_dir.name}/{a.target}")
    out.mkdir(parents=True, exist_ok=True)

    train_arr = H5Arrays(str(data_dir / "train.h5"), a.target)
    test_arr = H5Arrays(str(data_dir / "test.h5"), a.target) if (data_dir / "test.h5").exists() else None
    recon = ReconConfig.from_dict(train_arr.recon) if train_arr.recon else ReconConfig(mode=a.recon_mode or "uvz")
    recon.num_points = int(train_arr.points.shape[1])
    tr_idx, va_idx = train_arr.split_by_object(a.val_fraction, a.seed)

    cfg = MFP3DConfig(point_encoder=a.point_encoder, use_image=not a.no_image, k=a.k, curve_setting=a.curve_setting,
                      num_points=recon.num_points, image_pretrained=not a.no_pretrained, target=a.target)
    model = MFP3D(cfg).to(device)
    model.init_output_bias(float(np.mean(train_arr.targets[tr_idx])))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs) if a.cosine else None
    loss_fn = torch.nn.L1Loss()

    train_loader = DataLoader(MFP3DTorchDataset(train_arr, tr_idx, augment=True, up_axis=a.up_axis,
                                                scale_range=a.scale_aug, seed=a.seed),
                              batch_size=a.batch_size, shuffle=True, drop_last=len(tr_idx) > a.batch_size,
                              num_workers=a.num_workers)
    val_loader = DataLoader(MFP3DTorchDataset(train_arr, va_idx), batch_size=a.batch_size) if len(va_idx) else None

    best, log = float("inf"), []
    ckpt = out / "best.pt"
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for pts, img, y in train_loader:
            pts, y = pts.to(device), y.to(device)
            img = img.to(device) if cfg.use_image else None
            loss = loss_fn(model(pts, img), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(y), n + len(y)
        if sched:
            sched.step()
        rec = {"epoch": epoch, "train_l1": total / max(n, 1), "sec": round(time.time() - t0, 1)}
        if val_loader is not None:
            rec["val"] = evaluate(model, val_loader, device)[0]
            score = rec["val"]["MAE"]
        else:
            score = rec["train_l1"]
        if score < best:
            best = score
            save_checkpoint(str(ckpt), model, recon, {"epoch": epoch, "val": rec.get("val"), "args": vars(a)})
        log.append(rec)
        print(json.dumps(rec))
    (out / "train_log.json").write_text(json.dumps(log, indent=2))

    if test_arr is not None:
        from foodagent.mfp3d.predictor import MFP3DPredictor

        model = MFP3DPredictor.load(str(ckpt), device).model
        metrics, pred, gt = evaluate(model, DataLoader(MFP3DTorchDataset(test_arr), batch_size=a.batch_size), device)
        (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        np.savetxt(out / "test_predictions.csv", np.c_[gt, pred], delimiter=",", header="gt,pred", comments="")
        unit = {"volume": "ml", "energy": "kcal"}.get(a.target, "g")
        print(f"TEST {a.target}: MAE {metrics['MAE']:.2f} {unit}, MAPE {metrics['MAPE']:.2f}% (n={metrics['n']})")
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
