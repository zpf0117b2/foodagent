"""Build MFP3D train/test H5 files from raw MetaFood3D.

Step 0 - look at the extracted archives and pick glob patterns:
    python scripts/mfp3d_build_dataset.py --inspect /data/MetaFood3D
Step 1 - manifest (one row per object):
    python scripts/mfp3d_build_dataset.py manifest --objects_glob "/data/MetaFood3D/RGBD_videos/*/*" \
        --frame_pattern "**/rgb/*.jpg" --mask_pattern "**/mask*/*.png" --depth_pattern "**/depth/*.png" \
        --mesh_pattern "**/*.obj" --nutrition_csv /data/MetaFood3D/nutrition.csv --id_col id \
        --out data/metafood3d/manifest.csv
  (or write the manifest CSV yourself - columns in foodagent/mfp3d/metafood3d.py)
Step 2 - reconstruct point clouds + crops, split 510/127 by object, write H5:
    python scripts/mfp3d_build_dataset.py build --manifest data/metafood3d/manifest.csv \
        --out_dir data/metafood3d/depth_metric --cloud depth --mode metric --anchor sensor
  --cloud depth          ZoeDepth point cloud (MFP3D "Depth Point Clouds", default)
  --cloud gt             surface samples of the scanned mesh (MFP3D "GTPC" upper bound)
  --cloud gt_normalized  GTPC rescaled to the unit cube (MFP3D "Normalized GTPC")
  --cloud provided       the manifest's `pointcloud` file (MetaFood3D point clouds, TripoSR samples, ...)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodagent.geometry import CameraIntrinsics, anchor_depth  # noqa: E402
from foodagent.mfp3d import metafood3d as mf  # noqa: E402
from foodagent.mfp3d.data import write_h5  # noqa: E402
from foodagent.mfp3d.reconstruct import ReconConfig, depth_point_cloud, masked_crop, normalize_unit_cube, sample_points  # noqa: E402,E501


def load_mask(path: str, shape) -> np.ndarray:
    from PIL import Image

    m = np.asarray(Image.open(path).convert("L").resize((shape[1], shape[0]), Image.NEAREST)) > 127
    return m


def cmd_manifest(a):
    nutrition = None
    if a.nutrition_csv:
        colmap = dict(kv.split("=") for kv in a.colmap) if a.colmap else {k: k for k in mf.NUTRITION_TARGETS}
        nutrition = mf.read_nutrition_table(a.nutrition_csv, a.id_col, colmap)
    rows = mf.discover_manifest(a.objects_glob, a.frame_pattern, a.mask_pattern, a.depth_pattern, a.mesh_pattern,
                                a.pointcloud_pattern, nutrition, a.frames_per_object, a.prefer_last)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    mf.write_manifest(a.out, rows)
    n_mask = sum(all(r.masks) for r in rows)
    n_lab = sum("energy" in r.targets or "weight" in r.targets for r in rows)
    print(f"{len(rows)} objects -> {a.out}  (all masks found: {n_mask}, with nutrition labels: {n_lab})")


def cmd_build(a):
    from PIL import Image

    recon = ReconConfig(mode=a.mode, num_points=a.num_points, coord_scale=a.coord_scale,
                        image_size=a.image_size, crop_to_mask=not a.no_crop)
    rows = mf.read_manifest(a.manifest)
    train_ids, test_ids = mf.split_objects([r.object_id for r in rows], a.n_test, a.seed)
    rng = np.random.default_rng(a.seed)
    vision = None
    if a.cloud == "depth" or any(not m for r in rows for m in r.masks):
        from foodagent.experts.volume import MFP3DVolumeEstimator

        vision = MFP3DVolumeEstimator(a.sam_model, a.depth_model, device=a.device)

    buckets = {"train": [], "test": []}
    for r in rows:
        targets = dict(r.targets)
        verts = faces = None
        if r.mesh and (a.cloud in ("gt", "gt_normalized") or "volume" not in targets):
            verts, faces = mf.load_mesh(r.mesh)
            verts = verts * a.mesh_scale_cm
            targets.setdefault("volume", mf.mesh_volume(verts, faces))  # cm^3 == ml
        obj_cloud = None
        if a.cloud in ("gt", "gt_normalized"):
            if verts is None:
                print(f"skip {r.object_id}: no mesh for GT point cloud")
                continue
            obj_cloud = mf.sample_surface(verts, faces, recon.num_points, rng) / recon.coord_scale
            obj_cloud = obj_cloud - obj_cloud.mean(0, keepdims=True)
            if a.cloud == "gt_normalized":
                obj_cloud = normalize_unit_cube(obj_cloud)
        elif a.cloud == "provided":
            if not r.pointcloud:
                print(f"skip {r.object_id}: no point cloud file")
                continue
            obj_cloud = sample_points(mf.load_point_file(r.pointcloud), recon.num_points, rng)
            obj_cloud = obj_cloud - obj_cloud.mean(0, keepdims=True)

        for fi, frame in enumerate(r.frames):
            image = Image.open(frame).convert("RGB")
            h, w = image.height, image.width
            if r.masks[fi]:
                mask = load_mask(r.masks[fi], (h, w))
            else:  # no released mask: SAM with an almost full-frame box
                mask = vision.segment(image, [(0.02 * w, 0.02 * h, 0.98 * w, 0.98 * h)])[0]
            if mask.sum() < 50:
                continue
            cloud = obj_cloud
            if cloud is None:
                depth = vision.predict_depth_cm(image)
                if a.anchor == "sensor" and r.depths[fi]:
                    sensor = np.asarray(Image.open(r.depths[fi]), np.float64) * a.sensor_depth_to_cm
                    if sensor.shape != depth.shape:
                        sensor = np.asarray(Image.fromarray(sensor.astype(np.float32)).resize((w, h), Image.NEAREST))
                    bg = (~mask) & (sensor > 0)
                    target_cm = float(np.median(sensor[bg])) if bg.any() else None
                    depth, _ = anchor_depth(depth, ~mask, target_cm)
                intr = (CameraIntrinsics(fx=a.fx, fy=a.fx, ref_width=w) if a.fx
                        else CameraIntrinsics.from_fov(w, h, a.hfov_deg)).for_size(w, h)
                cloud = depth_point_cloud(depth, mask, recon, intr, rng)
            split = "train" if r.object_id in train_ids else "test"
            buckets[split].append((cloud, masked_crop(image, mask, recon), targets, frame, r.object_id))
        print(f"{r.object_id}: {len(r.frames)} frames, targets {sorted(targets)}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"recon": recon.to_dict(), "cloud": a.cloud, "anchor": a.anchor, "hfov_deg": a.hfov_deg, "fx": a.fx,
            "mesh_scale_cm": a.mesh_scale_cm, "n_test_objects": len(test_ids), "seed": a.seed}
    for split, items in buckets.items():
        if not items:
            continue
        names = sorted({k for it in items for k in it[2]})
        targets = {k: np.array([it[2].get(k, np.nan) for it in items], np.float32) for k in names}
        write_h5(str(out / f"{split}.h5"), np.stack([it[0] for it in items]), np.stack([it[1] for it in items]),
                 targets, [it[3] for it in items], [it[4] for it in items], recon=recon.to_dict())
        print(f"{split}: {len(items)} samples from {len({it[4] for it in items})} objects")
    (out / "build_info.json").write_text(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspect", metavar="ROOT", help="print the directory structure and exit")
    sub = p.add_subparsers(dest="cmd")
    m = sub.add_parser("manifest")
    m.add_argument("--objects_glob", required=True)
    m.add_argument("--frame_pattern", required=True)
    m.add_argument("--mask_pattern", default="")
    m.add_argument("--depth_pattern", default="")
    m.add_argument("--mesh_pattern", default="")
    m.add_argument("--pointcloud_pattern", default="")
    m.add_argument("--nutrition_csv", default=None)
    m.add_argument("--id_col", default="id")
    m.add_argument("--colmap", nargs="*", help="ours=theirs, e.g. energy=calories weight=weight_g")
    m.add_argument("--frames_per_object", type=int, default=4)
    m.add_argument("--prefer_last", action="store_true", help="take the last frames (overhead end of the video)")
    m.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--manifest", required=True)
    b.add_argument("--out_dir", required=True)
    b.add_argument("--cloud", default="depth", choices=["depth", "gt", "gt_normalized", "provided"])
    b.add_argument("--mode", default="metric", choices=["metric", "uvz"])
    b.add_argument("--num_points", type=int, default=1024)
    b.add_argument("--coord_scale", type=float, default=10.0)
    b.add_argument("--image_size", type=int, default=224)
    b.add_argument("--no_crop", action="store_true", help="resize the whole masked frame instead of a mask crop")
    b.add_argument("--anchor", default="none", choices=["none", "sensor"],
                   help="sensor: rescale ZoeDepth so the background matches the RGB-D sensor median depth")
    b.add_argument("--sensor_depth_to_cm", type=float, default=0.1, help="sensor depth unit -> cm (mm: 0.1)")
    b.add_argument("--hfov_deg", type=float, default=60.0)
    b.add_argument("--fx", type=float, default=None)
    b.add_argument("--mesh_scale_cm", type=float, default=100.0, help="mesh unit -> cm (metres: 100, mm: 0.1)")
    b.add_argument("--n_test", type=int, default=127)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--sam_model", default="facebook/sam-vit-huge")
    b.add_argument("--depth_model", default="Intel/zoedepth-nyu-kitti")
    b.add_argument("--device", default=None)
    a = p.parse_args()
    if a.inspect:
        print(mf.inspect_tree(a.inspect))
    elif a.cmd == "manifest":
        cmd_manifest(a)
    elif a.cmd == "build":
        cmd_build(a)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
