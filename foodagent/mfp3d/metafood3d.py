"""MetaFood3D (https://lorenz.ecn.purdue.edu/~food3d/) -> MFP3D training manifests.

MetaFood3D: 637 single-food objects / 108 categories (MFP3D paper version), each with a textured 3D scan, RGB-D
video frames with segmentation masks and depth, FNDDS code, weight (g) and energy / protein / carb / fat.
Access requires an approved request form; the archives are 3D_Mesh, Blender render images, RGBD videos (+flipped),
Point cloud, Nutrition values and a Readme.

Because the internal folder naming is only documented inside the archive Readme, discovery here is driven by
glob patterns relative to each object's folder (run `scripts/mfp3d_build_dataset.py --inspect ROOT` first and
adapt the patterns). The manifest is a plain CSV, one row per object:

  object_id, category, frames, masks, depths, mesh, pointcloud, weight, volume, energy, protein, fat, carb
  (frames / masks / depths are ';'-separated, aligned lists; any of mesh / pointcloud / volume may be empty)

GT volume: if the nutrition file has no volume, it is computed from the (watertight) scanned mesh.
"""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MANIFEST_FIELDS = ["object_id", "category", "frames", "masks", "depths", "mesh", "pointcloud",
                   "weight", "volume", "energy", "protein", "fat", "carb"]
NUTRITION_TARGETS = ("weight", "volume", "energy", "protein", "fat", "carb")


@dataclass
class ManifestRow:
    object_id: str
    category: str = ""
    frames: List[str] = field(default_factory=list)
    masks: List[str] = field(default_factory=list)
    depths: List[str] = field(default_factory=list)
    mesh: str = ""
    pointcloud: str = ""
    targets: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- meshes
def load_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """(V (n,3), F (m,3)) - OBJ parsed natively; other formats via trimesh if installed."""
    if path.lower().endswith(".obj"):
        verts, faces = [], []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    verts.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("f "):
                    ids = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                    ids = [i - 1 if i > 0 else len(verts) + i for i in ids]
                    faces.extend([ids[0], ids[j], ids[j + 1]] for j in range(1, len(ids) - 1))  # fan triangulation
        return np.asarray(verts, np.float64), np.asarray(faces, np.int64)
    import trimesh  # optional dependency for .ply / .glb / .stl

    mesh = trimesh.load(path, force="mesh")
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)


def mesh_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Enclosed volume via the divergence theorem (sum of signed tetrahedra); needs a closed mesh."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def sample_surface(v: np.ndarray, f: np.ndarray, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Area-weighted uniform surface samples (GTPC)."""
    rng = rng or np.random.default_rng(0)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
    return ((1 - r1) * a[tri] + r1 * (1 - r2) * b[tri] + r1 * r2 * c[tri]).astype(np.float32)


def load_point_file(path: str) -> np.ndarray:
    """Pre-computed clouds (MetaFood3D 'Point cloud' archive, TripoSR samples, ...): .npy / .xyz / .txt / .ply."""
    p = path.lower()
    if p.endswith(".npy"):
        return np.load(path)[:, :3].astype(np.float32)
    if p.endswith((".xyz", ".txt", ".pts")):
        return np.loadtxt(path, ndmin=2)[:, :3].astype(np.float32)
    if p.endswith(".ply"):
        import trimesh

        return np.asarray(trimesh.load(path).vertices, np.float32)[:, :3]
    raise ValueError(f"unsupported point file: {path}")


# --------------------------------------------------------------------------- discovery
def inspect_tree(root: str, max_depth: int = 4, max_entries: int = 6) -> str:
    """Compact view of a directory tree (a few entries per folder) to design the glob patterns."""
    lines = []
    root_p = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root_p).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames.sort()
        indent = "  " * depth
        lines.append(f"{indent}{Path(dirpath).name}/  ({len(dirnames)} dirs, {len(filenames)} files)")
        for fn in sorted(filenames)[:max_entries]:
            lines.append(f"{indent}  {fn}")
        if len(dirnames) > max_entries:
            dirnames[:] = dirnames[:max_entries]
            lines.append(f"{indent}  ... (showing {max_entries} sub-folders)")
    return "\n".join(lines)


def _evenly(items: List[str], k: int, prefer_last: bool) -> List[str]:
    if k <= 0 or len(items) <= k:
        return items
    if prefer_last:  # MetaFood3D videos end with an overhead view
        return items[-k:]
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def read_nutrition_table(path: str, id_col: str, colmap: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """{object_id: row}; `colmap` maps our target names to the file's column names (case-insensitive)."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        lower = {c.lower().strip(): c for c in reader.fieldnames or []}
        id_key = lower.get(id_col.lower())
        if id_key is None:
            raise KeyError(f"id column '{id_col}' not in {reader.fieldnames}")
        for row in reader:
            rec = {}
            for ours, theirs in colmap.items():
                key = lower.get(theirs.lower())
                if key is not None:
                    rec[ours] = row[key]
            for extra in ("category", "food_name", "name"):
                if extra in lower and "category" not in rec:
                    rec["category"] = row[lower[extra]]
            out[str(row[id_key]).strip()] = rec
    return out


def discover_manifest(objects_glob: str, frame_pattern: str, mask_pattern: str = "", depth_pattern: str = "",
                      mesh_pattern: str = "", pointcloud_pattern: str = "", nutrition: Optional[Dict] = None,
                      frames_per_object: int = 4, prefer_last: bool = False) -> List[ManifestRow]:
    """Build manifest rows. `objects_glob` matches one folder per object (its name is the object id unless a
    nutrition table maps it); the other patterns are globs relative to that folder. Frames and masks are paired by
    file stem when possible, else by sorted order."""
    rows = []
    for obj_dir in sorted(glob.glob(objects_glob)):
        if not os.path.isdir(obj_dir):
            continue
        oid = Path(obj_dir).name

        def find(pattern: str) -> List[str]:
            return sorted(glob.glob(os.path.join(obj_dir, pattern), recursive=True)) if pattern else []

        frames = find(frame_pattern)
        if not frames:
            continue
        masks, depths = find(mask_pattern), find(depth_pattern)
        frames = _evenly(frames, frames_per_object, prefer_last)
        masks = _pair(frames, masks)
        depths = _pair(frames, depths)
        meshes, clouds = find(mesh_pattern), find(pointcloud_pattern)
        nut = (nutrition or {}).get(oid, {})
        targets = {}
        for k in NUTRITION_TARGETS:
            try:
                targets[k] = float(nut[k])
            except (KeyError, TypeError, ValueError):
                pass
        rows.append(ManifestRow(oid, nut.get("category", ""), frames, masks, depths,
                                meshes[0] if meshes else "", clouds[0] if clouds else "", targets))
    return rows


def _pair(frames: Sequence[str], others: Sequence[str]) -> List[str]:
    """Align `others` to `frames` (stem match, then numeric/order fallback); '' where missing."""
    if not others:
        return [""] * len(frames)
    by_stem = {Path(o).stem: o for o in others}
    out = []
    for i, fr in enumerate(frames):
        stem = Path(fr).stem
        cand = by_stem.get(stem) or next((o for s, o in by_stem.items() if stem in s or s in stem), None)
        out.append(cand or "")
    if all(out):
        return out
    if len(others) == len(frames):
        return list(others)
    return out


def write_manifest(path: str, rows: Sequence[ManifestRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_FIELDS)
        for r in rows:
            w.writerow([r.object_id, r.category, ";".join(r.frames), ";".join(r.masks), ";".join(r.depths),
                        r.mesh, r.pointcloud] + [("" if k not in r.targets else f"{r.targets[k]:.6g}")
                                                 for k in NUTRITION_TARGETS])


def read_manifest(path: str) -> List[ManifestRow]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            split = lambda s: [x for x in (s or "").split(";")]  # noqa: E731 - keep empty slots aligned
            frames = [x for x in split(rec.get("frames")) if x]
            masks, depths = split(rec.get("masks")), split(rec.get("depths"))
            targets = {k: float(rec[k]) for k in NUTRITION_TARGETS if rec.get(k) not in (None, "")}
            rows.append(ManifestRow(rec["object_id"], rec.get("category", ""), frames,
                                    (masks + [""] * len(frames))[:len(frames)],
                                    (depths + [""] * len(frames))[:len(frames)],
                                    rec.get("mesh", ""), rec.get("pointcloud", ""), targets))
    return rows


def split_objects(object_ids: Sequence[str], n_test: int = 127, seed: int = 42) -> Tuple[set, set]:
    """MFP3D protocol: random object-level split, 510 train / 127 test for the 637-object release."""
    ids = sorted(set(object_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = min(n_test, len(ids) - 1) if len(ids) > 1 else 0
    return set(ids[n_test:]), set(ids[:n_test])
