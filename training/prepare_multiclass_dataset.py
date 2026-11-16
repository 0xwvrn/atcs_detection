"""
ATCS Multiclass Dataset Preprocessing, Sequence-Aware Splitting & Quality Audit.

Combines UA-DETRAC and VisDrone2019-DET into a reproducible multiclass YOLO
dataset with deterministic sequence-level train/val/test splits, cross-split
duplicate detection, bounding-box integrity checks, and class-imbalance metrics.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
import yaml

_VISION_ROOT = Path(__file__).resolve().parent.parent
if str(_VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(_VISION_ROOT))

from converters.convert_detrac import process_split as convert_detrac_split
from converters.convert_visdrone import extract_and_convert_split as convert_visdrone_split
from perception.vehicle_taxonomy import CANONICAL_ID_TO_NAME


def _sequence_key_detrac(filename_stem: str) -> str:
    # Format: MVI_20011_img00001 -> MVI_20011
    idx = filename_stem.rfind("_img")
    return filename_stem[:idx] if idx > 0 else filename_stem


def _sequence_key_visdrone(filename_stem: str) -> str:
    # VisDrone filenames typically begin with <seq_id>_<frame>_...
    parts = filename_stem.split("_")
    return parts[0] if len(parts) > 1 else filename_stem


def split_sequences_deterministically(seq_to_items, val_ratio: float = 0.15, seed: int = 42):
    """Splits sequence groups deterministically into (train_items, val_items, train_seqs, val_seqs)."""
    seqs = sorted(seq_to_items.keys())
    rng = random.Random(seed)
    shuffled = list(seqs)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        val_count = 0
    else:
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1

    val_seqs = sorted(shuffled[:val_count])
    train_seqs = sorted(shuffled[val_count:])

    train_items = []
    for s in train_seqs:
        train_items.extend(sorted(seq_to_items[s]))
    val_items = []
    for s in val_seqs:
        val_items.extend(sorted(seq_to_items[s]))
    return train_items, val_items, train_seqs, val_seqs


def _place_file(src: Path, dst: Path, symlink: bool):
    if dst.exists() or dst.is_symlink():
        return
    if symlink:
        try:
            os.symlink(os.path.realpath(src), dst)
            return
        except OSError:
            pass
    shutil.copy2(os.path.realpath(src), dst)


def _file_blake2b(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def audit_yolo_dataset(dataset_dir: Path, valid_class_ids=None, check_hashes: bool = True):
    """Runs integrity, box-validity, class-distribution, and cross-split duplicate checks."""
    if valid_class_ids is None:
        valid_class_ids = set(CANONICAL_ID_TO_NAME.keys())

    splits = [d.name for d in sorted((dataset_dir / "images").iterdir()) if d.is_dir()] if (dataset_dir / "images").exists() else []
    split_reports = {}
    hash_to_splits = defaultdict(list)

    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        lbls = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}

        missing_labels = sorted(img_stems - lbl_stems)
        missing_images = sorted(lbl_stems - img_stems)

        invalid_class_ids = 0
        invalid_boxes = 0
        zero_area_boxes = 0
        out_of_bounds_boxes = 0
        duplicate_annotations = 0
        tiny_boxes = 0
        huge_boxes = 0
        empty_label_files = 0
        class_counts = Counter()

        for img_path in imgs:
            if check_hashes:
                h = _file_blake2b(img_path)
                hash_to_splits[h].append((split, img_path.name))

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            lines = [ln.strip() for ln in lbl_path.read_text().splitlines() if ln.strip()]
            if not lines:
                empty_label_files += 1
                continue

            seen_lines = set()
            for ln in lines:
                if ln in seen_lines:
                    duplicate_annotations += 1
                seen_lines.add(ln)

                parts = ln.split()
                if len(parts) < 5:
                    invalid_boxes += 1
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, nw, nh = map(float, parts[1:5])
                except ValueError:
                    invalid_boxes += 1
                    continue

                if cid not in valid_class_ids:
                    invalid_class_ids += 1
                    continue
                if not all(math.isfinite(v) for v in (cx, cy, nw, nh)):
                    invalid_boxes += 1
                    continue
                if nw <= 0.0 or nh <= 0.0:
                    zero_area_boxes += 1
                    continue
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
                    out_of_bounds_boxes += 1
                    continue

                area = nw * nh
                if area < 4e-5:
                    tiny_boxes += 1
                elif area > 0.95:
                    huge_boxes += 1

                class_counts[CANONICAL_ID_TO_NAME.get(cid, str(cid))] += 1

        split_reports[split] = {
            "images": len(imgs),
            "labels": len(lbls),
            "empty_label_files": empty_label_files,
            "missing_labels": len(missing_labels),
            "missing_images": len(missing_images),
            "invalid_class_ids": invalid_class_ids,
            "invalid_boxes": invalid_boxes,
            "zero_area_boxes": zero_area_boxes,
            "out_of_bounds_boxes": out_of_bounds_boxes,
            "duplicate_annotations": duplicate_annotations,
            "tiny_boxes_lt_4e5": tiny_boxes,
            "huge_boxes_gt_095": huge_boxes,
            "class_counts": {name: class_counts.get(name, 0) for name in CANONICAL_ID_TO_NAME.values()},
        }

    cross_split_duplicates = []
    for h, entries in hash_to_splits.items():
        distinct_splits = sorted({sp for sp, _ in entries})
        if len(distinct_splits) > 1:
            cross_split_duplicates.append({"hash": h, "splits": distinct_splits, "files": entries[:10]})

    total_class_counts = Counter()
    for rep in split_reports.values():
        for k, v in rep["class_counts"].items():
            total_class_counts[k] += v

    nonzero_counts = [v for v in total_class_counts.values() if v > 0]
    imbalance_ratio = round(max(nonzero_counts) / min(nonzero_counts), 2) if nonzero_counts else None

    return {
        "splits": split_reports,
        "total_class_counts": {name: total_class_counts.get(name, 0) for name in CANONICAL_ID_TO_NAME.values()},
        "imbalance_ratio_max_over_min_nonzero": imbalance_ratio,
        "cross_split_duplicate_count": len(cross_split_duplicates),
        "cross_split_duplicates": cross_split_duplicates[:20],
    }


def write_multiclass_yaml(output_dir: Path, yaml_path: Path):
    """Writes canonical multiclass YOLO dataset YAML configuration."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CANONICAL_ID_TO_NAME),
        "names": dict(CANONICAL_ID_TO_NAME),
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def prepare_dataset(
    detrac_dir: Path,
    visdrone_dir: Path,
    output_dir: Path,
    yaml_path: Path,
    stride: int = 8,
    val_ratio: float = 0.15,
    seed: int = 42,
    symlink: bool = True,
    keep_staging: bool = False,
):
    """Converts, splits without sequence leakage, audits, and generates YAML + manifest."""
    output_dir = output_dir.resolve()
    staging_dir = output_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    source_summaries = {}

    # 1. Convert DETRAC if present
    detrac_train_xml = detrac_dir / "DETRAC-Train-Annotations-XML"
    detrac_test_xml = detrac_dir / "DETRAC-Test-Annotations-XML"
    detrac_imgs = detrac_dir / "DETRAC-Images"
    if detrac_train_xml.exists() and detrac_imgs.exists():
        source_summaries["detrac_train"] = convert_detrac_split(
            "detrac_train", detrac_train_xml, detrac_imgs, staging_dir, stride, 960.0, 540.0, symlink, mode="multiclass"
        )
        if detrac_test_xml.exists():
            source_summaries["detrac_test"] = convert_detrac_split(
                "detrac_test", detrac_test_xml, detrac_imgs, staging_dir, stride, 960.0, 540.0, symlink, mode="multiclass"
            )

    # 2. Convert VisDrone if present
    for vd_split, vd_folder in [("train", "VisDrone2019-DET-train"), ("val", "VisDrone2019-DET-val")]:
        raw_split = visdrone_dir / vd_folder
        if (raw_split / "images").exists() and (raw_split / "annotations").exists():
            source_summaries[f"visdrone_{vd_split}"] = convert_visdrone_split(
                vd_split, raw_split, staging_dir, symlink, mode="multiclass"
            )

    # Prepare final train/val/test folders
    for sp in ("train", "val", "test"):
        img_sp = output_dir / "images" / sp
        lbl_sp = output_dir / "labels" / sp
        if img_sp.exists():
            shutil.rmtree(img_sp)
        if lbl_sp.exists():
            shutil.rmtree(lbl_sp)
        img_sp.mkdir(parents=True, exist_ok=True)
        lbl_sp.mkdir(parents=True, exist_ok=True)

    split_sequences = {}

    # 3. Sequence-aware split for DETRAC train -> train / val
    dt_img_dir = staging_dir / "images" / "detrac_train"
    dt_lbl_dir = staging_dir / "labels" / "detrac_train"
    if dt_img_dir.exists():
        seq_map = defaultdict(list)
        for img_p in sorted(dt_img_dir.glob("*.jpg")):
            seq_map[_sequence_key_detrac(img_p.stem)].append(img_p)
        tr_items, va_items, tr_seqs, va_seqs = split_sequences_deterministically(seq_map, val_ratio, seed)
        split_sequences["detrac_train_seqs"] = tr_seqs
        split_sequences["detrac_val_seqs"] = va_seqs
        for sp_name, items in [("train", tr_items), ("val", va_items)]:
            for img_p in items:
                lbl_p = dt_lbl_dir / f"{img_p.stem}.txt"
                _place_file(img_p, output_dir / "images" / sp_name / f"detrac_{img_p.name}", symlink)
                shutil.copy2(lbl_p, output_dir / "labels" / sp_name / f"detrac_{img_p.stem}.txt")

    # DETRAC test -> test
    dt_test_img = staging_dir / "images" / "detrac_test"
    dt_test_lbl = staging_dir / "labels" / "detrac_test"
    if dt_test_img.exists():
        test_seqs = set()
        for img_p in sorted(dt_test_img.glob("*.jpg")):
            test_seqs.add(_sequence_key_detrac(img_p.stem))
            lbl_p = dt_test_lbl / f"{img_p.stem}.txt"
            _place_file(img_p, output_dir / "images" / "test" / f"detrac_{img_p.name}", symlink)
            shutil.copy2(lbl_p, output_dir / "labels" / "test" / f"detrac_{img_p.stem}.txt")
        split_sequences["detrac_test_seqs"] = sorted(test_seqs)

    # 4. Sequence-aware split for VisDrone train -> train / val
    vd_tr_img = staging_dir / "images" / "visdrone_train"
    vd_tr_lbl = staging_dir / "labels" / "visdrone_train"
    if vd_tr_img.exists():
        seq_map = defaultdict(list)
        for img_p in sorted(list(vd_tr_img.glob("*.jpg")) + list(vd_tr_img.glob("*.png"))):
            seq_map[_sequence_key_visdrone(img_p.stem)].append(img_p)
        tr_items, va_items, tr_seqs, va_seqs = split_sequences_deterministically(seq_map, val_ratio, seed)
        split_sequences["visdrone_train_seq_count"] = len(tr_seqs)
        split_sequences["visdrone_val_seq_count"] = len(va_seqs)
        for sp_name, items in [("train", tr_items), ("val", va_items)]:
            for img_p in items:
                lbl_p = vd_tr_lbl / f"{img_p.stem}.txt"
                _place_file(img_p, output_dir / "images" / sp_name / f"visdrone_{img_p.name}", symlink)
                shutil.copy2(lbl_p, output_dir / "labels" / sp_name / f"visdrone_{img_p.stem}.txt")

    # VisDrone val -> test
    vd_val_img = staging_dir / "images" / "visdrone_val"
    vd_val_lbl = staging_dir / "labels" / "visdrone_val"
    if vd_val_img.exists():
        vd_test_seqs = set()
        for img_p in sorted(list(vd_val_img.glob("*.jpg")) + list(vd_val_img.glob("*.png"))):
            vd_test_seqs.add(_sequence_key_visdrone(img_p.stem))
            lbl_p = vd_val_lbl / f"{img_p.stem}.txt"
            _place_file(img_p, output_dir / "images" / "test" / f"visdrone_{img_p.name}", symlink)
            shutil.copy2(lbl_p, output_dir / "labels" / "test" / f"visdrone_{img_p.stem}.txt")
        split_sequences["visdrone_test_seq_count"] = len(vd_test_seqs)

    if not keep_staging and staging_dir.exists():
        shutil.rmtree(staging_dir)

    write_multiclass_yaml(output_dir, yaml_path)
    audit = audit_yolo_dataset(output_dir, check_hashes=True)

    manifest = {
        "dataset_version": "multiclass_v1",
        "seed": seed,
        "val_ratio": val_ratio,
        "detrac_stride": stride,
        "symlink_images": symlink,
        "canonical_classes": dict(CANONICAL_ID_TO_NAME),
        "source_conversion_summaries": source_summaries,
        "split_sequences": split_sequences,
        "audit": audit,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args():
    p = argparse.ArgumentParser(description="Prepare ATCS multiclass YOLO dataset (DETRAC + VisDrone).")
    p.add_argument("--detrac-dir", type=str, default=str(_VISION_ROOT.parent / "data" / "DETRAC"))
    p.add_argument("--visdrone-dir", type=str, default=str(_VISION_ROOT.parent / "data" / "VisDrone"))
    p.add_argument("--output-dir", type=str, default=str(_VISION_ROOT / "dataset_multiclass"))
    p.add_argument("--yaml-path", type=str, default=str(_VISION_ROOT / "configs" / "data_multiclass.yaml"))
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    p.add_argument("--keep-staging", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = prepare_dataset(
        detrac_dir=Path(args.detrac_dir),
        visdrone_dir=Path(args.visdrone_dir),
        output_dir=Path(args.output_dir),
        yaml_path=Path(args.yaml_path),
        stride=args.stride,
        val_ratio=args.val_ratio,
        seed=args.seed,
        symlink=not args.copy,
        keep_staging=args.keep_staging,
    )
    print(f"Prepared multiclass dataset at: {Path(args.output_dir).resolve()}")
    print(f"YAML config: {Path(args.yaml_path).resolve()}")
    print("Split image counts:", {k: v["images"] for k, v in manifest["audit"]["splits"].items()})
    print("Total class counts:", manifest["audit"]["total_class_counts"])
    print("Cross-split duplicates:", manifest["audit"]["cross_split_duplicate_count"])


if __name__ == "__main__":
    main()
