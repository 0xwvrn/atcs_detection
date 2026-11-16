import os
import sys
import json
import math
import shutil
import argparse
import zipfile
from collections import Counter
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml

_VISION_ROOT = Path(__file__).resolve().parent.parent
if str(_VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(_VISION_ROOT))

from perception.vehicle_taxonomy import CANONICAL_ID_TO_NAME, VISDRONE_CATEGORIES, map_source_label

VISDRONE_URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
}


def is_valid_zip(file_path):
    """Validates file existence and zip archive integrity."""
    if not file_path.exists() or file_path.stat().st_size < 1024 * 1024:
        return False
    return zipfile.is_zipfile(file_path)


def download_dataset(split, url, target_zip):
    """Downloads raw dataset archive via ultralytics utility or fallback urllib."""
    print(f"\n[1/2] Downloading VisDrone {split} set...")
    try:
        from ultralytics.utils.downloads import download
        download([url], dir=target_zip.parent, threads=4)
    except Exception:
        import urllib.request
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, target_zip)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VisDrone2019-DET to YOLO format.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_VISION_ROOT.parent / "data" / "VisDrone"),
        help="Directory containing raw VisDrone splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_VISION_ROOT / "dataset_multiclass"),
        help="Target output directory for the converted dataset (separate from legacy dataset/).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["multiclass", "single_class"],
        default="multiclass",
        help="Class mapping mode: 'multiclass' (canonical taxonomy) or 'single_class' (0: vehicle).",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks for images instead of copying files (saves disk space).",
    )
    return parser.parse_args()


def _new_stats_dict():
    return {
        "images_processed": 0,
        "images_skipped": 0,
        "empty_label_images": 0,
        "annotations_processed": 0,
        "annotations_rejected": 0,
        "annotations_ignored": 0,
        "instances_per_canonical_class": {name: 0 for name in CANONICAL_ID_TO_NAME.values()},
        "excluded_ambiguous": Counter(),
        "excluded_non_vehicle": Counter(),
        "unsupported_labels": Counter(),
    }


def extract_and_convert_split(split_name, raw_split_dir, output_dir, symlink, mode="multiclass"):
    """Converts VisDrone annotations into normalized YOLO format with canonical mapping and stats."""
    img_src_dir = Path(raw_split_dir) / "images"
    ann_src_dir = Path(raw_split_dir) / "annotations"

    img_dst_dir = Path(output_dir) / "images" / f"visdrone_{split_name}"
    lbl_dst_dir = Path(output_dir) / "labels" / f"visdrone_{split_name}"

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(img_src_dir.glob("*.jpg")) + list(img_src_dir.glob("*.png")))
    stats = _new_stats_dict()

    for img_path in tqdm(img_files, desc=f"Converting VisDrone {split_name}"):
        ann_path = ann_src_dir / f"{img_path.stem}.txt"
        if not ann_path.exists():
            stats["images_skipped"] += 1
            continue

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            stats["images_skipped"] += 1
            continue

        if img_w <= 0 or img_h <= 0:
            stats["images_skipped"] += 1
            continue

        boxes_yolo = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    if line.strip():
                        stats["annotations_rejected"] += 1
                    continue

                try:
                    bbox_left = float(parts[0])
                    bbox_top = float(parts[1])
                    bbox_w = float(parts[2])
                    bbox_h = float(parts[3])
                    score = int(parts[4])
                    category = int(parts[5])
                except ValueError:
                    stats["annotations_rejected"] += 1
                    continue

                if score == 0:
                    stats["annotations_ignored"] += 1
                    continue

                if not all(math.isfinite(v) for v in (bbox_left, bbox_top, bbox_w, bbox_h)) or bbox_w <= 1.0 or bbox_h <= 1.0:
                    stats["annotations_rejected"] += 1
                    continue

                raw_name = VISDRONE_CATEGORIES.get(category, str(category))
                if mode == "single_class":
                    if category in {3, 4, 5, 6, 7, 8, 9, 10}:
                        cls_id, cls_name, status = 0, "vehicle", "mapped"
                    else:
                        continue
                else:
                    cls_id, cls_name, status = map_source_label("visdrone", category)
                    if status == "ignored":
                        stats["annotations_ignored"] += 1
                        continue
                    if status == "non_vehicle":
                        stats["excluded_non_vehicle"][f"visdrone:{raw_name}"] += 1
                        continue
                    if status == "ambiguous":
                        stats["excluded_ambiguous"][f"visdrone:{raw_name}"] += 1
                        continue
                    if status != "mapped" or cls_id is None:
                        stats["unsupported_labels"][f"visdrone:{raw_name}"] += 1
                        continue

                x1 = max(0.0, min(float(img_w), bbox_left))
                y1 = max(0.0, min(float(img_h), bbox_top))
                x2 = max(0.0, min(float(img_w), bbox_left + bbox_w))
                y2 = max(0.0, min(float(img_h), bbox_top + bbox_h))
                clipped_w = x2 - x1
                clipped_h = y2 - y1
                if clipped_w <= 1.0 or clipped_h <= 1.0:
                    stats["annotations_rejected"] += 1
                    continue

                cx = ((x1 + x2) / 2.0) / img_w
                cy = ((y1 + y2) / 2.0) / img_h
                nw = clipped_w / img_w
                nh = clipped_h / img_h

                boxes_yolo.append((cls_id, cx, cy, nw, nh))
                stats["annotations_processed"] += 1
                if mode == "multiclass" and cls_name in stats["instances_per_canonical_class"]:
                    stats["instances_per_canonical_class"][cls_name] += 1

        dst_lbl_path = lbl_dst_dir / f"{img_path.stem}.txt"
        with open(dst_lbl_path, "w") as f:
            for cls_id, nx, ny, nw, nh in boxes_yolo:
                f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

        dst_img_path = img_dst_dir / img_path.name
        if not dst_img_path.exists():
            if symlink:
                try:
                    os.symlink(os.path.abspath(img_path), dst_img_path)
                except OSError:
                    shutil.copy2(img_path, dst_img_path)
            else:
                shutil.copy2(img_path, dst_img_path)

        stats["images_processed"] += 1
        if not boxes_yolo:
            stats["empty_label_images"] += 1

    stats["excluded_ambiguous"] = dict(stats["excluded_ambiguous"])
    stats["excluded_non_vehicle"] = dict(stats["excluded_non_vehicle"])
    stats["unsupported_labels"] = dict(stats["unsupported_labels"])
    return stats


def create_combined_yaml(output_dir, config_path, mode="multiclass"):
    """Creates dataset YAML configuration."""
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    names = {0: "vehicle"} if mode == "single_class" else dict(CANONICAL_ID_TO_NAME)

    yaml_data = {
        "path": str(Path(output_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    with open(config_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)


def main():
    args = parse_args()
    raw_dir = Path(args.data_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_dir).resolve()

    splits_stats = {}
    for split, url in VISDRONE_URLS.items():
        split_folder_name = f"VisDrone2019-DET-{split}"
        split_dir = raw_dir / split_folder_name
        zip_file = raw_dir / f"{split_folder_name}.zip"

        if not split_dir.exists():
            if zip_file.exists() and not is_valid_zip(zip_file):
                zip_file.unlink()
            if not zip_file.exists():
                download_dataset(split, url, zip_file)
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(raw_dir)

        splits_stats[split] = extract_and_convert_split(split, split_dir, output_root, args.symlink, args.mode)

    summary = {
        "source_dataset": "VisDrone2019-DET",
        "mode": args.mode,
        "splits": splits_stats,
    }
    summary_path = output_root / "conversion_summary_visdrone.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved VisDrone conversion summary to: {summary_path}")


if __name__ == "__main__":
    main()
