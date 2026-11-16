import os
import sys
import json
import math
import shutil
import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import yaml

_VISION_ROOT = Path(__file__).resolve().parent.parent
if str(_VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(_VISION_ROOT))

from perception.vehicle_taxonomy import CANONICAL_ID_TO_NAME, map_source_label


def parse_args():
    parser = argparse.ArgumentParser(description="Convert UA-DETRAC dataset to YOLO format.")
    parser.add_argument(
        "--detrac-dir",
        type=str,
        default=str(_VISION_ROOT.parent / "data" / "DETRAC"),
        help="Path to DETRAC root directory containing DETRAC-Images and XML folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_VISION_ROOT / "dataset_multiclass"),
        help="Target output directory for the converted YOLO dataset (separate from legacy dataset/).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["multiclass", "single_class"],
        default="multiclass",
        help="Class mapping mode: 'multiclass' (canonical taxonomy) or 'single_class' (0: vehicle).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Frame sampling stride (default: 8, i.e., take every 8th frame).",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks for images instead of copying files (saves disk space and time).",
    )
    parser.add_argument(
        "--img-width",
        type=float,
        default=960.0,
        help="Default width of DETRAC images (960px).",
    )
    parser.add_argument(
        "--img-height",
        type=float,
        default=540.0,
        help="Default height of DETRAC images (540px).",
    )
    return parser.parse_args()


def is_inside_ignored_region(box_center_x, box_center_y, ignored_boxes):
    """Checks if a target vehicle center falls inside an annotated ignored region."""
    for left, top, right, bottom in ignored_boxes:
        if left <= box_center_x <= right and top <= box_center_y <= bottom:
            return True
    return False


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
        "unsupported_labels": Counter(),
    }


def parse_xml_sequence(xml_path, img_w, img_h, stride=8, mode="multiclass", stats=None):
    """Parses DETRAC XML sequence, validates boxes, applies canonical mapping, and normalizes to YOLO."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if stats is None:
        stats = _new_stats_dict()

    ignored_boxes = []
    ignored_region_tag = root.find("ignored_region")
    if ignored_region_tag is not None:
        for box in ignored_region_tag.findall("box"):
            left = float(box.get("left", 0))
            top = float(box.get("top", 0))
            w = float(box.get("width", 0))
            h = float(box.get("height", 0))
            ignored_boxes.append((left, top, left + w, top + h))

    frames_data = {}

    for frame in root.findall("frame"):
        frame_num = int(frame.get("num", 0))
        if stride > 1 and frame_num % stride != 1:
            continue

        target_list = frame.find("target_list")
        if target_list is None:
            frames_data[frame_num] = []
            continue

        frame_boxes = []
        for target in target_list.findall("target"):
            box = target.find("box")
            if box is None:
                stats["annotations_rejected"] += 1
                continue

            try:
                left = float(box.get("left", 0))
                top = float(box.get("top", 0))
                w = float(box.get("width", 0))
                h = float(box.get("height", 0))
            except (TypeError, ValueError):
                stats["annotations_rejected"] += 1
                continue

            if not all(math.isfinite(v) for v in (left, top, w, h)) or w <= 1.0 or h <= 1.0:
                stats["annotations_rejected"] += 1
                continue

            center_x = left + w / 2.0
            center_y = top + h / 2.0

            if is_inside_ignored_region(center_x, center_y, ignored_boxes):
                stats["annotations_ignored"] += 1
                continue

            attr = target.find("attribute")
            if attr is not None:
                truncation = float(attr.get("truncation_ratio", 0.0))
                if truncation > 0.85:
                    stats["annotations_ignored"] += 1
                    continue
                vehicle_type = attr.get("vehicle_type", "")
            else:
                vehicle_type = ""

            if mode == "single_class":
                cls_id, cls_name, status = 0, "vehicle", "mapped"
            else:
                cls_id, cls_name, status = map_source_label("detrac", vehicle_type)
                if status == "ambiguous":
                    stats["excluded_ambiguous"][f"detrac:{vehicle_type}"] += 1
                    continue
                if status != "mapped" or cls_id is None:
                    stats["unsupported_labels"][f"detrac:{vehicle_type}"] += 1
                    continue

            x1 = max(0.0, min(img_w, left))
            y1 = max(0.0, min(img_h, top))
            x2 = max(0.0, min(img_w, left + w))
            y2 = max(0.0, min(img_h, top + h))
            clipped_w = x2 - x1
            clipped_h = y2 - y1
            if clipped_w <= 1.0 or clipped_h <= 1.0:
                stats["annotations_rejected"] += 1
                continue

            norm_x = ((x1 + x2) / 2.0) / img_w
            norm_y = ((y1 + y2) / 2.0) / img_h
            norm_w = clipped_w / img_w
            norm_h = clipped_h / img_h

            frame_boxes.append((cls_id, norm_x, norm_y, norm_w, norm_h))
            stats["annotations_processed"] += 1
            if mode == "multiclass" and cls_name in stats["instances_per_canonical_class"]:
                stats["instances_per_canonical_class"][cls_name] += 1

        frames_data[frame_num] = frame_boxes

    return frames_data


def process_split(split_name, xml_dir, images_root, output_dir, stride, img_w, img_h, symlink, mode="multiclass"):
    """Converts a DETRAC split directory into YOLO images/labels and returns conversion statistics."""
    img_out_dir = Path(output_dir) / "images" / split_name
    lbl_out_dir = Path(output_dir) / "labels" / split_name

    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(Path(xml_dir).glob("*.xml"))
    stats = _new_stats_dict()

    for xml_file in tqdm(xml_files, desc=f"Converting DETRAC {split_name}"):
        seq_name = xml_file.stem
        seq_img_dir = Path(images_root) / seq_name

        if not seq_img_dir.exists():
            stats["images_skipped"] += 1
            continue

        frames_data = parse_xml_sequence(xml_file, img_w, img_h, stride=stride, mode=mode, stats=stats)

        for frame_num in sorted(frames_data.keys()):
            boxes = frames_data[frame_num]
            img_filename = f"img{frame_num:05d}.jpg"
            src_img_path = seq_img_dir / img_filename

            if not src_img_path.exists():
                stats["images_skipped"] += 1
                continue

            out_basename = f"{seq_name}_{img_filename}"
            dst_img_path = img_out_dir / out_basename
            dst_lbl_path = lbl_out_dir / f"{seq_name}_img{frame_num:05d}.txt"

            if not dst_img_path.exists():
                if symlink:
                    try:
                        os.symlink(os.path.abspath(src_img_path), dst_img_path)
                    except OSError:
                        shutil.copy2(src_img_path, dst_img_path)
                else:
                    shutil.copy2(src_img_path, dst_img_path)

            with open(dst_lbl_path, "w") as f:
                for cls_id, nx, ny, nw, nh in boxes:
                    f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

            stats["images_processed"] += 1
            if not boxes:
                stats["empty_label_images"] += 1

    stats["excluded_ambiguous"] = dict(stats["excluded_ambiguous"])
    stats["unsupported_labels"] = dict(stats["unsupported_labels"])
    return stats


def create_yaml_config(output_dir, config_path, mode="multiclass"):
    """Generates YOLO dataset configuration file."""
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    names = {0: "vehicle"} if mode == "single_class" else dict(CANONICAL_ID_TO_NAME)

    yaml_data = {
        "path": str(Path(output_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    with open(config_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)


def main():
    args = parse_args()
    detrac_root = Path(args.detrac_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    train_xml_dir = detrac_root / "DETRAC-Train-Annotations-XML"
    test_xml_dir = detrac_root / "DETRAC-Test-Annotations-XML"
    images_root = detrac_root / "DETRAC-Images"

    if not train_xml_dir.exists() or not images_root.exists():
        print(f"Error: DETRAC directories not found under: {detrac_root}")
        sys.exit(1)

    train_stats = process_split(
        "detrac_train", train_xml_dir, images_root, output_root, args.stride, args.img_width, args.img_height, args.symlink, args.mode
    )
    val_stats = process_split(
        "detrac_test", test_xml_dir, images_root, output_root, args.stride, args.img_width, args.img_height, args.symlink, args.mode
    )

    summary = {
        "source_dataset": "UA-DETRAC",
        "mode": args.mode,
        "stride": args.stride,
        "splits": {"detrac_train": train_stats, "detrac_test": val_stats},
    }
    summary_path = output_root / "conversion_summary_detrac.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved DETRAC conversion summary to: {summary_path}")


if __name__ == "__main__":
    main()
