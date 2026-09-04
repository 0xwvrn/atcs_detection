import os
import sys
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Convert UA-DETRAC dataset to YOLO format.")
    parser.add_argument(
        "--detrac-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "data" / "DETRAC"),
        help="Path to DETRAC root directory containing DETRAC-Images and XML folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "dataset"),
        help="Target output directory for the converted YOLO dataset.",
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


# Checks if a target vehicle center coordinate falls within any annotated ignored region bounding box
def is_inside_ignored_region(box_center_x, box_center_y, ignored_boxes):
    for left, top, right, bottom in ignored_boxes:
        if left <= box_center_x <= right and top <= box_center_y <= bottom:
            return True
    return False


# Parses DETRAC XML sequence, filters ignored regions, and normalizes box coordinates to YOLO format
def parse_xml_sequence(xml_path, img_w, img_h, stride=8):
    tree = ET.parse(xml_path)
    root = tree.getroot()

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
        if frame_num % stride != 1 and stride > 1:
            continue

        target_list = frame.find("target_list")
        if target_list is None:
            frames_data[frame_num] = []
            continue

        frame_boxes = []
        for target in target_list.findall("target"):
            box = target.find("box")
            if box is None:
                continue

            left = float(box.get("left", 0))
            top = float(box.get("top", 0))
            w = float(box.get("width", 0))
            h = float(box.get("height", 0))

            if w <= 1.0 or h <= 1.0:
                continue

            center_x = left + w / 2.0
            center_y = top + h / 2.0

            if is_inside_ignored_region(center_x, center_y, ignored_boxes):
                continue

            attr = target.find("attribute")
            if attr is not None:
                truncation = float(attr.get("truncation_ratio", 0.0))
                if truncation > 0.85:
                    continue

            norm_x = max(0.0, min(1.0, center_x / img_w))
            norm_y = max(0.0, min(1.0, center_y / img_h))
            norm_w = max(0.0, min(1.0, w / img_w))
            norm_h = max(0.0, min(1.0, h / img_h))

            frame_boxes.append((0, norm_x, norm_y, norm_w, norm_h))

        frames_data[frame_num] = frame_boxes

    return frames_data


# Iterates over split XMLs and pairs them with corresponding frame images
def process_split(split_name, xml_dir, images_root, output_dir, stride, img_w, img_h, symlink):
    img_out_dir = Path(output_dir) / "images" / split_name
    lbl_out_dir = Path(output_dir) / "labels" / split_name

    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(Path(xml_dir).glob("*.xml"))
    print(f"\nProcessing {len(xml_files)} sequences for '{split_name}' split (stride={stride})...")

    total_images = 0
    total_annotations = 0

    for xml_file in tqdm(xml_files, desc=f"Converting {split_name}"):
        seq_name = xml_file.stem
        seq_img_dir = Path(images_root) / seq_name

        if not seq_img_dir.exists():
            print(f"[Warning] Image folder missing for sequence: {seq_img_dir}, skipping.")
            continue

        frames_data = parse_xml_sequence(xml_file, img_w, img_h, stride=stride)

        for frame_num, boxes in frames_data.items():
            img_filename = f"img{frame_num:05d}.jpg"
            src_img_path = seq_img_dir / img_filename

            if not src_img_path.exists():
                continue

            out_basename = f"{seq_name}_{img_filename}"
            dst_img_path = img_out_dir / out_basename
            dst_lbl_path = lbl_out_dir / f"{seq_name}_img{frame_num:05d}.txt"

            if dst_img_path.exists():
                pass
            elif symlink:
                try:
                    os.symlink(os.path.abspath(src_img_path), dst_img_path)
                except OSError:
                    shutil.copy2(src_img_path, dst_img_path)
            else:
                shutil.copy2(src_img_path, dst_img_path)

            with open(dst_lbl_path, "w") as f:
                for cls_id, nx, ny, nw, nh in boxes:
                    f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

            total_images += 1
            total_annotations += len(boxes)

    print(f"[{split_name.upper()} COMPLETE] Processed {total_images} images with {total_annotations} vehicle bounding boxes.")
    return total_images, total_annotations


# Generates YOLO dataset configuration file
def create_yaml_config(output_dir, config_path):
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)

    yaml_data = {
        "path": str(Path(output_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "vehicle"
        }
    }

    with open(config_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nCreated YOLO data configuration at: {config_path}")


def main():
    args = parse_args()
    detrac_root = Path(args.detrac_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    train_xml_dir = detrac_root / "DETRAC-Train-Annotations-XML"
    test_xml_dir = detrac_root / "DETRAC-Test-Annotations-XML"
    images_root = detrac_root / "DETRAC-Images"

    if not train_xml_dir.exists() or not images_root.exists():
        print(f"Error: DETRAC directories not found under: {detrac_root}")
        print("Expected structure: data/DETRAC/DETRAC-Train-Annotations-XML and data/DETRAC/DETRAC-Images")
        sys.exit(1)

    print("=" * 70)
    print(" ATCS: UA-DETRAC to YOLO Dataset Converter")
    print("=" * 70)
    print(f"Source DETRAC: {detrac_root}")
    print(f"Target Output: {output_root}")
    print(f"Frame Stride:  {args.stride} (Sample every {args.stride}th frame)")
    print(f"Class Mapping: Single Unified Class -> '0: vehicle'")
    print(f"Mode:          {'Symlink images' if args.symlink else 'Copy images'}")
    print("=" * 70)

    train_imgs, train_boxes = process_split(
        "train", train_xml_dir, images_root, output_root, args.stride, args.img_width, args.img_height, args.symlink
    )

    val_imgs, val_boxes = process_split(
        "val", test_xml_dir, images_root, output_root, args.stride, args.img_width, args.img_height, args.symlink
    )

    config_file = Path(__file__).resolve().parent.parent / "configs" / "data.yaml"
    create_yaml_config(output_root, config_file)

    print("\n" + "=" * 70)
    print(f" Dataset Conversion Finished Successfully!")
    print(f" Total Train Images: {train_imgs:,} | Train Vehicle Boxes: {train_boxes:,}")
    print(f" Total Val Images:   {val_imgs:,} | Val Vehicle Boxes:   {val_boxes:,}")
    print(f" YOLO Config File:   {config_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
