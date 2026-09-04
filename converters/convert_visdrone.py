import os
import sys
import shutil
import argparse
import zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml

VEHICLE_CATEGORIES = {
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

VISDRONE_URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
}


# Validates file existence and zip archive integrity
def is_valid_zip(file_path):
    if not file_path.exists() or file_path.stat().st_size < 1024 * 1024:
        return False
    return zipfile.is_zipfile(file_path)


# Downloads raw dataset archive via ultralytics utility or fallback urllib
def download_dataset(split, url, target_zip):
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
    parser = argparse.ArgumentParser(description="Download and convert VisDrone to single-class YOLO format.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "data" / "VisDrone"),
        help="Directory to store raw VisDrone downloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "dataset"),
        help="Target output directory for the converted dataset.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks for images instead of copying files (saves disk space).",
    )
    return parser.parse_args()


# Converts VisDrone annotations into normalized YOLO format filtering for motorized and non-motorized vehicles
def extract_and_convert_split(split_name, raw_split_dir, output_dir, symlink):
    img_src_dir = raw_split_dir / "images"
    ann_src_dir = raw_split_dir / "annotations"

    img_dst_dir = Path(output_dir) / "images" / f"{split_name}_visdrone"
    lbl_dst_dir = Path(output_dir) / "labels" / f"{split_name}_visdrone"

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(img_src_dir.glob("*.jpg")) + list(img_src_dir.glob("*.png")))
    print(f"\nConverting VisDrone '{split_name}' split ({len(img_files)} images)...")

    total_vehicles = 0
    valid_images = 0

    for img_path in tqdm(img_files, desc=f"Processing {split_name}"):
        ann_path = ann_src_dir / f"{img_path.stem}.txt"
        if not ann_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            continue

        boxes_yolo = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue

                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_w = float(parts[2])
                bbox_h = float(parts[3])
                score = int(parts[4])
                category = int(parts[5])

                if score == 0:
                    continue

                if category in VEHICLE_CATEGORIES:
                    if bbox_w <= 1.0 or bbox_h <= 1.0:
                        continue

                    cx = (bbox_left + bbox_w / 2.0) / img_w
                    cy = (bbox_top + bbox_h / 2.0) / img_h
                    nw = bbox_w / img_w
                    nh = bbox_h / img_h

                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    nw = max(0.0, min(1.0, nw))
                    nh = max(0.0, min(1.0, nh))

                    boxes_yolo.append((0, cx, cy, nw, nh))

        dst_lbl_path = lbl_dst_dir / f"{img_path.stem}.txt"
        with open(dst_lbl_path, "w") as f:
            for cls_id, nx, ny, nw, nh in boxes_yolo:
                f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

        dst_img_path = img_dst_dir / img_path.name
        if dst_img_path.exists():
            pass
        elif symlink:
            try:
                os.symlink(os.path.abspath(img_path), dst_img_path)
            except OSError:
                shutil.copy2(img_path, dst_img_path)
        else:
            shutil.copy2(img_path, dst_img_path)

        total_vehicles += len(boxes_yolo)
        valid_images += 1

    print(f"[{split_name.upper()} COMPLETE] Processed {valid_images} images with {total_vehicles:,} vehicle annotations.")
    return valid_images, total_vehicles


# Creates multi-dataset YAML configuration pointing to both DETRAC and VisDrone splits
def create_combined_yaml(output_dir, config_path):
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)

    yaml_data = {
        "path": str(Path(output_dir).resolve()),
        "train": [
            "images/train",
            "images/train_visdrone",
        ],
        "val": [
            "images/val",
            "images/val_visdrone",
        ],
        "names": {
            0: "vehicle"
        }
    }

    with open(config_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nGenerated combined dataset configuration at: {config_path}")


def main():
    args = parse_args()
    raw_dir = Path(args.data_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_dir).resolve()

    print("=" * 70)
    print(" ATCS: VisDrone Dataset Downloader & Converter")
    print("=" * 70)
    print(f"Raw Storage:   {raw_dir}")
    print(f"Target Output: {output_root}")
    print(f"Class Mapping: [car, van, truck, bus, motor, tricycle, bicycle] -> Single Class '0: vehicle'")
    print("=" * 70)

    for split, url in VISDRONE_URLS.items():
        split_folder_name = f"VisDrone2019-DET-{split}"
        split_dir = raw_dir / split_folder_name
        zip_file = raw_dir / f"{split_folder_name}.zip"

        if not split_dir.exists():
            if zip_file.exists() and not is_valid_zip(zip_file):
                print(f"[Notice] Removing incomplete/corrupted zip file: {zip_file.name}")
                zip_file.unlink()

            if not zip_file.exists():
                download_dataset(split, url, zip_file)

            print(f"\n[2/2] Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(raw_dir)

        extract_and_convert_split(split, split_dir, output_root, args.symlink)

    combined_config = Path(__file__).resolve().parent.parent / "configs" / "data_combined.yaml"
    create_combined_yaml(output_root, combined_config)

    print("\n" + "=" * 70)
    print(" VisDrone Conversion & Dataset Combination Complete!")
    print(f" Combined Configuration: {combined_config}")
    print("=" * 70)


if __name__ == "__main__":
    main()
