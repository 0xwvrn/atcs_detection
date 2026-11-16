import argparse
import sys
from pathlib import Path
import cv2
import yaml

_VISION_ROOT = Path(__file__).resolve().parent.parent
if str(_VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(_VISION_ROOT))

from perception.vehicle_taxonomy import CANONICAL_ID_TO_NAME


CLASS_COLORS = {
    0: (0, 255, 0),      # car: green
    1: (0, 165, 255),    # bus: orange
    2: (255, 255, 0),    # motorcycle: cyan
    3: (255, 0, 255),    # auto_rickshaw: magenta
    4: (255, 128, 0),    # van: blue-orange
    5: (0, 0, 255),      # truck: red
    6: (180, 255, 100),  # bicycle: light green
    7: (200, 200, 200),  # pedestrian: light gray
    8: (255, 180, 200),  # people: pink
}


def load_class_names(data_yaml: str = None):
    if data_yaml and Path(data_yaml).exists():
        cfg = yaml.safe_load(Path(data_yaml).read_text()) or {}
        names = cfg.get("names")
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
    return dict(CANONICAL_ID_TO_NAME)


def render_annotated_sample(img_path: Path, lbl_path: Path, out_file: Path, class_names: dict):
    """Draws ground-truth bounding boxes with class IDs and names onto an image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    h, w = img.shape[:2]
    box_count = 0

    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                nx, ny, nw, nh = map(float, parts[1:5])

                cx, cy, bw, bh = nx * w, ny * h, nw * w, nh * h
                x1 = int(cx - bw / 2.0)
                y1 = int(cy - bh / 2.0)
                x2 = int(cx + bw / 2.0)
                y2 = int(cy + bh / 2.0)

                color = CLASS_COLORS.get(cls_id, (0, 255, 0))
                cls_name = class_names.get(cls_id, f"cls_{cls_id}")
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    f"{cls_id}:{cls_name}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
                box_count += 1

    cv2.putText(
        img,
        f"Verified Boxes: {box_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 200, 255),
        2,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), img)
    return box_count


def select_per_class_samples(img_files, lbl_dir: Path, class_names: dict, per_class: int = 1):
    """Selects at least `per_class` sample images covering every populated class."""
    needed = {cid: per_class for cid in class_names}
    selected = []
    seen = set()

    for img_path in img_files:
        if all(v <= 0 for v in needed.values()):
            break
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        present = set()
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                present.add(int(float(parts[0])))
        if any(needed.get(cid, 0) > 0 for cid in present):
            if img_path not in seen:
                selected.append(img_path)
                seen.add(img_path)
            for cid in present:
                if needed.get(cid, 0) > 0:
                    needed[cid] -= 1
    return selected


def parse_args():
    base_dir = _VISION_ROOT
    parser = argparse.ArgumentParser(description="ATCS Dataset Annotation Verification Tool.")
    parser.add_argument("--dataset-dir", type=str, default=str(base_dir / "dataset_multiclass"))
    parser.add_argument("--data", type=str, default=None, help="Optional YAML to read class names.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--per-class", action="store_true", help="Sample at least one image per populated class.")
    parser.add_argument("--output-dir", type=str, default=str(base_dir / "runs" / "sanity_check"))
    return parser.parse_args()


def main():
    args = parse_args()
    img_dir = Path(args.dataset_dir) / "images" / args.split
    lbl_dir = Path(args.dataset_dir) / "labels" / args.split
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(args.data)

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not img_files:
        print(f"Warning: No images found in directory: {img_dir}")
        return

    if args.per_class:
        samples = select_per_class_samples(img_files, lbl_dir, class_names, per_class=1)
    else:
        step = max(1, len(img_files) // args.num_samples)
        samples = img_files[::step][: args.num_samples]

    for img_path in samples:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        out_file = out_dir / f"verified_{img_path.name}"
        box_count = render_annotated_sample(img_path, lbl_path, out_file, class_names)
        print(f"Verified {img_path.name}: {box_count} boxes -> {out_file.name}")

    print(f"Sanity check complete. Saved {len(samples)} annotated images to: {out_dir}")


if __name__ == "__main__":
    main()
