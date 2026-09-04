import argparse
from pathlib import Path
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="ATCS Dataset Annotation Verification Tool.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "dataset"),
        help="Path to dataset directory containing images and labels.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to verify ('train' or 'val').",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample images to visualize (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "sanity_check"),
        help="Directory to save verification visualization images.",
    )
    return parser.parse_args()


# Draws ground truth bounding boxes on sample images to visually verify label alignment
def main():
    args = parse_args()
    img_dir = Path(args.dataset_dir) / "images" / args.split
    lbl_dir = Path(args.dataset_dir) / "labels" / args.split
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" ATCS: Ground Truth Dataset Annotation Verifier")
    print("=" * 70)
    print(f" Dataset Directory: {args.dataset_dir}")
    print(f" Dataset Split:     {args.split}")
    print(f" Sample Images:     {args.num_samples}")
    print(f" Output Directory:  {out_dir}")
    print("=" * 70)

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not img_files:
        print(f"Warning: No images found in directory: {img_dir}")
        return

    step = max(1, len(img_files) // args.num_samples)
    samples = img_files[::step][: args.num_samples]

    for img_path in samples:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        box_count = 0

        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id, nx, ny, nw, nh = map(float, parts[:5])

                    cx, cy, bw, bh = nx * w, ny * h, nw * w, nh * h
                    x1 = int(cx - bw / 2.0)
                    y1 = int(cy - bh / 2.0)
                    x2 = int(cx + bw / 2.0)
                    y2 = int(cy + bh / 2.0)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        "vehicle",
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    box_count += 1

        cv2.putText(
            img,
            f"Verified Vehicles: {box_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
        )

        out_file = out_dir / f"verified_{img_path.name}"
        cv2.imwrite(str(out_file), img)
        print(f"Verified {img_path.name}: {box_count} vehicle bounding boxes -> {out_file.name}")

    print("\n" + "=" * 70)
    print(f" Sanity check complete. Saved {len(samples)} annotated images to: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
