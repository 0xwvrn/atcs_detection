import sys
import time
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


def parse_args():
    default_weights = Path(__file__).resolve().parent / "runs" / "detect" / "model_p2_960_safe" / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = Path(__file__).resolve().parent / "runs" / "detect" / "atcs_high_recall_model" / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = "yolov8n.pt"

    default_data_dir = Path(__file__).resolve().parent.parent / "data" / "DETRAC"

    parser = argparse.ArgumentParser(description="ATCS Traffic Counting & Multi-Object Tracking Benchmark Suite.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_weights),
        help="Path to trained YOLO weights checkpoint (.pt file).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(default_data_dir),
        help="Path to UA-DETRAC root directory containing DETRAC-Test-Annotations-XML and DETRAC-Images.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracking algorithm configuration file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.50,
        help="IoU threshold for bipartite ground-truth matching (default: 0.50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image resolution (default: 960).",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=None,
        help="Maximum number of test sequences to benchmark (default: all available).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on ('0', 'mps', 'cpu'). Defaults to cpu for safe high-resolution anchor decoding.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "benchmark"),
        help="Directory to save benchmark metrics and summary tables.",
    )
    return parser.parse_args()


# Checks if a target vehicle center coordinate falls within any annotated ignored region bounding box
def is_inside_ignored_region(box_center_x, box_center_y, ignored_boxes):
    for left, top, right, bottom in ignored_boxes:
        if left <= box_center_x <= right and top <= box_center_y <= bottom:
            return True
    return False


# Parses ground truth bounding boxes and vehicle IDs from UA-DETRAC XML sequence file
def parse_ground_truth_sequence(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ignored_boxes = []
    ignored_tag = root.find("ignored_region")
    if ignored_tag is not None:
        for box in ignored_tag.findall("box"):
            left = float(box.get("left", 0))
            top = float(box.get("top", 0))
            w = float(box.get("width", 0))
            h = float(box.get("height", 0))
            ignored_boxes.append((left, top, left + w, top + h))

    frames_gt = {}
    total_unique_gt_ids = set()

    for frame in root.findall("frame"):
        frame_num = int(frame.get("num", 0))
        target_list = frame.find("target_list")
        frame_targets = []

        if target_list is not None:
            for target in target_list.findall("target"):
                target_id = int(target.get("id", -1))
                box = target.find("box")
                if box is None:
                    continue

                left = float(box.get("left", 0))
                top = float(box.get("top", 0))
                w = float(box.get("width", 0))
                h = float(box.get("height", 0))

                if w <= 1.0 or h <= 1.0:
                    continue

                cx = left + w / 2.0
                cy = top + h / 2.0
                if is_inside_ignored_region(cx, cy, ignored_boxes):
                    continue

                attr = target.find("attribute")
                if attr is not None:
                    truncation = float(attr.get("truncation_ratio", 0.0))
                    if truncation > 0.85:
                        continue

                frame_targets.append({
                    "id": target_id,
                    "bbox": [left, top, left + w, top + h],
                })
                total_unique_gt_ids.add(target_id)

        frames_gt[frame_num] = frame_targets

    return frames_gt, len(total_unique_gt_ids)


# Calculates pairwise intersection over union between two sets of bounding boxes
def compute_iou_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1 = np.array(boxes1, dtype=np.float32)
    b2 = np.array(boxes2, dtype=np.float32)

    x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection

    return np.where(union > 0, intersection / union, 0.0)


# Benchmarks vehicle detection and tracking across a single UA-DETRAC sequence
def evaluate_sequence(model, tracker_cfg, xml_path, img_dir, conf, iou_thresh, imgsz, device):
    frames_gt, unique_gt_count = parse_ground_truth_sequence(xml_path)
    seq_name = xml_path.stem

    total_gt_instances = 0
    total_pred_instances = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    absolute_count_errors = []
    squared_count_errors = []
    latency_records = []

    gt_to_track_mapping = {}
    frame_nums = sorted(frames_gt.keys())

    for fnum in frame_nums:
        img_path = img_dir / f"img{fnum:05d}.jpg"
        if not img_path.exists():
            continue

        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            continue

        t_start = time.perf_counter()
        results = model.track(
            source=frame_bgr,
            persist=True,
            tracker=tracker_cfg,
            conf=conf,
            iou=iou_thresh,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        t_end = time.perf_counter()
        latency_records.append((t_end - t_start) * 1000.0)

        pred_boxes = []
        pred_ids = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            pred_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            if results[0].boxes.id is not None:
                pred_ids = results[0].boxes.id.int().cpu().numpy().tolist()
            else:
                pred_ids = list(range(len(pred_boxes)))

        gt_targets = frames_gt[fnum]
        gt_boxes = [t["bbox"] for t in gt_targets]
        gt_ids = [t["id"] for t in gt_targets]

        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        total_gt_instances += n_gt
        total_pred_instances += n_pred

        abs_err = abs(n_pred - n_gt)
        absolute_count_errors.append(abs_err)
        squared_count_errors.append(abs_err ** 2)

        # Computes bipartite matching to identify True Positives, False Positives, and ID switches
        if n_gt > 0 and n_pred > 0:
            iou_matrix = compute_iou_matrix(gt_boxes, pred_boxes)
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_gt = set()
            matched_pred = set()

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= iou_thresh:
                    matched_gt.add(r)
                    matched_pred.add(c)
                    total_tp += 1

                    current_gt_id = gt_ids[r]
                    current_track_id = pred_ids[c]

                    # Checks if this ground truth vehicle switched to a different Track ID
                    if current_gt_id in gt_to_track_mapping:
                        if gt_to_track_mapping[current_gt_id] != current_track_id:
                            total_id_switches += 1
                            gt_to_track_mapping[current_gt_id] = current_track_id
                    else:
                        gt_to_track_mapping[current_gt_id] = current_track_id

            total_fp += (n_pred - len(matched_pred))
            total_fn += (n_gt - len(matched_gt))
        else:
            total_fp += n_pred
            total_fn += n_gt

    mae = float(np.mean(absolute_count_errors)) if absolute_count_errors else 0.0
    rmse = float(np.sqrt(np.mean(squared_count_errors))) if squared_count_errors else 0.0
    sum_abs_err = float(np.sum(absolute_count_errors)) if absolute_count_errors else 0.0

    counting_accuracy = max(0.0, (1.0 - sum_abs_err / max(1, total_gt_instances))) * 100.0
    counting_error = 100.0 - counting_accuracy
    false_count_rate = (total_fp / max(1, total_pred_instances)) * 100.0
    missed_vehicle_rate = (total_fn / max(1, total_gt_instances)) * 100.0
    duplicate_count_rate = (total_id_switches / max(1, unique_gt_count)) * 100.0
    avg_latency = float(np.mean(latency_records)) if latency_records else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "sequence": seq_name,
        "frames_evaluated": len(absolute_count_errors),
        "total_gt_instances": total_gt_instances,
        "total_pred_instances": total_pred_instances,
        "unique_gt_vehicles": unique_gt_count,
        "counting_accuracy_pct": round(counting_accuracy, 2),
        "counting_error_pct": round(counting_error, 2),
        "mae_per_frame": round(mae, 3),
        "rmse_per_frame": round(rmse, 3),
        "false_count_rate_pct": round(false_count_rate, 2),
        "missed_vehicle_rate_pct": round(missed_vehicle_rate, 2),
        "duplicate_count_rate_pct": round(duplicate_count_rate, 2),
        "id_switches": total_id_switches,
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 1),
    }


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: Ultralytics is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"Error: Model weights not found at: {weights_path}")
        sys.exit(1)

    data_root = Path(args.data_dir).resolve()
    xml_dir = data_root / "DETRAC-Test-Annotations-XML"
    images_root = data_root / "DETRAC-Images"

    if not xml_dir.exists() or not images_root.exists():
        print(f"Error: UA-DETRAC test directories not found under: {data_root}")
        print("Expected: DETRAC-Test-Annotations-XML and DETRAC-Images")
        sys.exit(1)

    xml_files = sorted(list(xml_dir.glob("*.xml")))
    if args.num_sequences is not None and args.num_sequences > 0:
        xml_files = xml_files[: args.num_sequences]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" ATCS Traffic Counting & Multi-Object Tracking Benchmark Suite")
    print("=" * 85)
    print(f" Model Weights:     {weights_path}")
    print(f" Tracker Config:    {args.tracker}")
    print(f" Test Sequences:    {len(xml_files)} sequences")
    print(f" Matching IoU:      {args.iou_thresh}")
    print(f" Confidence Thresh: {args.conf}")
    print(f" Resolution:        {args.imgsz}px")
    print(f" Execution Device:  {args.device.upper()}")
    print("=" * 85)

    model = YOLO(str(weights_path))

    sequence_results = []
    pbar = tqdm(xml_files, desc="Benchmarking Sequences")

    for xml_path in pbar:
        seq_name = xml_path.stem
        seq_img_dir = images_root / seq_name
        if not seq_img_dir.exists():
            continue

        pbar.set_postfix({"seq": seq_name})
        res = evaluate_sequence(
            model=model,
            tracker_cfg=args.tracker,
            xml_path=xml_path,
            img_dir=seq_img_dir,
            conf=args.conf,
            iou_thresh=args.iou_thresh,
            imgsz=args.imgsz,
            device=args.device,
        )
        sequence_results.append(res)

    if not sequence_results:
        print("Error: No test sequences were evaluated.")
        sys.exit(1)

    # Aggregates metrics across all evaluated sequences
    total_frames = sum(r["frames_evaluated"] for r in sequence_results)
    total_gt = sum(r["total_gt_instances"] for r in sequence_results)
    total_pred = sum(r["total_pred_instances"] for r in sequence_results)
    total_id_switches = sum(r["id_switches"] for r in sequence_results)
    total_unique_gt = sum(r["unique_gt_vehicles"] for r in sequence_results)

    avg_counting_acc = float(np.mean([r["counting_accuracy_pct"] for r in sequence_results]))
    avg_counting_err = float(np.mean([r["counting_error_pct"] for r in sequence_results]))
    avg_mae = float(np.mean([r["mae_per_frame"] for r in sequence_results]))
    avg_rmse = float(np.mean([r["rmse_per_frame"] for r in sequence_results]))
    avg_false_count_rate = float(np.mean([r["false_count_rate_pct"] for r in sequence_results]))
    avg_missed_rate = float(np.mean([r["missed_vehicle_rate_pct"] for r in sequence_results]))
    avg_dup_rate = (total_id_switches / max(1, total_unique_gt)) * 100.0
    overall_latency = float(np.mean([r["avg_latency_ms"] for r in sequence_results]))
    overall_fps = 1000.0 / overall_latency if overall_latency > 0 else 0.0

    summary_metrics = {
        "model": str(weights_path.name),
        "total_sequences": len(sequence_results),
        "total_frames_evaluated": total_frames,
        "total_ground_truth_vehicles": total_gt,
        "total_predicted_vehicles": total_pred,
        "counting_accuracy_pct": round(avg_counting_acc, 2),
        "counting_error_pct": round(avg_counting_err, 2),
        "mae_vehicle_count": round(avg_mae, 3),
        "rmse_vehicle_count": round(avg_rmse, 3),
        "false_count_rate_pct": round(avg_false_count_rate, 2),
        "missed_vehicle_rate_pct": round(avg_missed_rate, 2),
        "duplicate_count_rate_pct": round(avg_dup_rate, 2),
        "id_switches": total_id_switches,
        "average_latency_ms": round(overall_latency, 2),
        "fps": round(overall_fps, 1),
    }

    # Exports JSON and CSV benchmark logs
    json_path = out_dir / "counting_metrics.json"
    csv_path = out_dir / "per_sequence_summary.csv"

    with open(json_path, "w") as f:
        json.dump({"summary": summary_metrics, "sequences": sequence_results}, f, indent=2)

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sequence_results[0].keys()))
        writer.writeheader()
        writer.writerows(sequence_results)

    # Displays formatted benchmark report
    print("\n" + "=" * 85)
    print(" ATCS TRAFFIC COUNTING & TRACKING BENCHMARK REPORT")
    print("=" * 85)
    print(f" Evaluated Frames:           {total_frames:,} across {len(sequence_results)} sequences")
    print(f" Total Ground Truth Count:   {total_gt:,} vehicle instances")
    print(f" Total Predicted Count:      {total_pred:,} vehicle instances")
    print("-" * 85)
    print(f" Counting Accuracy:        {summary_metrics['counting_accuracy_pct']}%")
    print(f" Counting Error (MAPE):    {summary_metrics['counting_error_pct']}%")
    print(f" MAE (Vehicle Count):      {summary_metrics['mae_vehicle_count']} vehicles / frame")
    print(f" RMSE (Vehicle Count):     {summary_metrics['rmse_vehicle_count']} vehicles / frame")
    print(f" False Count Rate:         {summary_metrics['false_count_rate_pct']}%  (Ghost detections)")
    print(f" Missed Vehicle Rate:       {summary_metrics['missed_vehicle_rate_pct']}%  (Omissions)")
    print(f" Duplicate Count Rate:      {summary_metrics['duplicate_count_rate_pct']}%  (ID switches: {total_id_switches})")
    print(f" Per-Frame Latency:        {summary_metrics['average_latency_ms']} ms  ({summary_metrics['fps']} FPS)")
    print("=" * 85)
    print(f" Benchmark JSON: {json_path}")
    print(f" Sequence CSV:   {csv_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
