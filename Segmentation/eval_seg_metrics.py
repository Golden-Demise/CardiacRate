r"""
eval_seg_metrics.py

Evaluate cardiac CT segmentation predictions against ground-truth labels.

Default labels:
  1 = myocardium
  2 = aortic_valve
  3 = calcification

Outputs:
  1) per-case CSV: Dice and HD95 for each class + calcification detection result
  2) summary CSV: mean metrics + calcification detection confusion matrix

Example:
python eval_seg_metrics.py ^
  --pred_dir D:\CardiacRate\dataset\predict ^
  --gt_dir D:\CardiacRate\dataset\label ^
  --out_csv D:\CardiacRate\dataset\eval_seg_metrics.csv ^
  --summary_csv D:\CardiacRate\dataset\eval_seg_summary.csv ^
  --calci_min_vox 20
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


DEFAULT_CLASSES = {
    1: "myocardium",
    2: "aortic_valve",
    3: "calcification",
}

KNOWN_SUFFIXES = [
    "_predict",
    "_pred",
    "_seg",
    "_label",
    "_labels",
    "_gt",
    "_mask",
    "_masks",
]


def strip_nii_suffix(path: Path) -> str:
    """Return file name without .nii or .nii.gz."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def normalize_case_id(path: Path) -> str:
    """Normalize common CT/label/prediction filenames to the same case id."""
    stem = strip_nii_suffix(path)
    for suffix in KNOWN_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def list_nifti_files(folder: Path, recursive: bool = False) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    files = []
    for p in folder.glob(pattern):
        if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii")):
            files.append(p)
    return sorted(files)


def build_case_map(folder: Path, recursive: bool = False) -> Dict[str, Path]:
    case_map: Dict[str, Path] = {}
    for p in list_nifti_files(folder, recursive=recursive):
        case_id = normalize_case_id(p)
        if case_id in case_map:
            print(f"[WARN] duplicate case id '{case_id}', keep first: {case_map[case_id]}, ignore: {p}")
            continue
        case_map[case_id] = p
    return case_map


def parse_classes(text: str) -> Dict[int, str]:
    """
    Parse class mapping string, e.g.
      "1:myocardium,2:aortic_valve,3:calcification"
    """
    mapping: Dict[int, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid class mapping item: {item}. Expected format label:name")
        label_text, name = item.split(":", 1)
        mapping[int(label_text.strip())] = name.strip()
    if not mapping:
        raise ValueError("No valid classes were provided.")
    return mapping


def load_label(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    arr = np.rint(arr).astype(np.int16)
    zooms = img.header.get_zooms()[:3]
    spacing = tuple(float(z) for z in zooms)
    return arr, spacing


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    denom = pred_sum + gt_sum
    if denom == 0:
        return float("nan")
    inter = int(np.logical_and(pred, gt).sum())
    return 2.0 * inter / denom


def get_surface(mask: np.ndarray) -> np.ndarray:
    """Return binary surface voxels. For tiny objects, the object itself is treated as surface."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    surface = np.logical_xor(mask, eroded)
    if not surface.any():
        surface = mask.astype(bool)
    return surface


def hd95_score(pred: np.ndarray, gt: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    """
    Compute symmetric 95th percentile Hausdorff distance in mm.

    Return policy:
      - both empty: 0.0
      - one empty : NaN, because HD95 is undefined when only one mask exists
    """
    pred_any = bool(pred.any())
    gt_any = bool(gt.any())
    if not pred_any and not gt_any:
        return 0.0
    if pred_any != gt_any:
        return float("nan")

    pred_surface = get_surface(pred)
    gt_surface = get_surface(gt)

    # Distance transform is computed on the inverse surface.
    # At a surface voxel, distance is 0; elsewhere, it is distance to nearest surface voxel.
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing)

    distances_gt_to_pred = dt_to_pred[gt_surface]
    distances_pred_to_gt = dt_to_gt[pred_surface]
    distances = np.concatenate([distances_gt_to_pred, distances_pred_to_gt])

    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def safe_float(value: float) -> str:
    if value is None:
        return ""
    try:
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf"
    except TypeError:
        pass
    return f"{value:.6f}"


def mean_ignore_nan(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def detection_label(gt_has: bool, pred_has: bool) -> str:
    if gt_has and pred_has:
        return "TP"
    if (not gt_has) and pred_has:
        return "FP"
    if gt_has and (not pred_has):
        return "FN"
    return "TN"


def evaluate_case(
    case_id: str,
    pred_path: Path,
    gt_path: Path,
    classes: Dict[int, str],
    calci_label: int,
    calci_min_vox: int,
) -> Dict[str, object]:
    pred, pred_spacing = load_label(pred_path)
    gt, gt_spacing = load_label(gt_path)

    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, gt={gt.shape}")

    # Usually pred and GT should share spacing. HD95 uses GT spacing as reference.
    spacing = gt_spacing
    if any(abs(a - b) > 1e-5 for a, b in zip(pred_spacing, gt_spacing)):
        print(f"[WARN] {case_id}: spacing mismatch, use GT spacing. pred={pred_spacing}, gt={gt_spacing}")

    row: Dict[str, object] = {
        "case_id": case_id,
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "shape": "x".join(map(str, gt.shape)),
        "spacing_mm": "x".join(f"{s:.6g}" for s in spacing),
    }

    for label, name in classes.items():
        pred_mask = pred == label
        gt_mask = gt == label
        row[f"{name}_gt_voxels"] = int(gt_mask.sum())
        row[f"{name}_pred_voxels"] = int(pred_mask.sum())
        row[f"{name}_dice"] = dice_score(pred_mask, gt_mask)
        row[f"{name}_hd95_mm"] = hd95_score(pred_mask, gt_mask, spacing)

    gt_calci_vox = int((gt == calci_label).sum())
    pred_calci_vox = int((pred == calci_label).sum())
    gt_has_calci = gt_calci_vox >= calci_min_vox
    pred_has_calci = pred_calci_vox >= calci_min_vox

    row["calci_min_vox"] = calci_min_vox
    row["gt_has_calcification"] = int(gt_has_calci)
    row["pred_has_calcification"] = int(pred_has_calci)
    row["calcification_detection"] = detection_label(gt_has_calci, pred_has_calci)

    return row


def write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, float):
                    cleaned[k] = safe_float(v)
                else:
                    cleaned[k] = v
            writer.writerow(cleaned)


def build_summary(rows: List[Dict[str, object]], classes: Dict[int, str]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []

    for _, name in classes.items():
        dice_key = f"{name}_dice"
        hd95_key = f"{name}_hd95_mm"
        gt_vox_key = f"{name}_gt_voxels"

        all_dice = [float(r[dice_key]) for r in rows]
        all_hd95 = [float(r[hd95_key]) for r in rows]
        fg_rows = [r for r in rows if int(r.get(gt_vox_key, 0)) > 0]

        summary.append({
            "metric_group": name,
            "num_cases_all": len(rows),
            "num_cases_gt_positive": len(fg_rows),
            "mean_dice_all_cases": mean_ignore_nan(all_dice),
            "mean_hd95_mm_all_cases": mean_ignore_nan(all_hd95),
            "mean_dice_gt_positive": mean_ignore_nan(float(r[dice_key]) for r in fg_rows),
            "mean_hd95_mm_gt_positive": mean_ignore_nan(float(r[hd95_key]) for r in fg_rows),
        })

    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for r in rows:
        counts[str(r["calcification_detection"])] += 1

    tp, fp, tn, fn = counts["TP"], counts["FP"], counts["TN"], counts["FN"]
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    accuracy = (tp + tn) / (tp + fp + tn + fn) if rows else float("nan")

    summary.append({
        "metric_group": "calcification_detection",
        "num_cases_all": len(rows),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
    })

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation Dice, HD95, and calcification detection.")
    parser.add_argument("--pred_dir", required=True, type=Path, help="Folder containing prediction .nii/.nii.gz files.")
    parser.add_argument("--gt_dir", required=True, type=Path, help="Folder containing ground-truth .nii/.nii.gz files.")
    parser.add_argument("--out_csv", default=Path("eval_seg_metrics.csv"), type=Path, help="Output per-case CSV.")
    parser.add_argument("--summary_csv", default=Path("eval_seg_summary.csv"), type=Path, help="Output summary CSV.")
    parser.add_argument("--classes", default="1:myocardium,2:aortic_valve,3:calcification", help="Class mapping, e.g. '1:myocardium,2:aortic_valve,3:calcification'.")
    parser.add_argument("--calci_label", default=3, type=int, help="Label id for calcification.")
    parser.add_argument("--calci_min_vox", default=1, type=int, help="Minimum voxels to count as having calcification.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search prediction and GT folders.")
    parser.add_argument("--allow_missing", action="store_true", help="Skip cases without matching prediction/GT instead of stopping.")
    args = parser.parse_args()

    classes = parse_classes(args.classes)

    pred_map = build_case_map(args.pred_dir, recursive=args.recursive)
    gt_map = build_case_map(args.gt_dir, recursive=args.recursive)

    pred_cases = set(pred_map.keys())
    gt_cases = set(gt_map.keys())
    common_cases = sorted(pred_cases & gt_cases)
    missing_pred = sorted(gt_cases - pred_cases)
    missing_gt = sorted(pred_cases - gt_cases)

    print("=" * 80)
    print("Segmentation evaluation")
    print("=" * 80)
    print(f"Prediction folder : {args.pred_dir}")
    print(f"GT folder         : {args.gt_dir}")
    print(f"Prediction files  : {len(pred_map)}")
    print(f"GT files          : {len(gt_map)}")
    print(f"Matched cases     : {len(common_cases)}")

    if missing_pred:
        print(f"[WARN] GT cases without prediction: {len(missing_pred)}")
        print("       " + ", ".join(missing_pred[:20]) + (" ..." if len(missing_pred) > 20 else ""))
    if missing_gt:
        print(f"[WARN] Prediction cases without GT: {len(missing_gt)}")
        print("       " + ", ".join(missing_gt[:20]) + (" ..." if len(missing_gt) > 20 else ""))

    if not args.allow_missing and (missing_pred or missing_gt):
        raise SystemExit("Missing matched files. Use --allow_missing to evaluate matched cases only.")
    if not common_cases:
        raise SystemExit("No matched cases found. Please check file names and folders.")

    rows: List[Dict[str, object]] = []
    for i, case_id in enumerate(common_cases, start=1):
        print(f"[{i}/{len(common_cases)}] {case_id}")
        try:
            row = evaluate_case(
                case_id=case_id,
                pred_path=pred_map[case_id],
                gt_path=gt_map[case_id],
                classes=classes,
                calci_label=args.calci_label,
                calci_min_vox=args.calci_min_vox,
            )
            rows.append(row)
        except Exception as e:
            if args.allow_missing:
                print(f"[ERROR] skip {case_id}: {e}")
                continue
            raise

    write_csv(rows, args.out_csv)
    summary_rows = build_summary(rows, classes)
    write_csv(summary_rows, args.summary_csv)

    print("=" * 80)
    print("Done")
    print(f"Per-case CSV : {args.out_csv}")
    print(f"Summary CSV  : {args.summary_csv}")
    print("=" * 80)

    # Console summary
    for row in summary_rows:
        group = row.get("metric_group", "")
        if group == "calcification_detection":
            print(
                f"{group}: TP={row.get('TP')}, FP={row.get('FP')}, TN={row.get('TN')}, FN={row.get('FN')}, "
                f"accuracy={safe_float(float(row.get('accuracy', float('nan'))))}, "
                f"sensitivity={safe_float(float(row.get('sensitivity_recall', float('nan'))))}, "
                f"specificity={safe_float(float(row.get('specificity', float('nan'))))}"
            )
        else:
            print(
                f"{group}: mean Dice(all)={safe_float(float(row.get('mean_dice_all_cases', float('nan'))))}, "
                f"mean HD95(all)={safe_float(float(row.get('mean_hd95_mm_all_cases', float('nan'))))} mm, "
                f"mean Dice(GT+)={safe_float(float(row.get('mean_dice_gt_positive', float('nan'))))}, "
                f"mean HD95(GT+)={safe_float(float(row.get('mean_hd95_mm_gt_positive', float('nan'))))} mm"
            )


if __name__ == "__main__":
    main()
