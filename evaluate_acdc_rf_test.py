import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

CLASS_ORDER = ["NOR", "MINF", "DCM", "HCM", "RV"]


def parse_info_cfg(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return data


def normalize_group(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().upper()
    aliases = {
        "NORMAL": "NOR",
        "MYOCARDIAL INFARCTION": "MINF",
        "PREVIOUS MYOCARDIAL INFARCTION": "MINF",
        "DILATED CARDIOMYOPATHY": "DCM",
        "HYPERTROPHIC CARDIOMYOPATHY": "HCM",
        "ABNORMAL RIGHT VENTRICLE": "RV",
    }
    value = aliases.get(value, value)
    return value if value in CLASS_ORDER else None


def find_info_cfg(testing_dir: Path, case_id: str) -> Optional[Path]:
    case_dir = testing_dir / case_id
    for name in ("Info.cfg", "info.cfg", "INFO.CFG"):
        candidate = case_dir / name
        if candidate.exists():
            return candidate
    if case_dir.exists():
        for candidate in case_dir.rglob("*"):
            if candidate.is_file() and candidate.name.lower() == "info.cfg":
                return candidate
    for candidate in testing_dir.rglob("*"):
        if (
            candidate.is_file()
            and candidate.name.lower() == "info.cfg"
            and candidate.parent.name.lower() == case_id.lower()
        ):
            return candidate
    return None


def find_prediction_json(prediction_dir: Path, case_id: str) -> Optional[Path]:
    for name in (
        f"{case_id}_rf_prediction.json",
        f"{case_id}_prediction.json",
        f"{case_id}.json",
    ):
        candidate = prediction_dir / name
        if candidate.exists():
            return candidate
    candidates = sorted(prediction_dir.glob(f"{case_id}*.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(prediction_dir.rglob(f"{case_id}*.json"))
    return candidates[0] if candidates else None


def extract_prediction(
    obj: Dict[str, Any],
) -> Tuple[Optional[str], Optional[float], Dict[str, Optional[float]]]:
    payload = obj.get("classification_prediction")
    if not isinstance(payload, dict):
        payload = obj

    predicted_group = normalize_group(payload.get("predicted_group"))

    confidence = payload.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    raw_probs = payload.get("class_probabilities", {})
    if not isinstance(raw_probs, dict):
        raw_probs = {}

    probabilities: Dict[str, Optional[float]] = {}
    for label in CLASS_ORDER:
        value = raw_probs.get(label)
        try:
            probabilities[label] = float(value) if value is not None else None
        except (TypeError, ValueError):
            probabilities[label] = None

    if confidence is None and predicted_group:
        confidence = probabilities.get(predicted_group)

    return predicted_group, confidence, probabilities


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    prediction_dir: Path,
    testing_dir: Path,
    out_dir: Path,
    start: int,
    end: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []

    for number in range(start, end + 1):
        case_id = f"patient{number:03d}"
        pred_path = find_prediction_json(prediction_dir, case_id)
        info_path = find_info_cfg(testing_dir, case_id)

        row: Dict[str, Any] = {
            "case_id": case_id,
            "true_group": "",
            "predicted_group": "",
            "correct": "",
            "confidence": "",
            "prob_NOR": "",
            "prob_MINF": "",
            "prob_DCM": "",
            "prob_HCM": "",
            "prob_RV": "",
            "status": "ok",
            "prediction_path": str(pred_path) if pred_path else "",
            "info_cfg_path": str(info_path) if info_path else "",
        }

        problems: List[str] = []
        true_group: Optional[str] = None
        predicted_group: Optional[str] = None

        if info_path is None:
            problems.append("info_cfg_not_found")
        else:
            try:
                info = parse_info_cfg(info_path)
                true_group = normalize_group(info.get("Group"))
                row["true_group"] = true_group or ""
                if true_group is None:
                    problems.append(f"invalid_or_missing_group:{info.get('Group')}")
            except Exception as exc:
                problems.append(f"info_cfg_read_error:{exc}")

        if pred_path is None:
            problems.append("prediction_json_not_found")
        else:
            try:
                with pred_path.open("r", encoding="utf-8-sig") as f:
                    prediction_obj = json.load(f)
                predicted_group, confidence, probabilities = extract_prediction(
                    prediction_obj
                )
                row["predicted_group"] = predicted_group or ""
                row["confidence"] = confidence if confidence is not None else ""
                for label in CLASS_ORDER:
                    value = probabilities.get(label)
                    row[f"prob_{label}"] = value if value is not None else ""
                if predicted_group is None:
                    problems.append("invalid_or_missing_predicted_group")
            except Exception as exc:
                problems.append(f"prediction_json_read_error:{exc}")

        if problems:
            row["status"] = ";".join(problems)
            invalid_rows.append(row)
        else:
            row["correct"] = int(true_group == predicted_group)
            valid_rows.append(row)

        rows.append(row)

    fields = [
        "case_id",
        "true_group",
        "predicted_group",
        "correct",
        "confidence",
        "prob_NOR",
        "prob_MINF",
        "prob_DCM",
        "prob_HCM",
        "prob_RV",
        "status",
        "prediction_path",
        "info_cfg_path",
    ]
    write_csv(out_dir / "test_comparison.csv", rows, fields)
    if invalid_rows:
        write_csv(out_dir / "missing_or_invalid_cases.csv", invalid_rows, fields)

    if not valid_rows:
        raise RuntimeError(
            "No valid case pairs found. Check the folder paths and JSON/Info.cfg format."
        )

    y_true = [row["true_group"] for row in valid_rows]
    y_pred = [row["predicted_group"] for row in valid_rows]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)

    metrics = {
        "range": {"start": start, "end": end, "expected_cases": end - start + 1},
        "valid_cases": len(valid_rows),
        "missing_or_invalid_cases": len(invalid_rows),
        "correct_predictions": int(sum(row["correct"] for row in valid_rows)),
        "incorrect_predictions": int(
            len(valid_rows) - sum(row["correct"] for row in valid_rows)
        ),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "true_class_counts": {
            label: int(sum(value == label for value in y_true))
            for label in CLASS_ORDER
        },
        "predicted_class_counts": {
            label: int(sum(value == label for value in y_pred))
            for label in CLASS_ORDER
        },
    }
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (out_dir / "test_classification_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    confusion_rows = []
    for i, true_label in enumerate(CLASS_ORDER):
        confusion_rows.append(
            {
                "true_group": true_label,
                **{
                    f"pred_{pred_label}": int(matrix[i, j])
                    for j, pred_label in enumerate(CLASS_ORDER)
                },
            }
        )
    write_csv(
        out_dir / "test_confusion_matrix.csv",
        confusion_rows,
        ["true_group"] + [f"pred_{label}" for label in CLASS_ORDER],
    )

    per_class_rows = []
    for label in CLASS_ORDER:
        values = report.get(label, {})
        per_class_rows.append(
            {
                "group": label,
                "precision": values.get("precision", 0.0),
                "recall": values.get("recall", 0.0),
                "f1_score": values.get("f1-score", 0.0),
                "support": int(values.get("support", 0)),
            }
        )
    write_csv(
        out_dir / "test_per_class_metrics.csv",
        per_class_rows,
        ["group", "precision", "recall", "f1_score", "support"],
    )

    misclassified = [row for row in valid_rows if row["correct"] == 0]
    write_csv(out_dir / "test_misclassified_cases.csv", misclassified, fields)

    print("=" * 70)
    print("ACDC Random Forest test evaluation")
    print("=" * 70)
    print(f"Expected cases        : {end - start + 1}")
    print(f"Valid cases           : {len(valid_rows)}")
    print(f"Missing/invalid cases : {len(invalid_rows)}")
    print(f"Correct predictions   : {sum(row['correct'] for row in valid_rows)}")
    print(f"Incorrect predictions : {len(misclassified)}")
    print("-" * 70)
    print(f"Accuracy              : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Balanced accuracy     : {balanced_accuracy:.4f}")
    print(f"Macro precision       : {macro_precision:.4f}")
    print(f"Macro recall          : {macro_recall:.4f}")
    print(f"Macro F1              : {macro_f1:.4f}")
    print("-" * 70)
    for label in CLASS_ORDER:
        values = report.get(label, {})
        print(
            f"{label:4s} | precision={values.get('precision', 0.0):.4f} "
            f"recall={values.get('recall', 0.0):.4f} "
            f"f1={values.get('f1-score', 0.0):.4f} "
            f"support={int(values.get('support', 0))}"
        )
    print("-" * 70)
    print(f"Saved evaluation results to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ACDC RF predictions against Info.cfg Group labels."
    )
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--testing_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--start", type=int, default=101)
    parser.add_argument("--end", type=int, default=150)
    args = parser.parse_args()

    if args.start > args.end:
        raise ValueError("--start must be <= --end")

    evaluate(
        prediction_dir=Path(args.prediction_dir),
        testing_dir=Path(args.testing_dir),
        out_dir=Path(args.out_dir),
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
