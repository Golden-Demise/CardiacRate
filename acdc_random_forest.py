from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


# Fixed display order used in reports and confusion matrices.
CLASS_NAMES: List[str] = ["NOR", "MINF", "DCM", "HCM", "RV"]

# These features are intentionally derived only from measurable facts.
# evaluation_metadata.ground_truth_group is used only as the training target.
FEATURE_COLUMNS: List[str] = [
    "lv_edv_index_ml_m2",
    "lv_esv_index_ml_m2",
    "lv_ejection_fraction_percent",
    "lv_stroke_volume_index_ml_m2",
    "rv_edv_index_ml_m2",
    "rv_esv_index_ml_m2",
    "rv_ejection_fraction_percent",
    "rv_stroke_volume_index_ml_m2",
    "lv_myocardial_mass_index_g_m2",
    "approximate_max_thickness_ed_mm",
    "rv_lv_edv_ratio",
    "rv_lv_esv_ratio",
]

IDENTIFIER_COLUMNS: List[str] = ["case_id", "patient_id", "facts_path"]
TARGET_COLUMN = "group"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return obj


def nested_get(obj: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    current: Any = obj
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    result = numerator / denominator
    return float(result) if np.isfinite(result) else None


def normalize_group(value: Any) -> Optional[str]:
    if value is None:
        return None
    group = str(value).strip().upper()
    aliases = {
        "NORMAL": "NOR",
        "HEALTHY": "NOR",
        "MYOCARDIAL INFARCTION": "MINF",
        "PREVIOUS MYOCARDIAL INFARCTION": "MINF",
        "DILATED CARDIOMYOPATHY": "DCM",
        "HYPERTROPHIC CARDIOMYOPATHY": "HCM",
        "ABNORMAL RIGHT VENTRICLE": "RV",
    }
    group = aliases.get(group, group)
    return group if group in CLASS_NAMES else None


def extract_features(facts: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Extract the first-version ACDC Random Forest features from one facts JSON."""
    lv = nested_get(facts, ["cardiac_function", "left_ventricle"]) or {}
    rv = nested_get(facts, ["cardiac_function", "right_ventricle"]) or {}
    metadata = nested_get(facts, ["metadata"]) or {}
    myo = nested_get(facts, ["myocardial_measurements"]) or {}

    bsa = as_float(metadata.get("bsa_m2_mosteller"))

    lv_edv = as_float(lv.get("edv_ml"))
    lv_esv = as_float(lv.get("esv_ml"))
    lv_sv = as_float(lv.get("stroke_volume_ml"))
    rv_edv = as_float(rv.get("edv_ml"))
    rv_esv = as_float(rv.get("esv_ml"))
    rv_sv = as_float(rv.get("stroke_volume_ml"))
    mass_g = as_float(myo.get("estimated_lv_myocardial_mass_g"))

    # Prefer values already written by make_facts_acdc.py; derive them only if absent.
    lv_edvi = as_float(lv.get("edv_index_ml_m2"))
    lv_esvi = as_float(lv.get("esv_index_ml_m2"))
    rv_edvi = as_float(rv.get("edv_index_ml_m2"))
    rv_esvi = as_float(rv.get("esv_index_ml_m2"))

    if lv_edvi is None:
        lv_edvi = safe_div(lv_edv, bsa)
    if lv_esvi is None:
        lv_esvi = safe_div(lv_esv, bsa)
    if rv_edvi is None:
        rv_edvi = safe_div(rv_edv, bsa)
    if rv_esvi is None:
        rv_esvi = safe_div(rv_esv, bsa)

    values: Dict[str, Optional[float]] = {
        "lv_edv_index_ml_m2": lv_edvi,
        "lv_esv_index_ml_m2": lv_esvi,
        "lv_ejection_fraction_percent": as_float(lv.get("ejection_fraction_percent")),
        "lv_stroke_volume_index_ml_m2": safe_div(lv_sv, bsa),
        "rv_edv_index_ml_m2": rv_edvi,
        "rv_esv_index_ml_m2": rv_esvi,
        "rv_ejection_fraction_percent": as_float(rv.get("ejection_fraction_percent")),
        "rv_stroke_volume_index_ml_m2": safe_div(rv_sv, bsa),
        "lv_myocardial_mass_index_g_m2": safe_div(mass_g, bsa),
        "approximate_max_thickness_ed_mm": as_float(
            myo.get("approximate_max_thickness_ed_mm")
        ),
        "rv_lv_edv_ratio": safe_div(rv_edv, lv_edv),
        "rv_lv_esv_ratio": safe_div(rv_esv, lv_esv),
    }

    return values


def extract_training_row(facts: Mapping[str, Any], facts_path: Path) -> Dict[str, Any]:
    dataset = str(facts.get("dataset", "")).strip().upper()
    modality = str(facts.get("modality", "")).strip().lower()
    if dataset and dataset != "ACDC":
        raise ValueError(f"Not an ACDC facts file: dataset={dataset}")
    if modality and modality not in {"cine_mri", "cine-mri", "mri"}:
        raise ValueError(f"Unexpected modality for ACDC: {modality}")

    case_id = str(facts.get("case_id") or facts.get("patient_id") or facts_path.stem)
    patient_id = str(facts.get("patient_id") or case_id)
    group = normalize_group(
        nested_get(facts, ["evaluation_metadata", "ground_truth_group"])
    )

    row: Dict[str, Any] = {
        "case_id": case_id,
        "patient_id": patient_id,
        "facts_path": str(facts_path.resolve()),
        **extract_features(facts),
        "group": group,
    }
    return row


def find_acdc_facts(facts_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in sorted(facts_dir.rglob("*.json")):
        try:
            obj = load_json(path)
        except Exception:
            continue
        if str(obj.get("dataset", "")).strip().upper() == "ACDC" or str(
            obj.get("schema_version", "")
        ).startswith("acdc_facts"):
            candidates.append(path)
    return candidates


def build_dataset(facts_dir: Path, out_csv: Path, strict: bool = False) -> pd.DataFrame:
    paths = find_acdc_facts(facts_dir)
    if not paths:
        raise FileNotFoundError(f"No ACDC facts JSON files found under: {facts_dir}")

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for path in paths:
        try:
            row = extract_training_row(load_json(path), path)
            if row[TARGET_COLUMN] is None:
                raise ValueError(
                    "Missing/invalid evaluation_metadata.ground_truth_group"
                )
            rows.append(row)
        except Exception as exc:
            message = f"{path}: {exc}"
            if strict:
                raise RuntimeError(message) from exc
            skipped.append(message)

    if not rows:
        raise RuntimeError("No usable labeled ACDC facts files were found.")

    df = pd.DataFrame(rows)
    ordered_columns = IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df.reindex(columns=ordered_columns)

    duplicate_mask = df.duplicated(subset=["patient_id"], keep=False)
    if duplicate_mask.any():
        duplicates = sorted(df.loc[duplicate_mask, "patient_id"].astype(str).unique())
        raise ValueError(
            "Duplicate patient_id values were found. Keep one facts JSON per patient "
            f"for this training run: {duplicates}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Saved feature dataset: {out_csv}")
    print(f"Usable patients: {len(df)}")
    print("Class counts:")
    print(df[TARGET_COLUMN].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())
    print("Missing values per feature:")
    print(df[FEATURE_COLUMNS].isna().sum().to_string())
    if skipped:
        print(f"Skipped files: {len(skipped)}")
        for message in skipped[:20]:
            print(f"  - {message}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
    return df


def make_pipeline(
    n_estimators: int,
    random_state: int,
    min_samples_leaf: int,
    max_depth: Optional[int],
    n_jobs: int,
) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("classifier", classifier),
        ]
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    return value


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(obj), f, ensure_ascii=False, indent=2)


def train_model(
    csv_path: Path,
    out_dir: Path,
    n_splits: int,
    n_estimators: int,
    random_state: int,
    min_samples_leaf: int,
    max_depth: Optional[int],
    n_jobs: int,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    missing_columns = [
        c for c in IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(normalize_group)
    if df[TARGET_COLUMN].isna().any():
        bad_rows = df.index[df[TARGET_COLUMN].isna()].tolist()
        raise ValueError(f"Invalid or missing group labels at rows: {bad_rows}")

    class_counts = Counter(df[TARGET_COLUMN].tolist())
    missing_classes = [name for name in CLASS_NAMES if class_counts.get(name, 0) == 0]
    if missing_classes:
        raise ValueError(
            "A five-class model requires at least one patient from every class. "
            f"Missing classes: {missing_classes}"
        )

    minimum_class_count = min(class_counts.values())
    if minimum_class_count < 2:
        raise ValueError(
            "Each class needs at least 2 patients for stratified cross-validation. "
            f"Class counts: {dict(class_counts)}"
        )

    effective_splits = min(n_splits, minimum_class_count)
    if effective_splits < n_splits:
        print(
            f"Warning: requested {n_splits} folds, but the smallest class has "
            f"{minimum_class_count} patients. Using {effective_splits} folds."
        )

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET_COLUMN].astype(str).to_numpy()

    base_pipeline = make_pipeline(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        n_jobs=n_jobs,
    )
    cv = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )

    n_samples = len(df)
    oof_pred = np.empty(n_samples, dtype=object)
    oof_proba = np.zeros((n_samples, len(CLASS_NAMES)), dtype=float)
    fold_ids = np.zeros(n_samples, dtype=int)
    fold_metrics: List[Dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        model = clone(base_pipeline)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        proba = model.predict_proba(X.iloc[test_idx])
        model_classes = list(model.named_steps["classifier"].classes_)

        aligned = np.zeros((len(test_idx), len(CLASS_NAMES)), dtype=float)
        for source_col, class_name in enumerate(model_classes):
            aligned[:, CLASS_NAMES.index(class_name)] = proba[:, source_col]

        oof_pred[test_idx] = pred
        oof_proba[test_idx] = aligned
        fold_ids[test_idx] = fold

        fold_record = {
            "fold": fold,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "accuracy": accuracy_score(y[test_idx], pred),
            "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
            "macro_f1": f1_score(y[test_idx], pred, average="macro", zero_division=0),
        }
        fold_metrics.append(fold_record)
        print(
            f"Fold {fold}/{effective_splits}: "
            f"accuracy={fold_record['accuracy']:.4f}, "
            f"macro_f1={fold_record['macro_f1']:.4f}"
        )

    overall_metrics: Dict[str, Any] = {
        "n_patients": n_samples,
        "n_splits": effective_splits,
        "class_counts": {name: int(class_counts.get(name, 0)) for name in CLASS_NAMES},
        "accuracy": accuracy_score(y, oof_pred),
        "balanced_accuracy": balanced_accuracy_score(y, oof_pred),
        "macro_precision": precision_score(
            y, oof_pred, labels=CLASS_NAMES, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y, oof_pred, labels=CLASS_NAMES, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y, oof_pred, labels=CLASS_NAMES, average="macro", zero_division=0
        ),
        "fold_metrics": fold_metrics,
    }

    report = classification_report(
        y,
        oof_pred,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y, oof_pred, labels=CLASS_NAMES)

    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = df[IDENTIFIER_COLUMNS + [TARGET_COLUMN]].copy()
    prediction_df["cv_fold"] = fold_ids
    prediction_df["predicted_group"] = oof_pred
    prediction_df["correct"] = prediction_df[TARGET_COLUMN] == prediction_df["predicted_group"]
    for idx, class_name in enumerate(CLASS_NAMES):
        prediction_df[f"prob_{class_name}"] = oof_proba[:, idx]
    prediction_df.to_csv(
        out_dir / "cv_predictions.csv", index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    save_json(out_dir / "classification_report.json", report)
    save_json(out_dir / "metrics.json", overall_metrics)

    final_model = clone(base_pipeline)
    final_model.fit(X, y)

    imputer = final_model.named_steps["imputer"]
    classifier = final_model.named_steps["classifier"]
    transformed_names = list(imputer.get_feature_names_out(FEATURE_COLUMNS))
    importance_df = pd.DataFrame(
        {
            "feature": transformed_names,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(
        out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    model_metadata: Dict[str, Any] = {
        "model_type": "RandomForestClassifier",
        "task": "ACDC five-class prediction",
        "classes": CLASS_NAMES,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "training_csv": str(csv_path.resolve()),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_patient_count": n_samples,
        "class_counts": overall_metrics["class_counts"],
        "cross_validation": {
            "type": "StratifiedKFold",
            "n_splits": effective_splits,
            "shuffle": True,
            "random_state": random_state,
        },
        "random_forest_parameters": classifier.get_params(),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "This first-version model uses global ED/ES measurements only.",
            "Regional LV contraction features are not included, so DCM and MINF may be difficult to distinguish.",
            "Cross-validation estimates classification performance; the final saved model is fitted on all rows in the CSV.",
            "Do not evaluate a patient with a model that was trained on that same patient.",
        ],
    }

    bundle = {
        "pipeline": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "class_names": CLASS_NAMES,
        "metadata": model_metadata,
    }
    model_path = out_dir / "acdc_random_forest.joblib"
    joblib.dump(bundle, model_path)
    save_json(out_dir / "model_info.json", model_metadata)

    print("\nOut-of-fold evaluation")
    print(f"Accuracy:          {overall_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {overall_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {overall_metrics['macro_f1']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved reports: {out_dir}")
    return overall_metrics


def predict_one(
    model_path: Path,
    facts_path: Path,
    out_json: Optional[Path],
) -> Dict[str, Any]:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Unsupported model file: expected the bundle created by this script.")

    pipeline: Pipeline = bundle["pipeline"]
    feature_columns: List[str] = list(bundle.get("feature_columns", FEATURE_COLUMNS))
    metadata: Dict[str, Any] = dict(bundle.get("metadata", {}))

    facts = load_json(facts_path)
    features = extract_features(facts)
    X = pd.DataFrame([{name: features.get(name) for name in feature_columns}])
    X = X.apply(pd.to_numeric, errors="coerce")

    predicted_group = str(pipeline.predict(X)[0])
    probabilities = pipeline.predict_proba(X)[0]
    model_classes = list(pipeline.named_steps["classifier"].classes_)
    probability_map = {
        class_name: float(probabilities[model_classes.index(class_name)])
        if class_name in model_classes
        else 0.0
        for class_name in CLASS_NAMES
    }

    importance = pipeline.named_steps["classifier"].feature_importances_
    transformed_names = list(
        pipeline.named_steps["imputer"].get_feature_names_out(feature_columns)
    )
    top_global_features = [
        {"feature": name, "importance": float(score)}
        for name, score in sorted(
            zip(transformed_names, importance), key=lambda item: item[1], reverse=True
        )[:5]
    ]

    result: Dict[str, Any] = {
        "case_id": facts.get("case_id") or facts.get("patient_id") or facts_path.stem,
        "model": "ACDC first-version Random Forest",
        "predicted_group": predicted_group,
        "confidence": probability_map[predicted_group],
        "class_probabilities": probability_map,
        "input_features": {
            key: (None if pd.isna(X.iloc[0][key]) else float(X.iloc[0][key]))
            for key in feature_columns
        },
        "missing_features": [key for key in feature_columns if pd.isna(X.iloc[0][key])],
        "top_global_model_features": top_global_features,
        "model_training_patient_count": metadata.get("training_patient_count"),
        "limitations": metadata.get("limitations", []),
    }

    if out_json is not None:
        save_json(out_json, result)
        print(f"Saved prediction: {out_json}")
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "First-version ACDC five-class Random Forest pipeline. "
            "Build a feature CSV, train/evaluate the classifier, or predict one facts JSON."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a labeled feature CSV")
    build_parser.add_argument("--facts_dir", required=True, type=Path)
    build_parser.add_argument("--out_csv", required=True, type=Path)
    build_parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first unusable JSON instead of skipping it.",
    )

    train_parser = subparsers.add_parser("train", help="Train and evaluate the Random Forest")
    train_parser.add_argument("--csv", required=True, type=Path)
    train_parser.add_argument("--out_dir", required=True, type=Path)
    train_parser.add_argument("--n_splits", type=int, default=5)
    train_parser.add_argument("--n_estimators", type=int, default=500)
    train_parser.add_argument("--random_state", type=int, default=42)
    train_parser.add_argument("--min_samples_leaf", type=int, default=2)
    train_parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Default None lets each tree grow until other stopping criteria apply.",
    )
    train_parser.add_argument("--n_jobs", type=int, default=-1)

    predict_parser = subparsers.add_parser("predict", help="Predict one ACDC facts JSON")
    predict_parser.add_argument("--model", required=True, type=Path)
    predict_parser.add_argument("--facts_path", required=True, type=Path)
    predict_parser.add_argument("--out_json", type=Path, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        build_dataset(args.facts_dir, args.out_csv, strict=args.strict)
    elif args.command == "train":
        train_model(
            csv_path=args.csv,
            out_dir=args.out_dir,
            n_splits=args.n_splits,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            min_samples_leaf=args.min_samples_leaf,
            max_depth=args.max_depth,
            n_jobs=args.n_jobs,
        )
    elif args.command == "predict":
        predict_one(args.model, args.facts_path, args.out_json)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
