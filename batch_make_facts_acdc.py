from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_info_cfg(path: Path) -> Dict[str, str]:
    raw: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            raw[key.strip()] = value.strip()
    for key in ("ED", "ES"):
        if key not in raw:
            raise ValueError(f"{path} is missing {key}")
    return raw


def find_info_cfg(patient_dir: Path) -> Path:
    candidates = [p for p in patient_dir.iterdir() if p.is_file() and p.name.lower() == "info.cfg"]
    if not candidates:
        raise FileNotFoundError(f"Info.cfg not found in {patient_dir}")
    return candidates[0]


def find_frame_file(
    patient_dir: Path,
    case_id: str,
    frame: int,
    ground_truth: bool,
) -> Path:
    suffix = "_gt" if ground_truth else ""
    expected_names = [
        f"{case_id}_frame{frame:02d}{suffix}.nii.gz",
        f"{case_id}_frame{frame:02d}{suffix}.nii",
        f"{case_id}_frame{frame}{suffix}.nii.gz",
        f"{case_id}_frame{frame}{suffix}.nii",
    ]
    by_lower_name = {p.name.lower(): p for p in patient_dir.iterdir() if p.is_file()}
    for name in expected_names:
        found = by_lower_name.get(name.lower())
        if found is not None:
            return found

    marker = f"_frame{frame:02d}{suffix}".lower()
    candidates = [
        p
        for p in patient_dir.glob("*.nii*")
        if marker in p.name.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple files matched frame {frame} ({'GT' if ground_truth else 'image'}): {candidates}"
        )
    raise FileNotFoundError(
        f"Could not find {'ground-truth mask' if ground_truth else 'image'} "
        f"for {case_id} frame {frame} in {patient_dir}"
    )


def run_one(
    make_facts_script: Path,
    patient_dir: Path,
    out_dir: Path,
    overwrite: bool,
) -> Tuple[str, Path]:
    case_id = patient_dir.name
    info_path = find_info_cfg(patient_dir)
    info = parse_info_cfg(info_path)
    ed = int(float(info["ED"]))
    es = int(float(info["ES"]))

    ed_image = find_frame_file(patient_dir, case_id, ed, ground_truth=False)
    es_image = find_frame_file(patient_dir, case_id, es, ground_truth=False)
    ed_mask = find_frame_file(patient_dir, case_id, ed, ground_truth=True)
    es_mask = find_frame_file(patient_dir, case_id, es, ground_truth=True)

    out_path = out_dir / f"{case_id}_facts.json"
    if out_path.exists() and not overwrite:
        return "reused", out_path

    cmd = [
        sys.executable,
        str(make_facts_script),
        "--ed_image_path",
        str(ed_image),
        "--ed_mask_path",
        str(ed_mask),
        "--es_image_path",
        str(es_image),
        "--es_mask_path",
        str(es_mask),
        "--info_path",
        str(info_path),
        "--out_path",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"make_facts_acdc.py failed for {case_id}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    if not out_path.exists():
        raise FileNotFoundError(f"Expected facts file was not created: {out_path}")
    return "created", out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-create ACDC training facts from ED/ES images and ground-truth masks. "
            "These facts are suitable for the first Random Forest baseline."
        )
    )
    parser.add_argument(
        "--training_dir",
        required=True,
        type=Path,
        help=r"For example: D:\CardiacRate\dataset_acdc\training",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=Path,
        help=r"For example: D:\CardiacRate\dataset_acdc\facts_gt",
    )
    parser.add_argument(
        "--make_facts_script",
        type=Path,
        default=Path(__file__).resolve().parent / "make_facts_acdc.py",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    training_dir = args.training_dir.resolve()
    out_dir = args.out_dir.resolve()
    make_facts_script = args.make_facts_script.resolve()

    if not training_dir.is_dir():
        raise NotADirectoryError(f"Training directory not found: {training_dir}")
    if not make_facts_script.is_file():
        raise FileNotFoundError(f"make_facts_acdc.py not found: {make_facts_script}")

    patient_dirs = sorted(
        p for p in training_dir.iterdir() if p.is_dir() and p.name.lower().startswith("patient")
    )
    if not patient_dirs:
        raise FileNotFoundError(f"No patient directories found under: {training_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"created": 0, "reused": 0, "failed": 0}
    failures: List[Dict[str, str]] = []

    for index, patient_dir in enumerate(patient_dirs, start=1):
        try:
            status, out_path = run_one(
                make_facts_script=make_facts_script,
                patient_dir=patient_dir,
                out_dir=out_dir,
                overwrite=args.overwrite,
            )
            summary[status] += 1
            print(f"[{index}/{len(patient_dirs)}] {patient_dir.name}: {status} -> {out_path}")
        except Exception as exc:
            summary["failed"] += 1
            failures.append({"patient": patient_dir.name, "error": str(exc)})
            print(f"[{index}/{len(patient_dirs)}] {patient_dir.name}: FAILED\n{exc}")

    summary_path = out_dir / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "training_dir": str(training_dir),
                "out_dir": str(out_dir),
                "make_facts_script": str(make_facts_script),
                "total_patient_directories": len(patient_dirs),
                **summary,
                "failures": failures,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nBatch summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary: {summary_path}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
