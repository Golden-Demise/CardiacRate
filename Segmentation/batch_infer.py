"""
batch_infer.py

Run Segmentation/infer.py repeatedly for a folder or a list of cardiac CT volumes.

Example:
    python batch_infer.py ^
      --infer_script D:\\CardiacRate\\Segmentation\\infer.py ^
      --input_dir D:\\CardiacRate\\dataset\\ct ^
      --infer_dir D:\\CardiacRate\\dataset\\predict ^
      --model_name unetcnx_a1 ^
      --checkpoint D:\\CardiacRate\\Segmentation\\model\\unetcnx_a1\\best_model.pth

You can forward any extra infer.py arguments after the batch arguments, for example:
    python batch_infer.py --infer_script Segmentation\\infer.py --input_dir dataset\\ct \
      --infer_dir dataset\\predict --model_name unetcnx_a1 --checkpoint model.pth \
      --roi_x 128 --roi_y 128 --roi_z 128 --a_min -42 --a_max 423
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List


def strip_nii_suffix(path: Path) -> str:
    """Return filename stem while correctly handling .nii.gz."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def is_probably_label_or_prediction(path: Path) -> bool:
    """Avoid accidentally sending ground-truth labels or previous predictions to infer.py."""
    stem = strip_nii_suffix(path).lower()
    return stem.endswith("_gt") or stem.endswith("_label") or stem.endswith("_predict") or stem.endswith("_pred")


def read_list_file(list_file: Path) -> List[Path]:
    images: List[Path] = []
    base_dir = list_file.parent
    for line_no, line in enumerate(list_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"List file line {line_no}: image does not exist: {p}")
        images.append(p)
    return images


def collect_images(input_dir: Path, recursive: bool, include_labels: bool) -> List[Path]:
    patterns = ("*.nii.gz", "*.nii")
    images: List[Path] = []
    for pattern in patterns:
        found: Iterable[Path] = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        images.extend(found)

    images = sorted({p.resolve() for p in images})
    if not include_labels:
        images = [p for p in images if not is_probably_label_or_prediction(p)]
    return images


def build_command(
    python_exe: str,
    infer_script: Path,
    img_path: Path,
    infer_dir: Path,
    model_name: str,
    checkpoint: Path,
    extra_args: List[str],
) -> List[str]:
    return [
        python_exe,
        str(infer_script),
        "--model_name",
        model_name,
        "--checkpoint",
        str(checkpoint),
        "--img_pth",
        str(img_path),
        "--infer_dir",
        str(infer_dir),
        *extra_args,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch runner for infer.py. Unknown arguments are forwarded to infer.py."
    )
    parser.add_argument(
        "--infer_script",
        default="Segmentation/infer.py",
        help="Path to infer.py, e.g. D:\\CardiacRate\\Segmentation\\infer.py",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir", help="Folder containing .nii or .nii.gz images.")
    src.add_argument("--list_file", help="Text file containing one image path per line.")
    parser.add_argument("--infer_dir", required=True, help="Output directory passed to infer.py.")
    parser.add_argument("--model_name", required=True, help="Model name passed to infer.py, e.g. unetcnx_a1.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth path passed to infer.py.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run infer.py.")
    parser.add_argument("--recursive", action="store_true", help="Search input_dir recursively.")
    parser.add_argument(
        "--include_labels",
        action="store_true",
        help="Do not filter files ending with _gt, _label, _predict, or _pred.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "Skip a case if infer_dir already contains <case_stem>_predict.nii.gz. "
            "Use only if this matches your run_infering() output naming rule."
        ),
    )
    parser.add_argument(
        "--output_suffix",
        default="_predict.nii.gz",
        help="Expected output suffix used only by --skip_existing. Default: _predict.nii.gz",
    )
    parser.add_argument("--stop_on_error", action="store_true", help="Stop immediately when one case fails.")
    parser.add_argument("--log_dir", default=None, help="Optional folder to save stdout/stderr logs per case.")

    args, extra_args = parser.parse_known_args()

    infer_script = Path(args.infer_script).expanduser().resolve()
    infer_dir = Path(args.infer_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()

    if not infer_script.exists():
        raise FileNotFoundError(f"infer.py not found: {infer_script}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    infer_dir.mkdir(parents=True, exist_ok=True)

    if args.list_file:
        images = read_list_file(Path(args.list_file).expanduser().resolve())
    else:
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"input_dir not found: {input_dir}")
        images = collect_images(input_dir, args.recursive, args.include_labels)

    if not images:
        print("No .nii or .nii.gz images found.")
        return 1

    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Batch inference")
    print("=" * 80)
    print(f"infer.py   : {infer_script}")
    print(f"checkpoint : {checkpoint}")
    print(f"infer_dir  : {infer_dir}")
    print(f"model_name : {args.model_name}")
    print(f"num_cases  : {len(images)}")
    if extra_args:
        print(f"forwarded  : {' '.join(extra_args)}")
    print("=" * 80)

    failed: List[Path] = []
    skipped = 0
    start_all = time.time()

    for idx, img_path in enumerate(images, start=1):
        case_stem = strip_nii_suffix(img_path)
        expected_output = infer_dir / f"{case_stem}{args.output_suffix}"

        if args.skip_existing and expected_output.exists():
            skipped += 1
            print(f"[{idx}/{len(images)}] SKIP existing: {expected_output.name}")
            continue

        print("-" * 80)
        print(f"[{idx}/{len(images)}] infer: {img_path}")
        cmd = build_command(
            args.python,
            infer_script,
            img_path,
            infer_dir,
            args.model_name,
            checkpoint,
            extra_args,
        )

        t0 = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(infer_script.parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        elapsed = time.time() - t0

        if log_dir:
            log_file = log_dir / f"{case_stem}.log"
            log_file.write_text(result.stdout, encoding="utf-8", errors="replace")

        print(result.stdout)
        if result.returncode == 0:
            print(f"[{idx}/{len(images)}] DONE in {elapsed:.1f}s")
        else:
            failed.append(img_path)
            print(f"[{idx}/{len(images)}] FAILED with return code {result.returncode} after {elapsed:.1f}s")
            if args.stop_on_error:
                break

    total_elapsed = time.time() - start_all
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total cases : {len(images)}")
    print(f"Skipped     : {skipped}")
    print(f"Succeeded   : {len(images) - skipped - len(failed)}")
    print(f"Failed      : {len(failed)}")
    print(f"Total time  : {total_elapsed:.1f}s")

    if failed:
        print("Failed cases:")
        for p in failed:
            print(f"  - {p}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
