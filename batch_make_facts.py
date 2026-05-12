import json
import argparse
from pathlib import Path

from make_facts import build_facts, get_patient_id


def find_nii_files(input_dir: str):
    """
    找出資料夾內所有 .nii 或 .nii.gz 檔案。
    """
    input_dir = Path(input_dir)

    nii_files = list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz"))
    nii_files = sorted(nii_files)

    return nii_files


def find_matching_image(mask_path: Path, image_dir: str | None):
    """
    根據 mask 檔名尋找對應的原始 CT image。

    example:
    mask:  example_predict.nii.gz
    image: example.nii.gz

    如果找不到，回傳 None。
    """
    if image_dir is None:
        return None

    image_dir = Path(image_dir)
    patient_id = get_patient_id(str(mask_path))

    candidates = [
        image_dir / f"{patient_id}.nii.gz",
        image_dir / f"{patient_id}.nii",
        image_dir / f"{patient_id}_image.nii.gz",
        image_dir / f"{patient_id}_ct.nii.gz",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def save_json(data, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def batch_make_facts(mask_dir: str, out_dir: str, image_dir: str | None = None):
    mask_dir = Path(mask_dir)
    out_dir = Path(out_dir)

    mask_files = find_nii_files(mask_dir)

    if len(mask_files) == 0:
        print(f"[WARN] No .nii or .nii.gz files found in: {mask_dir}")
        return

    print(f"[INFO] Found {len(mask_files)} mask files.")
    print(f"[INFO] Mask dir : {mask_dir}")
    print(f"[INFO] Output dir: {out_dir}")

    if image_dir is not None:
        print(f"[INFO] Image dir : {image_dir}")

    success_count = 0
    fail_count = 0

    for idx, mask_path in enumerate(mask_files, start=1):
        try:
            patient_id = get_patient_id(str(mask_path))

            image_path = find_matching_image(
                mask_path=mask_path,
                image_dir=image_dir,
            )

            facts = build_facts(
                mask_path=str(mask_path),
                image_path=image_path,
            )

            out_path = out_dir / f"{patient_id}_facts.json"
            save_json(facts, out_path)

            success_count += 1

            print(f"[{idx}/{len(mask_files)}] [OK] {mask_path.name} -> {out_path.name}")

        except Exception as e:
            fail_count += 1
            print(f"[{idx}/{len(mask_files)}] [FAIL] {mask_path.name}")
            print(f"    Error: {e}")

    print()
    print("========== Batch finished ==========")
    print(f"Success: {success_count}")
    print(f"Failed : {fail_count}")
    print(f"Total  : {len(mask_files)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask_dir",
        required=True,
        help="Directory containing predicted segmentation masks.",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save generated facts.json files.",
    )

    parser.add_argument(
        "--image_dir",
        default=None,
        help="Optional directory containing original CT images.",
    )

    args = parser.parse_args()

    batch_make_facts(
        mask_dir=args.mask_dir,
        out_dir=args.out_dir,
        image_dir=args.image_dir,
    )


if __name__ == "__main__":
    main()