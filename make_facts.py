import os
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


LABEL_MAP = {
    1: "myocardium",
    2: "aortic_valve",
    3: "aortic_valve_calcification",
}


def get_patient_id(path: str) -> str:
    name = Path(path).name

    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = Path(path).stem

    name = name.replace("_predict", "")
    name = name.replace("_pred", "")
    name = name.replace("_gt", "")

    return name


def get_spacing_from_nifti(nifti_img):
    """
    Returns voxel spacing in mm.
    For NIfTI, header.get_zooms() usually gives spacing for x, y, z.
    """
    spacing = nifti_img.header.get_zooms()[:3]
    return [float(x) for x in spacing]


def safe_float(x):
    if x is None:
        return None
    return float(np.round(x, 6))


def compute_structure_facts(mask: np.ndarray, label: int, spacing_mm):
    binary = mask == label
    voxel_count = int(binary.sum())

    if voxel_count == 0:
        return {
            "label": label,
            "present": False,
            "voxel_count": 0,
            "volume_mm3": 0.0,
            "volume_ml": 0.0,
            "bbox_voxel": None,
            "slice_range_z": None,
            "centroid_voxel": None,
        }

    coords = np.argwhere(binary)

    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)
    size_xyz = max_xyz - min_xyz + 1

    centroid = coords.mean(axis=0)

    voxel_volume_mm3 = float(np.prod(spacing_mm))
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0

    z_values = coords[:, 2]
    slice_range_z = [int(z_values.min()), int(z_values.max())]

    return {
        "label": label,
        "present": True,
        "voxel_count": voxel_count,
        "volume_mm3": safe_float(volume_mm3),
        "volume_ml": safe_float(volume_ml),
        "bbox_voxel": {
            "min": [int(x) for x in min_xyz],
            "max": [int(x) for x in max_xyz],
            "size": [int(x) for x in size_xyz],
        },
        "slice_range_z": slice_range_z,
        "centroid_voxel": [safe_float(x) for x in centroid],
    }


def classify_calcification_severity(calc_volume_mm3, valve_volume_mm3):
    """
    Rule-based severity.
    這裡的 threshold 先當作 heuristic。
    之後如果有醫師標註或文獻標準，可以再調整。

    建議論文寫法：
    Since expert severity labels were not available, a rule-based severity
    estimate was defined based on the calcification-to-aortic-valve volume ratio.
    """

    if calc_volume_mm3 <= 0:
        return "none"

    if valve_volume_mm3 <= 0:
        return "unknown"

    ratio = calc_volume_mm3 / valve_volume_mm3

    if ratio < 0.05:
        return "mild"
    elif ratio < 0.20:
        return "moderate"
    else:
        return "severe"


def compute_slice_count(structure_fact):
    if not structure_fact["present"] or structure_fact["slice_range_z"] is None:
        return 0

    z_min, z_max = structure_fact["slice_range_z"]
    return int(z_max - z_min + 1)


def build_facts(mask_path: str, image_path: str = None):
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata().astype(np.int16)

    spacing_mm = get_spacing_from_nifti(mask_nii)
    patient_id = get_patient_id(mask_path)

    facts = {
        "patient_id": patient_id,
        "mask_path": str(mask_path),
        "image_path": str(image_path) if image_path else None,
        "image_shape": [int(x) for x in mask.shape],
        "spacing_mm": spacing_mm,
        "structures": {},
        "derived_metrics": {},
        "answerable_findings": {},
        "limitations": [],
    }

    for label, name in LABEL_MAP.items():
        facts["structures"][name] = compute_structure_facts(
            mask=mask,
            label=label,
            spacing_mm=spacing_mm,
        )

    myocardium = facts["structures"]["myocardium"]
    valve = facts["structures"]["aortic_valve"]
    calc = facts["structures"]["aortic_valve_calcification"]

    calc_volume = calc["volume_mm3"]
    valve_volume = valve["volume_mm3"]
    myocardium_volume = myocardium["volume_mm3"]

    if valve_volume > 0:
        calc_to_valve_ratio = calc_volume / valve_volume
    else:
        calc_to_valve_ratio = None

    if myocardium_volume > 0:
        calc_to_myocardium_ratio = calc_volume / myocardium_volume
    else:
        calc_to_myocardium_ratio = None

    severity = classify_calcification_severity(
        calc_volume_mm3=calc_volume,
        valve_volume_mm3=valve_volume,
    )

    facts["derived_metrics"] = {
        "calcification_to_aortic_valve_volume_ratio": safe_float(calc_to_valve_ratio),
        "calcification_to_myocardium_volume_ratio": safe_float(calc_to_myocardium_ratio),
        "calcification_severity_rule_based": severity,
        "calcification_slice_count": compute_slice_count(calc),
    }

    facts["answerable_findings"] = {
        "has_myocardium_segmentation": myocardium["present"],
        "has_aortic_valve_segmentation": valve["present"],
        "has_aortic_valve_calcification": calc["present"],
        "can_answer_calcification_volume": calc["present"],
        "can_answer_calcification_severity": calc["present"] and valve["present"],
        "can_answer_calcification_location": calc["present"],
        "can_answer_structure_volume": True,
        "can_answer_slice_range": True,

        # 這些目前不應該讓 LLM 亂回答
        "can_answer_cardiac_enlargement": False,
        "can_answer_coronary_stenosis": False,
        "can_answer_cardiac_function": False,
        "can_answer_ejection_fraction": False,
    }

    facts["limitations"] = [
        "Cardiac enlargement cannot be reliably determined without validated reference ranges, body-size normalization, or clinical diagnostic labels.",
        "Coronary artery stenosis cannot be assessed from the current segmentation labels.",
        "Functional cardiac information such as ejection fraction cannot be estimated from a single static CT segmentation.",
        "The calcification severity is rule-based and should not be interpreted as a definitive clinical diagnosis.",
    ]

    return facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path", required=True, help="Path to predicted segmentation mask .nii.gz")
    parser.add_argument("--image_path", default=None, help="Optional original CT image path")
    parser.add_argument("--out_path", required=True, help="Output facts.json path")
    args = parser.parse_args()

    facts = build_facts(
        mask_path=args.mask_path,
        image_path=args.image_path,
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved facts to: {args.out_path}")


if __name__ == "__main__":
    main()