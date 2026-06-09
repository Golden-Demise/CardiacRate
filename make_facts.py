import os
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label


LABEL_MAP = {
    1: "myocardium",
    2: "aortic_valve",
    3: "aortic_valve_calcification",
}


CALC_LABEL = 3


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
    Rule-based calcification burden based on calcification-to-valve volume ratio.
    This is kept only as a heuristic description of calcification burden.
    Do not use this as a clinical aortic stenosis severity classifier.
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


def density_weight(max_hu):
    if max_hu < 130:
        return 0
    elif max_hu < 200:
        return 1
    elif max_hu < 300:
        return 2
    elif max_hu < 400:
        return 3
    else:
        return 4


def compute_agatston_like_score_from_arrays(
    ct: np.ndarray,
    mask: np.ndarray,
    spacing_mm,
    calc_label: int = CALC_LABEL,
    hu_threshold: int = 130,
    min_area_mm2: float = 1.0,
):
    """
    Compute an Agatston-like aortic valve calcium score from CT and segmentation.

    This function uses only calc_label voxels and HU >= hu_threshold.
    It returns both the raw slice-wise score and a simple 3-mm normalized score.
    """
    if ct.shape != mask.shape:
        raise ValueError(f"Shape mismatch: CT {ct.shape}, mask {mask.shape}")

    sx, sy, sz = [float(x) for x in spacing_mm]
    pixel_area_mm2 = sx * sy

    calcium = (mask == calc_label) & (ct >= hu_threshold)

    total_score_raw = 0.0
    total_score_3mm = 0.0
    total_area_mm2 = 0.0
    total_volume_mm3 = 0.0
    component_count = 0
    max_hu_global = None

    for z in range(ct.shape[2]):
        calcium_slice = calcium[:, :, z]

        if not np.any(calcium_slice):
            continue

        labeled, num = cc_label(calcium_slice)
        ct_slice = ct[:, :, z]

        for comp_id in range(1, num + 1):
            comp = labeled == comp_id
            pixel_count = int(np.count_nonzero(comp))
            area_mm2 = pixel_count * pixel_area_mm2

            if area_mm2 < min_area_mm2:
                continue

            max_hu = float(ct_slice[comp].max())
            w = density_weight(max_hu)

            if w == 0:
                continue

            score_raw = area_mm2 * w
            score_3mm = score_raw * (sz / 3.0)

            total_score_raw += score_raw
            total_score_3mm += score_3mm
            total_area_mm2 += area_mm2
            total_volume_mm3 += area_mm2 * sz
            component_count += 1
            max_hu_global = max(max_hu_global, max_hu) if max_hu_global is not None else max_hu

    return {
        "available": True,
        "agatston_like_score_raw": safe_float(total_score_raw),
        "agatston_like_score_3mm_normalized": safe_float(total_score_3mm),
        "calcification_area_mm2": safe_float(total_area_mm2),
        "calcification_volume_mm3_from_hu_mask": safe_float(total_volume_mm3),
        "calcification_volume_ml_from_hu_mask": safe_float(total_volume_mm3 / 1000.0),
        "component_count": int(component_count),
        "max_hu": safe_float(max_hu_global),
        "spacing_mm": [float(sx), float(sy), float(sz)],
        "hu_threshold": int(hu_threshold),
        "min_area_mm2": float(min_area_mm2),
        "calc_label": int(calc_label),
        "method": "slice-wise calcification area multiplied by peak-HU density weight; also reported with simple 3-mm slice-thickness normalization",
        "note": "Agatston-like score computed from segmentation masks; not a clinically validated CT-AVC score.",
    }


def classify_aortic_stenosis_risk_from_agatston(score_3mm, sex=None):
    """
    Rule-based severe aortic stenosis likelihood from CT-AVC-style thresholds.
    sex can be "male", "female", or None.
    """
    if score_3mm is None:
        return {
            "risk_level": "unknown",
            "severe_aortic_stenosis_likelihood": "unknown",
            "reason": "Agatston-like score is unavailable.",
        }

    sex = sex.lower() if isinstance(sex, str) else None
    if sex in ["m", "man"]:
        sex = "male"
    elif sex in ["f", "woman"]:
        sex = "female"

    if sex == "female":
        if score_3mm < 800:
            likelihood = "unlikely"
        elif score_3mm <= 1200:
            likelihood = "indeterminate"
        elif score_3mm <= 1600:
            likelihood = "likely"
        else:
            likelihood = "highly_likely"
    elif sex == "male":
        if score_3mm < 1600:
            likelihood = "unlikely"
        elif score_3mm <= 2000:
            likelihood = "indeterminate"
        elif score_3mm <= 3000:
            likelihood = "likely"
        else:
            likelihood = "highly_likely"
    else:
        # Conservative classification when sex is unknown.
        if score_3mm < 800:
            likelihood = "unlikely_for_both_sexes"
        else:
            likelihood = "sex_required_for_guideline_based_classification"

    if "unlikely" in likelihood:
        risk_level = "low"
    elif likelihood == "indeterminate":
        risk_level = "indeterminate"
    elif likelihood in ["likely", "highly_likely"]:
        risk_level = "increased"
    else:
        risk_level = "unknown"

    return {
        "risk_level": risk_level,
        "severe_aortic_stenosis_likelihood": likelihood,
        "sex_used_for_thresholds": sex,
        "score_used": safe_float(score_3mm),
        "score_type": "agatston_like_score_3mm_normalized",
        "important_note": "This does not diagnose aortic stenosis. Echocardiographic peak velocity, mean pressure gradient, and aortic valve area are required for definitive severity assessment.",
    }


def compute_slice_count(structure_fact):
    if not structure_fact["present"] or structure_fact["slice_range_z"] is None:
        return 0

    z_min, z_max = structure_fact["slice_range_z"]
    return int(z_max - z_min + 1)


def build_facts(mask_path: str, image_path: str = None, sex: str = None):
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata().astype(np.int16)

    spacing_mm = get_spacing_from_nifti(mask_nii)
    patient_id = get_patient_id(mask_path)

    facts = {
        "patient_id": patient_id,
        "mask_path": str(mask_path),
        "image_path": str(image_path) if image_path else None,
        "sex": sex,
        "image_shape": [int(x) for x in mask.shape],
        "spacing_mm": spacing_mm,
        "structures": {},
        "derived_metrics": {},
        "diagnostic_findings": {},
        "answerable_findings": {},
        "limitations": [],
        "qc_flags": [],
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

    agatston = {
        "available": False,
        "reason": "image_path was not provided, so HU-based Agatston-like scoring cannot be computed.",
    }
    as_risk = {
        "risk_level": "unknown",
        "severe_aortic_stenosis_likelihood": "unknown",
        "reason": "Agatston-like score is unavailable.",
    }

    if image_path and calc["present"]:
        ct_nii = nib.load(image_path)
        ct = ct_nii.get_fdata()
        ct_spacing_mm = get_spacing_from_nifti(ct_nii)

        if ct.shape != mask.shape:
            facts["qc_flags"].append(
                f"shape_mismatch_ct_mask: CT {ct.shape}, mask {mask.shape}"
            )
            agatston = {
                "available": False,
                "reason": f"Shape mismatch: CT {ct.shape}, mask {mask.shape}",
            }
        else:
            agatston = compute_agatston_like_score_from_arrays(
                ct=ct,
                mask=mask,
                spacing_mm=ct_spacing_mm,
                calc_label=CALC_LABEL,
                hu_threshold=130,
                min_area_mm2=1.0,
            )
            as_risk = classify_aortic_stenosis_risk_from_agatston(
                score_3mm=agatston["agatston_like_score_3mm_normalized"],
                sex=sex,
            )

            # Simple QC guard: extremely large valve calcium volumes usually indicate mask/label issues.
            if agatston["calcification_volume_mm3_from_hu_mask"] is not None and agatston["calcification_volume_mm3_from_hu_mask"] > 10000:
                facts["qc_flags"].append(
                    "unusually_large_aortic_valve_calcification_volume_check_mask_or_alignment"
                )
    elif image_path and not calc["present"]:
        agatston = {
            "available": True,
            "agatston_like_score_raw": 0.0,
            "agatston_like_score_3mm_normalized": 0.0,
            "calcification_area_mm2": 0.0,
            "calcification_volume_mm3_from_hu_mask": 0.0,
            "calcification_volume_ml_from_hu_mask": 0.0,
            "component_count": 0,
            "max_hu": None,
            "hu_threshold": 130,
            "calc_label": CALC_LABEL,
            "note": "No aortic valve calcification label was present.",
        }
        as_risk = classify_aortic_stenosis_risk_from_agatston(
            score_3mm=0.0,
            sex=sex,
        )

    facts["derived_metrics"] = {
        "calcification_to_aortic_valve_volume_ratio": safe_float(calc_to_valve_ratio),
        "calcification_to_myocardium_volume_ratio": safe_float(calc_to_myocardium_ratio),
        "calcification_severity_rule_based": severity,
        "calcification_slice_count": compute_slice_count(calc),
        "aortic_valve_calcification_agatston_like": agatston,
    }

    facts["diagnostic_findings"] = {
        "aortic_stenosis_risk": {
            "assessment_basis": "aortic valve calcification on CT",
            "calcification_present": bool(calc["present"]),
            "calcification_volume_mm3": calc["volume_mm3"],
            "calcification_volume_ml": calc["volume_ml"],
            "calcification_burden_rule_based": severity,
            "agatston_like": agatston,
            "risk_assessment": as_risk,
            "limitations": [
                "This is an Agatston-like score derived from segmentation masks, not a clinically validated CT-AVC score.",
                "Aortic stenosis cannot be diagnosed from calcification burden alone.",
                "Echocardiographic peak velocity, mean pressure gradient, and aortic valve area are required for definitive severity assessment.",
            ],
        }
    }

    can_answer_as_risk = bool(agatston.get("available", False))

    facts["answerable_findings"] = {
        "has_myocardium_segmentation": myocardium["present"],
        "has_aortic_valve_segmentation": valve["present"],
        "has_aortic_valve_calcification": calc["present"],
        "can_answer_calcification_volume": calc["present"],
        "can_answer_calcification_severity": calc["present"] and valve["present"],
        "can_answer_calcification_location": calc["present"],
        "can_answer_aortic_stenosis_risk": can_answer_as_risk,
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
        "Aortic stenosis risk is estimated from an Agatston-like aortic valve calcification score and does not replace echocardiographic evaluation.",
    ]

    return facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path", required=True, help="Path to predicted segmentation mask .nii.gz")
    parser.add_argument("--image_path", default=None, help="Optional original CT image path")
    parser.add_argument("--out_path", required=True, help="Output facts.json path")
    parser.add_argument("--sex", default=None, choices=["male", "female"], help="Optional patient sex for sex-specific CT-AVC thresholding")
    args = parser.parse_args()

    facts = build_facts(
        mask_path=args.mask_path,
        image_path=args.image_path,
        sex=args.sex,
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved facts to: {args.out_path}")


if __name__ == "__main__":
    main()
