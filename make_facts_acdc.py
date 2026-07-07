import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None


LABEL_MAP = {
    1: "right_ventricle",
    2: "myocardium",
    3: "left_ventricle",
}


def parse_info_cfg(path: str) -> Dict[str, Any]:
    """Parse an ACDC Info.cfg file into normalized keys."""
    raw: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            raw[key.strip()] = value.strip()

    required = ["ED", "ES", "NbFrame"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Info.cfg is missing required fields: {missing}")

    def to_int(key: str) -> Optional[int]:
        value = raw.get(key)
        return int(float(value)) if value not in (None, "") else None

    def to_float(key: str) -> Optional[float]:
        value = raw.get(key)
        return float(value) if value not in (None, "") else None

    return {
        "ed_frame": to_int("ED"),
        "es_frame": to_int("ES"),
        "group": raw.get("Group") or None,
        "height_cm": to_float("Height"),
        "number_of_frames": to_int("NbFrame"),
        "weight_kg": to_float("Weight"),
        "raw": raw,
    }


def strip_nii(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return filename[:-7]
    if lower.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def infer_case_id(image_path: str) -> str:
    stem = strip_nii(Path(image_path).name)
    if "_frame" in stem:
        return stem.split("_frame", 1)[0]
    for suffix in ("_ed", "_es", "_ED", "_ES"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def load_volume(path: str, is_mask: bool = False) -> Tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(path)
    array = np.asanyarray(image.dataobj)
    if array.ndim == 4:
        if array.shape[-1] != 1:
            raise ValueError(
                f"Expected a 3D ED/ES volume, but got a 4D volume at {path}: {array.shape}"
            )
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume at {path}, got shape {array.shape}")
    if is_mask:
        array = np.rint(array).astype(np.int16, copy=False)
    return image, np.asarray(array)


def voxel_geometry(
    image: nib.Nifti1Image,
) -> Tuple[List[float], float, float]:
    """Return header spacing, header-derived voxel volume and affine-derived volume.

    ACDC files may contain an affine whose determinant does not reflect the voxel
    spacing stored in the NIfTI header. Volumetric measurements therefore use
    the header zooms, while the affine-derived value is retained for QC only.
    """
    zooms = image.header.get_zooms()
    if len(zooms) < 3:
        raise ValueError(f"Invalid voxel spacing: {zooms}")

    spacing = [float(v) for v in zooms[:3]]
    if any(not np.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError(f"Invalid voxel spacing: {spacing}")

    voxel_mm3 = float(np.prod(spacing))
    affine_voxel_mm3 = float(abs(np.linalg.det(image.affine[:3, :3])))

    if not np.isfinite(affine_voxel_mm3) or affine_voxel_mm3 <= 0:
        affine_voxel_mm3 = float("nan")

    return spacing, voxel_mm3, affine_voxel_mm3


def append_voxel_geometry_qc(
    phase_name: str,
    header_voxel_mm3: float,
    affine_voxel_mm3: float,
    qc_flags: List[str],
) -> None:
    """Record header/affine disagreement without changing volume calculations."""
    if not np.isfinite(affine_voxel_mm3):
        qc_flags.append(f"invalid_{phase_name}_affine_voxel_volume")
        return

    if not np.isclose(
        header_voxel_mm3,
        affine_voxel_mm3,
        rtol=0.01,
        atol=1e-6,
    ):
        qc_flags.append(
            f"{phase_name}_affine_header_voxel_volume_mismatch: "
            f"header={header_voxel_mm3:.6f}, "
            f"affine={affine_voxel_mm3:.6f}; using header spacing"
        )


def round_or_none(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def bbox_dhw(mask: np.ndarray) -> Optional[List[List[int]]]:
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    x, y, z = coords
    return [
        [int(z.min()), int(y.min()), int(x.min())],
        [int(z.max()), int(y.max()), int(x.max())],
    ]


def centroid_dhw(mask: np.ndarray) -> Optional[List[float]]:
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    x, y, z = coords
    return [round(float(z.mean()), 3), round(float(y.mean()), 3), round(float(x.mean()), 3)]


def structure_stats(mask_array: np.ndarray, label_id: int, voxel_mm3: float) -> Dict[str, Any]:
    mask = mask_array == label_id
    voxels = int(mask.sum())
    volume_mm3 = float(voxels * voxel_mm3)
    z_indices = np.where(mask)[2]
    return {
        "label_id": label_id,
        "voxel_count": voxels,
        "volume_mm3": round(volume_mm3, 3),
        "volume_ml": round(volume_mm3 / 1000.0, 3),
        "bbox_dhw": bbox_dhw(mask),
        "slice_range_z": [int(z_indices.min()), int(z_indices.max())] if voxels else None,
        "centroid_dhw": centroid_dhw(mask),
    }


def phase_stats(mask_array: np.ndarray, voxel_mm3: float) -> Dict[str, Any]:
    return {
        name: structure_stats(mask_array, label_id, voxel_mm3)
        for label_id, name in LABEL_MAP.items()
    }


def functional_metric(edv_ml: float, esv_ml: float) -> Dict[str, Optional[float]]:
    stroke_volume = edv_ml - esv_ml
    ef = (stroke_volume / edv_ml * 100.0) if edv_ml > 0 else None
    return {
        "edv_ml": round_or_none(edv_ml),
        "esv_ml": round_or_none(esv_ml),
        "stroke_volume_ml": round_or_none(stroke_volume),
        "ejection_fraction_percent": round_or_none(ef),
    }


def approximate_max_wall_thickness_mm(
    myocardium_mask: np.ndarray,
    spacing_xyz: Iterable[float],
) -> Optional[float]:
    """
    Estimate the maximum myocardial wall thickness slice by slice.

    For each short-axis slice, 2 * max(distance-to-boundary) is used as an
    approximate local wall thickness. This is a geometric approximation and
    is not a standardized clinical segment-based thickness measurement.
    """
    if distance_transform_edt is None or not myocardium_mask.any():
        return None

    sx, sy, _ = [float(v) for v in spacing_xyz]
    maximum = 0.0
    for z in range(myocardium_mask.shape[2]):
        current = myocardium_mask[:, :, z]
        if not current.any():
            continue
        distance = distance_transform_edt(current, sampling=(sx, sy))
        maximum = max(maximum, float(distance.max()) * 2.0)
    return round_or_none(maximum)


def bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if not height_cm or not weight_kg or height_cm <= 0 or weight_kg <= 0:
        return None
    return round_or_none(weight_kg / ((height_cm / 100.0) ** 2))


def bsa_mosteller(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if not height_cm or not weight_kg or height_cm <= 0 or weight_kg <= 0:
        return None
    return round_or_none(math.sqrt(height_cm * weight_kg / 3600.0))


def indexed_volume(volume_ml: Optional[float], bsa_m2: Optional[float]) -> Optional[float]:
    if volume_ml is None or bsa_m2 is None or bsa_m2 <= 0:
        return None
    return round_or_none(volume_ml / bsa_m2)


def compare_geometry(
    reference_image: nib.Nifti1Image,
    reference_array: np.ndarray,
    other_image: nib.Nifti1Image,
    other_array: np.ndarray,
    name: str,
    qc_flags: List[str],
) -> None:
    if reference_array.shape != other_array.shape:
        qc_flags.append(
            f"shape_mismatch_{name}: reference={reference_array.shape}, other={other_array.shape}"
        )
    if not np.allclose(reference_image.affine, other_image.affine, atol=1e-3):
        qc_flags.append(f"affine_mismatch_{name}")


def build_facts(
    ed_image_path: str,
    ed_mask_path: str,
    es_image_path: str,
    es_mask_path: str,
    info_path: str,
    myocardial_density_g_ml: float = 1.05,
) -> Dict[str, Any]:
    info = parse_info_cfg(info_path)
    case_id = infer_case_id(ed_image_path)

    ed_image, ed_array = load_volume(ed_image_path, is_mask=False)
    ed_mask_image, ed_mask = load_volume(ed_mask_path, is_mask=True)
    es_image, es_array = load_volume(es_image_path, is_mask=False)
    es_mask_image, es_mask = load_volume(es_mask_path, is_mask=True)

    qc_flags: List[str] = []
    compare_geometry(ed_image, ed_array, ed_mask_image, ed_mask, "ed_image_mask", qc_flags)
    compare_geometry(es_image, es_array, es_mask_image, es_mask, "es_image_mask", qc_flags)
    if ed_array.shape != es_array.shape:
        qc_flags.append(f"ed_es_shape_mismatch: ed={ed_array.shape}, es={es_array.shape}")

    for phase_name, mask in (("ed", ed_mask), ("es", es_mask)):
        labels = set(int(v) for v in np.unique(mask))
        unexpected = sorted(labels.difference({0, 1, 2, 3}))
        if unexpected:
            qc_flags.append(f"unexpected_{phase_name}_labels:{unexpected}")
        for label_id, label_name in LABEL_MAP.items():
            if not np.any(mask == label_id):
                qc_flags.append(f"missing_{phase_name}_{label_name}")

    spacing_mm, voxel_mm3, affine_voxel_mm3 = voxel_geometry(ed_image)
    es_spacing_mm, es_voxel_mm3, es_affine_voxel_mm3 = voxel_geometry(es_image)

    append_voxel_geometry_qc("ed", voxel_mm3, affine_voxel_mm3, qc_flags)
    append_voxel_geometry_qc("es", es_voxel_mm3, es_affine_voxel_mm3, qc_flags)

    if not np.allclose(spacing_mm, es_spacing_mm, atol=1e-5):
        qc_flags.append(f"ed_es_spacing_mismatch: ed={spacing_mm}, es={es_spacing_mm}")
    if not np.isclose(voxel_mm3, es_voxel_mm3, rtol=0.0, atol=1e-6):
        qc_flags.append(
            f"ed_es_voxel_volume_mismatch: ed={voxel_mm3:.6f}, "
            f"es={es_voxel_mm3:.6f}"
        )

    ed_structures = phase_stats(ed_mask, voxel_mm3)
    es_structures = phase_stats(es_mask, es_voxel_mm3)

    lv = functional_metric(
        ed_structures["left_ventricle"]["volume_ml"],
        es_structures["left_ventricle"]["volume_ml"],
    )
    rv = functional_metric(
        ed_structures["right_ventricle"]["volume_ml"],
        es_structures["right_ventricle"]["volume_ml"],
    )

    if lv["stroke_volume_ml"] is not None and lv["stroke_volume_ml"] < 0:
        qc_flags.append("negative_lv_stroke_volume")
    if rv["stroke_volume_ml"] is not None and rv["stroke_volume_ml"] < 0:
        qc_flags.append("negative_rv_stroke_volume")

    height_cm = info["height_cm"]
    weight_kg = info["weight_kg"]
    calculated_bmi = bmi(height_cm, weight_kg)
    calculated_bsa = bsa_mosteller(height_cm, weight_kg)

    lv["edv_index_ml_m2"] = indexed_volume(lv["edv_ml"], calculated_bsa)
    lv["esv_index_ml_m2"] = indexed_volume(lv["esv_ml"], calculated_bsa)
    rv["edv_index_ml_m2"] = indexed_volume(rv["edv_ml"], calculated_bsa)
    rv["esv_index_ml_m2"] = indexed_volume(rv["esv_ml"], calculated_bsa)

    ed_myo_ml = ed_structures["myocardium"]["volume_ml"]
    es_myo_ml = es_structures["myocardium"]["volume_ml"]
    myocardial_mass_g = ed_myo_ml * myocardial_density_g_ml

    myocardial_phase_difference_percent = None
    if ed_myo_ml > 0:
        myocardial_phase_difference_percent = abs(es_myo_ml - ed_myo_ml) / ed_myo_ml * 100.0
        if myocardial_phase_difference_percent > 15.0:
            qc_flags.append(
                "large_ed_es_myocardial_volume_difference:"
                f"{myocardial_phase_difference_percent:.3f}%"
            )

    facts: Dict[str, Any] = {
        "schema_version": "acdc_facts_v1",
        "case_id": case_id,
        "patient_id": case_id,
        "modality": "cine_mri",
        "dataset": "ACDC",
        "metadata": {
            "ed_frame": info["ed_frame"],
            "es_frame": info["es_frame"],
            "number_of_frames": info["number_of_frames"],
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi_kg_m2": calculated_bmi,
            "bsa_m2_mosteller": calculated_bsa,
        },
        "image_info": {
            "ed_image_path": str(Path(ed_image_path)),
            "es_image_path": str(Path(es_image_path)),
            "ed_mask_path": str(Path(ed_mask_path)),
            "es_mask_path": str(Path(es_mask_path)),
            "info_cfg_path": str(Path(info_path)),
            "shape_xyz": [int(v) for v in ed_array.shape],
            "shape_dhw": [int(ed_array.shape[2]), int(ed_array.shape[1]), int(ed_array.shape[0])],
            "spacing_mm_xyz": [round(v, 6) for v in spacing_mm],
            "voxel_volume_mm3": round(voxel_mm3, 6),
            "affine_voxel_volume_mm3_qc": round_or_none(affine_voxel_mm3, 6),
            "volume_geometry_source": "NIfTI header zooms",
        },
        "phases": {
            "ED": {
                "frame": info["ed_frame"],
                "structures": ed_structures,
            },
            "ES": {
                "frame": info["es_frame"],
                "structures": es_structures,
            },
        },
        # A compact top-level copy makes the facts easier for the QA prompt to use.
        "structures": {
            "right_ventricle": {
                "ed_volume_ml": rv["edv_ml"],
                "es_volume_ml": rv["esv_ml"],
            },
            "left_ventricle": {
                "ed_volume_ml": lv["edv_ml"],
                "es_volume_ml": lv["esv_ml"],
            },
            "myocardium": {
                "ed_volume_ml": round_or_none(ed_myo_ml),
                "es_volume_ml": round_or_none(es_myo_ml),
            },
        },
        "cardiac_function": {
            "left_ventricle": lv,
            "right_ventricle": rv,
        },
        "myocardial_measurements": {
            "ed_myocardial_volume_ml": round_or_none(ed_myo_ml),
            "es_myocardial_volume_ml": round_or_none(es_myo_ml),
            "estimated_lv_myocardial_mass_g": round_or_none(myocardial_mass_g),
            "myocardial_density_g_ml": round_or_none(myocardial_density_g_ml),
            "ed_es_myocardial_volume_difference_percent": round_or_none(
                myocardial_phase_difference_percent
            ),
            "approximate_max_thickness_ed_mm": approximate_max_wall_thickness_mm(
                ed_mask == 2, spacing_mm
            ),
            "approximate_max_thickness_es_mm": approximate_max_wall_thickness_mm(
                es_mask == 2, es_spacing_mm
            ),
            "thickness_method": "2 × maximum in-mask Euclidean distance, computed slice by slice",
        },
        "derived_metrics": {
            "lv_edv_ml": lv["edv_ml"],
            "lv_esv_ml": lv["esv_ml"],
            "lv_stroke_volume_ml": lv["stroke_volume_ml"],
            "lv_ejection_fraction_percent": lv["ejection_fraction_percent"],
            "rv_edv_ml": rv["edv_ml"],
            "rv_esv_ml": rv["esv_ml"],
            "rv_stroke_volume_ml": rv["stroke_volume_ml"],
            "rv_ejection_fraction_percent": rv["ejection_fraction_percent"],
            "estimated_lv_myocardial_mass_g": round_or_none(myocardial_mass_g),
        },
        "answerable_findings": {
            "can_report_ed_es_volumes": True,
            "can_report_stroke_volume": True,
            "can_report_ejection_fraction": True,
            "can_report_myocardial_volume_and_estimated_mass": True,
            "can_report_approximate_maximum_myocardial_thickness": distance_transform_edt is not None,
            "can_confirm_clinical_diagnosis": False,
        },
        "limitations": [
            "Measurements are derived from automated ED and ES segmentation masks and may be affected by segmentation error.",
            "Ejection fraction is a segmentation-derived estimate and is not a substitute for clinical interpretation.",
            "Maximum myocardial thickness is an approximate geometric measurement, not a standardized segment-based clinical measurement.",
            "The ACDC Group field is a dataset reference label and is excluded from patient-specific QA evidence.",
        ],
        "qc_flags": qc_flags,
        # Keep the dataset label separate so app_gradio.py can exclude it from the LLM facts block.
        "evaluation_metadata": {
            "ground_truth_group": info["group"],
            "use_for_patient_qa": False,
        },
    }
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build structured ED/ES cardiac function facts from ACDC segmentations."
    )
    parser.add_argument("--ed_image_path", required=True)
    parser.add_argument("--ed_mask_path", required=True)
    parser.add_argument("--es_image_path", required=True)
    parser.add_argument("--es_mask_path", required=True)
    parser.add_argument("--info_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--myocardial_density_g_ml", type=float, default=1.05)
    args = parser.parse_args()

    facts = build_facts(
        ed_image_path=args.ed_image_path,
        ed_mask_path=args.ed_mask_path,
        es_image_path=args.es_image_path,
        es_mask_path=args.es_mask_path,
        info_path=args.info_path,
        myocardial_density_g_ml=args.myocardial_density_g_ml,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)

    print(f"Saved ACDC facts: {out_path}")


if __name__ == "__main__":
    main()
