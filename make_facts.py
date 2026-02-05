import os
import json
import argparse
import numpy as np

import nibabel as nib

try:
    from scipy.ndimage import label as cc_label
except Exception:
    cc_label = None

LABEL_MAP = {
    1: "myocardium",
    2: "aortic_valve",
    3: "aortic_valve_calcification",
}

def voxel_volume_mm3(spacing_xyz):
    sx, sy, sz = spacing_xyz
    return float(sx * sy * sz)

def to_dhw_shape(shape_xyz):
    # NIfTI commonly loads as (X,Y,Z); we report (D,H,W)=(Z,Y,X)
    x, y, z = shape_xyz
    return [int(z), int(y), int(x)]

def compute_bbox(indices_xyz):
    x, y, z = indices_xyz
    return [[int(z.min()), int(y.min()), int(x.min())],
            [int(z.max()), int(y.max()), int(x.max())]]

def compute_centroid(indices_xyz):
    x, y, z = indices_xyz
    return [float(z.mean()), float(y.mean()), float(x.mean())]  # D,H,W order

def connected_components_stats(mask: np.ndarray, spacing_xyz):
    if mask.sum() == 0:
        return 0, 0.0
    if cc_label is None:
        # fallback: treat as one component
        largest_mm3 = float(mask.sum() * voxel_volume_mm3(spacing_xyz))
        return 1, largest_mm3

    labeled, num = cc_label(mask.astype(np.uint8))
    if num == 0:
        return 0, 0.0
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_vox = int(counts.max())
    largest_mm3 = float(largest_vox * voxel_volume_mm3(spacing_xyz))
    return int(num), largest_mm3

def strip_nii_gz(name: str):
    # patient_0001.nii.gz -> patient_0001
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]

def build_facts(ct_path: str,
                label_path: str,
                min_calci_vox: int = 20):
    ct_img = nib.load(ct_path)
    spacing_xyz = ct_img.header.get_zooms()[:3]

    lbl_img = nib.load(label_path)
    lbl = lbl_img.get_fdata().astype(np.int16)

    # 基本一致性檢查（不阻擋流程，但會寫 qc flag）
    qc_flags = []
    if lbl.shape != ct_img.shape:
        qc_flags.append("shape_mismatch_ct_label")

    vvox = voxel_volume_mm3(spacing_xyz)
    case_id = strip_nii_gz(os.path.basename(ct_path))

    facts = {
        "schema_version": "1.0",
        "case_id": case_id,
        "modality": "CT",
        "image_path": ct_path.replace("\\", "/"),
        "label_path": label_path.replace("\\", "/"),
        "shape_dhw": to_dhw_shape(lbl.shape),
        "spacing_mm": [float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])],
        "labels": {str(k): v for k, v in LABEL_MAP.items()},
        "structures": {},
        "derived": {},
        "prompts": [],
        "qc_flags": qc_flags
    }

    # structures
    for lid, name in LABEL_MAP.items():
        mask = (lbl == lid)
        vox = int(mask.sum())
        vol_mm3 = float(vox * vvox)
        vol_ml = float(vol_mm3 / 1000.0)

        st = {
            "present": bool(vox > 0),
            "voxel_count": vox,
            "volume_mm3": vol_mm3,
            "volume_ml": vol_ml
        }

        if vox > 0:
            idx = np.where(mask)  # (x,y,z)
            st["centroid_voxel"] = compute_centroid(idx)
            st["bbox_voxel"] = compute_bbox(idx)

        if name == "aortic_valve_calcification":
            num_cc, largest_mm3 = connected_components_stats(mask, spacing_xyz)
            st["num_connected_components"] = num_cc
            st["largest_component_mm3"] = largest_mm3

        facts["structures"][name] = st

    # 鈣化小物件門檻：小於 min_calci_vox 視為不存在
    calci = facts["structures"]["aortic_valve_calcification"]
    if 0 < calci["voxel_count"] < min_calci_vox:
        facts["qc_flags"].append("calcification_below_threshold")
        calci["present"] = False
        calci["voxel_count"] = 0
        calci["volume_mm3"] = 0.0
        calci["volume_ml"] = 0.0
        calci["num_connected_components"] = 0
        calci["largest_component_mm3"] = 0.0

    # 邏輯一致性 QC
    if facts["structures"]["aortic_valve"]["voxel_count"] == 0:
        facts["qc_flags"].append("empty_aortic_valve")
        if calci["present"]:
            facts["qc_flags"].append("inconsistent_calci_without_valve")

    if len(facts["qc_flags"]) == 0:
        facts["qc_flags"].append("ok")

    # derived（給報告/QA 直接用）
    valve = facts["structures"]["aortic_valve"]
    facts["derived"] = {
        "myocardium_volume_ml": float(facts["structures"]["myocardium"]["volume_ml"]),
        "aortic_valve_volume_ml": float(valve["volume_ml"]),
        "calcification_present": bool(calci["present"]),
        "calcification_volume_mm3": float(calci["volume_mm3"]),
        "calcification_volume_ml": float(calci["volume_ml"]),
        "calcification_to_valve_ratio": float(calci["volume_mm3"] / max(valve["volume_mm3"], 1e-6))
    }

    # prompts（你後面要做 ROI prompt 時很方便）
    # 這裡先放 bbox prompt（不用額外存檔），以 aortic_valve 為 ROI
    if "bbox_voxel" in valve:
        facts["prompts"].append({
            "type": "bbox",
            "target": "aortic_valve",
            "prompt_bbox_voxel": valve["bbox_voxel"]
        })

    return facts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct_dir", required=True, help="放 CT: patient_0001.nii.gz 的資料夾")
    ap.add_argument("--label_dir", required=True, help="放 label: patient_0001_gt.nii.gz 的資料夾")
    ap.add_argument("--out_dir", required=True, help="輸出 facts.json 的資料夾")
    ap.add_argument("--min_calci_vox", type=int, default=20, help="鈣化最小 voxel 門檻")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 掃描 label_dir 找 *_gt.nii.gz / *_gt.nii
    for fn in os.listdir(args.label_dir):
        if not (fn.endswith(".nii.gz") or fn.endswith(".nii")):
            continue
        if "_gt" not in fn:
            continue

        label_path = os.path.join(args.label_dir, fn)
        base = strip_nii_gz(fn).replace("_gt", "")  # patient_0001_gt -> patient_0001
        ct_candidate_1 = os.path.join(args.ct_dir, base + ".nii.gz")
        ct_candidate_2 = os.path.join(args.ct_dir, base + ".nii")

        if os.path.exists(ct_candidate_1):
            ct_path = ct_candidate_1
        elif os.path.exists(ct_candidate_2):
            ct_path = ct_candidate_2
        else:
            print(f"[SKIP] 找不到對應CT：{base}.nii(.gz) for label {fn}")
            continue

        facts = build_facts(ct_path, label_path, min_calci_vox=args.min_calci_vox)
        out_path = os.path.join(args.out_dir, f"{facts['case_id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(facts, f, ensure_ascii=False, indent=2)

        print(f"[OK] {out_path}")

    print("Done.")

if __name__ == "__main__":
    main()
