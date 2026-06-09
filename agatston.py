import numpy as np
import nibabel as nib
from scipy.ndimage import label


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


def compute_agatston_like_score(
    ct_path,
    calc_mask_path,
    hu_threshold=130,
    min_area_mm2=1.0,
    thickness_normalize=False
):
    """
    Compute Agatston-like score from CT image and calcification mask.

    Assumption:
    - CT and mask have the same shape.
    - NIfTI shape is treated as (X, Y, Z).
    - spacing from nibabel header is (sx, sy, sz).
    """

    ct_nii = nib.load(ct_path)
    mask_nii = nib.load(calc_mask_path)

    ct = ct_nii.get_fdata()
    mask = mask_nii.get_fdata()

    if ct.shape != mask.shape:
        raise ValueError(f"Shape mismatch: CT {ct.shape}, mask {mask.shape}")

    spacing = ct_nii.header.get_zooms()[:3]
    sx, sy, sz = spacing

    pixel_area_mm2 = sx * sy

    # Keep calcification mask and HU >= 130
    calcium = (mask == 3) & (ct >= hu_threshold)

    total_score = 0.0
    total_area_mm2 = 0.0
    total_volume_mm3 = 0.0
    component_count = 0

    # axial slices along Z
    for z in range(ct.shape[2]):
        calcium_slice = calcium[:, :, z]

        if not np.any(calcium_slice):
            continue

        labeled, num = label(calcium_slice)

        for comp_id in range(1, num + 1):
            comp = labeled == comp_id
            pixel_count = np.count_nonzero(comp)

            area_mm2 = pixel_count * pixel_area_mm2

            # reduce tiny noise
            if area_mm2 < min_area_mm2:
                continue

            max_hu = float(ct[:, :, z][comp].max())
            w = density_weight(max_hu)

            if w == 0:
                continue

            score = area_mm2 * w

            # optional normalization if slice thickness is not 3 mm
            if thickness_normalize:
                score = score * (sz / 3.0)

            total_score += score
            total_area_mm2 += area_mm2
            total_volume_mm3 += area_mm2 * sz
            component_count += 1

    return {
        "agatston_like_score": round(total_score, 3),
        "calcification_area_mm2": round(total_area_mm2, 3),
        "calcification_volume_mm3": round(total_volume_mm3, 3),
        "component_count": component_count,
        "spacing_mm": [float(sx), float(sy), float(sz)],
        "hu_threshold": hu_threshold,
        "min_area_mm2": min_area_mm2,
        "thickness_normalize": thickness_normalize,
        "note": "This is an Agatston-like score computed from segmentation masks, not a clinically validated CT-AVC score."
    }

print(compute_agatston_like_score(ct_path="D:\\CardiacRate\\dataset\\ct\\patient0001.nii.gz",calc_mask_path="D:\\CardiacRate\\dataset\\label\\patient0001_gt.nii.gz"))