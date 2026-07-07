import os
import json
import glob
import argparse
import shutil, hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import gradio as gr

import nibabel as nib
try:
    from scipy.ndimage import label as cc_label
except Exception:
    cc_label = None

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DATASET_CONFIGS = {
    "Cardiac CT": {
        "key": "ct",
        "modality": "cardiac_ct",
        "model_name": "unetcnx_a1",
        "checkpoint": r"D:\CardiacRate\Segmentation\model\unetcnx_a1\best_model.pth",
        "infer_dir": r"D:\CardiacRate\dataset\predict",
        "infer_py": os.path.join("Segmentation", "infer.py"),
        "rf_model_path": "",
        "label_map": {
            1: "myocardium",
            2: "aortic_valve",
            3: "aortic_valve_calcification",
        },
        "label_colors": {
            1: [0, 255, 0],
            2: [255, 255, 0],
            3: [255, 0, 0],
        },
        "best_slice_labels": [2],
    },
    "ACDC cine-MRI": {
        "key": "acdc",
        "modality": "cine_mri",
        "model_name": "unetcnx_a1",
        "checkpoint": r"D:\CardiacRate\Segmentation\model\unetcnx_a1\best_model_acdc.pth",
        "infer_dir": r"D:\CardiacRate\dataset_acdc\predict",
        "facts_out_dir": r"D:\CardiacRate\dataset_acdc\facts",
        "infer_py": os.path.join("Segmentation", "infer_acdc.py"),
        "rf_model_path": (
            r"D:\CardiacRate\dataset_acdc\classification\rf_v1"
            r"\acdc_random_forest.joblib"
        ),
        # Standard ACDC labels: 1=RV cavity, 2=myocardium, 3=LV cavity.
        "label_map": {
            1: "right_ventricle",
            2: "myocardium",
            3: "left_ventricle",
        },
        "label_colors": {
            1: [0, 128, 255],
            2: [0, 255, 0],
            3: [255, 0, 0],
        },
        "best_slice_labels": [1, 2, 3],
    },
}

DEFAULT_DATASET = "Cardiac CT"


def get_dataset_config(dataset_name: str, ct_facts_dir: str = "") -> dict:
    """Return one dataset configuration without mutating the global constant."""
    name = dataset_name if dataset_name in DATASET_CONFIGS else DEFAULT_DATASET
    cfg = dict(DATASET_CONFIGS[name])
    if cfg["key"] == "ct":
        cfg["facts_out_dir"] = ct_facts_dir
    return cfg

SYSTEM = (
    "You are an evidence-grounded cardiac imaging health consultation assistant."
    "Your role is to help users understand cardiac imaging analysis results in clear and natural language. You do not replace a physician and must not make unsupported diagnoses or treatment decisions."
    "Evidence rules:"
    "1. Use the provided case-specific structured facts as the only source for statements about this patient."
    "2. General medical education may be used only to explain medical terms or the usual meaning of a finding."
    "3. Do not invent findings, symptoms, medical history, diagnoses, test results, or treatment recommendations."
    "4. Preserve numerical values and units exactly as provided in the facts."
    "5. Do not expose raw JSON unless the user explicitly asks for it."
    "Answering strategy:"
    "1. Answer in the same language as the user's question."
    "2. Adapt the explanation to the user's apparent level of medical knowledge."
    "3. For a simple factual question, answer directly and briefly."
    "4. For an explanatory or risk-related question, when appropriate:"
    "* state what was found;"
    "* explain what it means in plain language;"
    "* state what cannot be concluded;"
    "* mention what additional clinical information may be needed."
    "5. If the question is only partially supported, answer the supported portion and clearly identify the missing information."
    "6. If the question cannot be answered from the available evidence, explain why rather than giving only a generic refusal."
    "7. When discussing aortic stenosis, distinguish a CT calcium-based risk estimate from a confirmed diagnosis. A confirmed assessment generally requires clinical evaluation and echocardiographic information such as blood-flow velocity, mean pressure gradient, and aortic valve area."
    "8. Do not decide whether the user needs medication, surgery, or another treatment."
    "9. Use calm, patient-friendly wording. Avoid unnecessary technical terminology, but include technical measurements when they are relevant to the question."
    "10. Do not repeatedly add the same warning when a brief limitation statement is sufficient."
    "11. Do not mention patient findings unless the user asks about the CT result or a specific finding."
    "12. For greetings, acknowledgements, or casual conversation, respond briefly and naturally without introducing measurements or findings from the facts."
    "13. If the user asks what the system can do, briefly describe the supported question types and provide a few examples."
    "14. Do not answer a greeting with an unrelated patient-specific measurement."
    "The goal is to be accurate, helpful, understandable, and appropriately cautious while remaining grounded in the provided evidence."
)

SYSTEM += (
    "\nACDC cine-MRI classification rules:"
    "\n15. When classification_prediction is available, treat predicted_group as the output of a machine-learning classifier, not as a confirmed clinical diagnosis."
    "\n16. Do not independently replace or override the Random Forest predicted group."
    "\n17. Explain the prediction using only the supplied MRI measurements, class probabilities, and limitations."
    "\n18. The ACDC group abbreviations are NOR (normal), MINF (previous myocardial infarction), DCM (dilated cardiomyopathy), HCM (hypertrophic cardiomyopathy), and RV (abnormal right ventricle)."
    "\n19. If the highest and second-highest class probabilities are close, explicitly state that the classification is uncertain."
    "\n20. The first-version classifier uses global ED/ES measurements and may have difficulty distinguishing DCM from MINF because regional LV contraction is unavailable."
    "\n21. Never use or mention evaluation_metadata or ground_truth_group as patient evidence."
)

MODEL = None
TOKENIZER = None
FACTS_INDEX = {}   # case_id -> facts_path
DEVICE = None
_VOL_CACHE = {"image": None, "label": None}
_ACDC_RF_CACHE = {"path": None, "mtime": None, "bundle": None}

def load_facts_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_facts_block(facts_obj: dict, derived_only: bool = True) -> str:
    """Build the patient-specific evidence block and exclude evaluation labels."""
    case_id = facts_obj.get("case_id") or facts_obj.get("patient_id", "")

    payload = {
        "case_id": case_id,
        "patient_id": facts_obj.get("patient_id", case_id),
        "modality": facts_obj.get("modality", "cardiac_ct"),
        "dataset": facts_obj.get("dataset"),
        "metadata": facts_obj.get("metadata", {}),
        "image_info": facts_obj.get("image_info", {}),
        "image_shape": facts_obj.get("image_shape"),
        "spacing_mm": facts_obj.get("spacing_mm"),
        "structures": facts_obj.get("structures", {}),
        "phases": facts_obj.get("phases", {}),
        "cardiac_function": facts_obj.get("cardiac_function", {}),
        "myocardial_measurements": facts_obj.get("myocardial_measurements", {}),
        "derived": facts_obj.get("derived", {}),
        "derived_metrics": facts_obj.get("derived_metrics", {}),
        "classification_prediction": facts_obj.get("classification_prediction", {}),
        "answerable_findings": facts_obj.get("answerable_findings", {}),
        "limitations": facts_obj.get("limitations", []),
        "qc_flags": facts_obj.get("qc_flags", []),
    }

    # evaluation_metadata intentionally stays outside the LLM prompt. In ACDC,
    # this contains the dataset Group ground-truth label and must not leak into QA.
    return json.dumps(payload, ensure_ascii=False, indent=2)

def build_prompt(facts_block: str, question: str) -> str:
    return (
        f"### SYSTEM\n{SYSTEM}\n\n"
        f"### FACTS\n{facts_block}\n\n"
        f"### QUESTION\n{question}\n\n"
        f"### ANSWER\n"
    )

@torch.no_grad()
def generate_answer(prompt: str, max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0):
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    out = MODEL.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        repetition_penalty=1.0,
    )
    text = TOKENIZER.decode(out[0], skip_special_tokens=True)
    # Return only answer part
    if "### ANSWER" in text:
        return text.split("### ANSWER", 1)[-1].strip()
    return text.strip()

def build_index(facts_dir: str):
    idx = {}
    paths = sorted(glob.glob(os.path.join(facts_dir, "*.json")))
    for p in paths:
        try:
            obj = load_facts_file(p)
            cid = obj.get("case_id") or os.path.splitext(os.path.basename(p))[0]
            idx[cid] = p
        except Exception:
            pass
    return idx

def load_model(base_model: str, lora_dir: str, cache_dir: str, trust_remote_code: bool):
    global MODEL, TOKENIZER, DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TOKENIZER = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code
    )
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code
    )
    MODEL = PeftModel.from_pretrained(base, lora_dir)
    MODEL.eval()
    if DEVICE == "cpu":
        MODEL.to("cpu")

def voxel_volume_mm3(spacing_xyz):
    sx, sy, sz = spacing_xyz
    return float(sx * sy * sz)

def to_dhw_shape(shape_xyz):
    x, y, z = shape_xyz
    return [int(z), int(y), int(x)]

def compute_bbox(indices_xyz):
    x, y, z = indices_xyz
    return [[int(z.min()), int(y.min()), int(x.min())],
            [int(z.max()), int(y.max()), int(x.max())]]

def compute_centroid(indices_xyz):
    x, y, z = indices_xyz
    return [float(z.mean()), float(y.mean()), float(x.mean())]

def connected_components_stats(mask: np.ndarray, spacing_xyz):
    if mask.sum() == 0:
        return 0, 0.0
    if cc_label is None:
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

def make_facts_from_ct_and_label(
    ct_path: str,
    label_path: str,
    out_facts_path: str,
    min_calci_vox: int = 20,
):
    """
    Execute make_facts.py to generate facts.json, then read and return it.

    Args:
        ct_path:
            Original CT image path.
        label_path:
            Segmentation label / prediction mask path.
        out_facts_path:
            Output facts.json path.
        min_calci_vox:
            Kept for compatibility with the old function.
            Note: current make_facts.py does not use this argument unless you add it there.

    Returns:
        facts: dict
    """

    ct_path = str(Path(ct_path))
    label_path = str(Path(label_path))
    out_facts_path = str(Path(out_facts_path))

    # 假設 make_facts.py 跟目前執行的 script 放在同一個資料夾
    current_dir = Path(__file__).resolve().parent
    make_facts_script = current_dir / "make_facts.py"

    if not make_facts_script.exists():
        raise FileNotFoundError(f"make_facts.py not found: {make_facts_script}")

    if not Path(ct_path).exists():
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    if not Path(label_path).exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    out_dir = Path(out_facts_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(make_facts_script),
        "--mask_path",
        label_path,
        "--image_path",
        ct_path,
        "--out_path",
        out_facts_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            "make_facts.py failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if not Path(out_facts_path).exists():
        raise FileNotFoundError(
            f"make_facts.py finished, but output facts file was not found: {out_facts_path}"
        )

    with open(out_facts_path, "r", encoding="utf-8") as f:
        facts = json.load(f)

    return facts


def make_facts_acdc_case(
    ed_image_path: str,
    ed_mask_path: str,
    es_image_path: str,
    es_mask_path: str,
    info_path: str,
    out_facts_path: str,
):
    """Execute make_facts_acdc.py and return the generated facts object."""
    script = Path(__file__).resolve().parent / "make_facts_acdc.py"
    if not script.exists():
        raise FileNotFoundError(f"make_facts_acdc.py not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--ed_image_path", str(ed_image_path),
        "--ed_mask_path", str(ed_mask_path),
        "--es_image_path", str(es_image_path),
        "--es_mask_path", str(es_mask_path),
        "--info_path", str(info_path),
        "--out_path", str(out_facts_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            "make_facts_acdc.py failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    if not Path(out_facts_path).exists():
        raise FileNotFoundError(
            f"make_facts_acdc.py finished but did not create: {out_facts_path}"
        )
    return load_facts_file(out_facts_path), result.stdout



def _load_acdc_rf_bundle(model_path: str):
    """Load and cache the Random Forest bundle created by acdc_random_forest.py."""
    if not model_path:
        raise ValueError("ACDC Random Forest model path is empty.")

    model_path = str(Path(model_path))
    if not Path(model_path).exists():
        raise FileNotFoundError(f"ACDC Random Forest model not found: {model_path}")

    mtime = Path(model_path).stat().st_mtime
    cached = _ACDC_RF_CACHE
    if (
        cached.get("bundle") is not None
        and cached.get("path") == model_path
        and cached.get("mtime") == mtime
    ):
        return cached["bundle"]

    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required for ACDC classification. "
            "Install it with: pip install joblib scikit-learn pandas"
        ) from exc

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(
            "Unsupported Random Forest file. Expected the model bundle created "
            "by acdc_random_forest.py."
        )

    _ACDC_RF_CACHE.update(
        {"path": model_path, "mtime": mtime, "bundle": bundle}
    )
    return bundle


def predict_acdc_group_from_facts(facts_obj: dict, model_path: str) -> dict:
    """
    Predict NOR/MINF/DCM/HCM/RV from make_facts_acdc.py measurements.

    ground_truth_group is never used as an input feature.
    """
    try:
        import pandas as pd
        from acdc_random_forest import (
            CLASS_NAMES,
            FEATURE_COLUMNS,
            extract_features,
        )
    except ImportError as exc:
        raise ImportError(
            "ACDC classification requires acdc_random_forest.py, pandas, "
            "joblib, and scikit-learn in the app directory/environment."
        ) from exc

    bundle = _load_acdc_rf_bundle(model_path)
    pipeline = bundle["pipeline"]
    feature_columns = list(bundle.get("feature_columns", FEATURE_COLUMNS))
    class_names = list(bundle.get("class_names", CLASS_NAMES))
    model_metadata = dict(bundle.get("metadata", {}))

    features = extract_features(facts_obj)
    input_frame = pd.DataFrame(
        [{name: features.get(name) for name in feature_columns}]
    ).apply(pd.to_numeric, errors="coerce")

    missing_features = [
        name for name in feature_columns if pd.isna(input_frame.iloc[0][name])
    ]
    if len(missing_features) == len(feature_columns):
        raise ValueError(
            "All ACDC Random Forest input features are missing from facts.json."
        )

    predicted_group = str(pipeline.predict(input_frame)[0])
    probabilities = pipeline.predict_proba(input_frame)[0]

    classifier = pipeline.named_steps.get("classifier")
    model_classes = list(
        getattr(classifier, "classes_", getattr(pipeline, "classes_", []))
    )
    probability_map = {
        class_name: (
            float(probabilities[model_classes.index(class_name)])
            if class_name in model_classes
            else 0.0
        )
        for class_name in class_names
    }

    ranked = sorted(
        probability_map.items(), key=lambda item: item[1], reverse=True
    )
    second_group = ranked[1][0] if len(ranked) > 1 else None
    second_probability = ranked[1][1] if len(ranked) > 1 else None
    probability_margin = (
        ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
    )

    top_global_features = []
    try:
        importance = classifier.feature_importances_
        imputer = pipeline.named_steps.get("imputer")
        transformed_names = list(
            imputer.get_feature_names_out(feature_columns)
            if imputer is not None
            else feature_columns
        )
        top_global_features = [
            {"feature": name, "importance": round(float(score), 6)}
            for name, score in sorted(
                zip(transformed_names, importance),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
        ]
    except Exception:
        top_global_features = []

    default_limitations = [
        "This is a machine-learning group prediction, not a confirmed clinical diagnosis.",
        "The prediction is derived from ED/ES cine-MRI segmentation measurements.",
        "The first-version classifier uses global ED/ES measurements and does not include regional left-ventricular contraction features.",
        "DCM and MINF may be difficult to distinguish when regional contraction information is unavailable.",
    ]
    metadata_limitations = model_metadata.get("limitations", [])
    limitations = []
    for item in [*default_limitations, *metadata_limitations]:
        if item and item not in limitations:
            limitations.append(item)

    return {
        "model": "ACDC first-version Random Forest",
        "predicted_group": predicted_group,
        "confidence": round(float(probability_map[predicted_group]), 6),
        "second_most_likely_group": second_group,
        "second_probability": (
            round(float(second_probability), 6)
            if second_probability is not None
            else None
        ),
        "top_two_probability_margin": (
            round(float(probability_margin), 6)
            if probability_margin is not None
            else None
        ),
        "class_probabilities": {
            name: round(float(probability_map.get(name, 0.0)), 6)
            for name in class_names
        },
        "input_features": {
            name: (
                None
                if pd.isna(input_frame.iloc[0][name])
                else round(float(input_frame.iloc[0][name]), 6)
            )
            for name in feature_columns
        },
        "missing_features": missing_features,
        "top_global_model_features": top_global_features,
        "model_training_patient_count": model_metadata.get(
            "training_patient_count"
        ),
        "evidence_source": (
            "make_facts_acdc.py ED/ES cine-MRI segmentation-derived measurements"
        ),
        "is_confirmed_diagnosis": False,
        "limitations": limitations,
    }


def add_acdc_classification_to_facts(
    facts_obj: dict,
    facts_path: str,
    rf_model_path: str,
):
    """Run the Random Forest and persist its output inside the same facts JSON."""
    prediction = predict_acdc_group_from_facts(facts_obj, rf_model_path)
    facts_obj["classification_prediction"] = prediction

    with open(facts_path, "w", encoding="utf-8") as f:
        json.dump(facts_obj, f, ensure_ascii=False, indent=2)

    return facts_obj, prediction


def run_segmentation_infer(
    infer_py: str,
    model_name: str,
    checkpoint: str,
    image_path: str,
    infer_dir: str,
):
    """Run a segmentation script and return the newest output NIfTI path."""
    os.makedirs(infer_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(infer_dir, "*.nii*")))

    infer_script = Path(infer_py)
    if not infer_script.is_absolute():
        infer_script = Path(__file__).resolve().parent / infer_script
    if not infer_script.exists():
        raise FileNotFoundError(f"Inference script not found: {infer_script}")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cmd = [
        sys.executable,
        str(infer_script),
        "--model_name", model_name,
        "--checkpoint", checkpoint,
        "--img_pth", image_path,
        "--infer_dir", infer_dir,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            "Segmentation inference failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    after = set(glob.glob(os.path.join(infer_dir, "*.nii*")))
    new_files = list(after - before)
    candidates = new_files or list(after)
    if not candidates:
        raise RuntimeError(
            f"Inference finished but no .nii/.nii.gz was found in {infer_dir}.\n"
            f"STDOUT:\n{proc.stdout}"
        )
    return max(candidates, key=os.path.getmtime), proc.stdout


def _strip_nii(name: str):
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def _is_nii(path: str):
    lower = str(path).lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _quick_sig(path: str, nbytes: int = 2 * 1024 * 1024):
    size = os.path.getsize(path)
    h = hashlib.md5()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
        if size > nbytes:
            f.seek(max(0, size - nbytes))
            h.update(f.read(nbytes))
    return h.hexdigest()


def _copy_if_changed(src: str, dst: str) -> bool:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and _quick_sig(src) == _quick_sig(dst):
        return False
    shutil.copy2(src, dst)
    return True


def _uploaded_paths(upload_value):
    if upload_value is None:
        return []
    values = upload_value if isinstance(upload_value, (list, tuple)) else [upload_value]
    paths = []
    for value in values:
        if isinstance(value, (str, os.PathLike)):
            path = str(value)
        else:
            path = getattr(value, "name", None)
        if path and os.path.isfile(path):
            paths.append(path)
    return paths


def get_dataset_root(infer_dir: str):
    return os.path.dirname(infer_dir.rstrip("\\/"))


def get_ct_store_dir(infer_dir: str):
    return os.path.join(get_dataset_root(infer_dir), "ct")


def ensure_ct_in_store(src_image: str, infer_dir: str):
    store_dir = get_ct_store_dir(infer_dir)
    os.makedirs(store_dir, exist_ok=True)
    base = os.path.basename(src_image)
    if not _is_nii(base):
        raise ValueError("Cardiac CT input must be a .nii or .nii.gz file.")
    dst = os.path.join(store_dir, base)
    changed = _copy_if_changed(src_image, dst)
    return dst, _strip_nii(base), changed


def find_existing_pred(infer_dir: str, case_id: str):
    exact = [
        os.path.join(infer_dir, f"{case_id}_predict.nii.gz"),
        os.path.join(infer_dir, f"{case_id}_predict.nii"),
    ]
    for path in exact:
        if os.path.exists(path):
            return path
    candidates = glob.glob(os.path.join(infer_dir, f"{case_id}*.nii*"))
    return max(candidates, key=os.path.getmtime) if candidates else None


def parse_acdc_info(path: str):
    values = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    for required in ("ED", "ES", "NbFrame"):
        if required not in values:
            raise ValueError(f"Info.cfg is missing required field: {required}")
    return {
        "ed_frame": int(float(values["ED"])),
        "es_frame": int(float(values["ES"])),
        "number_of_frames": int(float(values["NbFrame"])),
        "group": values.get("Group"),
        "height_cm": float(values["Height"]) if values.get("Height") else None,
        "weight_kg": float(values["Weight"]) if values.get("Weight") else None,
    }


def _acdc_frame_records(paths):
    import re

    records = []
    pattern = re.compile(
        r"^(?P<case>.+)_frame(?P<frame>\d+)(?P<gt>_gt)?\.nii(?:\.gz)?$",
        re.IGNORECASE,
    )
    for path in paths:
        match = pattern.match(os.path.basename(path))
        if not match:
            continue
        records.append(
            {
                "path": path,
                "case_id": match.group("case"),
                "frame": int(match.group("frame")),
                "is_gt": bool(match.group("gt")),
            }
        )
    return records


def _save_4d_frame(source_4d: str, frame_number: int, destination: str):
    image = nib.load(source_4d)
    data = np.asanyarray(image.dataobj)
    if data.ndim != 4:
        raise ValueError(f"Expected a 4D ACDC cine-MRI file, got {data.shape}: {source_4d}")
    frame_index = int(frame_number) - 1
    if frame_index < 0 or frame_index >= data.shape[3]:
        raise IndexError(
            f"Frame {frame_number} is outside 1..{data.shape[3]} for {source_4d}"
        )
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data[..., frame_index]), image.affine, image.header), destination)


def prepare_acdc_case(acdc_upload, infer_dir: str, facts_out_dir: str):
    """Store one uploaded ACDC patient directory and resolve ED/ES images."""
    paths = _uploaded_paths(acdc_upload)
    if not paths:
        raise ValueError("Please upload one ACDC patient directory.")

    info_candidates = [p for p in paths if os.path.basename(p).lower() == "info.cfg"]
    if not info_candidates:
        info_candidates = [p for p in paths if p.lower().endswith(".cfg")]
    if len(info_candidates) != 1:
        raise ValueError(
            f"Expected exactly one Info.cfg in the selected directory, found {len(info_candidates)}."
        )
    source_info = info_candidates[0]
    info = parse_acdc_info(source_info)

    records = _acdc_frame_records(paths)
    image_records = [record for record in records if not record["is_gt"]]
    case_ids = sorted(set(record["case_id"] for record in image_records))

    source_4d = next(
        (
            p for p in paths
            if _is_nii(p) and _strip_nii(os.path.basename(p)).lower().endswith("_4d")
        ),
        None,
    )
    if len(case_ids) == 1:
        case_id = case_ids[0]
    elif source_4d:
        case_id = _strip_nii(os.path.basename(source_4d))[:-3]
    else:
        raise ValueError(
            "Could not determine one ACDC patient ID from files such as patient001_frame01.nii.gz."
        )

    def find_phase(frame_number: int):
        matches = [r["path"] for r in image_records if r["frame"] == frame_number]
        if len(matches) > 1:
            matches = [p for p in matches if _strip_nii(os.path.basename(p)).startswith(case_id)]
        return matches[0] if matches else None

    source_ed = find_phase(info["ed_frame"])
    source_es = find_phase(info["es_frame"])
    if (source_ed is None or source_es is None) and source_4d is None:
        raise FileNotFoundError(
            "The directory must contain the ED and ES frame files, or a patient *_4d.nii.gz file. "
            f"Expected frame {info['ed_frame']} and frame {info['es_frame']}."
        )

    case_dir = os.path.join(get_dataset_root(infer_dir), "cases", case_id)
    os.makedirs(case_dir, exist_ok=True)
    stored_info = os.path.join(case_dir, "Info.cfg")
    changed = _copy_if_changed(source_info, stored_info)

    ed_ext = ".nii.gz" if (source_ed is None or source_ed.lower().endswith(".nii.gz")) else ".nii"
    es_ext = ".nii.gz" if (source_es is None or source_es.lower().endswith(".nii.gz")) else ".nii"
    ed_name = f"{case_id}_frame{info['ed_frame']:02d}{ed_ext}"
    es_name = f"{case_id}_frame{info['es_frame']:02d}{es_ext}"
    stored_ed = os.path.join(case_dir, ed_name)
    stored_es = os.path.join(case_dir, es_name)

    if source_ed:
        changed = _copy_if_changed(source_ed, stored_ed) or changed
    elif not os.path.exists(stored_ed):
        _save_4d_frame(source_4d, info["ed_frame"], stored_ed)
        changed = True

    if source_es:
        changed = _copy_if_changed(source_es, stored_es) or changed
    elif not os.path.exists(stored_es):
        _save_4d_frame(source_4d, info["es_frame"], stored_es)
        changed = True

    facts_path = os.path.join(facts_out_dir, f"{case_id}.json")
    pred_case_dir = os.path.join(infer_dir, case_id)
    if changed:
        if os.path.isdir(pred_case_dir):
            shutil.rmtree(pred_case_dir, ignore_errors=True)
        if os.path.exists(facts_path):
            try:
                os.remove(facts_path)
            except OSError:
                pass

    return {
        "case_id": case_id,
        "info": info,
        "info_path": stored_info,
        "ED": {"image": stored_ed, "frame": info["ed_frame"]},
        "ES": {"image": stored_es, "frame": info["es_frame"]},
        "facts_path": facts_path,
        "changed": changed,
    }


def find_acdc_prediction(infer_dir: str, case_id: str, phase: str):
    phase = phase.upper()
    phase_dir = os.path.join(infer_dir, case_id, phase)
    exact = [
        os.path.join(phase_dir, f"{case_id}_{phase}_predict.nii.gz"),
        os.path.join(phase_dir, f"{case_id}_{phase}_predict.nii"),
    ]
    for path in exact:
        if os.path.exists(path):
            return path
    candidates = glob.glob(os.path.join(phase_dir, "*.nii*"))
    return max(candidates, key=os.path.getmtime) if candidates else None


def run_acdc_phase(
    case_id: str,
    phase: str,
    image_path: str,
    infer_py: str,
    model_name: str,
    checkpoint: str,
    infer_dir: str,
):
    phase = phase.upper()
    existing = find_acdc_prediction(infer_dir, case_id, phase)
    if existing:
        return existing, f"Reused existing {phase} prediction: {existing}", True

    phase_dir = os.path.join(infer_dir, case_id, phase)
    os.makedirs(phase_dir, exist_ok=True)
    prediction, stdout = run_segmentation_infer(
        infer_py=infer_py,
        model_name=model_name,
        checkpoint=checkpoint,
        image_path=image_path,
        infer_dir=phase_dir,
    )
    extension = ".nii.gz" if prediction.lower().endswith(".nii.gz") else ".nii"
    fixed = os.path.join(phase_dir, f"{case_id}_{phase}_predict{extension}")
    if os.path.abspath(prediction) != os.path.abspath(fixed):
        if os.path.exists(fixed):
            os.remove(fixed)
        os.replace(prediction, fixed)
    return fixed, stdout, False


def _facts_preview(facts_obj: dict):
    if not facts_obj:
        return ""
    return json.dumps(
        {
            "case_id": facts_obj.get("case_id"),
            "modality": facts_obj.get("modality"),
            "metadata": facts_obj.get("metadata", {}),
            "cardiac_function": facts_obj.get("cardiac_function", {}),
            "myocardial_measurements": facts_obj.get("myocardial_measurements", {}),
            "derived_metrics": facts_obj.get("derived_metrics", facts_obj.get("derived", {})),
            "classification_prediction": facts_obj.get("classification_prediction", {}),
            "qc_flags": facts_obj.get("qc_flags", []),
        },
        ensure_ascii=False,
        indent=2,
    )


def pipeline_new_case(
    dataset_name: str,
    ct_upload,
    acdc_upload,
    model_name: str,
    checkpoint: str,
    infer_dir: str,
    facts_out_dir: str,
    rf_model_path: str,
    min_calci_vox: int,
    progress=gr.Progress(track_tqdm=True),
):
    cfg = get_dataset_config(dataset_name, facts_out_dir)
    os.makedirs(infer_dir, exist_ok=True)
    os.makedirs(facts_out_dir, exist_ok=True)

    if cfg["key"] == "ct":
        paths = _uploaded_paths(ct_upload)
        if len(paths) != 1:
            raise gr.Error("Please upload one Cardiac CT .nii or .nii.gz file.")

        progress(0.05, desc="Preparing Cardiac CT...")
        image_path, case_id, changed = ensure_ct_in_store(paths[0], infer_dir)
        facts_path = os.path.join(facts_out_dir, f"{case_id}.json")
        if changed:
            old_pred = find_existing_pred(infer_dir, case_id)
            if old_pred and os.path.exists(old_pred):
                os.remove(old_pred)
            if os.path.exists(facts_path):
                os.remove(facts_path)

        pred_path = find_existing_pred(infer_dir, case_id)
        reused = bool(pred_path)
        if pred_path:
            infer_log = f"Reused existing prediction: {pred_path}"
        else:
            progress(0.25, desc="Running Cardiac CT segmentation...")
            pred_path, infer_log = run_segmentation_infer(
                cfg["infer_py"], model_name, checkpoint, image_path, infer_dir
            )
            extension = ".nii.gz" if pred_path.lower().endswith(".nii.gz") else ".nii"
            fixed = os.path.join(infer_dir, f"{case_id}_predict{extension}")
            if os.path.abspath(pred_path) != os.path.abspath(fixed):
                if os.path.exists(fixed):
                    os.remove(fixed)
                os.replace(pred_path, fixed)
                pred_path = fixed

        progress(0.70, desc="Building Cardiac CT facts...")
        if not os.path.exists(facts_path):
            facts_obj = make_facts_from_ct_and_label(
                image_path, pred_path, facts_path, min_calci_vox=min_calci_vox
            )
        else:
            facts_obj = load_facts_file(facts_path)

        best_z = pick_best_z_from_label(pred_path, dataset_name)
        preview, preview_info = render_image_preview(image_path, best_z)
        overlay, overlay_info = render_seg_overlay(image_path, pred_path, best_z, dataset_name)
        volume = _get_cached("image", image_path)
        slider = gr.update(minimum=0, maximum=volume.shape[-1] - 1, value=best_z, step=1, interactive=True)
        progress(1.0, desc="Done.")
        return (
            image_path, pred_path, facts_path, _facts_preview(facts_obj), infer_log,
            slider, preview, preview_info, overlay, overlay_info, best_z,
            f"OK ({'reused cache' if reused else 'new inference'})",
            {}, gr.update(visible=False, value="ED"),
        )

    progress(0.05, desc="Reading ACDC patient directory and Info.cfg...")
    try:
        case = prepare_acdc_case(acdc_upload, infer_dir, facts_out_dir)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    progress(0.20, desc=f"Running ED frame {case['ED']['frame']} segmentation...")
    ed_pred, ed_log, ed_reused = run_acdc_phase(
        case["case_id"], "ED", case["ED"]["image"], cfg["infer_py"],
        model_name, checkpoint, infer_dir,
    )
    progress(0.52, desc=f"Running ES frame {case['ES']['frame']} segmentation...")
    es_pred, es_log, es_reused = run_acdc_phase(
        case["case_id"], "ES", case["ES"]["image"], cfg["infer_py"],
        model_name, checkpoint, infer_dir,
    )
    case["ED"]["pred"] = ed_pred
    case["ES"]["pred"] = es_pred

    progress(0.78, desc="Calculating ED/ES cardiac function facts...")
    facts_path = case["facts_path"]
    if not os.path.exists(facts_path):
        facts_obj, facts_log = make_facts_acdc_case(
            case["ED"]["image"], ed_pred,
            case["ES"]["image"], es_pred,
            case["info_path"], facts_path,
        )
    else:
        facts_obj = load_facts_file(facts_path)
        facts_log = f"Reused existing facts: {facts_path}"

    progress(0.90, desc="Running ACDC five-class Random Forest...")
    try:
        facts_obj, classification_prediction = add_acdc_classification_to_facts(
            facts_obj=facts_obj,
            facts_path=facts_path,
            rf_model_path=rf_model_path,
        )
    except Exception as exc:
        raise gr.Error(f"ACDC Random Forest classification failed: {exc}") from exc

    classification_log = json.dumps(
        {
            "predicted_group": classification_prediction.get("predicted_group"),
            "confidence": classification_prediction.get("confidence"),
            "second_most_likely_group": classification_prediction.get(
                "second_most_likely_group"
            ),
            "second_probability": classification_prediction.get(
                "second_probability"
            ),
            "class_probabilities": classification_prediction.get(
                "class_probabilities", {}
            ),
            "missing_features": classification_prediction.get(
                "missing_features", []
            ),
        },
        ensure_ascii=False,
        indent=2,
    )

    selected_phase = "ED"
    image_path = case[selected_phase]["image"]
    pred_path = case[selected_phase]["pred"]
    best_z = pick_best_z_from_label(pred_path, dataset_name)
    preview, preview_info = render_image_preview(image_path, best_z)
    overlay, overlay_info = render_seg_overlay(image_path, pred_path, best_z, dataset_name)
    volume = _get_cached("image", image_path)
    slider = gr.update(minimum=0, maximum=volume.shape[-1] - 1, value=best_z, step=1, interactive=True)

    state = {
        "case_id": case["case_id"],
        "info_path": case["info_path"],
        "facts_path": facts_path,
        "ED": case["ED"],
        "ES": case["ES"],
    }
    infer_log = (
        f"=== ED ===\n{ed_log}"
        f"\n\n=== ES ===\n{es_log}"
        f"\n\n=== FACTS ===\n{facts_log}"
        f"\n\n=== RANDOM FOREST ===\n{classification_log}"
    )
    cache_status = "reused ED/ES predictions" if ed_reused and es_reused else "completed ED/ES inference"
    predicted_group = classification_prediction.get("predicted_group")
    confidence = classification_prediction.get("confidence")
    progress(1.0, desc="Done.")
    return (
        image_path, pred_path, facts_path, _facts_preview(facts_obj), infer_log,
        slider, preview, preview_info, overlay, overlay_info, best_z,
        (
            f"OK ({cache_status}); RF prediction={predicted_group} "
            f"(confidence={confidence:.4f}); "
            f"showing ED frame {case['ED']['frame']}."
        ),
        state, gr.update(visible=True, value="ED", choices=["ED", "ES"]),
    )


def _load_nii_full(path: str):
    image = nib.load(path)
    array = np.asanyarray(image.dataobj)
    if array.ndim == 4:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got {array.shape}: {path}")
    return np.asarray(array)


def _get_cached(kind: str, path: str):
    mtime = os.path.getmtime(path)
    entry = _VOL_CACHE.get(kind)
    if entry and entry["path"] == path and entry["mtime"] == mtime:
        return entry["vol"]
    volume = _load_nii_full(path)
    _VOL_CACHE[kind] = {"path": path, "mtime": mtime, "vol": volume}
    return volume


def _auto_window_to_uint8(x2d: np.ndarray):
    x = x2d.astype(np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max() + 1e-6)
    x = np.nan_to_num(x, nan=lo, posinf=hi, neginf=lo)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)
    return (x * 255).astype(np.uint8)


def render_image_preview(image_path: str, z: int = None):
    volume = _get_cached("image", image_path)
    z_count = volume.shape[-1]
    z = z_count // 2 if z is None else int(np.clip(z, 0, z_count - 1))
    image = _auto_window_to_uint8(volume[:, :, z])
    image = np.rot90(image, k=3)
    return image, f"Image shape={volume.shape}, z={z}/{z_count - 1}"


def render_seg_overlay(image_path: str, label_path: str, z: int, dataset_name: str = DEFAULT_DATASET):
    image = _get_cached("image", image_path)
    label = _get_cached("label", label_path)
    if image.shape != label.shape:
        raise ValueError(
            f"Image and prediction shapes do not match: image={image.shape}, prediction={label.shape}"
        )

    z_count = image.shape[-1]
    z = int(np.clip(z, 0, z_count - 1))
    gray = _auto_window_to_uint8(image[:, :, z])
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    cfg = get_dataset_config(dataset_name)
    colored = np.zeros_like(rgb)
    foreground = np.zeros(gray.shape, dtype=bool)
    legends = []
    for label_id, name in cfg["label_map"].items():
        current = label[:, :, z] == label_id
        if current.any():
            colored[current] = cfg["label_colors"][label_id]
            foreground |= current
        legends.append(f"{label_id}={name}")

    overlay = rgb.copy()
    alpha = 0.35
    overlay[foreground] = (1 - alpha) * rgb[foreground] + alpha * colored[foreground]
    overlay = np.rot90(overlay.clip(0, 255).astype(np.uint8), k=3)
    return overlay, f"Overlay z={z}/{z_count - 1} ({', '.join(legends)})"


def pick_best_z_from_label(label_path: str, dataset_name: str = DEFAULT_DATASET):
    label = _get_cached("label", label_path)
    cfg = get_dataset_config(dataset_name)
    target_labels = cfg.get("best_slice_labels") or list(cfg["label_map"].keys())
    area = np.isin(label, target_labels).sum(axis=(0, 1))
    return int(area.argmax()) if area.max() > 0 else label.shape[-1] // 2


def on_z_change_update_overlay(z, image_path, pred_path, dataset_name):
    if not image_path or not os.path.exists(image_path):
        return None, "", None, "No image loaded.", -1
    z = int(z)
    preview, preview_info = render_image_preview(image_path, z)
    if pred_path and os.path.exists(pred_path):
        overlay, overlay_info = render_seg_overlay(image_path, pred_path, z, dataset_name)
        return preview, preview_info, overlay, overlay_info, z
    return preview, preview_info, None, "Run segmentation first.", -1


def on_acdc_phase_change(phase, acdc_state, dataset_name):
    if dataset_name != "ACDC cine-MRI" or not acdc_state:
        return "", "", gr.update(interactive=False), None, "", None, "", -1
    phase = str(phase or "ED").upper()
    phase_data = acdc_state.get(phase, {})
    image_path = phase_data.get("image", "")
    pred_path = phase_data.get("pred", "")
    if not image_path or not os.path.exists(image_path):
        return "", "", gr.update(interactive=False), None, "", None, f"Missing {phase} image.", -1

    if pred_path and os.path.exists(pred_path):
        best_z = pick_best_z_from_label(pred_path, dataset_name)
    else:
        volume = _get_cached("image", image_path)
        best_z = volume.shape[-1] // 2
    volume = _get_cached("image", image_path)
    slider = gr.update(
        minimum=0,
        maximum=volume.shape[-1] - 1,
        value=best_z,
        step=1,
        interactive=True,
    )
    preview, preview_info = render_image_preview(image_path, best_z)
    if pred_path and os.path.exists(pred_path):
        overlay, overlay_info = render_seg_overlay(image_path, pred_path, best_z, dataset_name)
    else:
        overlay, overlay_info = None, f"No {phase} prediction loaded."
    frame = phase_data.get("frame")
    preview_info = f"{phase} frame={frame}; {preview_info}"
    return image_path, pred_path, slider, preview, preview_info, overlay, overlay_info, best_z


def reset_case_outputs(dataset_name):
    is_acdc = dataset_name == "ACDC cine-MRI"
    return (
        "", "", "", "", "",
        gr.update(minimum=0, maximum=1, value=0, interactive=False),
        None, "", None, "", -1,
        "Files selected. Click Run Segmentation + Build facts.json.",
        {}, gr.update(visible=is_acdc, value="ED"),
        [],
    )


def on_dataset_change(dataset_name, ct_facts_dir):
    cfg = get_dataset_config(dataset_name, ct_facts_dir)
    is_ct = cfg["key"] == "ct"
    description = (
        "Upload one Cardiac CT NIfTI file."
        if is_ct
        else "Upload one ACDC patient directory containing Info.cfg and ED/ES frame files (or the 4D cine file)."
    )
    return (
        gr.update(value=cfg["model_name"]),
        gr.update(value=cfg["checkpoint"]),
        gr.update(value=cfg["infer_dir"]),
        gr.update(value=cfg["facts_out_dir"]),
        gr.update(value=cfg["infer_py"]),
        gr.update(
            value=cfg.get("rf_model_path", ""),
            visible=not is_ct,
            interactive=not is_ct,
        ),
        gr.update(visible=is_ct, interactive=is_ct),
        gr.update(visible=is_ct, value=None),
        gr.update(visible=not is_ct, value=None),
        gr.update(visible=not is_ct, value="ED"),
        gr.update(minimum=0, maximum=1, value=0, interactive=False),
        None, None, "", "", description,
        "", "", "", "", "", {}, [],
    )


def messages_to_turns(history_messages):
    """Convert Gradio 'messages' history -> list[(user, assistant)] for prompt building."""
    turns = []
    pending_user = None
    for m in history_messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant":
            if pending_user is not None:
                turns.append((pending_user, content))
                pending_user = None
    return turns

def build_chat_prompt(facts_block: str, history_messages, user_msg: str, keep_last_k: int = 2) -> str:
    turns = messages_to_turns(history_messages)
    turns = turns[-keep_last_k:] if turns else []

    history_text = ""
    for u, a in turns:
        history_text += f"Previous Question: {u}\nPrevious Answer: {a}\n\n"

    return (
        f"### System:\n"
        f"{SYSTEM}\n\n"
        f"### User:\n"
        f"{history_text}"
        f"Question:\n"
        f"{user_msg}\n\n"
        f"FACTS:\n"
        f"{facts_block}\n\n"
        f"Instruction:\n"
        f"Answer the question using only the FACTS. "
        f"Write a natural-language answer. Do not output raw JSON or dictionaries.\n\n"
        f"### Assistant:\n"
    )

@torch.no_grad()
def generate_chat(prompt: str, max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0) -> str:
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]

    out = MODEL.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        pad_token_id=TOKENIZER.eos_token_id,
        eos_token_id=TOKENIZER.eos_token_id,
        repetition_penalty=1.05,
    )

    gen_ids = out[0][input_len:]
    ans = TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()

    stop_tokens = [
        "\n###",
        "\nUser:",
        "\nQuestion:",
        "\nFACTS:",
        "### User:",
        "### System:",
        "### Assistant:",
    ]

    for stop in stop_tokens:
        if stop in ans:
            ans = ans.split(stop, 1)[0].strip()

    # Optional cleanup
    ans = ans.strip()

    return ans

def chat_respond(user_msg, history, facts_path, max_new_tokens, temperature, top_p, derived_only):
    history = history or []
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return history, ""

    if (not facts_path) or (not os.path.exists(facts_path)):
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "No facts loaded. Upload a cardiac image and run segmentation (or load cached facts) first."}
        ]
        return history, ""

    facts_obj = load_facts_file(facts_path)
    facts_block = build_facts_block(facts_obj, derived_only=derived_only)

    prompt = build_chat_prompt(facts_block, history, user_msg, keep_last_k=4)
    print("Prompt：", prompt)
    ans = generate_chat(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    print("Ans：", ans)

    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ans}
    ]
    print("History：", history)
    return history, ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="HF model id")
    parser.add_argument("--lora_dir", required=True, help="LoRA adapter directory")
    parser.add_argument("--facts_dir", required=True, help="Cardiac CT facts output directory")
    parser.add_argument("--cache_dir", default=r"D:\CardiacRate\hf_cache")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    print("Loading model...")
    load_model(args.base_model, args.lora_dir, args.cache_dir, args.trust_remote_code)

    global FACTS_INDEX
    FACTS_INDEX = build_index(args.facts_dir)

    ct_facts_dir_state_value = args.facts_dir
    initial_cfg = get_dataset_config(DEFAULT_DATASET, ct_facts_dir_state_value)

    with gr.Blocks(title="Cardiac Agent (CT + ACDC)") as demo:
        gr.Markdown(
            "# Cardiac Agent (Segmentation + facts.json + LLM QA)\n"
            "Cardiac CT uses one 3D NIfTI image. ACDC processes ED and ES, "
            "calculates MRI cardiac measurements, runs five-class Random Forest "
            "prediction, and provides both evidence sources to the LLM."
        )

        ct_facts_dir_state = gr.State(ct_facts_dir_state_value)
        acdc_case_state = gr.State({})
        overlay_z_state = gr.State(-1)

        with gr.Tabs():
            with gr.TabItem("New cardiac case"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=390):
                        gr.Markdown("## 1) Select dataset → Upload → Segment → Build facts")

                        dataset_selector = gr.Dropdown(
                            choices=list(DATASET_CONFIGS.keys()),
                            value=DEFAULT_DATASET,
                            label="Dataset / modality",
                            interactive=True,
                        )

                        ct_upload = gr.File(
                            label="Upload Cardiac CT (.nii/.nii.gz)",
                            file_count="single",
                            file_types=[".nii", ".gz"],
                            type="filepath",
                            visible=True,
                        )
                        acdc_upload = gr.File(
                            label="Upload one ACDC patient directory",
                            file_count="directory",
                            type="filepath",
                            visible=False,
                        )

                        phase_selector = gr.Radio(
                            choices=["ED", "ES"],
                            value="ED",
                            label="ACDC cardiac phase",
                            visible=False,
                            interactive=True,
                        )

                        z_slider = gr.Slider(
                            0, 1, value=0, step=1, label="Z slice", interactive=False
                        )

                        run_btn = gr.Button(
                            "Run Segmentation + Build facts.json", variant="primary"
                        )

                        image_preview_img = gr.Image(label="Image preview", height=260)
                        seg_overlay_img = gr.Image(label="Segmentation overlay", height=260)
                        image_preview_info = gr.Textbox(label="Image preview info", interactive=False)
                        overlay_info = gr.Textbox(label="Overlay info", interactive=False)
                        status_box = gr.Textbox(
                            value="Upload one Cardiac CT NIfTI file.",
                            label="Status / Error",
                            interactive=False,
                        )

                        model_name = gr.Textbox(
                            value=initial_cfg["model_name"], label="Segmentation model_name"
                        )
                        checkpoint = gr.Textbox(
                            value=initial_cfg["checkpoint"], label="Checkpoint path"
                        )
                        infer_dir = gr.Textbox(
                            value=initial_cfg["infer_dir"], label="Prediction output directory"
                        )
                        facts_out_dir = gr.Textbox(
                            value=initial_cfg["facts_out_dir"], label="facts output directory"
                        )
                        infer_script = gr.Textbox(
                            value=initial_cfg["infer_py"],
                            label="Inference script",
                            interactive=False,
                        )
                        rf_model_path = gr.Textbox(
                            value=initial_cfg.get("rf_model_path", ""),
                            label="ACDC Random Forest model path",
                            visible=False,
                            interactive=True,
                        )
                        min_calci_vox = gr.Slider(
                            0,
                            200,
                            value=20,
                            step=1,
                            label="min_calci_vox (Cardiac CT only)",
                            visible=True,
                        )

                        out_image_path = gr.Textbox(label="Displayed image path", interactive=False)
                        out_pred_path = gr.Textbox(label="Displayed prediction path", interactive=False)
                        out_facts_path = gr.Textbox(label="facts.json path", interactive=False)

                        with gr.Accordion("Debug / Preview", open=False):
                            out_facts_preview = gr.Textbox(label="facts preview", lines=14)
                            out_infer_log = gr.Textbox(label="Inference / facts log", lines=12)

                    with gr.Column(scale=2, min_width=520):
                        gr.Markdown("## 2) Chat (grounded on generated facts.json)")
                        chatbot = gr.Chatbot(label="Chat", height=520)
                        user_msg = gr.Textbox(
                            label="Message",
                            placeholder="Ask about this case using the generated facts...",
                            lines=2,
                        )
                        with gr.Row():
                            send_btn = gr.Button("Send", variant="primary")
                            clear_btn = gr.Button("Clear chat")
                        with gr.Row():
                            max_new = gr.Slider(32, 512, value=128, step=8, label="max_new_tokens")
                            temp = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="temperature")
                            top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top_p")
                        derived_only = gr.Checkbox(value=False, label="Use compact facts only")

                dataset_selector.change(
                    fn=on_dataset_change,
                    inputs=[dataset_selector, ct_facts_dir_state],
                    outputs=[
                        model_name, checkpoint, infer_dir, facts_out_dir, infer_script,
                        rf_model_path, min_calci_vox, ct_upload, acdc_upload,
                        phase_selector, z_slider,
                        image_preview_img, seg_overlay_img, image_preview_info, overlay_info,
                        status_box, out_image_path, out_pred_path, out_facts_path,
                        out_facts_preview, out_infer_log, acdc_case_state, chatbot,
                    ],
                )

                reset_outputs = [
                    out_image_path, out_pred_path, out_facts_path, out_facts_preview,
                    out_infer_log, z_slider, image_preview_img, image_preview_info,
                    seg_overlay_img, overlay_info, overlay_z_state, status_box,
                    acdc_case_state, phase_selector, chatbot,
                ]
                ct_upload.change(
                    fn=reset_case_outputs,
                    inputs=[dataset_selector],
                    outputs=reset_outputs,
                )
                acdc_upload.change(
                    fn=reset_case_outputs,
                    inputs=[dataset_selector],
                    outputs=reset_outputs,
                )

                run_btn.click(
                    fn=pipeline_new_case,
                    inputs=[
                        dataset_selector, ct_upload, acdc_upload, model_name, checkpoint,
                        infer_dir, facts_out_dir, rf_model_path, min_calci_vox,
                    ],
                    outputs=[
                        out_image_path, out_pred_path, out_facts_path, out_facts_preview,
                        out_infer_log, z_slider, image_preview_img, image_preview_info,
                        seg_overlay_img, overlay_info, overlay_z_state, status_box,
                        acdc_case_state, phase_selector,
                    ],
                )

                phase_selector.change(
                    fn=on_acdc_phase_change,
                    inputs=[phase_selector, acdc_case_state, dataset_selector],
                    outputs=[
                        out_image_path, out_pred_path, z_slider, image_preview_img,
                        image_preview_info, seg_overlay_img, overlay_info, overlay_z_state,
                    ],
                )

                z_slider.change(
                    fn=on_z_change_update_overlay,
                    inputs=[z_slider, out_image_path, out_pred_path, dataset_selector],
                    outputs=[
                        image_preview_img, image_preview_info, seg_overlay_img,
                        overlay_info, overlay_z_state,
                    ],
                )

                send_btn.click(
                    fn=chat_respond,
                    inputs=[user_msg, chatbot, out_facts_path, max_new, temp, top_p, derived_only],
                    outputs=[chatbot, user_msg],
                )
                user_msg.submit(
                    fn=chat_respond,
                    inputs=[user_msg, chatbot, out_facts_path, max_new, temp, top_p, derived_only],
                    outputs=[chatbot, user_msg],
                )
                clear_btn.click(lambda: [], outputs=[chatbot])

        demo.launch(share=args.share)


if __name__ == "__main__":
    main()
