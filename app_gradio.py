import os
import json
import glob
import argparse
import time
import shutil
import subprocess
import sys

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

LABEL_MAP = {
    1: "myocardium",
    2: "aortic_valve",
    3: "aortic_valve_calcification",
}

SYSTEM = (
    "You are a cardiac assistant. Answer in English. "
    "Use ONLY the provided FACTS. If the answer is not available in FACTS, "
    "reply exactly: Not available in provided facts."
)

MODEL = None
TOKENIZER = None
FACTS_INDEX = {}   # case_id -> facts_path
DEVICE = None
_VOL_CACHE = {"ct": None, "lbl": None}

def load_facts_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_facts_block(facts_obj: dict, derived_only: bool = True) -> str:
    """Keep it short to reduce hallucination."""
    if derived_only:
        payload = {
            "case_id": facts_obj.get("case_id", ""),
            "qc_flags": facts_obj.get("qc_flags", []),
            "derived": facts_obj.get("derived", {})
        }
    else:
        payload = facts_obj
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


def answer_from_case(case_id: str, question: str, max_new_tokens: int, temperature: float, top_p: float, derived_only: bool):
    if not question or not question.strip():
        return "", "Please enter a question."

    if case_id not in FACTS_INDEX:
        return "", "Case not found."

    facts_path = FACTS_INDEX[case_id]
    facts_obj = load_facts_file(facts_path)
    facts_block = build_facts_block(facts_obj, derived_only=derived_only)
    prompt = build_prompt(facts_block, question)

    ans = generate_answer(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

    # Show evidence snippet (facts block) for transparency
    evidence = f"Facts file: {facts_path}\n\n{facts_block}"
    return ans, evidence


def answer_from_upload(upload_file, question: str, max_new_tokens: int, temperature: float, top_p: float, derived_only: bool):
    if not question or not question.strip():
        return "", "Please enter a question."

    if upload_file is None:
        return "", "Please upload a facts.json file."

    facts_obj = load_facts_file(upload_file.name)
    facts_block = build_facts_block(facts_obj, derived_only=derived_only)
    prompt = build_prompt(facts_block, question)

    ans = generate_answer(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

    evidence = f"Uploaded: {upload_file.name}\n\n{facts_block}"
    return ans, evidence


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

def _strip_nii(name: str):
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


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


def make_facts_from_ct_and_label(ct_path: str, label_path: str, out_facts_path: str, min_calci_vox: int = 20):
    ct_img = nib.load(ct_path)
    spacing_xyz = ct_img.header.get_zooms()[:3]

    lbl_img = nib.load(label_path)
    lbl = lbl_img.get_fdata().astype(np.int16)

    qc_flags = []
    if lbl.shape != ct_img.shape:
        qc_flags.append("shape_mismatch_ct_label")

    vvox = voxel_volume_mm3(spacing_xyz)
    case_id = _strip_nii(os.path.basename(ct_path))

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
        "qc_flags": qc_flags,
    }

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
            idx = np.where(mask)
            st["centroid_voxel"] = compute_centroid(idx)
            st["bbox_voxel"] = compute_bbox(idx)

        if name == "aortic_valve_calcification":
            num_cc, largest_mm3 = connected_components_stats(mask, spacing_xyz)
            st["num_connected_components"] = num_cc
            st["largest_component_mm3"] = largest_mm3

        facts["structures"][name] = st

    # threshold calcification tiny noise
    calci = facts["structures"]["aortic_valve_calcification"]
    if 0 < calci["voxel_count"] < min_calci_vox:
        facts["qc_flags"].append("calcification_below_threshold")
        calci["present"] = False
        calci["voxel_count"] = 0
        calci["volume_mm3"] = 0.0
        calci["volume_ml"] = 0.0
        calci["num_connected_components"] = 0
        calci["largest_component_mm3"] = 0.0

    valve = facts["structures"]["aortic_valve"]
    if valve["voxel_count"] == 0:
        facts["qc_flags"].append("empty_aortic_valve")
        if calci["present"]:
            facts["qc_flags"].append("inconsistent_calci_without_valve")

    if len(facts["qc_flags"]) == 0:
        facts["qc_flags"].append("ok")

    facts["derived"] = {
        "myocardium_volume_ml": float(facts["structures"]["myocardium"]["volume_ml"]),
        "aortic_valve_volume_ml": float(valve["volume_ml"]),
        "calcification_present": bool(calci["present"]),
        "calcification_volume_mm3": float(calci["volume_mm3"] if calci["present"] else 0.0),
        "calcification_volume_ml": float(calci["volume_ml"] if calci["present"] else 0.0),
        "calcification_to_valve_ratio": float((calci["volume_mm3"] if calci["present"] else 0.0) / max(valve["volume_mm3"], 1e-6))
    }

    # store a bbox prompt for valve (optional)
    if "bbox_voxel" in valve:
        facts["prompts"].append({
            "type": "bbox",
            "target": "aortic_valve",
            "prompt_bbox_voxel": valve["bbox_voxel"]
        })

    os.makedirs(os.path.dirname(out_facts_path), exist_ok=True)
    with open(out_facts_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)

    return facts


def run_segmentation_infer(
    infer_py: str,
    model_name: str,
    checkpoint: str,
    ct_path: str,
    infer_dir: str
):
    """
    Calls: python Segmentation\\infer.py --model_name ... --checkpoint ... --img_pth ... --infer_dir ...
    Returns: predicted label path (best guess: newest nii/nii.gz in infer_dir).
    """
    os.makedirs(infer_dir, exist_ok=True)

    before = set(glob.glob(os.path.join(infer_dir, "*.nii*")))

    cmd = [
        sys.executable, infer_py,
        "--model_name", model_name,
        "--checkpoint", checkpoint,
        "--img_pth", ct_path,
        "--infer_dir", infer_dir
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"infer.py failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    after = set(glob.glob(os.path.join(infer_dir, "*.nii*")))
    new_files = list(after - before)

    if new_files:
        # pick newest among new files
        pred = max(new_files, key=lambda p: os.path.getmtime(p))
        return pred, proc.stdout
    else:
        # fallback: pick newest overall
        all_files = list(after)
        if not all_files:
            raise RuntimeError(f"infer.py finished but no nii/nii.gz found in {infer_dir}\nSTDOUT:\n{proc.stdout}")
        pred = max(all_files, key=lambda p: os.path.getmtime(p))
        return pred, proc.stdout


def answer_from_facts_path(facts_path: str, question: str, max_new_tokens: int, temperature: float, top_p: float, derived_only: bool):
    if not facts_path or not os.path.exists(facts_path):
        return "", "facts.json not found. Please run segmentation + facts first."
    if not question or not question.strip():
        return "", "Please enter a question."

    facts_obj = load_facts_file(facts_path)
    facts_block = build_facts_block(facts_obj, derived_only=derived_only)
    prompt = build_prompt(facts_block, question)
    ans = generate_answer(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    evidence = f"Facts file: {facts_path}\n\n{facts_block}"
    return ans, evidence


def pipeline_new_ct(
    ct_upload,
    model_name: str,
    checkpoint: str,
    infer_dir: str,
    facts_out_dir: str,
    min_calci_vox: int,
    progress=gr.Progress(track_tqdm=True)
):
    """
    1) copy uploaded CT to a stable D: working dir
    2) run infer.py to get pred label
    3) build facts.json
    """
    if ct_upload is None:
        return "", "", "", "", "Please upload a CT (.nii/.nii.gz)."

    # Make sure paths exist
    os.makedirs(infer_dir, exist_ok=True)
    os.makedirs(facts_out_dir, exist_ok=True)

    # Copy CT to working dir to avoid Gradio temp path issues
    progress(0.05, desc="Preparing CT file...")
    src = ct_upload.name
    base = os.path.basename(src)
    case_id = _strip_nii(base)
    # Avoid collisions
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst_ct = os.path.join(infer_dir, f"{case_id}_{stamp}.nii.gz" if base.endswith(".nii.gz") else f"{case_id}_{stamp}.nii")
    shutil.copyfile(src, dst_ct)

    # Run segmentation
    progress(0.25, desc="Running segmentation inference...")
    infer_py = os.path.join("Segmentation", "infer.py")
    pred_label_path, infer_stdout = run_segmentation_infer(
        infer_py=infer_py,
        model_name=model_name,
        checkpoint=checkpoint,
        ct_path=dst_ct,
        infer_dir=infer_dir
    )

    # Build facts
    progress(0.7, desc="Building facts.json...")
    out_facts_path = os.path.join(facts_out_dir, f"{_strip_nii(os.path.basename(dst_ct))}.json")
    facts_obj = make_facts_from_ct_and_label(
        ct_path=dst_ct,
        label_path=pred_label_path,
        out_facts_path=out_facts_path,
        min_calci_vox=min_calci_vox
    )

    progress(1.0, desc="Done.")
    facts_preview = json.dumps(
        {"case_id": facts_obj.get("case_id"), "qc_flags": facts_obj.get("qc_flags"), "derived": facts_obj.get("derived")},
        ensure_ascii=False, indent=2
    )
    # after facts built

    best_z = pick_best_z_from_valve(pred_label_path)

    ct_img, ct_info = render_ct_preview(dst_ct, best_z)
    overlay, ov_info = render_seg_overlay(dst_ct, pred_label_path, best_z)

    ct_vol = _get_cached("ct", dst_ct)
    Z = ct_vol.shape[-1]
    slider_upd = gr.update(minimum=0, maximum=Z-1, value=best_z, step=1, interactive=True)

    # 注意：這裡最後多回傳 overlay_z_state
    return (
        dst_ct, pred_label_path, out_facts_path, facts_preview, infer_stdout,
        slider_upd, ct_img, ct_info,
        overlay, ov_info,
        dst_ct, pred_label_path, best_z, ""   # ct_path_state, pred_path_state, overlay_z_state, status
    )

def _load_nii_full(path: str):
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 4:
        arr = arr[..., 0]
    return arr

def _get_cached(kind: str, path: str):
    mtime = os.path.getmtime(path)
    entry = _VOL_CACHE.get(kind)
    if entry and entry["path"] == path and entry["mtime"] == mtime:
        return entry["vol"]
    vol = _load_nii_full(path)
    if isinstance(vol, tuple):  # 防呆
        vol = vol[0]
    _VOL_CACHE[kind] = {"path": path, "mtime": mtime, "vol": vol}
    return vol

def _auto_window_to_uint8(x2d: np.ndarray):
    x = x2d.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max() + 1e-6)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)
    return (x * 255).astype(np.uint8)

def render_ct_preview(ct_path: str, z: int = None):
    ct = _get_cached("ct", ct_path)  # (X,Y,Z)
    Z = ct.shape[-1]
    if z is None:
        z = Z // 2
    z = int(np.clip(z, 0, Z - 1))
    sl = _auto_window_to_uint8(ct[:, :, z])
    sl = np.rot90(sl, k=3)  # clockwise 90°
    return sl, f"CT shape={ct.shape}, z={z}/{Z-1}"


def render_seg_overlay(ct_path: str, lbl_path: str, z: int):
    ct = _get_cached("ct", ct_path)
    lbl = _get_cached("lbl", lbl_path)

    Z = ct.shape[-1]
    z = int(np.clip(z, 0, Z - 1))
    if z is None:
        # default: choose best valve slice
        area = (lbl == 2).sum(axis=(0, 1))
        z = int(area.argmax()) if area.max() > 0 else (Z // 2)

    z = int(np.clip(z, 0, Z - 1))

    ct_u8 = _auto_window_to_uint8(ct[:, :, z])
    rgb = np.stack([ct_u8, ct_u8, ct_u8], axis=-1).astype(np.float32)

    m1 = (lbl[:, :, z] == 1)  # myocardium
    m2 = (lbl[:, :, z] == 2)  # valve
    m3 = (lbl[:, :, z] == 3)  # calcification

    mask_rgb = np.zeros_like(rgb)
    mask_rgb[m1] = [0, 255, 0]
    mask_rgb[m2] = [255, 255, 0]
    mask_rgb[m3] = [255, 0, 0]

    overlay = rgb.copy()
    alpha = 0.35
    anym = m1 | m2 | m3
    overlay[anym] = (1 - alpha) * rgb[anym] + alpha * mask_rgb[anym]

    overlay = overlay.clip(0, 255).astype(np.uint8)
    overlay = np.rot90(overlay, k=3)  # clockwise 90°
    info = f"Overlay z={z}/{Z-1} (myocardium=Green,aortic_valve=Yellow,aortic_valve_calcification=Red)"
    return overlay, info


def pick_best_z_from_valve(lbl_path: str):
    lbl = _get_cached("lbl", lbl_path)
    # 找 valve(label==2) 面積最大的那層
    area = (lbl == 2).sum(axis=(0, 1))
    return int(area.argmax()) if area.max() > 0 else (lbl.shape[-1] // 2)

def on_ct_upload_init(file):
    if file is None:
        return (gr.update(interactive=False), None, "", "", "", None, None, "")
    p = file.name
    pl = p.lower()
    # 前端放行 .gz 是為了 .nii.gz，後端這裡再嚴格擋掉
    if not (pl.endswith(".nii") or pl.endswith(".nii.gz")):
        return (gr.update(interactive=False), None, "", "", "", None, None, "Invalid file type. Please upload .nii/.nii.gz")

    ct = _get_cached("ct", p)
    Z = ct.shape[-1]
    z0 = Z // 2
    img, info = render_ct_preview(p, z0)

    slider_upd = gr.update(minimum=0, maximum=Z-1, value=z0, step=1, interactive=True)
    # 清掉 pred 狀態 & segmentation preview
    return slider_upd, img, info, p, "", None, None, ""

def on_z_change_update_overlay(z, ct_path, pred_path):
    # 拖 slider 後：更新 CT +（如果已有 pred）更新 overlay
    if not ct_path or not os.path.exists(ct_path):
        return None, "", None, "No CT loaded.", -1

    z = int(z)
    ct_img, ct_info = render_ct_preview(ct_path, z)

    if pred_path and os.path.exists(pred_path):
        overlay, ov_info = render_seg_overlay(ct_path, pred_path, z)
        return ct_img, ct_info, overlay, ov_info, z
    else:
        return ct_img, ct_info, None, "Run segmentation first to generate prediction.", -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--lora_dir", required=True, help="Your LoRA output dir, e.g. D:\\heart_lora")
    ap.add_argument("--facts_dir", required=True, help="Folder containing facts/*.json")
    ap.add_argument("--cache_dir", default=r"D:\CardiacRate\hf_cache")
    ap.add_argument("--trust_remote_code", action="store_true", help="Enable for some models like Qwen if needed")
    ap.add_argument("--share", action="store_true", help="Create a public share link (optional)")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print("Loading model...")
    load_model(args.base_model, args.lora_dir, args.cache_dir, args.trust_remote_code)

    global FACTS_INDEX
    FACTS_INDEX = build_index(args.facts_dir)
    case_ids = sorted(FACTS_INDEX.keys())

    # infer variable
    model_name = "unetcnx_a1"
    checkpoint = r"D:\CardiacRate\Segmentation\model\unetcnx_a1\best_model.pth"
    infer_dir = r"D:\CardiacRate\Segmentation\infer\predict"
    facts_out_dir = r"D:\CardiacRate\facts_new"

    with gr.Blocks(title="Cardiac Agent (Seg + Facts + LLM)") as demo:
        gr.Markdown("# Cardiac Agent (Segmentation + facts.json + LLM QA)\nAll answers are grounded on facts.json.")

        # --- Existing tabs (you already have) can stay; here I just keep it minimal
        with gr.Tabs():
            with gr.TabItem("New CT (Upload → Segment → facts → QA)"):
                with gr.Row():
                    # LEFT: (1) Upload/Segment/Facts  — smaller
                    with gr.Column(scale=1, min_width=380):
                        gr.Markdown("## 1) Upload CT → Segmentation → facts.json")

                        ct_upload = gr.File(
                            label="Upload CT (.nii/.nii.gz)",
                            file_types=[".nii", ".gz"]
                        )
                        ct_path_state = gr.State("")
                        pred_path_state = gr.State("")
                        overlay_z_state = gr.State(-1)
                        
                        z_slider = gr.Slider(0, 1, value=0, step=1, label="Z slice", interactive=False)
                        ct_preview_img = gr.Image(label="CT preview", height=260)
                        seg_overlay_img = gr.Image(label="Segmentation overlay", height=260)
                        
                        ct_preview_info = gr.Textbox(label="CT preview info", interactive=False)
                        overlay_info = gr.Textbox(label="Overlay info", interactive=False)

                        status_box = gr.Textbox(label="Status / Error", interactive=False)


                        ct_upload.change(
                            fn=on_ct_upload_init,  # 你原本的 init 可以保留；但 outputs 要對齊你現在元件
                            inputs=[ct_upload],
                            outputs=[z_slider, ct_preview_img, ct_preview_info, ct_path_state, pred_path_state, status_box]
                        )

                        z_slider.input(
                            fn=on_z_change_update_overlay,
                            inputs=[z_slider, ct_path_state, pred_path_state],
                            outputs=[ct_preview_img, ct_preview_info, seg_overlay_img, overlay_info, overlay_z_state]
                        )


                        model_name = gr.Textbox(value="unetcnx_a1", label="Segmentation model_name")
                        checkpoint = gr.Textbox(
                            value=r"D:\CardiacRate\Segmentation\model\unetcnx_a1\best_model.pth",
                            label="Checkpoint path"
                        )

                        infer_dir = gr.Textbox(
                            value=r"D:\CardiacRate\Segmentation\infer\predict",
                            label="infer_dir (pred output)"
                        )
                        facts_out_dir = gr.Textbox(
                            value=r"D:\CardiacRate\facts_new",
                            label="facts_out_dir"
                        )

                        min_calci_vox = gr.Slider(
                            0, 200, value=20, step=1,
                            label="min_calci_vox"
                        )

                        run_btn = gr.Button("Run Segmentation + Build facts.json", variant="primary")


                        out_ct_path = gr.Textbox(label="Saved CT path", interactive=False)
                        out_pred_path = gr.Textbox(label="Pred label path", interactive=False)
                        out_facts_path = gr.Textbox(label="facts.json path", interactive=False)

                        # 把 preview/log 放左邊會讓左邊變很長；如果你想更緊湊，可以先折疊：
                        with gr.Accordion("Debug / Preview", open=False):
                            out_facts_preview = gr.Textbox(label="facts preview (derived)", lines=10)
                            out_infer_log = gr.Textbox(label="infer.py STDOUT (debug)", lines=10)

                        run_btn.click(
                            fn=pipeline_new_ct,
                            inputs=[ct_upload, model_name, checkpoint, infer_dir, facts_out_dir, min_calci_vox],
                            outputs=[
                                out_ct_path, out_pred_path, out_facts_path, out_facts_preview, out_infer_log,
                                z_slider, ct_preview_img, ct_preview_info,
                                seg_overlay_img, overlay_info,
                                ct_path_state, pred_path_state, overlay_z_state, status_box
                            ]
                        )


                    # RIGHT: (2) Ask questions — larger
                    with gr.Column(scale=2, min_width=520):
                        gr.Markdown("## 2) Ask questions (grounded on the generated facts.json)")

                        q = gr.Textbox(
                            label="Question (English)",
                            placeholder="e.g., Is aortic valve calcification present?"
                        )

                        with gr.Row():
                            max_new = gr.Slider(32, 512, value=128, step=8, label="max_new_tokens")
                            temp = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="temperature (0 = deterministic)")
                            top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top_p")

                        derived_only = gr.Checkbox(value=True, label="Use derived-only facts (recommended)")
                        ask_btn = gr.Button("Answer", variant="primary")

                        ans = gr.Textbox(label="Answer", lines=6)
                        ev = gr.Textbox(label="Evidence (facts used)", lines=14)

                        ask_btn.click(
                            fn=answer_from_facts_path,
                            inputs=[out_facts_path, q, max_new, temp, top_p, derived_only],
                            outputs=[ans, ev]
                        )


        demo.launch(share=args.share)


if __name__ == "__main__":
    main()
