import os
import json
import glob
import argparse
import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM = (
    "You are a cardiac assistant. Answer in English. "
    "Use ONLY the provided FACTS. If the answer is not available in FACTS, "
    "reply exactly: Not available in provided facts."
)

MODEL = None
TOKENIZER = None
FACTS_INDEX = {}   # case_id -> facts_path
DEVICE = None


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
    if not case_ids:
        raise FileNotFoundError(f"No facts json found in {args.facts_dir}")

    with gr.Blocks(title="Cardiac LLM (Facts-grounded QA)") as demo:
        gr.Markdown("# Cardiac LLM (Facts-grounded QA)\nAnswer in English using ONLY facts.json evidence.")

        with gr.Tabs():
            with gr.TabItem("Select Case"):
                case_dropdown = gr.Dropdown(choices=case_ids, value=case_ids[0], label="Case ID")
                q1 = gr.Textbox(label="Question (English)", placeholder="e.g., Is aortic valve calcification present?")
                with gr.Row():
                    max_new = gr.Slider(32, 512, value=128, step=8, label="max_new_tokens")
                    temp = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="temperature (0 = deterministic)")
                    top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top_p")
                derived_only = gr.Checkbox(value=True, label="Use derived-only facts (recommended)")
                btn1 = gr.Button("Answer")
                ans1 = gr.Textbox(label="Answer", lines=5)
                ev1 = gr.Textbox(label="Evidence (facts used)", lines=12)

                btn1.click(
                    fn=answer_from_case,
                    inputs=[case_dropdown, q1, max_new, temp, top_p, derived_only],
                    outputs=[ans1, ev1]
                )

            with gr.TabItem("Upload facts.json"):
                upload = gr.File(label="Upload facts.json", file_types=[".json"])
                q2 = gr.Textbox(label="Question (English)")
                with gr.Row():
                    max_new2 = gr.Slider(32, 512, value=128, step=8, label="max_new_tokens")
                    temp2 = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="temperature")
                    top_p2 = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top_p")
                derived_only2 = gr.Checkbox(value=True, label="Use derived-only facts (recommended)")
                btn2 = gr.Button("Answer")
                ans2 = gr.Textbox(label="Answer", lines=5)
                ev2 = gr.Textbox(label="Evidence (facts used)", lines=12)

                btn2.click(
                    fn=answer_from_upload,
                    inputs=[upload, q2, max_new2, temp2, top_p2, derived_only2],
                    outputs=[ans2, ev2]
                )

        gr.Markdown("**Tip:** keep temperature=0 for strict, repeatable numeric answers.")

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
