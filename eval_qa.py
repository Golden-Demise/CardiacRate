import os
import re
import csv
import json
import random
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


SYSTEM = (
    "You are a cardiac CT assistant. Answer in English. "
    "Use ONLY the provided FACTS. "
    "Do not use external medical knowledge. "
    "Do not invent unsupported findings. "
    "If the answer is not available in FACTS, reply exactly: "
    "\"Not available in provided facts.\" "
    "When answering numeric questions, convert the relevant FACTS into a concise natural-language sentence. "
    "Do not output raw JSON, Python dictionaries, key-value pairs, or code. "
    "Do not repeat the prompt, FACTS, or dialogue. "
    "Output only the final answer."
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_facts_index(facts_dir):
    facts_dir = Path(facts_dir)
    index = {}

    for p in sorted(facts_dir.glob("*.json")):
        try:
            obj = load_json(p)
            patient_id = obj.get("patient_id") or obj.get("case_id") or p.stem
            case_id = obj.get("case_id") or patient_id

            index[patient_id] = str(p)
            index[case_id] = str(p)
            index[p.stem] = str(p)

        except Exception as e:
            print(f"[WARN] Failed to index {p}: {e}")

    return index


def build_facts_block(facts_obj):
    case_id = facts_obj.get("case_id") or facts_obj.get("patient_id", "")

    payload = {
        "case_id": case_id,
        "patient_id": facts_obj.get("patient_id", case_id),
        "modality": facts_obj.get("modality", "cardiac_ct"),
        "image_info": facts_obj.get("image_info", {}),
        "image_shape": facts_obj.get("image_shape", None),
        "spacing_mm": facts_obj.get("spacing_mm", None),
        "structures": facts_obj.get("structures", {}),
        "derived": facts_obj.get("derived", {}),
        "derived_metrics": facts_obj.get("derived_metrics", {}),
        "answerable_findings": facts_obj.get("answerable_findings", {}),
        "limitations": facts_obj.get("limitations", []),
        "qc_flags": facts_obj.get("qc_flags", []),
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_prompt(facts_block, question):
    return (
        f"### System:\n"
        f"{SYSTEM}\n\n"
        f"### User:\n"
        f"Question:\n"
        f"{question}\n\n"
        f"FACTS:\n"
        f"{facts_block}\n\n"
        f"Instruction:\n"
        f"Answer the question using only the FACTS. "
        f"Write a natural-language answer. Do not output raw JSON or dictionaries.\n\n"
        f"### Assistant:\n"
    )


def load_model(base_model, lora_dir, cache_dir=None, trust_remote_code=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()

    if device == "cpu":
        model.to("cpu")

    return model, tokenizer, device


@torch.no_grad()
def generate_answer(model, tokenizer, device, prompt, max_new_tokens=128, temperature=0.0, top_p=1.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.05,
    )

    gen_ids = out[0][input_len:]
    ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

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

    return ans.strip()


def normalize_text(s):
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_numbers(text):
    """
    Extract numeric values from text.
    For quick evaluation only.
    """
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    return [float(x) for x in nums]


def number_match_score(pred, gold, tolerance=1e-3):
    """
    Simple numeric matching:
    Checks whether every number in gold appears approximately in pred.
    """
    gold_nums = extract_numbers(gold)
    pred_nums = extract_numbers(pred)

    if len(gold_nums) == 0:
        return None

    matched = 0

    for g in gold_nums:
        ok = False
        for p in pred_nums:
            if abs(g - p) <= tolerance:
                ok = True
                break
        if ok:
            matched += 1

    return matched / max(len(gold_nums), 1)

def reason_score(pred, answerable):
    """
    For unanswerable questions, check whether the answer explains why.
    """
    if answerable:
        return None

    pred_l = normalize_text(pred)

    reason_markers = [
        "facts",
        "provided facts",
        "current facts",
        "dynamic imaging",
        "cardiac-cycle",
        "functional information",
        "functional labels",
        "clinical functional labels",
        "segmentation",
        "not include",
        "does not include",
        "missing",
        "insufficient",
    ]

    has_reason = any(marker in pred_l for marker in reason_markers)

    return 1.0 if has_reason else 0.0

def refusal_score(pred, answerable):
    """
    For unanswerable questions, check whether model refuses / says unavailable.
    """
    pred_l = normalize_text(pred)

    refusal_markers = [
        "not available",
        "cannot be reliably answered",
        "cannot be answered",
        "not provided",
        "not included",
        "insufficient",
        "not enough information",
    ]

    has_refusal = any(m in pred_l for m in refusal_markers)

    if answerable:
        return None

    return 1.0 if has_refusal else 0.0


def raw_json_penalty(pred):
    """
    Check whether model outputs raw dict/json-like answer.
    """
    pred_s = pred.strip()

    if pred_s.startswith("{") or pred_s.startswith("["):
        return 1

    if "{" in pred_s and "}" in pred_s:
        return 1

    return 0


def simple_contains_score(pred, gold):
    """
    Very rough lexical score.
    This is not a final metric, only for quick debugging.
    """
    pred_l = normalize_text(pred)
    gold_l = normalize_text(gold)

    gold_words = set(re.findall(r"[a-zA-Z_]+", gold_l))
    pred_words = set(re.findall(r"[a-zA-Z_]+", pred_l))

    if len(gold_words) == 0:
        return 0.0

    return len(gold_words & pred_words) / len(gold_words)


def evaluate_one(pred, gold, answerable):
    return {
        "lexical_overlap": simple_contains_score(pred, gold),
        "number_match": number_match_score(pred, gold),
        "refusal_score": refusal_score(pred, answerable),
        "reason_score": reason_score(pred, answerable),
        "raw_json_penalty": raw_json_penalty(pred),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_dir", required=True)
    parser.add_argument("--facts_dir", required=True)
    parser.add_argument("--qa_json", required=True)

    parser.add_argument("--out_json", default="eval_results.json")
    parser.add_argument("--out_csv", default="eval_results.csv")

    parser.add_argument("--cache_dir", default=r"D:\CardiacRate\hf_cache")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    args = parser.parse_args()

    print("[INFO] Loading model...")
    model, tokenizer, device = load_model(
        base_model=args.base_model,
        lora_dir=args.lora_dir,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    print("[INFO] Building facts index...")
    facts_index = build_facts_index(args.facts_dir)

    print("[INFO] Loading QA dataset...")
    qa_data = load_json(args.qa_json)

    random.seed(args.seed)
    random.shuffle(qa_data)

    if args.max_samples > 0:
        qa_data = qa_data[:args.max_samples]

    results = []

    for i, item in enumerate(qa_data, start=1):
        patient_id = item.get("patient_id")
        question = item.get("question", "")
        gold_answer = item.get("answer", "")
        category = item.get("category", "")
        answerable = bool(item.get("answerable", True))

        facts_path = facts_index.get(patient_id)

        if facts_path is None:
            print(f"[{i}/{len(qa_data)}] [SKIP] No facts for patient_id={patient_id}")
            continue

        facts_obj = load_json(facts_path)
        facts_block = build_facts_block(facts_obj)
        prompt = build_prompt(facts_block, question)

        pred_answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        scores = evaluate_one(
            pred=pred_answer,
            gold=gold_answer,
            answerable=answerable,
        )

        row = {
            "idx": i,
            "patient_id": patient_id,
            "category": category,
            "answerable": answerable,
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            **scores,
        }

        results.append(row)

        print(f"[{i}/{len(qa_data)}] {category}")
        print(f"Q   : {question}")
        print(f"Gold: {gold_answer}")
        print(f"Pred: {pred_answer}")
        print(f"Score: {scores}")
        print("-" * 80)

    save_json(results, args.out_json)
    save_csv(results, args.out_csv)

    # Summary
    if len(results) > 0:
        lexical = [r["lexical_overlap"] for r in results if r["lexical_overlap"] is not None]
        number = [r["number_match"] for r in results if r["number_match"] is not None]
        refusal = [r["refusal_score"] for r in results if r["refusal_score"] is not None]
        raw_json = [r["raw_json_penalty"] for r in results]

        print()
        print("========== Evaluation Summary ==========")
        print(f"Samples evaluated: {len(results)}")

        if lexical:
            print(f"Avg lexical overlap : {sum(lexical) / len(lexical):.4f}")

        if number:
            print(f"Avg number match    : {sum(number) / len(number):.4f}")

        if refusal:
            print(f"Refusal accuracy    : {sum(refusal) / len(refusal):.4f}")

        if raw_json:
            print(f"Raw JSON rate       : {sum(raw_json) / len(raw_json):.4f}")

        print(f"Saved JSON: {args.out_json}")
        print(f"Saved CSV : {args.out_csv}")


if __name__ == "__main__":
    main()