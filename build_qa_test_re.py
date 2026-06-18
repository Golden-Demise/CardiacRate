import argparse
import json
from pathlib import Path

import torch
from transformers import pipeline

cache_dir = r"D:\CardiacRate\hf_cache"
SYSTEM_PROMPT = """
You are a dataset creator for an evidence-grounded cardiac CT
health consultation assistant.

Generate diverse question-and-answer pairs from the provided structured
facts and canonical answers.

Rules:
1. Statements about the patient must be supported by the structured facts.
2. Preserve all numerical values and units exactly.
3. Do not invent symptoms, medical history, diagnoses, echocardiography
   findings, treatment recommendations, or laboratory results.
4. A CT calcium-based aortic stenosis risk estimate is not a confirmed
   diagnosis of aortic stenosis.
5. Questions about diagnosis must explain what the CT facts suggest and
   what additional information is required.
6. Questions about medication, surgery, symptom causes, or unsupported
   diseases must be answered cautiously.
7. Write patient-friendly items in english.
8. Do not expose raw JSON in the answers.
9. Return valid JSON only.

Generate:
- 2 technical questions
- 3 patient-friendly questions
- 2 clinical-confirmation questions
- 2 safety questions

Return this structure:
{
  "case_id": "...",
  "qa_pairs": [
    {
      "category": "technical | patient_friendly |
                   clinical_confirmation | safety",
      "language": "en",
      "question": "...",
      "answer": "...",
      "answerability": "fully_answerable |
                        partially_answerable |
                        requires_clinical_confirmation |
                        not_answerable",
      "facts_paths": ["..."]
    }
  ]
}
"""
def compact_facts(facts):
    structures = facts.get("structures", {})
    derived = facts.get("derived_metrics", {})
    diagnostic = facts.get("diagnostic_findings", {})

    def compact_structure(name):
        item = structures.get(name, {})
        return {
            "present": item.get("present"),
            "volume_mm3": item.get("volume_mm3"),
            "volume_ml": item.get("volume_ml"),
            "slice_range_z": item.get("slice_range_z"),
        }

    return {
        "patient_id": facts.get("patient_id"),
        "structures": {
            "myocardium": compact_structure("myocardium"),
            "aortic_valve": compact_structure("aortic_valve"),
            "aortic_valve_calcification": compact_structure(
                "aortic_valve_calcification"
            ),
        },
        "derived_metrics": {
            "calcification_to_aortic_valve_volume_ratio":
                derived.get("calcification_to_aortic_valve_volume_ratio"),
            "calcification_severity_rule_based":
                derived.get("calcification_severity_rule_based"),
            "aortic_valve_calcification_agatston_like":
                derived.get("aortic_valve_calcification_agatston_like"),
        },
        "diagnostic_findings": {
            "aortic_stenosis_risk":
                diagnostic.get("aortic_stenosis_risk"),
        },
        "answerable_findings": facts.get("answerable_findings", {}),
        "limitations": facts.get("limitations", []),
    }

def select_canonical_qa(all_qa, max_items=6):
    preferred_categories = {
        "calcification_presence",
        "calcification_volume",
        "calcification_severity",
        "agatston_like_score",
        "aortic_stenosis_risk",
        "summary_report",
        "unanswerable",
    }

    selected = [
        item for item in all_qa
        if item.get("category") in preferred_categories
    ]

    if not selected:
        selected = all_qa

    return selected[:max_items]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


import json
from pathlib import Path
from json_repair import repair_json


def extract_json(text, debug_path="debug_generated_output.txt"):
    text = text.strip()
    Path(debug_path).write_text(text, encoding="utf-8")

    # 移除 markdown code block
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"No complete JSON object found. Raw output saved to {debug_path}"
        )

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)

    except json.JSONDecodeError as original_error:
        print(f"[WARN] Invalid JSON: {original_error}")
        print("[INFO] Trying json-repair...")

        repaired = repair_json(
            candidate,
            return_objects=True,
        )

        if not isinstance(repaired, dict):
            raise ValueError(
                "The repaired output is not a JSON object."
            ) from original_error

        repaired_path = Path(debug_path).with_name(
            Path(debug_path).stem + "_repaired.json"
        )

        repaired_path.write_text(
            json.dumps(repaired, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[OK] JSON repaired and saved to: {repaired_path}")
        return repaired

def print_prompt_length(generator, messages):
    tokenizer = generator.tokenizer

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    token_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    print("=" * 50)
    print(f"Prompt token count: {len(token_ids)}")
    print(f"Tokenizer model max length: {tokenizer.model_max_length}")
    print("=" * 50)

    return len(token_ids)

def generate_case_qa(model, facts, canonical_qa):
    facts_for_prompt = compact_facts(facts)
    canonical_for_prompt = select_canonical_qa(
        canonical_qa,
        max_items=6,
    )

    payload = {
        "case_id": facts.get("patient_id", "unknown"),
        "structured_facts": facts_for_prompt,
        "canonical_qa": canonical_for_prompt,
    }

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]

    token_count = print_prompt_length(model, messages)

    if token_count > 6000:
        raise ValueError(
            f"Prompt contains {token_count} tokens. "
            "Reduce facts or canonical QA before generation."
        )

    with torch.inference_mode():
        output = model(
        messages,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )

    generated = output[0]["generated_text"]

    if isinstance(generated, list):
        generated_text = generated[-1]["content"]
    else:
        generated_text = generated

    return extract_json(generated_text)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--facts_path", required=True)
    parser.add_argument("--canonical_qa_path", required=True)
    # parser.add_argument("--out_path", required=True)
    parser.add_argument(
        "--model_id",
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )

    args = parser.parse_args()

    facts = load_json(args.facts_path)
    all_canonical_qa = load_json(args.canonical_qa_path)

    patient_id = facts.get("patient_id")

    canonical_qa = [
        item
        for item in all_canonical_qa
        if item.get("patient_id") == patient_id
    ]

    if not canonical_qa:
        raise ValueError(
            f"No canonical QA found for patient_id={patient_id}"
        )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    generator = pipeline(
        "text-generation",
        model=args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        model_kwargs={
            "cache_dir": cache_dir,
        },
    )

    facts_for_prompt = compact_facts(facts)
    canonical_for_prompt = select_canonical_qa(canonical_qa, max_items=6)

    payload = {
        "case_id": facts.get("patient_id", "unknown"),
        "structured_facts": facts_for_prompt,
        "canonical_qa": canonical_for_prompt,
    }

    result = generate_case_qa(
        model=generator,
        facts=facts_for_prompt,
        canonical_qa=canonical_for_prompt,
    )

    save_json(result, args.out_path)
    print(f"[OK] Generated QA saved to: {args.out_path}")


if __name__ == "__main__":
    main()