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
10. Use the correct volume conversion: 1 mL = 1000 mm³.
    When converting from mm³ to mL, divide by 1000.
    When converting from mL to mm³, multiply by 1000.
    If both values are provided in the structured facts, preserve them exactly
    and verify that they are consistent.
    Do not invent or recalculate a value unless conversion is required.

Generate exactly 14 question-and-answer pairs:
- exactly 2 technical questions;
- exactly 5 patient-friendly questions;
- exactly 5 clinical-confirmation questions;
- exactly 2 safety questions.

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

CATEGORY_TARGETS = {
    "technical": 2,
    "patient_friendly": 5,
    "clinical_confirmation": 5,
    "safety": 2,
}

CATEGORY_INSTRUCTIONS = {
    "technical": """
Generate technical questions about measurable case-specific findings,
including structure presence, volume, slice range, calcification burden,
Agatston-like score, or available risk assessment.
""",

    "patient_friendly": """
Generate natural patient-friendly questions in clear English.
Explain the CT result using language understandable to a person without
medical training.
""",

    "clinical_confirmation": """
Generate questions about whether the current CT findings confirm aortic
stenosis or another cardiac condition. Clearly distinguish a CT-based risk
estimate from a confirmed clinical diagnosis.
""",

    "safety": """
Generate questions about treatment, surgery, medication, symptoms, prognosis,
cardiac function, or other conclusions that cannot be reliably determined
from the available structured facts.
""",
}

ALLOWED_ANSWERABILITY = {
    "fully_answerable",
    "partially_answerable",
    "requires_clinical_confirmation",
    "not_answerable",
}

def normalize_question(text):
    return " ".join(
        str(text).lower().strip().split()
    )


def filter_valid_category_pairs(
    qa_pairs,
    expected_category,
    existing_questions,
):
    valid_pairs = []

    existing_normalized = {
        normalize_question(question)
        for question in existing_questions
    }

    for item in qa_pairs:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        if not question or not answer:
            continue

        # 類別不正確就不接受
        if item.get("category") != expected_category:
            continue

        # 強制英文語言欄位
        if item.get("language") != "en":
            continue

        if (
            item.get("answerability")
            not in ALLOWED_ANSWERABILITY
        ):
            continue

        normalized = normalize_question(question)

        # 去除重複問題
        if normalized in existing_normalized:
            continue

        existing_normalized.add(normalized)
        valid_pairs.append(item)

    return valid_pairs

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

def generate_category_once(
    model,
    facts,
    # canonical_qa,
    category,
    requested_count,
    existing_questions,
    debug_path,
):
    facts_for_prompt = compact_facts(facts)

    # canonical_for_prompt = select_canonical_qa(
    #     canonical_qa,
    #     max_items=6,
    # )

    category_instruction = CATEGORY_INSTRUCTIONS[category]

    system_prompt = f"""
You are a dataset creator for an evidence-grounded cardiac CT
health consultation assistant.

Evidence rules:
1. Statements about the patient must be supported by the structured facts.
2. Preserve all numerical values and units exactly.
3. Use the correct conversion: 1 mL = 1000 mm³.
4. When converting mm³ to mL, divide by 1000.
5. When converting mL to mm³, multiply by 1000.
6. Do not invent symptoms, medical history, diagnoses, echocardiography
   findings, treatment recommendations, or laboratory results.
7. A CT calcium-based aortic stenosis risk estimate is not a confirmed
   diagnosis of aortic stenosis.
8. All questions and answers must be written in English only.
9. Do not expose raw JSON in the answers.
10. Return exactly one valid JSON object.
11. Do not write Markdown or explanatory text outside the JSON.

Current category:
{category}

Category requirements:
{category_instruction}

Generate exactly {requested_count} new question-and-answer pairs.

Every generated item must have:
- category set exactly to "{category}";
- language set exactly to "en";
- a non-empty question;
- a non-empty answer;
- a valid answerability value;
- facts_paths based on the provided structured facts.

Do not repeat any question listed in existing_questions.

Return exactly this structure:

{{
  "qa_pairs": [
    {{
      "category": "{category}",
      "language": "en",
      "question": "...",
      "answer": "...",
      "answerability": "fully_answerable",
      "facts_paths": ["..."]
    }}
  ]
}}

Allowed answerability values:
- fully_answerable
- partially_answerable
- requires_clinical_confirmation
- not_answerable
"""

    payload = {
        "case_id": facts.get("patient_id", "unknown"),
        "structured_facts": facts_for_prompt,
        # "canonical_qa": canonical_for_prompt,
        "existing_questions": existing_questions,
    }

    messages = [
        {
            "role": "system",
            "content": system_prompt,
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

    with torch.inference_mode():
        output = model(
            messages,
            max_new_tokens=10000,
            do_sample=False,
        )

    generated = output[0]["generated_text"]

    if isinstance(generated, list):
        generated_text = generated[-1]["content"]
    else:
        generated_text = generated

    result = extract_json(
        generated_text,
        debug_path=debug_path,
    )

    qa_pairs = result.get("qa_pairs", [])

    if not isinstance(qa_pairs, list):
        raise ValueError("qa_pairs is not a list.")

    return qa_pairs

def generate_category_until_complete(
    model,
    facts,
    # canonical_qa,
    category,
    target_count,
    debug_dir,
    max_rounds=5,
):
    patient_id = facts.get(
        "patient_id",
        "unknown",
    )

    collected = []

    for round_number in range(1, max_rounds + 1):
        missing_count = target_count - len(collected)

        if missing_count <= 0:
            break

        existing_questions = [
            item["question"]
            for item in collected
        ]

        print(
            f"[INFO] {patient_id} | {category} | "
            f"round {round_number} | "
            f"need {missing_count} more"
        )

        debug_path = (
            Path(debug_dir)
            / (
                f"{patient_id}_{category}_"
                f"round{round_number}_raw.txt"
            )
        )

        try:
            generated_pairs = generate_category_once(
                model=model,
                facts=facts,
                # canonical_qa=canonical_qa,
                category=category,
                requested_count=missing_count,
                existing_questions=existing_questions,
                debug_path=debug_path,
            )

            valid_pairs = filter_valid_category_pairs(
                qa_pairs=generated_pairs,
                expected_category=category,
                existing_questions=existing_questions,
            )

            # 只加入還缺少的數量
            collected.extend(
                valid_pairs[:missing_count]
            )

            print(
                f"[INFO] Accepted {len(valid_pairs[:missing_count])} "
                f"new items; total {len(collected)}/{target_count}"
            )

        except Exception as error:
            print(
                f"[WARN] {patient_id} | {category} | "
                f"round {round_number} failed: {error}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(collected) != target_count:
        raise RuntimeError(
            f"{patient_id}: failed to generate exactly "
            f"{target_count} items for {category}. "
            f"Generated {len(collected)} valid items."
        )

    return collected

def generate_case_qa(
    model,
    facts,
    # canonical_qa,
    debug_dir="D://CardiacRate//dataset//generated_qa//debug",
):
    patient_id = facts.get(
        "patient_id",
        "unknown",
    )

    all_pairs = []

    for category, target_count in CATEGORY_TARGETS.items():
        category_pairs = generate_category_until_complete(
            model=model,
            facts=facts,
            # canonical_qa=canonical_qa,
            category=category,
            target_count=target_count,
            debug_dir=debug_dir,
            max_rounds=5,
        )

        all_pairs.extend(category_pairs)

    result = {
        "case_id": patient_id,
        "qa_pairs": all_pairs,
    }

    expected_total = sum(
        CATEGORY_TARGETS.values()
    )

    if len(all_pairs) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} total QA pairs, "
            f"got {len(all_pairs)}."
        )

    return result

from collections import Counter

def validate_final_counts(result):
    qa_pairs = result.get("qa_pairs", [])

    counts = Counter(
        item.get("category")
        for item in qa_pairs
    )

    errors = []

    for category, expected in CATEGORY_TARGETS.items():
        actual = counts.get(category, 0)

        if actual != expected:
            errors.append(
                f"{category}: expected {expected}, got {actual}"
            )

    expected_total = sum(
        CATEGORY_TARGETS.values()
    )

    if len(qa_pairs) != expected_total:
        errors.append(
            f"Total: expected {expected_total}, "
            f"got {len(qa_pairs)}"
        )

    if errors:
        raise ValueError(
            "Final QA count validation failed:\n"
            + "\n".join(errors)
        )

    print("\nFinal QA counts")
    print("------------------------------")

    for category, expected in CATEGORY_TARGETS.items():
        print(
            f"{category:24s}: "
            f"{counts.get(category, 0)}"
        )

    print(
        f"{'total':24s}: "
        f"{len(qa_pairs)}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts_path", required=True)
    # parser.add_argument("--canonical_qa_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument(
        "--model_id",
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )

    args = parser.parse_args()

    facts = load_json(args.facts_path)
    # all_canonical_qa = load_json(args.canonical_qa_path)

    # patient_id = facts.get("patient_id")

    # canonical_qa = [
    #     item
    #     for item in all_canonical_qa
    #     if item.get("patient_id") == patient_id
    # ]

    # if not canonical_qa:
    #     raise ValueError(
    #         f"No canonical QA found for patient_id={patient_id}"
    #     )

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
    # canonical_for_prompt = select_canonical_qa(canonical_qa, max_items=6)

    # payload = {
    #     "case_id": facts.get("patient_id", "unknown"),
    #     "structured_facts": facts_for_prompt,
    #     "canonical_qa": canonical_for_prompt,
    # }

    result = generate_case_qa(
        model=generator,
        facts=facts_for_prompt #,
        # canonical_qa=canonical_for_prompt,
    )
    validate_final_counts(result)
    save_json(result, args.out_path)
    print(f"[OK] Generated QA saved to: {args.out_path}")


if __name__ == "__main__":
    main()