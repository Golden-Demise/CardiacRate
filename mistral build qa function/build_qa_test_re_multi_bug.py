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
        item
        for item in all_qa
        if item.get("category") in preferred_categories
        and item.get("language", "en") == "en"
    ]

    if not selected:
        selected = [
            item
            for item in all_qa
            if item.get("language", "en") == "en"
        ]

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

def generate_case_qa(
    model,
    facts,
    canonical_qa,
    debug_path,
):
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

    return extract_json(
        generated_text,
        debug_path=debug_path,
    )

import re

def normalize_patient_id(value):
    if value is None:
        return None

    text = str(value).strip()

    numbers = re.findall(r"\d+", text)

    if numbers:
        number = int(numbers[-1])
        return f"patient{number:04d}"

    return text.lower()

def build_canonical_qa_index(all_canonical_qa):
    index = {}

    for item in all_canonical_qa:
        patient_id = normalize_patient_id(
            item.get("patient_id")
        )

        if patient_id is None:
            continue

        index.setdefault(patient_id, []).append(item)

    return index

def validate_generated_result(result):
    allowed_categories = {
        "technical",
        "patient_friendly",
        "clinical_confirmation",
        "safety",
    }

    allowed_answerability = {
        "fully_answerable",
        "partially_answerable",
        "requires_clinical_confirmation",
        "not_answerable",
    }

    expected_counts = {
        "technical": 2,
        "patient_friendly": 3,
        "clinical_confirmation": 2,
        "safety": 2,
    }

    errors = []

    if not isinstance(result, dict):
        return ["The generated result is not a JSON object."]

    qa_pairs = result.get("qa_pairs")

    if not isinstance(qa_pairs, list):
        return ["qa_pairs is missing or is not a list."]

    category_counts = {
        key: 0
        for key in expected_counts
    }

    for index, item in enumerate(qa_pairs):
        category = item.get("category")
        language = item.get("language")
        question = item.get("question", "")
        answer = item.get("answer", "")
        answerability = item.get("answerability")

        if category not in allowed_categories:
            errors.append(
                f"QA {index}: invalid category {category!r}"
            )
        else:
            category_counts[category] += 1

        if language != "en":
            errors.append(
                f"QA {index}: language must be 'en', got {language!r}"
            )

        if not question.strip():
            errors.append(f"QA {index}: empty question")

        if not answer.strip():
            errors.append(f"QA {index}: empty answer")

        if answerability not in allowed_answerability:
            errors.append(
                f"QA {index}: invalid answerability "
                f"{answerability!r}"
            )

        # 檢查中文字符
        if re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF]", question):
            errors.append(
                f"QA {index}: Chinese characters found in question"
            )

        if re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF]", answer):
            errors.append(
                f"QA {index}: Chinese characters found in answer"
            )

    for category, expected_count in expected_counts.items():
        actual_count = category_counts.get(category, 0)

        if actual_count != expected_count:
            errors.append(
                f"{category}: expected {expected_count}, "
                f"got {actual_count}"
            )

    return errors

def generate_case_qa_with_retry(
    model,
    facts,
    canonical_qa,
    debug_dir,
    max_retries=3,
):
    patient_id = normalize_patient_id(
        facts.get("patient_id")
    ) or "unknown"

    last_error = None

    for attempt in range(1, max_retries + 1):
        debug_path = (
            Path(debug_dir)
            / f"{patient_id}_attempt{attempt}_raw.txt"
        )

        try:
            print(
                f"[INFO] {patient_id}: "
                f"attempt {attempt}/{max_retries}"
            )

            result = generate_case_qa(
                model=model,
                facts=facts,
                canonical_qa=canonical_qa,
                debug_path=debug_path,
            )

            errors = validate_generated_result(result)

            if errors:
                raise ValueError(
                    "Generated QA validation failed:\n"
                    + "\n".join(errors)
                )

            return result

        except Exception as e:
            last_error = e

            print(
                f"[WARN] {patient_id}: "
                f"attempt {attempt} failed"
            )
            print(f"       {type(e).__name__}: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(
        f"Failed to generate QA for {patient_id} "
        f"after {max_retries} attempts."
    ) from last_error

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--facts_dir",
        required=True,
        help="Directory containing facts JSON files.",
    )

    parser.add_argument(
        "--canonical_qa_path",
        required=True,
        help="Path to the canonical QA dataset JSON.",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory for generated QA files.",
    )

    parser.add_argument(
        "--debug_dir",
        default=None,
        help="Directory for raw model outputs and repaired JSON.",
    )

    parser.add_argument(
        "--start_id",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--end_id",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--facts_pattern",
        default="patient{num:04d}_facts.json",
        help=(
            "Facts filename pattern. "
            "Example: patient{num:04d}_facts.json"
        ),
    )

    parser.add_argument(
        "--model_id",
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files that already exist.",
    )

    args = parser.parse_args()

    facts_dir = Path(args.facts_dir)
    out_dir = Path(args.out_dir)

    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
    else:
        debug_dir = out_dir / "debug"

    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # canonical QA 只讀一次
    all_canonical_qa = load_json(
        args.canonical_qa_path
    )

    if not isinstance(all_canonical_qa, list):
        raise ValueError(
            "canonical_qa_path must contain a JSON list."
        )

    canonical_index = build_canonical_qa_index(
        all_canonical_qa
    )

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    print("[INFO] Loading model...")

    # 模型只載入一次
    generator = pipeline(
        "text-generation",
        model=args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        model_kwargs={
            "cache_dir": cache_dir,
        },
    )

    print("[OK] Model loaded.")
    print(
        f"[INFO] Processing cases "
        f"{args.start_id} to {args.end_id}"
    )

    summary = {
        "success": [],
        "skipped_existing": [],
        "missing_facts": [],
        "missing_canonical_qa": [],
        "failed": [],
    }

    all_generated_results = []

    total_cases = args.end_id - args.start_id + 1

    for position, case_number in enumerate(
        range(args.start_id, args.end_id + 1),
        start=1,
    ):
        default_patient_id = f"patient{case_number:04d}"

        print()
        print("=" * 70)
        print(
            f"[{position}/{total_cases}] "
            f"Processing {default_patient_id}"
        )
        print("=" * 70)

        facts_filename = args.facts_pattern.format(
            num=case_number
        )

        facts_path = facts_dir / facts_filename

        if not facts_path.exists():
            print(
                f"[WARN] Facts file not found: {facts_path}"
            )
            summary["missing_facts"].append(
                str(facts_path)
            )
            continue

        try:
            facts = load_json(facts_path)

            patient_id = normalize_patient_id(
                facts.get("patient_id")
            ) or default_patient_id

            output_path = (
                out_dir
                / f"{patient_id}_generated_qa.json"
            )

            if output_path.exists() and not args.overwrite:
                print(
                    f"[SKIP] Output already exists: "
                    f"{output_path}"
                )

                summary["skipped_existing"].append(
                    patient_id
                )

                try:
                    existing_result = load_json(output_path)
                    all_generated_results.append(
                        existing_result
                    )
                except Exception:
                    pass

                continue

            canonical_qa = canonical_index.get(
                patient_id,
                [],
            )

            if not canonical_qa:
                print(
                    f"[WARN] No canonical QA found for "
                    f"{patient_id}"
                )
                summary["missing_canonical_qa"].append(
                    patient_id
                )
                continue

            result = generate_case_qa_with_retry(
                model=generator,
                facts=facts,
                canonical_qa=canonical_qa,
                debug_dir=debug_dir,
                max_retries=args.max_retries,
            )

            # 強制輸出 case_id 與 facts 一致
            result["case_id"] = patient_id

            save_json(result, output_path)

            all_generated_results.append(result)
            summary["success"].append(patient_id)

            print(
                f"[OK] Saved QA: {output_path}"
            )

        except Exception as e:
            print(
                f"[FAIL] {default_patient_id}: "
                f"{type(e).__name__}: {e}"
            )

            summary["failed"].append({
                "patient_id": default_patient_id,
                "facts_path": str(facts_path),
                "error_type": type(e).__name__,
                "error": str(e),
            })

            error_path = (
                debug_dir
                / f"{default_patient_id}_error.txt"
            )

            error_path.write_text(
                f"{type(e).__name__}: {e}",
                encoding="utf-8",
            )

        finally:
            # 避免長時間批次產生時累積記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 儲存成功病例的整合資料
    combined_path = out_dir / "all_generated_qa.json"

    save_json(
        all_generated_results,
        combined_path,
    )

    # 儲存批次執行摘要
    summary["counts"] = {
        key: len(value)
        for key, value in summary.items()
        if isinstance(value, list)
    }

    summary_path = out_dir / "batch_summary.json"

    save_json(
        summary,
        summary_path,
    )

    print()
    print("=" * 70)
    print("Batch generation finished")
    print("=" * 70)
    print(
        f"Success             : "
        f"{len(summary['success'])}"
    )
    print(
        f"Skipped existing    : "
        f"{len(summary['skipped_existing'])}"
    )
    print(
        f"Missing facts       : "
        f"{len(summary['missing_facts'])}"
    )
    print(
        f"Missing canonical QA: "
        f"{len(summary['missing_canonical_qa'])}"
    )
    print(
        f"Failed              : "
        f"{len(summary['failed'])}"
    )
    print(f"Combined output     : {combined_path}")
    print(f"Summary             : {summary_path}")


if __name__ == "__main__":
    main()