import argparse
import ast
import copy
import json
import re
from pathlib import Path
from typing import Any

import torch
from json_repair import repair_json
from transformers import pipeline


DEFAULT_PARAPHRASE_CATEGORIES = {
    "aortic_stenosis_risk",
    "aortic_stenosis_diagnosis_safety",
    "patient_friendly_explanation",
    "patient_friendly_calcification_meaning",
    "patient_friendly_seriousness",
    "patient_friendly_worry",
    "patient_friendly_diagnosis_safety",
    "patient_friendly_aortic_stenosis_safety",
    "patient_friendly_valve_blockage_safety",
    "patient_friendly_next_test",
    "patient_friendly_next_steps",
    "patient_friendly_symptom_safety",
    "unanswerable",
}


SYSTEM_PROMPT = """
You are a question-paraphrasing assistant for an evidence-grounded cardiac CT
question-answering dataset.

Your only task is to rewrite the supplied question into natural English
question variants.

Strict rules:
1. Generate questions only. Do not generate answers, explanations, evidence,
   labels, categories, diagnoses, recommendations, or JSON fields other than
   question_variants.
2. Preserve the exact medical intent of the original question.
3. Do not add or remove a clinical claim.
4. Do not add patient-specific findings, values, units, symptoms, diagnoses,
   medical history, treatment, prognosis, or follow-up recommendations.
5. Do not convert a current CT-based risk question into a prediction about
   future disease development, progression, or worsening.
6. Do not change a question about whether a condition can be confirmed into a
   statement that the condition is confirmed or excluded.
7. Do not change positive and negative meaning. In particular, do not add
   phrases such as "confirm the absence" or "rule out" unless they are already
   present in the original question.
8. Use "Agatston-like score", not "Agatston score", when referring to the
   segmentation-derived score.
9. Values in slice_range_z are z-slice indices, not millimeters. For slice
   questions, use "slice", "z-slice", or "slice range"; never use mm,
   millimeter, or physical distance.
10. Use natural English. Do not include Chinese characters.
11. Do not include patient IDs unless the original question contains one.
12. Each variant must be a complete question ending with a question mark.
13. The variants may be reused for both cases where a finding is present
    and cases where it is absent. Keep the wording neutral and do not imply
    that a finding was detected, found, present, absent, or shown unless the
    original question explicitly states that condition.
14. Output only the requested question variants. Never echo the input object,
    field names, category, original_question, requested_count, examples, or
    instructions.
15. Return exactly one valid JSON object and no Markdown.

Required output:
{
  "question_variants": [
    "Question variant 1?",
    "Question variant 2?"
  ]
}
""".strip()


SAFE_FALLBACKS: dict[tuple[str, str], list[str]] = {
    (
        "aortic_stenosis_diagnosis_safety",
        "does this patient have aortic stenosis",
    ): [
        "Can the current CT information determine whether this patient has aortic stenosis?",
        "Can aortic stenosis be diagnosed in this patient from the current CT information?",
        "Is it possible to determine from the current CT information whether this patient has aortic stenosis?",
    ],
    (
        "patient_friendly_aortic_stenosis_safety",
        "do i have aortic stenosis",
    ): [
        "Can the current CT information determine whether I have aortic stenosis?",
        "Can aortic stenosis be diagnosed from my current CT information?",
        "Do the current CT findings provide enough information to determine whether I have aortic stenosis?",
    ],
}


def get_safe_fallback_variants(
    original_question: str,
    category: str,
) -> list[str]:
    key = (
        category,
        normalize_text(original_question),
    )
    return SAFE_FALLBACKS.get(key, [])


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[?.!,;:]+$", "", text)
    return text


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF]", text))


def deduplicate_canonical_items(
    data: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """
    Remove exact duplicate QA samples while preserving the original answer,
    evidence, category, and answerable field.
    """
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    removed = 0

    for item in data:
        patient_id = str(item.get("patient_id", "unknown"))
        category = str(item.get("category", "unknown"))
        question = normalize_text(item.get("question", ""))
        answer = normalize_text(item.get("answer", ""))

        key = (
            patient_id,
            category,
            question,
            answer,
        )

        if key in seen:
            removed += 1
            continue

        seen.add(key)
        output.append(item)

    return output, removed


def extract_json(
    text: str,
    debug_path: str | Path,
) -> dict[str, Any]:
    """
    Parse Mistral output robustly.

    Accepted forms:
    1. {"question_variants": ["...", "..."]}
    2. ["...", "..."]
    3. A repaired JSON object/array
    4. A single JSON string, which is wrapped as one variant
    """
    debug_path = Path(debug_path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(text, encoding="utf-8")

    cleaned = str(text).strip()

    # Remove Markdown fences even when the model adds json after the opening fence.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    ).strip()

    candidates: list[str] = []

    # Try the full response first.
    if cleaned:
        candidates.append(cleaned)

    # Try the largest object.
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")

    if (
        object_start != -1
        and object_end != -1
        and object_end > object_start
    ):
        candidates.append(
            cleaned[object_start:object_end + 1]
        )

    # Try the largest array. Mistral sometimes returns a bare array.
    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")

    if (
        array_start != -1
        and array_end != -1
        and array_end > array_start
    ):
        candidates.append(
            cleaned[array_start:array_end + 1]
        )

    parsed: Any = None
    parse_errors: list[str] = []

    for candidate in dict.fromkeys(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as error:
            parse_errors.append(str(error))

            try:
                parsed = repair_json(
                    candidate,
                    return_objects=True,
                )
            except Exception as repair_error:
                parse_errors.append(str(repair_error))
                parsed = None

        if parsed is not None:
            break

    # json-repair may return a string containing JSON or a Python-style
    # literal such as "{'question_variants': ['...']}".
    for _ in range(3):
        if not isinstance(parsed, str):
            break

        stripped = parsed.strip()

        if not stripped:
            parsed = None
            break

        nested = None

        try:
            nested = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                nested = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                try:
                    nested = repair_json(
                        stripped,
                        return_objects=True,
                    )
                except Exception:
                    nested = None

        if nested is None or nested == stripped:
            # Do not turn serialized objects or echoed prompts into questions.
            if any(token in stripped for token in (
                "{",
                "}",
                "[",
                "]",
                "question_variants",
                "original_question",
                "requested_count",
                "existing_variants_to_avoid",
            )):
                parsed = None
            else:
                parsed = {
                    "question_variants": [stripped]
                }
            break

        parsed = nested

    if isinstance(parsed, list):
        parsed = {
            "question_variants": parsed
        }

    if not isinstance(parsed, dict):
        raise ValueError(
            "Model output could not be normalized to a JSON object. "
            f"Raw output saved to: {debug_path}. "
            f"Parser errors: {parse_errors[:3]}"
        )

    # Support a few common alternative keys without allowing the model
    # to control any medical fields.
    if "question_variants" not in parsed:
        for alternative_key in (
            "questions",
            "variants",
            "paraphrases",
        ):
            value = parsed.get(alternative_key)

            if isinstance(value, list):
                parsed = {
                    "question_variants": value
                }
                break

    return parsed


def concept_requirements(original_question: str) -> list[tuple[str, ...]]:
    """
    Each tuple is an OR group. At least one term in every returned group
    must appear in the paraphrase.
    """
    q = normalize_text(original_question)
    groups: list[tuple[str, ...]] = []

    if "aortic stenosis" in q:
        groups.append(("aortic stenosis", "aortic valve narrowing", "valve narrowing"))

    if "aortic valve calcification" in q:
        groups.append(("aortic valve calcification", "aortic valve calcium", "valve calcification"))

    if "myocardium" in q:
        groups.append(("myocardium", "heart muscle"))

    if "aortic valve" in q and "aortic valve calcification" not in q:
        groups.append(("aortic valve",))

    if "ejection fraction" in q:
        groups.append(("ejection fraction", " ef "))

    if "coronary artery stenosis" in q:
        groups.append(("coronary artery stenosis", "coronary narrowing"))

    if "cardiac function" in q:
        groups.append(("cardiac function", "heart function"))

    if "slice" in q:
        groups.append(("slice", "z-slice"))

    if "agatston-like" in q:
        groups.append(("agatston-like",))

    return groups


def validate_variant(
    original_question: str,
    variant: str,
    existing_normalized: set[str],
    category: str | None = None,
) -> tuple[bool, str]:
    variant = str(variant).strip()

    if not variant:
        return False, "empty"

    if contains_chinese(variant):
        return False, "contains Chinese characters"

    if len(variant) < 8 or len(variant) > 240:
        return False, "invalid length"

    if "\n" in variant:
        return False, "contains newline"

    # Reject serialized dictionaries, arrays, prompt echoes, or metadata.
    forbidden_structure_tokens = (
        "{",
        "}",
        "[",
        "]",
        "question_variants",
        "original_question",
        "requested_count",
        "existing_variants_to_avoid",
        "'category':",
        '"category":',
    )

    if any(token in variant for token in forbidden_structure_tokens):
        return False, "contains serialized object or prompt metadata"

    if not variant.endswith("?"):
        variant += "?"

    normalized = normalize_text(variant)

    if normalized in existing_normalized:
        return False, "duplicate"

    original_normalized = normalize_text(original_question)

    if normalized == original_normalized:
        return False, "identical to original"

    # Prevent slice index -> millimeter errors.
    if "slice" in original_normalized:
        if re.search(r"\b(mm|millimeter|millimeters)\b", normalized):
            return False, "slice index incorrectly expressed as millimeters"

    # Prevent Agatston-like -> clinical Agatston wording drift.
    if "agatston-like" in original_normalized:
        if "agatston-like" not in normalized:
            return False, "removed Agatston-like qualifier"

    # Prevent current assessment -> future prediction drift.
    future_terms = (
        "developing",
        "develop ",
        "progression",
        "progress ",
        "worsen",
        "future risk",
        "in the future",
    )

    if not any(term in original_normalized for term in future_terms):
        if any(term in normalized for term in future_terms):
            return False, "changed current assessment into future prediction"

    # Templates are reused across both positive and negative cases.
    # A generic meaning question must not assume that calcification exists.
    if category == "patient_friendly_calcification_meaning":
        presence_assumptions = (
            "found on",
            "found in",
            "detected on",
            "detected in",
            "presence of",
            "present on",
            "present in",
            "shown on",
            "shown in",
            "seen on",
            "seen in",
            "this finding",
        )

        if any(term in normalized for term in presence_assumptions):
            return False, "adds case-specific presence assumption"

    # Preserve diagnosis intent rather than changing it into an evidence-only
    # question.
    if category == "aortic_stenosis_diagnosis_safety":
        diagnosis_terms = (
            "have aortic stenosis",
            "has aortic stenosis",
            "is there aortic stenosis",
            "diagnos",
            "confirm whether",
            "determine whether",
        )

        if not any(term in normalized for term in diagnosis_terms):
            return False, "changed diagnosis intent"

    # Prevent polarity changes.
    if (
        "confirm the absence" not in original_normalized
        and "confirm absence" not in original_normalized
    ):
        if (
            "confirm the absence" in normalized
            or "confirm absence" in normalized
        ):
            return False, "added confirmation of absence"

    if "rule out" not in original_normalized and "rule out" in normalized:
        return False, "added rule-out meaning"

    # Basic concept-preservation checks.
    padded = f" {normalized} "

    for alternatives in concept_requirements(original_question):
        if not any(term in padded for term in alternatives):
            return False, f"missing required concept: {alternatives}"

    return True, variant
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

def build_template_key(
    category: str,
    question: str,
) -> str:
    return json.dumps(
        {
            "category": category,
            "question": normalize_text(question),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def collect_unique_templates(
    data: list[dict[str, Any]],
    categories: set[str],
) -> dict[str, dict[str, str]]:
    templates: dict[str, dict[str, str]] = {}

    for item in data:
        category = str(item.get("category", "unknown"))
        question = str(item.get("question", "")).strip()

        if category not in categories or not question:
            continue

        key = build_template_key(category, question)

        if key not in templates:
            templates[key] = {
                "category": category,
                "question": question,
            }

    return templates


def generate_variants_once(
    generator,
    original_question: str,
    category: str,
    requested_count: int,
    existing_variants: list[str],
    debug_path: str | Path,
) -> list[str]:
    category_instruction = ""

    if category == "aortic_stenosis_diagnosis_safety":
        category_instruction = (
            "Preserve the diagnosis question. Ask whether the patient has "
            "aortic stenosis or whether the current CT information can "
            "determine or diagnose it. Do not change the question into only "
            "asking whether the CT shows evidence or signs."
        )
    elif category == "patient_friendly_aortic_stenosis_safety":
        category_instruction = (
            "Preserve the first-person diagnosis question. Ask whether the "
            "current CT information can determine whether I have aortic "
            "stenosis. Do not ask only whether there are signs or evidence."
        )

    avoid_text = (
        "\n".join(f"- {value}" for value in existing_variants)
        if existing_variants
        else "- None"
    )

    user_instruction = f"""Original question:
{original_question}

Category:
{category}

Generate exactly {requested_count} distinct natural-English paraphrase(s).

Already accepted variants to avoid:
{avoid_text}

Additional category rule:
{category_instruction or "Preserve the original intent exactly."}

Return only:
{{"question_variants": ["...", "..."]}}
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_instruction,
        },
    ]

    with torch.inference_mode():
        output = generator(
            messages,
            max_new_tokens=320,
            do_sample=True,
            temperature=0.65,
            top_p=0.9,
        )

    generated = output[0]["generated_text"]

    if isinstance(generated, list):
        generated_text = generated[-1]["content"]
    else:
        generated_text = generated

    parsed = extract_json(
        generated_text,
        debug_path=debug_path,
    )

    variants = parsed.get("question_variants", [])

    if not isinstance(variants, list):
        raise ValueError("question_variants is not a list.")

    return [
        str(value).strip()
        for value in variants
        if str(value).strip()
    ]


def generate_variants_until_complete(
    generator,
    original_question: str,
    category: str,
    target_count: int,
    debug_dir: str | Path,
    template_number: int,
    max_rounds: int,
) -> list[str]:
    collected: list[str] = []
    existing_normalized = {
        normalize_text(original_question),
    }

    for round_number in range(1, max_rounds + 1):
        missing = target_count - len(collected)

        if missing <= 0:
            break

        debug_path = (
            Path(debug_dir)
            / (
                f"template_{template_number:03d}_"
                f"round_{round_number}_raw.txt"
            )
        )

        try:
            raw_variants = generate_variants_once(
                generator=generator,
                original_question=original_question,
                category=category,
                requested_count=missing,
                existing_variants=collected,
                debug_path=debug_path,
            )
        except Exception as error:
            print(
                f"[WARN] Paraphrase generation failed | "
                f"template={template_number} | "
                f"round={round_number} | "
                f"error={error}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

        for candidate in raw_variants:
            valid, value_or_reason = validate_variant(
                original_question=original_question,
                variant=candidate,
                existing_normalized=existing_normalized,
                category=category,
            )

            if not valid:
                print(
                    f"[REJECT] template={template_number} | "
                    f"round={round_number} | "
                    f"reason={value_or_reason} | "
                    f"candidate={candidate!r}"
                )
                continue

            accepted = value_or_reason
            collected.append(accepted)
            existing_normalized.add(
                normalize_text(accepted)
            )

            if len(collected) >= target_count:
                break

    if len(collected) < target_count:
        for fallback in get_safe_fallback_variants(
            original_question=original_question,
            category=category,
        ):
            valid, value_or_reason = validate_variant(
                original_question=original_question,
                variant=fallback,
                existing_normalized=existing_normalized,
                category=category,
            )

            if not valid:
                continue

            accepted = value_or_reason
            collected.append(accepted)
            existing_normalized.add(
                normalize_text(accepted)
            )

            print(
                f"[FALLBACK] template={template_number} | "
                f"accepted={accepted!r}"
            )

            if len(collected) >= target_count:
                break

    if len(collected) < target_count:
        print(
            f"[WARN] Only {len(collected)}/{target_count} valid variants "
            f"were produced for {original_question!r}. "
            "The batch will continue with the valid variants."
        )

    return collected[:target_count]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use Mistral to paraphrase canonical QA questions only. "
            "Answers, evidence, categories, and answerable labels are copied "
            "unchanged from the canonical dataset."
        )
    )

    parser.add_argument(
        "--qa_json",
        required=True,
        help="Canonical QA dataset, such as qa_dataset4_en.json.",
    )
    parser.add_argument(
        "--out_json",
        required=True,
        help="Output augmented QA dataset.",
    )
    parser.add_argument(
        "--model_id",
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument(
        "--cache_dir",
        default=r"D:\CardiacRate\hf_cache",
    )
    parser.add_argument(
        "--variants_per_question",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help=(
            "Categories to paraphrase. If omitted, patient-friendly, "
            "AS-risk, diagnosis-safety, and unanswerable categories are used."
        ),
    )
    parser.add_argument(
        "--include_all_categories",
        action="store_true",
        help="Paraphrase every category, including technical categories.",
    )
    parser.add_argument(
        "--cache_json",
        default=None,
        help="Resume cache for generated variants.",
    )
    parser.add_argument(
        "--debug_dir",
        default=None,
    )
    parser.add_argument(
        "--max_templates",
        type=int,
        default=None,
        help="Optional small test limit for unique question templates.",
    )
    parser.add_argument(
        "--exclude_patient_ids",
        nargs="*",
        default=["example"],
        help=(
            "Patient IDs to exclude from the canonical dataset. "
            "The default removes the non-patient example case."
        ),
    )

    args = parser.parse_args()

    if args.variants_per_question < 1:
        raise ValueError("--variants_per_question must be at least 1.")

    canonical_data = load_json(args.qa_json)

    if not isinstance(canonical_data, list):
        raise ValueError("The canonical QA dataset must be a JSON list.")

    excluded_patient_ids = set(args.exclude_patient_ids or [])
    excluded_sample_count = sum(
        1
        for item in canonical_data
        if str(item.get("patient_id", "unknown")) in excluded_patient_ids
    )

    canonical_data = [
        item
        for item in canonical_data
        if str(item.get("patient_id", "unknown"))
        not in excluded_patient_ids
    ]

    canonical_data, duplicate_count = deduplicate_canonical_items(
        canonical_data
    )

    all_categories = {
        str(item.get("category", "unknown"))
        for item in canonical_data
    }

    if args.include_all_categories:
        selected_categories = all_categories
    elif args.categories:
        selected_categories = set(args.categories)
    else:
        selected_categories = DEFAULT_PARAPHRASE_CATEGORIES

    templates = collect_unique_templates(
        data=canonical_data,
        categories=selected_categories,
    )

    template_items = list(templates.items())

    if args.max_templates is not None:
        template_items = template_items[:args.max_templates]

    out_json = Path(args.out_json)

    cache_path = (
        Path(args.cache_json)
        if args.cache_json
        else out_json.with_name(
            out_json.stem + "_paraphrase_cache.json"
        )
    )

    debug_dir = (
        Path(args.debug_dir)
        if args.debug_dir
        else out_json.with_name(
            out_json.stem + "_debug"
        )
    )

    debug_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cache_data = load_json(cache_path)

        if not isinstance(cache_data, dict):
            raise ValueError("The paraphrase cache must be a JSON object.")
    else:
        cache_data = {}

    # Revalidate cache entries so a previous malformed test cannot be reused.
    cleaned_cache: dict[str, list[str]] = {}

    for key, template in template_items:
        accepted: list[str] = []
        existing_normalized = {
            normalize_text(template["question"]),
        }

        for candidate in cache_data.get(key, []):
            valid, value_or_reason = validate_variant(
                original_question=template["question"],
                variant=str(candidate),
                existing_normalized=existing_normalized,
                category=template["category"],
            )

            if not valid:
                continue

            accepted_variant = value_or_reason
            accepted.append(accepted_variant)
            existing_normalized.add(
                normalize_text(accepted_variant)
            )

            if len(accepted) >= args.variants_per_question:
                break

        cleaned_cache[key] = accepted

    cache_data = cleaned_cache
    save_json(cache_data, cache_path)

    missing_templates = [
        (key, value)
        for key, value in template_items
        if len(cache_data.get(key, []))
        < args.variants_per_question
    ]

    generator = None

    if missing_templates:
        dtype = (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        print("[INFO] Loading Mistral once...")

        generator = pipeline(
            "text-generation",
            model=args.model_id,
            torch_dtype=dtype,
            device_map="auto",
            model_kwargs={
                "cache_dir": args.cache_dir,
            },
        )

        print("[OK] Model loaded.")

    for position, (key, template) in enumerate(
        missing_templates,
        start=1,
    ):
        print(
            f"[{position}/{len(missing_templates)}] "
            f"{template['category']} | {template['question']}"
        )

        variants = generate_variants_until_complete(
            generator=generator,
            original_question=template["question"],
            category=template["category"],
            target_count=args.variants_per_question,
            debug_dir=debug_dir,
            template_number=position,
            max_rounds=args.max_rounds,
        )

        cache_data[key] = variants
        save_json(cache_data, cache_path)

    augmented_data: list[dict[str, Any]] = []
    augmented_data.extend(canonical_data)

    seen_by_patient: set[tuple[str, str, str]] = {
        (
            str(item.get("patient_id", "unknown")),
            str(item.get("category", "unknown")),
            normalize_text(item.get("question", "")),
        )
        for item in canonical_data
    }

    augmented_count = 0

    for item in canonical_data:
        category = str(item.get("category", "unknown"))
        original_question = str(item.get("question", "")).strip()

        if category not in selected_categories:
            continue

        key = build_template_key(
            category=category,
            question=original_question,
        )

        variants = cache_data.get(key, [])

        for variant_number, variant in enumerate(
            variants,
            start=1,
        ):
            patient_id = str(
                item.get("patient_id", "unknown")
            )

            uniqueness_key = (
                patient_id,
                category,
                normalize_text(variant),
            )

            if uniqueness_key in seen_by_patient:
                continue

            new_item = copy.deepcopy(item)

            # Only the question is changed.
            new_item["question"] = variant
            new_item["augmentation"] = {
                "type": "question_paraphrase",
                "model_id": args.model_id,
                "canonical_question": original_question,
                "variant_number": variant_number,
            }

            augmented_data.append(new_item)
            seen_by_patient.add(uniqueness_key)
            augmented_count += 1

    save_json(
        augmented_data,
        out_json,
    )

    summary = {
        "input_canonical_samples": len(canonical_data),
        "excluded_patient_ids": sorted(excluded_patient_ids),
        "excluded_samples": excluded_sample_count,
        "removed_exact_duplicates": duplicate_count,
        "selected_categories": sorted(selected_categories),
        "unique_templates_considered": len(template_items),
        "variants_per_question": args.variants_per_question,
        "augmented_samples_added": augmented_count,
        "output_total_samples": len(augmented_data),
        "model_id": args.model_id,
        "cache_json": str(cache_path),
        "debug_dir": str(debug_dir),
        "preserved_fields": [
            "answer",
            "evidence",
            "category",
            "answerable",
            "language",
            "patient_id",
        ],
    }

    summary_path = out_json.with_name(
        out_json.stem + "_summary.json"
    )

    save_json(summary, summary_path)

    print()
    print("========== Question augmentation finished ==========")
    print(f"Canonical samples       : {len(canonical_data)}")
    print(f"Excluded samples        : {excluded_sample_count}")
    print(f"Removed duplicates      : {duplicate_count}")
    print(f"Unique templates        : {len(template_items)}")
    print(f"Augmented samples added : {augmented_count}")
    print(f"Output total samples    : {len(augmented_data)}")
    print(f"Output                  : {out_json}")
    print(f"Summary                 : {summary_path}")
    print(f"Resume cache            : {cache_path}")


if __name__ == "__main__":
    main()
