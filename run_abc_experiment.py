#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_abc_experiment.py

Run three ablation groups on the same test set:

A: Base Mistral, without case facts
B: Base Mistral, with case facts
C: Base Mistral + LoRA adapter, with case facts

Outputs:
  A_base_no_facts.jsonl
  B_base_with_facts.jsonl
  C_lora_with_facts.jsonl
  abc_all_predictions.csv
  abc_summary.csv
  abc_by_category.csv

Recommended test JSONL format (one JSON object per line):
{
  "case_id": "patient_0001",
  "category": "structure_volume",
  "question": "What is the volume of the myocardium?",
  "gold": "The myocardium volume is approximately 177.791 mL.",
  "facts": {
    "structures": {
      "myocardium": {"volume_ml": 177.791}
    }
  }
}

Supported aliases:
  question: question, query, instruction
  gold: gold, answer, reference, response, output
  facts: facts, context, evidence, facts_json
  category: category, type, question_type
  case id: case_id, patient_id, id

A facts_path field is also supported and will be loaded as JSON.
A messages-formatted record is partially supported:
  - last user message -> question
  - last assistant message -> gold
Explicit question/gold/facts fields are still strongly recommended.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


GROUPS = {
    "A": {
        "label": "Base model without facts",
        "filename": "A_base_no_facts.jsonl",
        "use_facts": False,
        "use_lora": False,
    },
    "B": {
        "label": "Base model with facts",
        "filename": "B_base_with_facts.jsonl",
        "use_facts": True,
        "use_lora": False,
    },
    "C": {
        "label": "LoRA model with facts",
        "filename": "C_lora_with_facts.jsonl",
        "use_facts": True,
        "use_lora": True,
    },
}

REFUSAL_PATTERNS = [
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bunable\b",
    r"\binsufficient\b",
    r"\bnot enough\b",
    r"\bnot available\b",
    r"\bnot provided\b",
    r"\bcannot be reliably answered\b",
    r"\bdo not have enough information\b",
    r"無法",
    r"不足",
    r"未提供",
    r"沒有足夠",
    r"不能可靠",
]

UNANSWERABLE_CATEGORY_NAMES = {
    "unanswerable",
    "unsupported",
    "insufficient_evidence",
    "out_of_scope",
    "cannot_answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A/B/C ablation experiments for Base Mistral and LoRA."
    )
    parser.add_argument(
        "--test_jsonl",
        required=True,
        help="Path to test JSONL.",
    )
    parser.add_argument(
        "--base_model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Hugging Face model id or local base-model directory.",
    )
    parser.add_argument(
        "--lora_path",
        required=True,
        help="Local LoRA adapter directory or Hugging Face adapter id.",
    )
    parser.add_argument(
        "--output_dir",
        default="./abc_results",
        help="Directory for predictions and summary files.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--groups",
        default="ABC",
        help="Groups to run, e.g. ABC, BC, or C.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of newly generated tokens.",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=6144,
        help="Maximum prompt length after tokenization.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N records, useful for a smoke test.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the base model in 4-bit. Requires bitsandbytes.",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Transformers device_map. Default: "auto". Use "none" to disable.',
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom model code from the model repository.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing group output files instead of resuming.",
    )
    parser.add_argument(
        "--relative_number_tolerance",
        type=float,
        default=1e-3,
        help="Relative tolerance used by the built-in numeric-match metric.",
    )
    parser.add_argument(
        "--absolute_number_tolerance",
        type=float,
        default=1e-3,
        help="Absolute tolerance used by the built-in numeric-match metric.",
    )
    return parser.parse_args()


def first_present(record: Dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def stringify_facts(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def extract_from_messages(messages: Any) -> Tuple[str, str]:
    question = ""
    gold = ""
    if not isinstance(messages, list):
        return question, gold

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = message.get("content", "")
        if isinstance(content, list):
            # Basic support for multimodal-style text content.
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)
        content = str(content).strip()

        if role == "user":
            question = content
        elif role == "assistant":
            gold = content

    return question, gold


def extract_from_sft_text(text: Any) -> Tuple[str, str, Any, str, str]:
    """
    Parse the single `text` field used by sft_train8/sft_val8/sft_test8.
    """
    if not isinstance(text, str) or not text.strip():
        return "", "", None, "", ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    if "### User:" not in normalized or "### Assistant:" not in normalized:
        return "", "", None, "", ""

    user_prefix, assistant_section = normalized.rsplit("### Assistant:", 1)
    user_section = user_prefix.split("### User:", 1)[1]

    try:
        question_start = user_section.index("Question:") + len("Question:")
        evidence_marker = user_section.index("\n\nEvidence:", question_start)
        question = user_section[question_start:evidence_marker].strip()

        evidence_start = evidence_marker + len("\n\nEvidence:")
        answerability_marker = user_section.index(
            "\n\nAnswerability:", evidence_start
        )
        evidence_text = user_section[evidence_start:answerability_marker].strip()

        answerability_start = (
            answerability_marker + len("\n\nAnswerability:")
        )
        category_marker = user_section.index(
            "\n\nCategory:", answerability_start
        )
        answerability = user_section[
            answerability_start:category_marker
        ].strip()

        category_start = category_marker + len("\n\nCategory:")
        category = user_section[category_start:].strip()
    except ValueError:
        question_match = re.search(
            r"Question:\s*(.*?)\s*Evidence:\s*",
            user_section,
            flags=re.DOTALL,
        )
        evidence_match = re.search(
            r"Evidence:\s*(.*?)\s*Answerability:\s*",
            user_section,
            flags=re.DOTALL,
        )
        answerability_match = re.search(
            r"Answerability:\s*(.*?)\s*Category:\s*",
            user_section,
            flags=re.DOTALL,
        )
        category_match = re.search(
            r"Category:\s*(.*?)\s*$",
            user_section,
            flags=re.DOTALL,
        )

        question = question_match.group(1).strip() if question_match else ""
        evidence_text = evidence_match.group(1).strip() if evidence_match else ""
        answerability = (
            answerability_match.group(1).strip()
            if answerability_match
            else ""
        )
        category = category_match.group(1).strip() if category_match else ""

    gold = assistant_section.strip()

    facts: Any = evidence_text
    if evidence_text:
        try:
            facts = json.loads(evidence_text)
        except json.JSONDecodeError:
            facts = evidence_text

    return question, gold, facts, category, answerability


def load_external_facts(record: Dict[str, Any], test_file: Path) -> Any:
    facts_path = record.get("facts_path")
    if not facts_path:
        return None

    path = Path(str(facts_path))
    if not path.is_absolute():
        path = test_file.parent / path
    if not path.exists():
        raise FileNotFoundError(f"facts_path does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_record(
    raw: Dict[str, Any],
    index: int,
    test_file: Path,
) -> Dict[str, Any]:
    question = first_present(raw, ["question", "query", "instruction"])
    gold = first_present(raw, ["gold", "answer", "reference", "response", "output"])
    facts = first_present(raw, ["facts", "context", "evidence", "facts_json"])
    category = first_present(raw, ["category", "type", "question_type"])
    answerability = first_present(raw, ["answerability"])

    if not question or not gold:
        msg_question, msg_gold = extract_from_messages(raw.get("messages"))
        question = question or msg_question
        gold = gold or msg_gold

    if (not question or not gold or facts is None) and raw.get("text"):
        (
            sft_question,
            sft_gold,
            sft_facts,
            sft_category,
            sft_answerability,
        ) = extract_from_sft_text(raw.get("text"))
        question = question or sft_question
        gold = gold or sft_gold
        if facts is None:
            facts = sft_facts
        category = category or sft_category
        answerability = answerability or sft_answerability

    if facts is None and raw.get("facts_path"):
        facts = load_external_facts(raw, test_file)

    case_id = first_present(raw, ["case_id", "patient_id", "id"])

    if not question:
        raise ValueError(
            f"Record {index} has no question. Add a 'question' field or use "
            "the supported SFT text format."
        )
    if not gold:
        raise ValueError(
            f"Record {index} has no gold answer. Add a 'gold'/'answer' field "
            "or use the supported SFT text format."
        )

    return {
        "record_id": str(raw.get("record_id", index)),
        "case_id": str(case_id) if case_id is not None else "",
        "category": str(category) if category is not None else "unknown",
        "answerability": (
            str(answerability) if answerability is not None else ""
        ),
        "question": str(question).strip(),
        "gold": str(gold).strip(),
        "facts": stringify_facts(facts),
        "source": raw,
    }


def read_jsonl(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}."
                )
            records.append(normalize_record(raw, len(records), path))
            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def build_messages(
    question: str,
    facts: str,
    use_facts: bool,
) -> List[Dict[str, str]]:
    if use_facts:
        system = (
            "You are an evidence-grounded cardiac imaging health information "
            "assistant. Answer only from the structured case facts supplied by "
            "the user. Do not invent measurements, findings, diagnoses, "
            "symptoms, treatments, or clinical history. If the facts do not "
            "support the requested conclusion, clearly state that the question "
            "cannot be reliably answered from the current facts. Explain the "
            "available evidence in clear language. A classifier output is a "
            "model prediction, not a confirmed clinical diagnosis."
        )
        user = (
            "Structured case facts:\n"
            f"{facts if facts else '[No structured facts were supplied]'}\n\n"
            "Question:\n"
            f"{question}"
        )
    else:
        system = (
            "You are a conservative cardiac imaging health information "
            "assistant. No case-specific structured facts are available in "
            "this experiment. Do not invent patient-specific measurements, "
            "findings, classifications, diagnoses, symptoms, treatments, or "
            "clinical history. When the question requires case-specific "
            "evidence, clearly state that it cannot be reliably answered "
            "without the case facts."
        )
        user = question

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_quantization_config(load_in_4bit: bool) -> Any:
    if not load_in_4bit:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "--load_in_4bit requires a Transformers version with "
            "BitsAndBytesConfig and the bitsandbytes package."
        ) from exc

    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_base_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = build_quantization_config(args.load_in_4bit)
    device_map = None if str(args.device_map).lower() == "none" else args.device_map

    model_kwargs: Dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device_map or "auto"
    else:
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = device_map or "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32
            if device_map is not None:
                model_kwargs["device_map"] = device_map

    print(f"Loading base model: {args.base_model}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )
    model.eval()
    return tokenizer, model


def model_input_device(model) -> torch.device:
    """
    Return the device that should receive input token tensors.

    Prefer the input-embedding device. This is safer than next(parameters())
    for PEFT models and models loaded with device_map="auto".
    """
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            device = embeddings.weight.device
            if device.type not in {"meta"}:
                return device
    except Exception:
        pass

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device_value in hf_device_map.values():
            if device_value in {"cpu", "disk", "meta"}:
                continue
            if isinstance(device_value, int):
                return torch.device(f"cuda:{device_value}")
            try:
                return torch.device(str(device_value))
            except (TypeError, RuntimeError):
                continue

    try:
        device = next(model.parameters()).device
        if device.type != "meta":
            return device
    except (StopIteration, AttributeError):
        pass

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    max_input_tokens: int,
) -> Tuple[str, int, int, float]:
    started = time.perf_counter()

    # Some Mistral tokenizer/chat-template versions do not accept a separate
    # system role. Fold the system instruction into the first user message
    # while preserving identical content across the compared groups.
    system_text = "\n\n".join(
        str(message.get("content", "")).strip()
        for message in messages
        if message.get("role") == "system"
    ).strip()
    user_text = "\n\n".join(
        str(message.get("content", "")).strip()
        for message in messages
        if message.get("role") == "user"
    ).strip()

    chat_messages = [
        {
            "role": "user",
            "content": (
                f"System instructions:\n{system_text}\n\n"
                f"User request:\n{user_text}"
                if system_text
                else user_text
            ),
        }
    ]

    if getattr(tokenizer, "chat_template", None):
        # Rendering to text first avoids compatibility problems where some
        # tokenizer versions return an unexpected object for tokenize=True.
        prompt = tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        prompt = (
            f"USER:\n{chat_messages[0]['content']}\n\n"
            "ASSISTANT:\n"
        )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_input_tokens,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    if input_ids.shape[-1] > max_input_tokens:
        input_ids = input_ids[:, -max_input_tokens:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -max_input_tokens:]

    device = model_input_device(model)
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_tokens = int(input_ids.shape[-1])

    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask

    output_ids = model.generate(**generation_kwargs)

    if not isinstance(output_ids, torch.Tensor):
        sequences = getattr(output_ids, "sequences", None)
        if sequences is None:
            raise TypeError(
                "model.generate() returned an unsupported output type: "
                f"{type(output_ids)!r}"
            )
        output_ids = sequences

    new_tokens = output_ids[0, prompt_tokens:]
    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    elapsed = time.perf_counter() - started
    return answer, prompt_tokens, int(new_tokens.shape[-1]), elapsed


def normalized_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]", text.lower())


def lexical_f1(gold: str, pred: str) -> float:
    gold_tokens = normalized_tokens(gold)
    pred_tokens = normalized_tokens(pred)
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_counts: Dict[str, int] = defaultdict(int)
    pred_counts: Dict[str, int] = defaultdict(int)
    for token in gold_tokens:
        gold_counts[token] += 1
    for token in pred_tokens:
        pred_counts[token] += 1

    overlap = sum(
        min(gold_counts[token], pred_counts[token])
        for token in gold_counts.keys() | pred_counts.keys()
    )
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


NUMBER_RE = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def extract_numbers(text: str) -> List[float]:
    values: List[float] = []
    for match in NUMBER_RE.findall(text):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def numbers_close(
    a: float,
    b: float,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def numeric_recall(
    gold: str,
    pred: str,
    rel_tol: float,
    abs_tol: float,
) -> Optional[float]:
    gold_numbers = extract_numbers(gold)
    if not gold_numbers:
        return None
    pred_numbers = extract_numbers(pred)
    if not pred_numbers:
        return 0.0

    unused = list(pred_numbers)
    matched = 0
    for gold_value in gold_numbers:
        found_index = None
        for idx, pred_value in enumerate(unused):
            if numbers_close(gold_value, pred_value, rel_tol, abs_tol):
                found_index = idx
                break
        if found_index is not None:
            matched += 1
            unused.pop(found_index)
    return matched / len(gold_numbers)


def contains_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in REFUSAL_PATTERNS)


def expected_unanswerable(category: str, gold: str) -> bool:
    normalized_category = category.strip().lower().replace(" ", "_")
    return (
        normalized_category in UNANSWERABLE_CATEGORY_NAMES
        or contains_refusal(gold)
    )


def evaluate_row(
    category: str,
    gold: str,
    pred: str,
    rel_tol: float,
    abs_tol: float,
) -> Dict[str, Any]:
    is_unanswerable = expected_unanswerable(category, gold)
    pred_refused = contains_refusal(pred)
    return {
        "lexical_f1": lexical_f1(gold, pred),
        "numeric_recall": numeric_recall(
            gold, pred, rel_tol=rel_tol, abs_tol=abs_tol
        ),
        "expected_unanswerable": is_unanswerable,
        "predicted_refusal": pred_refused,
        "refusal_correct": (
            int(pred_refused == is_unanswerable)
        ),
        "unwanted_refusal": int(pred_refused and not is_unanswerable),
    }


def load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    existing: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            existing[str(row["record_id"])] = row
    return existing


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def run_group(
    group_key: str,
    records: List[Dict[str, Any]],
    model,
    tokenizer,
    args: argparse.Namespace,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    spec = GROUPS[group_key]
    output_path = output_dir / spec["filename"]

    if args.overwrite and output_path.exists():
        output_path.unlink()

    existing = load_existing_results(output_path)
    results: List[Dict[str, Any]] = []

    print()
    print("=" * 78)
    print(f"Group {group_key}: {spec['label']}")
    print(f"Output: {output_path}")
    print("=" * 78)

    for position, record in enumerate(records, start=1):
        record_id = record["record_id"]
        if record_id in existing:
            row = existing[record_id]
            results.append(row)
            resume_status = (
                "ERROR"
                if row.get("error")
                else "OK"
            )
            print(
                f"[{position}/{len(records)}] "
                f"resume record_id={record_id} status={resume_status}"
            )
            continue

        messages = build_messages(
            question=record["question"],
            facts=record["facts"],
            use_facts=bool(spec["use_facts"]),
        )

        try:
            pred, prompt_tokens, generated_tokens, elapsed = generate_answer(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
            )
            error = ""
        except Exception as exc:
            pred = ""
            prompt_tokens = 0
            generated_tokens = 0
            elapsed = 0.0
            exception_repr = repr(exc)
            traceback_text = traceback.format_exc()
            error = (
                f"{type(exc).__name__}: {exception_repr}\n"
                f"{traceback_text}"
            )
            print(
                f"ERROR group={group_key} record_id={record_id}: "
                f"{type(exc).__name__}: {exception_repr}",
                file=sys.stderr,
            )
            print(traceback_text, file=sys.stderr)

        metrics = evaluate_row(
            category=record["category"],
            gold=record["gold"],
            pred=pred,
            rel_tol=args.relative_number_tolerance,
            abs_tol=args.absolute_number_tolerance,
        )

        row = {
            "group": group_key,
            "group_label": spec["label"],
            "record_id": record_id,
            "case_id": record["case_id"],
            "category": record["category"],
            "question": record["question"],
            "gold": record["gold"],
            "prediction": pred,
            "used_facts": bool(spec["use_facts"]),
            "used_lora": bool(spec["use_lora"]),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "generation_seconds": elapsed,
            "error": error,
            **metrics,
        }
        append_jsonl(output_path, row)
        results.append(row)

        number_display = metrics["numeric_recall"]
        number_text = "N/A" if number_display is None else f"{number_display:.3f}"
        print(
            f"[{position}/{len(records)}] "
            f"id={record_id} "
            f"lexical={metrics['lexical_f1']:.3f} "
            f"number={number_text} "
            f"refusal_ok={metrics['refusal_correct']} "
            f"time={elapsed:.2f}s"
        )

    return results


def mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return statistics.fmean(clean)


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    unanswerable = [row for row in valid if row["expected_unanswerable"]]
    answerable = [row for row in valid if not row["expected_unanswerable"]]
    numeric = [row for row in valid if row["numeric_recall"] is not None]

    return {
        "n_total": len(rows),
        "n_valid": len(valid),
        "n_errors": len(rows) - len(valid),
        "mean_lexical_f1": mean_optional(
            row["lexical_f1"] for row in valid
        ),
        "mean_numeric_recall": mean_optional(
            row["numeric_recall"] for row in numeric
        ),
        "numeric_questions": len(numeric),
        "refusal_accuracy_all": mean_optional(
            row["refusal_correct"] for row in valid
        ),
        "refusal_recall_unanswerable": mean_optional(
            int(row["predicted_refusal"]) for row in unanswerable
        ),
        "unwanted_refusal_rate_answerable": mean_optional(
            row["unwanted_refusal"] for row in answerable
        ),
        "mean_generation_seconds": mean_optional(
            row["generation_seconds"] for row in valid
        ),
        "mean_generated_tokens": mean_optional(
            row["generated_tokens"] for row in valid
        ),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_summaries(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> None:
    combined = [
        row
        for group_key in all_results
        for row in all_results[group_key]
    ]
    write_csv(output_dir / "abc_all_predictions.csv", combined)

    summary_rows = []
    for group_key, rows in all_results.items():
        summary_rows.append(
            {
                "group": group_key,
                "group_label": GROUPS[group_key]["label"],
                **summarize_rows(rows),
            }
        )
    write_csv(output_dir / "abc_summary.csv", summary_rows)

    category_rows = []
    for group_key, rows in all_results.items():
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["category"])].append(row)
        for category, category_group in sorted(grouped.items()):
            category_rows.append(
                {
                    "group": group_key,
                    "group_label": GROUPS[group_key]["label"],
                    "category": category,
                    **summarize_rows(category_group),
                }
            )
    write_csv(output_dir / "abc_by_category.csv", category_rows)

    def format_optional(
        value: Optional[float],
        decimals: int = 4,
        suffix: str = "",
    ) -> str:
        if value is None:
            return "N/A"
        return f"{float(value):.{decimals}f}{suffix}"

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    for row in summary_rows:
        print(
            f"{row['group']} | "
            f"valid={row['n_valid']}/{row['n_total']} | "
            f"errors={row['n_errors']} | "
            f"lexical_f1={format_optional(row['mean_lexical_f1'])} | "
            f"numeric_recall="
            f"{format_optional(row['mean_numeric_recall'])} | "
            f"refusal_accuracy="
            f"{format_optional(row['refusal_accuracy_all'])} | "
            f"mean_time="
            f"{format_optional(row['mean_generation_seconds'], 2, 's')}"
        )

        if row["n_valid"] == 0:
            print(
                "  WARNING: This group has no successful predictions. "
                "Check the 'error' column in the group JSONL file or in "
                "abc_all_predictions.csv. Re-run with --overwrite after "
                "fixing the underlying model-generation error."
            )

    print()
    print(f"Saved: {output_dir / 'abc_all_predictions.csv'}")
    print(f"Saved: {output_dir / 'abc_summary.csv'}")
    print(f"Saved: {output_dir / 'abc_by_category.csv'}")


def validate_groups(groups_text: str) -> List[str]:
    groups = []
    for char in groups_text.upper():
        if char.isspace() or char in {",", ";", "/"}:
            continue
        if char not in GROUPS:
            raise ValueError(
                f"Unknown group '{char}'. Valid groups are A, B, and C."
            )
        if char not in groups:
            groups.append(char)
    if not groups:
        raise ValueError("No groups selected.")
    return groups


def main() -> None:
    args = parse_args()
    test_path = Path(args.test_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = validate_groups(args.groups)
    if "train" in test_path.name.lower():
        print(
            "WARNING: The selected filename contains 'train'. This is suitable "
            "only for a smoke test, not for reporting final A/B/C evaluation "
            "results, because the LoRA model has already seen these examples."
        )

    records = read_jsonl(test_path, args.limit)

    print(f"Test records: {len(records)}")
    print(f"Selected groups: {', '.join(groups)}")
    print(f"Output directory: {output_dir}")
    print(f"PyTorch version: {torch.__version__}")
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except Exception:
        pass
    try:
        import peft
        print(f"PEFT version: {peft.__version__}")
    except Exception:
        pass

    # B must be generated before loading the adapter so that it remains a pure
    # base-model result. A and B share the same original base model.
    ordered_groups = [g for g in ["A", "B", "C"] if g in groups]

    tokenizer, base_model = load_base_model(args)
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for group_key in ordered_groups:
        if group_key == "C":
            if PeftModel is None:
                raise RuntimeError(
                    "PEFT is not installed. Run: pip install peft"
                )
            print()
            print(f"Loading LoRA adapter: {args.lora_path}")
            model = PeftModel.from_pretrained(
                base_model,
                args.lora_path,
                is_trainable=False,
            )
            model.eval()
        else:
            model = base_model

        all_results[group_key] = run_group(
            group_key=group_key,
            records=records,
            model=model,
            tokenizer=tokenizer,
            args=args,
            output_dir=output_dir,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    create_summaries(all_results, output_dir)


if __name__ == "__main__":
    main()
