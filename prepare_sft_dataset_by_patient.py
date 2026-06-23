import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are an evidence-grounded cardiac CT health consultation assistant.

Your role is to help users understand cardiac CT analysis results in clear and natural language. You do not replace a physician and must not make unsupported diagnoses or treatment decisions.

Evidence rules:

1. Use the provided case-specific structured facts as the only source for statements about this patient.
2. General medical education may be used only to explain medical terms or the usual meaning of a finding.
3. Do not invent findings, symptoms, medical history, diagnoses, test results, or treatment recommendations.
4. Preserve numerical values and units exactly as provided in the facts.
5. Use the correct volume conversion: 1 mL = 1000 mm³. Convert mm³ to mL by dividing by 1000, and convert mL to mm³ by multiplying by 1000.
6. Do not expose raw JSON unless the user explicitly asks for it.

Answering strategy:

1. Answer in the same language as the user's question.
2. Adapt the explanation to the user's apparent level of medical knowledge.
3. For a simple factual question, answer directly and briefly.
4. For an explanatory or risk-related question, state what was found, explain what it means, identify what cannot be concluded, and mention what additional clinical information may be needed.
5. If the question is only partially supported, answer the supported portion and clearly identify the missing information.
6. If the question cannot be answered from the available evidence, explain why rather than giving only a generic refusal.
7. A CT calcium-based aortic stenosis risk estimate is not a confirmed diagnosis. Confirmation generally requires clinical evaluation and echocardiographic information.
8. Do not decide whether the user needs medication, surgery, or another treatment.
9. Use calm, patient-friendly wording.
10. Remain grounded in the supplied evidence.
"""


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[OK] Saved {len(data)} samples to {path}")


def format_evidence(evidence: Any) -> str:
    return json.dumps(evidence, ensure_ascii=False, indent=2)


def qa_to_sft_text(item: dict[str, Any]) -> dict[str, Any]:
    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()
    evidence = item.get("evidence", {})
    answerability = item.get("answerability", "fully_answerable")
    category = item.get("category", "unknown")
    patient_id = (
        item.get("patient_id")
        or item.get("case_id")
        or "unknown"
    )

    evidence_text = format_evidence(evidence)

    user_prompt = f"""Question:
{question}

Evidence:
{evidence_text}

Answerability:
{answerability}

Category:
{category}
"""

    text = f"""### System:
{SYSTEM_PROMPT}

### User:
{user_prompt}

### Assistant:
{answer}
"""

    return {
        "text": text,
        "patient_id": patient_id,
        "category": category,
        "answerability": answerability,
    }


def split_by_patient(
    qa_data: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in qa_data:
        patient_id = str(
            item.get("patient_id")
            or item.get("case_id")
            or "unknown"
        )
        grouped[patient_id].append(item)

    patient_ids = sorted(grouped.keys())

    if len(patient_ids) < 2:
        raise ValueError(
            "At least two distinct patient IDs are required "
            "for a train/validation split."
        )

    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    val_patient_count = max(
        1,
        int(round(len(patient_ids) * val_ratio)),
    )
    val_patient_count = min(
        val_patient_count,
        len(patient_ids) - 1,
    )

    val_patient_ids = set(patient_ids[:val_patient_count])
    train_patient_ids = set(patient_ids[val_patient_count:])

    train_items: list[dict[str, Any]] = []
    val_items: list[dict[str, Any]] = []

    for patient_id, items in grouped.items():
        if patient_id in val_patient_ids:
            val_items.extend(items)
        else:
            train_items.extend(items)

    rng.shuffle(train_items)
    rng.shuffle(val_items)

    return (
        train_items,
        val_items,
        sorted(train_patient_ids),
        sorted(val_patient_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_json", required=True)
    parser.add_argument("--out_train", default="train.jsonl")
    parser.add_argument("--out_val", default="val.jsonl")
    parser.add_argument("--split_summary", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be between 0 and 1.")

    qa_data = load_json(args.qa_json)
    if not isinstance(qa_data, list):
        raise ValueError(
            "The merged QA dataset must be a JSON list."
        )

    (
        train_qa,
        val_qa,
        train_patient_ids,
        val_patient_ids,
    ) = split_by_patient(
        qa_data=qa_data,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_data = [qa_to_sft_text(item) for item in train_qa]
    val_data = [qa_to_sft_text(item) for item in val_qa]

    save_jsonl(train_data, args.out_train)
    save_jsonl(val_data, args.out_val)

    summary_path = (
        Path(args.split_summary)
        if args.split_summary
        else Path(args.out_train).with_name("split_summary.json")
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train_patient_count": len(train_patient_ids),
        "val_patient_count": len(val_patient_ids),
        "train_sample_count": len(train_data),
        "val_sample_count": len(val_data),
        "train_patient_ids": train_patient_ids,
        "val_patient_ids": val_patient_ids,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("========== SFT data prepared ==========")
    print(f"Total samples : {len(qa_data)}")
    print(f"Train samples : {len(train_data)}")
    print(f"Val samples   : {len(val_data)}")
    print(f"Train patients: {len(train_patient_ids)}")
    print(f"Val patients  : {len(val_patient_ids)}")
    print(f"Split summary : {summary_path}")


if __name__ == "__main__":
    main()
