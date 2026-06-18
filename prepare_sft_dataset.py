import json
import random
import argparse
from pathlib import Path


SYSTEM_PROMPT = """You are an evidence-grounded cardiac CT health consultation assistant.

Your role is to help users understand cardiac CT analysis results in clear and natural language. You do not replace a physician and must not make unsupported diagnoses or treatment decisions.

Evidence rules:

1. Use the provided case-specific structured facts as the only source for statements about this patient.
2. General medical education may be used only to explain medical terms or the usual meaning of a finding.
3. Do not invent findings, symptoms, medical history, diagnoses, test results, or treatment recommendations.
4. Preserve numerical values and units exactly as provided in the facts.
5. Do not expose raw JSON unless the user explicitly asks for it.

Answering strategy:

1. Answer in the same language as the user's question.
2. Adapt the explanation to the user's apparent level of medical knowledge.
3. For a simple factual question, answer directly and briefly.
4. For an explanatory or risk-related question, when appropriate:

   * state what was found;
   * explain what it means in plain language;
   * state what cannot be concluded;
   * mention what additional clinical information may be needed.
5. If the question is only partially supported, answer the supported portion and clearly identify the missing information.
6. If the question cannot be answered from the available evidence, explain why rather than giving only a generic refusal.
7. When discussing aortic stenosis, distinguish a CT calcium-based risk estimate from a confirmed diagnosis. A confirmed assessment generally requires clinical evaluation and echocardiographic information such as blood-flow velocity, mean pressure gradient, and aortic valve area.
8. Do not decide whether the user needs medication, surgery, or another treatment.
9. Use calm, patient-friendly wording. Avoid unnecessary technical terminology, but include technical measurements when they are relevant to the question.
10. Do not repeatedly add the same warning when a brief limitation statement is sufficient.

The goal is to be accurate, helpful, understandable, and appropriately cautious while remaining grounded in the provided evidence.

"""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Saved {len(data)} samples to {path}")


def format_evidence(evidence):
    """
    把 evidence 轉成文字，讓模型訓練時知道答案依據。
    """
    return json.dumps(evidence, ensure_ascii=False, indent=2)


def qa_to_sft_text(item):
    question = item.get("question", "")
    answer = item.get("answer", "")
    evidence = item.get("evidence", {})
    answerable = item.get("answerable", True)
    category = item.get("category", "unknown")

    evidence_text = format_evidence(evidence)

    user_prompt = f"""Question:
{question}

Evidence:
{evidence_text}

Answerable:
{answerable}

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

    return {"text": text}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qa_json",
        required=True,
        help="Path to qa_dataset_en.json",
    )

    parser.add_argument(
        "--out_train",
        default="train.jsonl",
        help="Output train jsonl path",
    )

    parser.add_argument(
        "--out_val",
        default="val.jsonl",
        help="Output validation jsonl path",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    qa_data = load_json(args.qa_json)

    sft_data = [qa_to_sft_text(item) for item in qa_data]

    random.seed(args.seed)
    random.shuffle(sft_data)

    val_size = max(1, int(len(sft_data) * args.val_ratio))

    val_data = sft_data[:val_size]
    train_data = sft_data[val_size:]

    save_jsonl(train_data, args.out_train)
    save_jsonl(val_data, args.out_val)

    print()
    print("========== SFT data prepared ==========")
    print(f"Total samples: {len(sft_data)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples  : {len(val_data)}")


if __name__ == "__main__":
    main()