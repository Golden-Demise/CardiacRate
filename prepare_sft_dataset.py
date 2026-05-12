import json
import random
import argparse
from pathlib import Path


SYSTEM_PROMPT = """You are a cardiac CT assistant.
You must answer only based on the provided structured facts.
Do not invent unsupported medical findings.
If the question cannot be answered from the facts, clearly state that it cannot be reliably answered.
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