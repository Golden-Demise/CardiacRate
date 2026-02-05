import os, json, argparse, random

SYSTEM = (
    "You are a cardiac assistant. Answer in English. "
    "Use ONLY the provided FACTS. If the answer is not available in FACTS, "
    "reply exactly: Not available in provided facts."
)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_facts_block(facts_obj, use_derived_only=True):
    if use_derived_only and "derived" in facts_obj:
        payload = {
            "case_id": facts_obj.get("case_id", ""),
            "qc_flags": facts_obj.get("qc_flags", []),
            "derived": facts_obj.get("derived", {})
        }
    else:
        payload = facts_obj
    return json.dumps(payload, ensure_ascii=False, indent=2)

def make_text(facts_block, question, answer):
    return (
        f"### SYSTEM\n{SYSTEM}\n\n"
        f"### FACTS\n{facts_block}\n\n"
        f"### QUESTION\n{question}\n\n"
        f"### ANSWER\n{answer}\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_jsonl", required=True)
    ap.add_argument("--out_train", default="train.jsonl")
    ap.add_argument("--out_val", default="val.jsonl")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_derived_only", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    rows = []
    with open(args.qa_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    def write(rows, out_path):
        with open(out_path, "w", encoding="utf-8") as out:
            for r in rows:
                facts_path = r.get("facts_file")
                if not facts_path or not os.path.exists(facts_path):
                    continue
                facts_obj = load_json(facts_path)
                facts_block = build_facts_block(facts_obj, use_derived_only=args.use_derived_only)

                q = r["question"]
                a = r["answer"]
                text = make_text(facts_block, q, a)

                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    write(train_rows, args.out_train)
    write(val_rows, args.out_val)

    print(f"Saved train: {args.out_train} ({len(train_rows)})")
    print(f"Saved val:   {args.out_val} ({len(val_rows)})")

if __name__ == "__main__":
    main()
