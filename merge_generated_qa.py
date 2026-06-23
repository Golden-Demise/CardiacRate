import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_patient_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    numbers = re.findall(r"\d+", text)
    if numbers:
        return f"patient{int(numbers[-1]):04d}"
    return text.lower() or None


def normalize_facts_path(path: str) -> str:
    path = str(path).strip()
    for prefix in ("structured_facts.", "facts."):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def get_by_path(data: Any, path: str) -> Any:
    value = data
    for key in normalize_facts_path(path).split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise KeyError(path)
    return value


def build_facts_index(facts_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in facts_dir.glob("*.json"):
        try:
            data = load_json(path)
        except Exception:
            continue
        patient_id = normalize_patient_id(
            data.get("patient_id") or data.get("case_id") or path.stem
        )
        if patient_id:
            index[patient_id] = path
    return index


def derive_answerable(answerability: str) -> bool:
    return answerability in {"fully_answerable", "partially_answerable"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-patient generated QA files into one flat QA dataset "
            "and attach evidence values from the corresponding facts files."
        )
    )
    parser.add_argument("--qa_dir", required=True)
    parser.add_argument("--facts_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--qa_glob", default="patient*_qa.json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Skip QA items when any facts_path cannot be resolved.",
    )
    args = parser.parse_args()

    qa_dir = Path(args.qa_dir)
    facts_dir = Path(args.facts_dir)
    out_json = Path(args.out_json)

    facts_index = build_facts_index(facts_dir)
    merged: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "qa_files_found": 0,
        "qa_files_processed": 0,
        "qa_items_saved": 0,
        "missing_facts_files": [],
        "invalid_qa_files": [],
        "missing_facts_paths": [],
        "category_counts": {},
        "answerability_counts": {},
    }

    category_counter: Counter[str] = Counter()
    answerability_counter: Counter[str] = Counter()

    qa_paths = sorted(qa_dir.glob(args.qa_glob))
    summary["qa_files_found"] = len(qa_paths)

    for qa_path in qa_paths:
        try:
            qa_data = load_json(qa_path)
        except Exception as e:
            summary["invalid_qa_files"].append({
                "file": str(qa_path),
                "error": str(e),
            })
            continue

        patient_id = normalize_patient_id(
            qa_data.get("case_id") or qa_data.get("patient_id") or qa_path.stem
        )
        if not patient_id:
            summary["invalid_qa_files"].append({
                "file": str(qa_path),
                "error": "Cannot determine patient_id.",
            })
            continue

        facts_path = facts_index.get(patient_id)
        if facts_path is None:
            summary["missing_facts_files"].append({
                "patient_id": patient_id,
                "qa_file": str(qa_path),
            })
            continue

        facts = load_json(facts_path)
        qa_pairs = qa_data.get("qa_pairs", [])
        if not isinstance(qa_pairs, list):
            summary["invalid_qa_files"].append({
                "file": str(qa_path),
                "error": "qa_pairs is not a list.",
            })
            continue

        summary["qa_files_processed"] += 1

        for qa_index, item in enumerate(qa_pairs):
            if not isinstance(item, dict):
                continue

            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer:
                continue

            raw_paths = item.get("facts_paths", [])
            if not isinstance(raw_paths, list):
                raw_paths = []

            evidence_values: dict[str, Any] = {}
            unresolved_paths: list[str] = []

            for raw_path in raw_paths:
                normalized_path = normalize_facts_path(raw_path)
                try:
                    evidence_values[normalized_path] = get_by_path(
                        facts, normalized_path
                    )
                except KeyError:
                    unresolved_paths.append(str(raw_path))

            if unresolved_paths:
                summary["missing_facts_paths"].append({
                    "patient_id": patient_id,
                    "qa_file": str(qa_path),
                    "qa_index": qa_index,
                    "paths": unresolved_paths,
                })
                if args.strict:
                    continue

            answerability = str(
                item.get("answerability", "fully_answerable")
            ).strip()
            category = str(item.get("category", "unknown")).strip()

            merged_item = {
                "patient_id": patient_id,
                "case_id": patient_id,
                "category": category,
                "language": item.get("language", "en"),
                "question": question,
                "answer": answer,
                "answerability": answerability,
                "answerable": derive_answerable(answerability),
                "facts_paths": [
                    normalize_facts_path(path)
                    for path in raw_paths
                ],
                "evidence": {
                    "patient_specific": evidence_values,
                },
                "source": {
                    "qa_file": str(qa_path),
                    "facts_file": str(facts_path),
                },
            }

            merged.append(merged_item)
            category_counter[category] += 1
            answerability_counter[answerability] += 1

    summary["qa_items_saved"] = len(merged)
    summary["category_counts"] = dict(category_counter)
    summary["answerability_counts"] = dict(answerability_counter)

    save_json(merged, out_json)
    save_json(
        summary,
        out_json.with_name(out_json.stem + "_summary.json"),
    )

    print("========== QA merge completed ==========")
    print(f"QA files found     : {summary['qa_files_found']}")
    print(f"QA files processed : {summary['qa_files_processed']}")
    print(f"QA items saved     : {summary['qa_items_saved']}")
    print(f"Output             : {out_json}")
    print(
        "Summary            : "
        f"{out_json.with_name(out_json.stem + '_summary.json')}"
    )


if __name__ == "__main__":
    main()
