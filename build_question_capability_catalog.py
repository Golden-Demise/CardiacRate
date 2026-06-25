import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a question capability catalog from a QA JSON dataset."
    )
    parser.add_argument("--input", required=True, help="Input QA JSON")
    parser.add_argument("--output_json", required=True, help="Output catalog JSON")
    parser.add_argument("--output_md", required=True, help="Output Markdown report")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")

    stats = defaultdict(lambda: {
        "samples": 0,
        "patients": set(),
        "answerable": Counter(),
        "questions": Counter(),
        "canonical_questions": Counter(),
        "sources": Counter(),
    })

    for item in data:
        category = item.get("category", "unknown")
        s = stats[category]
        s["samples"] += 1
        s["patients"].add(str(item.get("patient_id", "unknown")))
        s["answerable"][str(item.get("answerable"))] += 1

        question = str(item.get("question", "")).strip()
        if question:
            s["questions"][question] += 1

        augmentation = item.get("augmentation")
        if isinstance(augmentation, dict) and augmentation.get("canonical_question"):
            canonical = str(augmentation["canonical_question"]).strip()
        else:
            canonical = question

        if canonical:
            s["canonical_questions"][canonical] += 1

        source = item.get("evidence", {}).get("source")
        if source:
            s["sources"][str(source)] += 1

    def status(counter: Counter) -> str:
        t = counter.get("True", 0)
        f = counter.get("False", 0)
        if t and not f:
            return "answerable"
        if f and not t:
            return "requires_clinical_confirmation_or_refusal"
        return "mixed"

    catalog = []
    for category in sorted(stats):
        s = stats[category]
        questions = sorted(
            s["questions"].keys(),
            key=lambda q: (-s["questions"][q], q),
        )
        canonical = sorted(
            s["canonical_questions"].keys(),
            key=lambda q: (-s["canonical_questions"][q], q),
        )
        catalog.append({
            "category": category,
            "status": status(s["answerable"]),
            "sample_count": s["samples"],
            "patient_count": len(s["patients"]),
            "answerable_counts": dict(s["answerable"]),
            "canonical_questions": canonical,
            "example_questions": questions[:10],
            "evidence_sources": [x for x, _ in s["sources"].most_common()],
        })

    Path(args.output_json).write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# Question Capability Catalog", ""]
    for item in catalog:
        lines.append(f"## {item['category']}")
        lines.append("")
        lines.append(f"- Status: {item['status']}")
        lines.append(f"- Samples: {item['sample_count']}")
        lines.append(f"- Patients: {item['patient_count']}")
        lines.append("- Example questions:")
        for question in item["example_questions"]:
            lines.append(f"  - {question}")
        lines.append("")

    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")

    print(f"Categories: {len(catalog)}")
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
