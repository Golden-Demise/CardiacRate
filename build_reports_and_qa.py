import json
import argparse
from pathlib import Path


STRUCTURE_EN = {
    "myocardium": "myocardium",
    "aortic_valve": "aortic valve",
    "aortic_valve_calcification": "aortic valve calcification",
}


SEVERITY_EN = {
    "none": "none",
    "mild": "mild",
    "moderate": "moderate",
    "severe": "severe",
    "unknown": "unknown",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved QA dataset to: {path}")


def fmt_number(x, digits=3):
    if x is None:
        return "unknown"
    return f"{float(x):.{digits}f}"


def get_structure(facts, name):
    return facts.get("structures", {}).get(name, {})


def get_present_structures(facts):
    structures = facts.get("structures", {})
    present_names = []

    for key, value in structures.items():
        if value.get("present", False):
            present_names.append(STRUCTURE_EN.get(key, key))

    return present_names


def make_structure_presence_qa(facts):
    patient_id = facts.get("patient_id", "unknown")
    structures = facts.get("structures", {})

    evidence_value = {
        key: value.get("present", False)
        for key, value in structures.items()
    }

    present_names = get_present_structures(facts)

    if len(present_names) == 0:
        answer = "No target anatomical structures were segmented in this CT case."
    else:
        answer = "The segmented structures include " + ", ".join(present_names) + "."

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "structure_presence",
        "question": "Which anatomical structures were segmented in this CT case?",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": "structures.*.present",
            "value": evidence_value,
        },
    }


def make_structure_volume_qa(facts, structure_key):
    patient_id = facts.get("patient_id", "unknown")
    structure = get_structure(facts, structure_key)
    structure_name = STRUCTURE_EN.get(structure_key, structure_key)

    present = structure.get("present", False)
    volume_mm3 = structure.get("volume_mm3", None)
    volume_ml = structure.get("volume_ml", None)

    if present:
        answer = (
            f"The volume of the {structure_name} is approximately "
            f"{fmt_number(volume_mm3)} mm³, equivalent to {fmt_number(volume_ml)} mL."
        )
    else:
        answer = (
            f"The {structure_name} was not segmented in this CT case; "
            "therefore, its volume cannot be computed."
        )

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "structure_volume",
        "question": f"What is the volume of the {structure_name}?",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": f"structures.{structure_key}.volume_mm3 / volume_ml",
            "value": {
                "present": present,
                "volume_mm3": volume_mm3,
                "volume_ml": volume_ml,
            },
        },
    }


def make_structure_slice_range_qa(facts, structure_key):
    patient_id = facts.get("patient_id", "unknown")
    structure = get_structure(facts, structure_key)
    structure_name = STRUCTURE_EN.get(structure_key, structure_key)

    present = structure.get("present", False)
    slice_range = structure.get("slice_range_z", None)

    if present and slice_range is not None:
        z_min, z_max = slice_range
        slice_count = z_max - z_min + 1
        answer = (
            f"The {structure_name} appears approximately from z-slice {z_min} "
            f"to z-slice {z_max}, covering about {slice_count} slices."
        )
    else:
        answer = (
            f"The {structure_name} was not segmented in this CT case; "
            "therefore, no slice range is available."
        )

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "slice_range",
        "question": f"In which z-slices does the {structure_name} appear?",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": f"structures.{structure_key}.slice_range_z",
            "value": slice_range,
        },
    }


def make_calcification_presence_qa(facts):
    patient_id = facts.get("patient_id", "unknown")
    calc = get_structure(facts, "aortic_valve_calcification")

    present = calc.get("present", False)

    if present:
        answer = "Aortic valve calcification was segmented in this CT case."
    else:
        answer = "No aortic valve calcification was segmented in this CT case."

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "calcification_presence",
        "question": "Is aortic valve calcification present in this case?",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": "structures.aortic_valve_calcification.present",
            "value": present,
        },
    }


def make_calcification_volume_qa(facts):
    patient_id = facts.get("patient_id", "unknown")
    calc = get_structure(facts, "aortic_valve_calcification")

    present = calc.get("present", False)
    volume_mm3 = calc.get("volume_mm3", None)
    volume_ml = calc.get("volume_ml", None)

    if present:
        answer = (
            "The volume of the aortic valve calcification is approximately "
            f"{fmt_number(volume_mm3)} mm³, equivalent to {fmt_number(volume_ml)} mL."
        )
    else:
        answer = (
            "No aortic valve calcification was segmented in this CT case; "
            "therefore, the calcification volume is 0."
        )

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "calcification_volume",
        "question": "What is the volume of the aortic valve calcification?",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": "structures.aortic_valve_calcification.volume_mm3 / volume_ml",
            "value": {
                "present": present,
                "volume_mm3": volume_mm3,
                "volume_ml": volume_ml,
            },
        },
    }


def make_calcification_ratio_qa(facts):
    patient_id = facts.get("patient_id", "unknown")
    metrics = facts.get("derived_metrics", {})

    ratio = metrics.get("calcification_to_aortic_valve_volume_ratio", None)

    if ratio is None:
        answer = (
            "The calcification-to-aortic-valve volume ratio cannot be computed, "
            "possibly because the aortic valve was not segmented."
        )
    else:
        percentage = ratio * 100
        answer = (
            "The aortic valve calcification volume accounts for approximately "
            f"{fmt_number(percentage)}% of the aortic valve volume."
        )

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "calcification_ratio",
        "question": "What is the calcification-to-aortic-valve volume ratio?",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": "derived_metrics.calcification_to_aortic_valve_volume_ratio",
            "value": ratio,
        },
    }


def make_calcification_severity_qa(facts):
    patient_id = facts.get("patient_id", "unknown")
    metrics = facts.get("derived_metrics", {})

    severity = metrics.get("calcification_severity_rule_based", "unknown")
    severity_en = SEVERITY_EN.get(severity, severity)

    if severity == "none":
        answer = (
            "Based on the current segmentation result, no aortic valve calcification "
            "was identified in this case."
        )
    elif severity == "unknown":
        answer = (
            "The aortic valve calcification severity cannot be estimated, "
            "possibly because the required aortic valve or calcification information is missing."
        )
    else:
        answer = (
            f"Based on the rule-based calculation, the aortic valve calcification "
            f"severity is estimated as {severity_en}. "
            "This estimate is derived from the calcification-to-aortic-valve volume ratio "
            "and should not be interpreted as a definitive clinical diagnosis."
        )

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "calcification_severity",
        "question": "What is the estimated severity of aortic valve calcification in this case?",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": "derived_metrics.calcification_severity_rule_based",
            "value": severity,
        },
    }


def make_unanswerable_qa(facts, question, target_key, reason):
    patient_id = facts.get("patient_id", "unknown")
    answerable_findings = facts.get("answerable_findings", {})
    can_answer = answerable_findings.get(target_key, False)

    if can_answer:
        answer = (
            "The current facts.json indicates that this question is answerable, "
            "but no answer template has been defined for this question."
        )
        answerable = True
    else:
        answer = f"This question cannot be reliably answered from the current facts. {reason}"
        answerable = False

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "unanswerable",
        "question": question,
        "answer": answer,
        "answerable": answerable,
        "evidence": {
            "source": f"answerable_findings.{target_key}",
            "value": can_answer,
        },
    }


def make_summary_report_qa(facts):
    patient_id = facts.get("patient_id", "unknown")

    myocardium = get_structure(facts, "myocardium")
    valve = get_structure(facts, "aortic_valve")
    calc = get_structure(facts, "aortic_valve_calcification")
    metrics = facts.get("derived_metrics", {})

    parts = []

    if myocardium.get("present", False):
        parts.append(
            f"the myocardium volume is approximately {fmt_number(myocardium.get('volume_ml'))} mL"
        )
    else:
        parts.append("the myocardium was not segmented")

    if valve.get("present", False):
        parts.append(
            f"the aortic valve volume is approximately {fmt_number(valve.get('volume_ml'))} mL"
        )
    else:
        parts.append("the aortic valve was not segmented")

    if calc.get("present", False):
        severity = metrics.get("calcification_severity_rule_based", "unknown")
        severity_en = SEVERITY_EN.get(severity, severity)
        parts.append(
            "aortic valve calcification is present, with a volume of approximately "
            f"{fmt_number(calc.get('volume_mm3'))} mm³ and a rule-based severity of {severity_en}"
        )
    else:
        parts.append("no aortic valve calcification was segmented")

    answer = (
        "Based on the segmentation-derived facts, "
        + "; ".join(parts)
        + ". These findings are derived from automatic segmentation and may be affected by segmentation errors."
    )

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "summary_report",
        "question": "Generate a structured summary based on the current facts.",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": "structures + derived_metrics",
            "value": {
                "myocardium": myocardium,
                "aortic_valve": valve,
                "aortic_valve_calcification": calc,
                "derived_metrics": metrics,
            },
        },
    }


def build_qa_for_one_facts(facts):
    qa_list = []

    qa_list.append(make_structure_presence_qa(facts))

    qa_list.append(make_structure_volume_qa(facts, "myocardium"))
    qa_list.append(make_structure_volume_qa(facts, "aortic_valve"))
    qa_list.append(make_structure_volume_qa(facts, "aortic_valve_calcification"))

    qa_list.append(make_structure_slice_range_qa(facts, "myocardium"))
    qa_list.append(make_structure_slice_range_qa(facts, "aortic_valve"))
    qa_list.append(make_structure_slice_range_qa(facts, "aortic_valve_calcification"))

    qa_list.append(make_calcification_presence_qa(facts))
    qa_list.append(make_calcification_volume_qa(facts))
    qa_list.append(make_calcification_ratio_qa(facts))
    qa_list.append(make_calcification_severity_qa(facts))

    qa_list.append(
        make_unanswerable_qa(
            facts=facts,
            question="Can this system determine coronary artery stenosis?",
            target_key="can_answer_coronary_stenosis",
            reason=(
                "The current facts.json only contains segmentation-derived information "
                "for the myocardium, aortic valve, and aortic valve calcification. "
                "It does not include coronary artery lumen segmentation or stenosis annotations."
            ),
        )
    )

    qa_list.append(
        make_unanswerable_qa(
            facts=facts,
            question="Can this system estimate the ejection fraction?",
            target_key="can_answer_ejection_fraction",
            reason=(
                "The current facts are derived from a single static CT segmentation and do not include "
                "cardiac-cycle or functional information required to estimate ejection fraction."
            ),
        )
    )

    qa_list.append(
        make_unanswerable_qa(
            facts=facts,
            question="Can this system assess cardiac function?",
            target_key="can_answer_cardiac_function",
            reason=(
                "The current facts do not include dynamic imaging, cardiac-cycle information, "
                "or clinical functional labels."
            ),
        )
    )

    qa_list.append(make_summary_report_qa(facts))

    return qa_list


def build_qa_dataset(facts_dir):
    facts_dir = Path(facts_dir)
    facts_files = sorted(facts_dir.glob("*.json"))

    if len(facts_files) == 0:
        print(f"[WARN] No facts json files found in: {facts_dir}")
        return []

    dataset = []

    for idx, facts_path in enumerate(facts_files, start=1):
        try:
            facts = load_json(facts_path)
            qa_list = build_qa_for_one_facts(facts)
            dataset.extend(qa_list)

            print(f"[{idx}/{len(facts_files)}] [OK] {facts_path.name} -> {len(qa_list)} QA pairs")

        except Exception as e:
            print(f"[{idx}/{len(facts_files)}] [FAIL] {facts_path.name}")
            print(f"    Error: {e}")

    print()
    print("========== QA dataset finished ==========")
    print(f"Facts files: {len(facts_files)}")
    print(f"QA pairs   : {len(dataset)}")

    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--facts_dir",
        required=True,
        help="Directory containing facts json files.",
    )

    parser.add_argument(
        "--out_path",
        required=True,
        help="Output QA dataset json path.",
    )

    args = parser.parse_args()

    dataset = build_qa_dataset(args.facts_dir)
    save_json(dataset, args.out_path)


if __name__ == "__main__":
    main()