import os
import json
import glob
import argparse


LABELS = {
    "myocardium": "myocardium",
    "aortic_valve": "aortic valve",
    "aortic_valve_calcification": "aortic valve calcification",
}


def compute_derived_if_missing(facts: dict) -> dict:
    """Ensure facts['derived'] exists. Compute from facts['structures'] if missing."""
    derived = facts.get("derived", {}) or {}
    required = [
        "myocardium_volume_ml",
        "aortic_valve_volume_ml",
        "calcification_present",
        "calcification_volume_mm3",
        "calcification_volume_ml",
        "calcification_to_valve_ratio",
    ]
    if all(k in derived for k in required):
        return derived

    structures = facts.get("structures", {}) or {}
    myo = structures.get("myocardium", {}) or {}
    valve = structures.get("aortic_valve", {}) or {}
    calci = structures.get("aortic_valve_calcification", {}) or {}

    myo_ml = float(myo.get("volume_ml", 0.0) or 0.0)
    valve_ml = float(valve.get("volume_ml", 0.0) or 0.0)

    calci_present = bool(calci.get("present", False))
    calci_mm3 = float(calci.get("volume_mm3", 0.0) or 0.0)
    calci_ml = float(calci.get("volume_ml", 0.0) or 0.0)

    valve_mm3 = float(valve.get("volume_mm3", 0.0) or 0.0)
    ratio = calci_mm3 / max(valve_mm3, 1e-6)

    derived = {
        "myocardium_volume_ml": myo_ml,
        "aortic_valve_volume_ml": valve_ml,
        "calcification_present": calci_present,
        "calcification_volume_mm3": calci_mm3 if calci_present else 0.0,
        "calcification_volume_ml": calci_ml if calci_present else 0.0,
        "calcification_to_valve_ratio": ratio if calci_present else 0.0,
    }
    facts["derived"] = derived
    return derived


def format_report_en(facts: dict) -> str:
    case_id = facts.get("case_id", "unknown_case")
    qc_flags = facts.get("qc_flags", []) or []
    d = compute_derived_if_missing(facts)

    structures = facts.get("structures", {}) or {}
    calci = structures.get("aortic_valve_calcification", {}) or {}
    ncc = calci.get("num_connected_components", None)
    largest = calci.get("largest_component_mm3", None)

    lines = []
    lines.append(f"Case: {case_id}")

    spacing = facts.get("spacing_mm", None)
    shape = facts.get("shape_dhw", None)
    if spacing:
        lines.append(f"Spacing (mm): {spacing}")
    if shape:
        lines.append(f"Shape (D,H,W): {shape}")

    if qc_flags:
        lines.append(f"QC flags: {', '.join(qc_flags)}")
    lines.append("")

    lines.append("Findings (Quantitative):")
    lines.append(f"- Myocardium volume: {d['myocardium_volume_ml']:.3f} mL")
    lines.append(f"- Aortic valve volume: {d['aortic_valve_volume_ml']:.3f} mL")

    if d["calcification_present"]:
        lines.append("- Aortic valve calcification: Present")
        lines.append(f"  - Calcification volume: {d['calcification_volume_mm3']:.1f} mm³ ({d['calcification_volume_ml']:.3f} mL)")
        lines.append(f"  - Calcification/valve ratio: {d['calcification_to_valve_ratio']*100:.2f}%")
        if ncc is not None:
            lines.append(f"  - Connected components (calcification): {int(ncc)}")
        if largest is not None:
            lines.append(f"  - Largest component volume: {float(largest):.1f} mm³")
    else:
        lines.append("- Aortic valve calcification: Absent")

    lines.append("")
    lines.append("Impression:")
    if qc_flags and qc_flags != ["ok"]:
        lines.append("- Quantitative results are provided; please review QC flags for potential uncertainty.")
    else:
        lines.append("- Quantitative summary as above.")
    return "\n".join(lines)


def _yn(v: bool) -> str:
    return "Yes" if v else "No"


def make_qa_pairs_en(facts: dict):
    """
    Return list of (question, answer, meta_dict) in English.
    Expanded templates (18~30 per case).
    """
    d = compute_derived_if_missing(facts)
    structures = facts.get("structures", {}) or {}
    qc_flags = facts.get("qc_flags", []) or []

    myo_ml = float(d["myocardium_volume_ml"])
    valve_ml = float(d["aortic_valve_volume_ml"])
    calci_present = bool(d["calcification_present"])
    calci_mm3 = float(d["calcification_volume_mm3"]) if calci_present else 0.0
    calci_ml = float(d["calcification_volume_ml"]) if calci_present else 0.0
    ratio_pct = float(d["calcification_to_valve_ratio"] * 100.0) if calci_present else 0.0

    # Optional calcification morphology
    calci_st = structures.get("aortic_valve_calcification", {}) or {}
    ncc = calci_st.get("num_connected_components", None)
    largest = calci_st.get("largest_component_mm3", None)

    qa = []

    # ---- Presence (multiple phrasings)
    pres_qs = [
        "Is aortic valve calcification present?",
        "Is there any calcification in the aortic valve region?",
        "Does this case show aortic valve calcification?",
        "Is calcification detected at the aortic valve?",
    ]
    for q in pres_qs:
        qa.append((q, _yn(calci_present), {"type": "presence", "target": "aortic_valve_calcification"}))

    # ---- Volumes: myocardium (multiple phrasings)
    myo_qs = [
        "What is the myocardium volume (mL)?",
        "Report the myocardium volume in milliliters.",
        "Give the myocardium volume measured from segmentation (mL).",
    ]
    for q in myo_qs:
        qa.append((q, f"{myo_ml:.3f} mL", {"type": "volume", "target": "myocardium", "unit": "mL", "value": myo_ml}))

    # ---- Volumes: aortic valve
    valve_qs = [
        "What is the aortic valve volume (mL)?",
        "Report the aortic valve volume in milliliters.",
        "Give the aortic valve volume measured from segmentation (mL).",
    ]
    for q in valve_qs:
        qa.append((q, f"{valve_ml:.3f} mL", {"type": "volume", "target": "aortic_valve", "unit": "mL", "value": valve_ml}))

    # ---- Calcification volumes (always answer, absent -> 0)
    calci_vol_qs_mm3 = [
        "What is the aortic valve calcification volume (mm³)?",
        "Report calcification volume in cubic millimeters (mm³).",
        "How much aortic valve calcification is there in mm³?",
    ]
    for q in calci_vol_qs_mm3:
        qa.append((q, f"{calci_mm3:.1f} mm³", {"type": "volume", "target": "aortic_valve_calcification", "unit": "mm3", "value": calci_mm3}))

    calci_vol_qs_ml = [
        "What is the aortic valve calcification volume (mL)?",
        "Report calcification volume in milliliters (mL).",
        "How much aortic valve calcification is there in mL?",
    ]
    for q in calci_vol_qs_ml:
        qa.append((q, f"{calci_ml:.3f} mL", {"type": "volume", "target": "aortic_valve_calcification", "unit": "mL", "value": calci_ml}))

    # ---- Ratio
    ratio_qs = [
        "What percentage of the aortic valve volume is calcified?",
        "Report the calcification-to-valve volume ratio (%).",
        "What is the calcification burden as a percentage of valve volume?",
        "Provide calcification/valve volume ratio in percent.",
    ]
    for q in ratio_qs:
        qa.append((q, f"{ratio_pct:.2f}%", {"type": "ratio", "target": "aortic_valve_calcification/aortic_valve", "unit": "%", "value": ratio_pct}))

    # ---- Combined questions (more “agent-like”)
    qa.append((
        "Summarize aortic valve calcification status and volume.",
        f"{'Calcification is present' if calci_present else 'Calcification is absent'}. "
        f"Volume: {calci_mm3:.1f} mm³ ({calci_ml:.3f} mL).",
        {"type": "summary", "target": "aortic_valve_calcification"}
    ))

    qa.append((
        "Provide myocardium volume and aortic valve volume (mL).",
        f"Myocardium: {myo_ml:.3f} mL; Aortic valve: {valve_ml:.3f} mL.",
        {"type": "pair_volume", "targets": ["myocardium", "aortic_valve"], "unit": "mL"}
    ))

    qa.append((
        "List the key quantitative measurements for this case.",
        f"Myocardium volume: {myo_ml:.3f} mL; "
        f"Aortic valve volume: {valve_ml:.3f} mL; "
        f"Aortic valve calcification present: {_yn(calci_present)}; "
        f"Calcification volume: {calci_mm3:.1f} mm³ ({calci_ml:.3f} mL); "
        f"Calcification/valve ratio: {ratio_pct:.2f}%.",
        {"type": "case_summary"}
    ))

    # ---- QC question(s)
    qa.append((
        "Are there any QC warnings for this case?",
        "No (ok)" if (qc_flags == ["ok"] or len(qc_flags) == 0) else "Yes: " + ", ".join(qc_flags),
        {"type": "qc", "flags": qc_flags}
    ))

    # ---- Calcification morphology if available
    if ncc is not None:
        qa.append((
            "How many connected components are detected for aortic valve calcification?",
            f"{int(ncc)}",
            {"type": "morphology", "target": "aortic_valve_calcification", "field": "num_connected_components", "value": int(ncc)}
        ))
    if largest is not None:
        qa.append((
            "What is the volume of the largest calcification component (mm³)?",
            f"{float(largest):.1f} mm³",
            {"type": "morphology", "target": "aortic_valve_calcification", "field": "largest_component_mm3", "value": float(largest)}
        ))

    return qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts_dir", required=True, help="Folder containing facts JSON files (*.json)")
    ap.add_argument("--reports_dir", default="reports", help="Output folder for per-case reports")
    ap.add_argument("--qa_out", default="qa_dataset.jsonl", help="Output QA jsonl file path")
    ap.add_argument("--pos_oversample", type=int, default=1,
                    help="Replicate calcification-related QA for positive cases (>=1). Example: 3 means 3x for positive cases.")
    args = ap.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)

    facts_paths = sorted(glob.glob(os.path.join(args.facts_dir, "*.json")))
    if not facts_paths:
        raise FileNotFoundError(f"No JSON files found in: {args.facts_dir}")

    total_cases = 0
    total_qa = 0
    calci_pos = 0

    with open(args.qa_out, "w", encoding="utf-8") as fqa:
        for fp in facts_paths:
            with open(fp, "r", encoding="utf-8") as ff:
                facts = json.load(ff)

            case_id = facts.get("case_id", os.path.splitext(os.path.basename(fp))[0])
            total_cases += 1

            # Write report
            report_text = format_report_en(facts)
            report_path = os.path.join(args.reports_dir, f"{case_id}.txt")
            with open(report_path, "w", encoding="utf-8") as fr:
                fr.write(report_text)

            # Make QA
            qa_pairs = make_qa_pairs_en(facts)

            # Oversample calcification-related QA for positives (helps imbalance)
            d = compute_derived_if_missing(facts)
            is_pos = bool(d.get("calcification_present", False))
            if is_pos:
                calci_pos += 1

            for (q, a, meta) in qa_pairs:
                repeat = 1
                if is_pos and args.pos_oversample > 1:
                    # only oversample calcification-related questions
                    if ("calcification" in q.lower()) or (meta.get("target", "") == "aortic_valve_calcification"):
                        repeat = args.pos_oversample

                for _ in range(repeat):
                    obj = {
                        "case_id": case_id,
                        "question": q,
                        "answer": a,
                        "meta": meta,
                        "facts_file": fp.replace("\\", "/")
                    }
                    fqa.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    total_qa += 1

    print("Done.")
    print(f"- cases: {total_cases}")
    print(f"- calcification positive: {calci_pos} / {total_cases}")
    print(f"- QA lines: {total_qa}")
    print(f"- reports_dir: {args.reports_dir}")
    print(f"- qa_out: {args.qa_out}")


if __name__ == "__main__":
    main()
