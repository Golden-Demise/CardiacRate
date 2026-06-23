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

def get_as_risk_assessment(facts):
    """
    Read aortic stenosis risk result from facts.json if available.
    Compatible with your new diagnostic_findings structure.
    """
    finding = facts.get("diagnostic_findings", {}).get("aortic_stenosis_risk", {})
    return finding.get("risk_assessment", {})


def get_agatston_like_metrics(facts):
    """
    Read Agatston-like score from facts.json if available.
    """
    return facts.get("derived_metrics", {}).get(
        "aortic_valve_calcification_agatston_like", {}
    )


def format_risk_text(risk_level):
    mapping = {
        "low": "low",
        "increased": "increased",
        "high": "high",
        "indeterminate": "uncertain",
        "unknown": "unknown",
        None: "unknown",
    }
    return mapping.get(risk_level, str(risk_level))


def format_likelihood_text(likelihood):
    mapping = {
        "unlikely": "unlikely",
        "unlikely_for_both_sexes": "unlikely for both sexes",
        "indeterminate": "uncertain",
        "likely": "likely",
        "highly_likely": "highly likely",
        "sex_required_for_guideline_based_classification": "requires sex information for guideline-based classification",
        "unknown": "unknown",
        None: "unknown",
    }
    return mapping.get(likelihood, str(likelihood))


def make_patient_friendly_simple_explanation_qa(facts):
    patient_id = facts.get("patient_id", "unknown")

    calc = get_structure(facts, "aortic_valve_calcification")
    metrics = facts.get("derived_metrics", {})
    agatston = get_agatston_like_metrics(facts)
    as_risk = get_as_risk_assessment(facts)

    calc_present = calc.get("present", False)
    calc_volume_mm3 = calc.get("volume_mm3", None)
    severity = metrics.get("calcification_severity_rule_based", "unknown")

    score_3mm = agatston.get("agatston_like_score_3mm_normalized", None)
    risk_level = as_risk.get("risk_level", "unknown")
    likelihood = as_risk.get("severe_aortic_stenosis_likelihood", "unknown")

    if calc_present:
        answer = (
            "In simple terms, this CT analysis found calcification near the aortic valve. "
            "Calcification means that a calcium-like high-density deposit was detected around the valve. "
            f"The segmented calcification volume is approximately {fmt_number(calc_volume_mm3)} mm³, "
            f"and the rule-based calcification burden is estimated as {severity}. "
        )

        if score_3mm is not None:
            answer += (
                f"The 3-mm-normalized Agatston-like calcium score is approximately {fmt_number(score_3mm)}. "
            )

        if risk_level != "unknown":
            answer += (
                f"Based on the available CT-derived calcium information, the estimated aortic stenosis risk level is "
                f"{format_risk_text(risk_level)}, and the likelihood of severe aortic stenosis is "
                f"{format_likelihood_text(likelihood)}. "
            )

        answer += (
            "However, this system cannot diagnose aortic stenosis from CT facts alone. "
            "A doctor would still need echocardiography information, such as valve blood-flow velocity, "
            "pressure gradient, and valve opening area, to confirm the diagnosis."
        )

    else:
        answer = (
            "In simple terms, this CT analysis did not detect aortic valve calcification in the current segmentation result. "
            "This means the system did not find a calcium-like high-density deposit around the aortic valve. "
            "However, this does not replace a doctor's full interpretation of the CT scan or other clinical tests."
        )

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "patient_friendly_explanation",
        "question": "Can you explain this CT result in simple terms?",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": "structures.aortic_valve_calcification + derived_metrics + diagnostic_findings.aortic_stenosis_risk",
            "value": {
                "calcification_present": calc_present,
                "calcification_volume_mm3": calc_volume_mm3,
                "calcification_severity_rule_based": severity,
                "agatston_like": agatston,
                "aortic_stenosis_risk": as_risk,
            },
        },
    }


def make_patient_friendly_seriousness_qa(facts):
    patient_id = facts.get("patient_id", "unknown")

    calc = get_structure(facts, "aortic_valve_calcification")
    as_risk = get_as_risk_assessment(facts)

    calc_present = calc.get("present", False)
    risk_level = as_risk.get("risk_level", "unknown")
    likelihood = as_risk.get("severe_aortic_stenosis_likelihood", "unknown")

    if not calc_present:
        answer = (
            "Based on the current segmentation-derived facts, no aortic valve calcification was detected. "
            "Therefore, this specific CT-derived finding does not suggest a high calcification-related risk. "
            "However, this system cannot rule out all heart diseases from this information alone."
        )

    elif risk_level == "low":
        answer = (
            "Based on the current CT-derived facts, this result does not appear to suggest a high risk of severe "
            "aortic stenosis. Aortic valve calcification is present, but the estimated risk level is low. "
            "This does not mean the heart is completely normal; it only means that the current calcium-based analysis "
            "does not strongly support severe aortic valve narrowing."
        )

    elif risk_level in ["increased", "high"]:
        answer = (
            "Based on the current CT-derived calcium measurement, the result may suggest an increased risk of severe "
            "aortic valve narrowing. This should not be treated as a final diagnosis, but it would be reasonable to "
            "discuss the result with a doctor and consider echocardiography for confirmation."
        )

    elif risk_level == "indeterminate":
        answer = (
            "The result is not clearly low or high based on the current CT-derived calcium measurement. "
            "Further clinical evaluation, especially echocardiography, would be needed to better assess the aortic valve."
        )

    else:
        answer = (
            "The seriousness of this result cannot be reliably determined from the current facts alone. "
            "The system can describe the detected structures and calcification, but it cannot replace a doctor's diagnosis."
        )

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "patient_friendly_seriousness",
        "question": "Is this result serious?",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": "structures.aortic_valve_calcification.present + diagnostic_findings.aortic_stenosis_risk.risk_assessment",
            "value": {
                "calcification_present": calc_present,
                "risk_level": risk_level,
                "severe_aortic_stenosis_likelihood": likelihood,
            },
        },
    }


def make_patient_friendly_ask_doctor_qa(facts):
    patient_id = facts.get("patient_id", "unknown")

    calc = get_structure(facts, "aortic_valve_calcification")
    as_risk = get_as_risk_assessment(facts)

    calc_present = calc.get("present", False)
    risk_level = as_risk.get("risk_level", "unknown")

    if calc_present:
        answer = (
            "You may ask your doctor these questions: "
            "1) Does the aortic valve calcification need follow-up? "
            "2) Do I need an echocardiography exam to check the valve opening and blood-flow speed? "
            "3) Is this calcification related to symptoms such as chest discomfort, shortness of breath, dizziness, or fainting? "
            "4) How often should this finding be monitored? "
            f"In this case, the CT-derived aortic stenosis risk level is {format_risk_text(risk_level)}, "
            "but this is not a final diagnosis."
        )
    else:
        answer = (
            "You may ask your doctor whether the CT scan shows any other heart-related findings outside this system's segmentation targets. "
            "This system did not detect aortic valve calcification, but it cannot evaluate all possible heart conditions."
        )

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "patient_friendly_next_steps",
        "question": "What should I ask my doctor about this result?",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": "structures.aortic_valve_calcification + diagnostic_findings.aortic_stenosis_risk",
            "value": {
                "calcification_present": calc_present,
                "aortic_stenosis_risk": as_risk,
            },
        },
    }


def get_agatston_like_metrics(facts):
    """
    Read Agatston-like score from the new facts format.
    Expected path:
    derived_metrics.aortic_valve_calcification_agatston_like
    """
    return facts.get("derived_metrics", {}).get(
        "aortic_valve_calcification_agatston_like", {}
    )


def get_aortic_stenosis_risk_finding(facts):
    """
    Read AS risk information from the new facts format.
    Expected path:
    diagnostic_findings.aortic_stenosis_risk
    """
    return facts.get("diagnostic_findings", {}).get("aortic_stenosis_risk", {})


def format_as_likelihood(likelihood):
    mapping = {
        "unknown": "unknown",
        "unlikely": "unlikely",
        "unlikely_for_both_sexes": "unlikely for both sexes",
        "indeterminate": "indeterminate",
        "likely": "likely",
        "highly_likely": "highly likely",
        "sex_required_for_guideline_based_classification": "requires sex information for guideline-based classification",
    }
    return mapping.get(likelihood, likelihood if likelihood else "unknown")


def make_agatston_score_qa(facts):
    patient_id = facts.get("patient_id", "unknown")
    agatston = get_agatston_like_metrics(facts)

    available = agatston.get("available", False)
    raw_score = agatston.get("agatston_like_score_raw", None)
    norm_score = agatston.get("agatston_like_score_3mm_normalized", None)
    area_mm2 = agatston.get("calcification_area_mm2", None)
    volume_mm3 = agatston.get("calcification_volume_mm3_from_hu_mask", None)
    max_hu = agatston.get("max_hu", None)
    reason = agatston.get("reason", None)

    if available:
        answer = (
            "The Agatston-like aortic valve calcium score is approximately "
            f"{fmt_number(raw_score)} before slice-thickness normalization and "
            f"{fmt_number(norm_score)} after simple 3-mm normalization. "
            f"The HU-filtered calcification area is approximately {fmt_number(area_mm2)} mm², "
            f"the HU-filtered calcification volume is approximately {fmt_number(volume_mm3)} mm³, "
            f"and the maximum HU is {fmt_number(max_hu)}. "
            "This is an Agatston-like score derived from segmentation masks, not a clinically validated CT-AVC score."
        )
        answerable = True
    else:
        answer = (
            "The Agatston-like aortic valve calcium score cannot be computed from the current facts. "
            f"Reason: {reason if reason else 'HU-based CT image information is unavailable.'}"
        )
        answerable = False

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "aortic_valve_agatston_like_score",
        "question": "What is the Agatston-like score of the aortic valve calcification?",
        "answer": answer,
        "answerable": answerable,
        "evidence": {
            "source": "derived_metrics.aortic_valve_calcification_agatston_like",
            "value": agatston,
        },
    }


def make_aortic_stenosis_risk_qa(facts):
    patient_id = facts.get("patient_id", "unknown")
    finding = get_aortic_stenosis_risk_finding(facts)
    agatston = finding.get("agatston_like", get_agatston_like_metrics(facts))
    risk = finding.get("risk_assessment", {})

    available = agatston.get("available", False)
    raw_score = agatston.get("agatston_like_score_raw", None)
    norm_score = agatston.get("agatston_like_score_3mm_normalized", None)
    risk_level = risk.get("risk_level", "unknown")
    likelihood = risk.get("severe_aortic_stenosis_likelihood", "unknown")
    sex_used = risk.get("sex_used_for_thresholds", None)
    important_note = risk.get(
        "important_note",
        "This does not diagnose aortic stenosis. Echocardiographic peak velocity, mean pressure gradient, and aortic valve area are required for definitive severity assessment.",
    )

    if available:
        answer = (
            "Based on the segmentation-derived Agatston-like aortic valve calcium score, "
            f"the estimated aortic stenosis risk level is {risk_level}. "
            f"The likelihood of severe aortic stenosis is {format_as_likelihood(likelihood)}. "
            f"The raw Agatston-like score is approximately {fmt_number(raw_score)}, "
            f"and the 3-mm-normalized score is approximately {fmt_number(norm_score)}. "
        )

        if sex_used:
            answer += f"Sex-specific thresholds were applied using sex = {sex_used}. "
        else:
            answer += "No sex-specific threshold was applied because sex was not provided. "

        answer += important_note
        answerable = True
    else:
        reason = agatston.get("reason", "Agatston-like score is unavailable.")
        answer = (
            "Aortic stenosis risk cannot be reliably estimated from the current facts. "
            f"Reason: {reason}"
        )
        answerable = False

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "aortic_stenosis_risk",
        "question": "What is the estimated aortic stenosis risk based on this CT case?",
        "answer": answer,
        "answerable": answerable,
        "evidence": {
            "source": "diagnostic_findings.aortic_stenosis_risk",
            "value": finding,
        },
    }


def make_aortic_stenosis_diagnosis_safety_qa(facts):
    patient_id = facts.get("patient_id", "unknown")
    finding = get_aortic_stenosis_risk_finding(facts)
    agatston = finding.get("agatston_like", get_agatston_like_metrics(facts))
    risk = finding.get("risk_assessment", {})

    available = agatston.get("available", False)
    risk_level = risk.get("risk_level", "unknown")
    likelihood = risk.get("severe_aortic_stenosis_likelihood", "unknown")
    norm_score = agatston.get("agatston_like_score_3mm_normalized", None)

    if available:
        answer = (
            "This system cannot definitively diagnose aortic stenosis from the current CT facts alone. "
            f"It can only provide a risk-oriented interpretation based on aortic valve calcification. "
            f"In this case, the 3-mm-normalized Agatston-like score is approximately {fmt_number(norm_score)}, "
            f"with an estimated risk level of {risk_level} and severe AS likelihood of "
            f"{format_as_likelihood(likelihood)}. Definitive assessment still requires echocardiographic "
            "parameters such as peak velocity, mean pressure gradient, and aortic valve area."
        )
        answerable = True
    else:
        answer = (
            "This system cannot determine whether the patient has aortic stenosis from the current facts. "
            "The current facts do not provide an available Agatston-like aortic valve calcium score, and they also lack echocardiographic peak velocity, mean pressure gradient, and aortic valve area."
        )
        answerable = False

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "aortic_stenosis_diagnosis_safety",
        "question": "Does this patient have aortic stenosis?",
        "answer": answer,
        "answerable": answerable,
        "evidence": {
            "source": "diagnostic_findings.aortic_stenosis_risk + limitations",
            "value": {
                "aortic_stenosis_risk": finding,
                "limitations": facts.get("limitations", []),
            },
        },
    }

def get_as_risk_assessment(facts):
    finding = facts.get("diagnostic_findings", {}).get("aortic_stenosis_risk", {})
    return finding.get("risk_assessment", {})


def get_agatston_like_metrics(facts):
    return facts.get("derived_metrics", {}).get(
        "aortic_valve_calcification_agatston_like", {}
    )


def risk_to_plain_text(risk_level):
    mapping = {
        "low": "low",
        "increased": "increased",
        "high": "high",
        "indeterminate": "uncertain",
        "unknown": "unknown",
        None: "unknown",
    }
    return mapping.get(risk_level, str(risk_level))


def likelihood_to_plain_text(likelihood):
    mapping = {
        "unlikely": "unlikely",
        "unlikely_for_both_sexes": "unlikely for both sexes",
        "indeterminate": "uncertain",
        "likely": "likely",
        "highly_likely": "highly likely",
        "sex_required_for_guideline_based_classification": "requires sex information for guideline-based classification",
        "unknown": "unknown",
        None: "unknown",
    }
    return mapping.get(likelihood, str(likelihood))


def make_patient_friendly_qa_list(facts):
    patient_id = facts.get("patient_id", "unknown")

    calc = get_structure(facts, "aortic_valve_calcification")
    metrics = facts.get("derived_metrics", {})
    agatston = get_agatston_like_metrics(facts)
    as_risk = get_as_risk_assessment(facts)

    calc_present = calc.get("present", False)
    calc_volume_mm3 = calc.get("volume_mm3", None)
    calc_volume_ml = calc.get("volume_ml", None)

    severity = metrics.get("calcification_severity_rule_based", "unknown")
    score_3mm = agatston.get("agatston_like_score_3mm_normalized", None)

    risk_level = as_risk.get("risk_level", "unknown")
    severe_as_likelihood = as_risk.get("severe_aortic_stenosis_likelihood", "unknown")

    risk_text = risk_to_plain_text(risk_level)
    likelihood_text = likelihood_to_plain_text(severe_as_likelihood)

    base_evidence = {
        "calcification_present": calc_present,
        "calcification_volume_mm3": calc_volume_mm3,
        "calcification_volume_ml": calc_volume_ml,
        "calcification_severity_rule_based": severity,
        "agatston_like_score_3mm_normalized": score_3mm,
        "aortic_stenosis_risk": as_risk,
    }

    qa_list = []

    def add_qa(category, question, answer, answerable=True):
        qa_list.append({
            "language": "en",
            "patient_id": patient_id,
            "category": category,
            "question": question,
            "answer": answer,
            "answerable": answerable,
            "evidence": {
                "source": (
                    "structures.aortic_valve_calcification + "
                    "derived_metrics + diagnostic_findings.aortic_stenosis_risk"
                ),
                "value": base_evidence,
            },
        })

    # 1. Simple explanation
    if calc_present:
        simple_answer = (
            "In simple terms, this CT analysis found calcification near the aortic valve. "
            "Calcification means that a calcium-like high-density deposit was detected around the valve. "
            f"The segmented calcification volume is approximately {fmt_number(calc_volume_mm3)} mm³. "
        )

        if score_3mm is not None:
            simple_answer += (
                f"The 3-mm-normalized Agatston-like calcium score is approximately {fmt_number(score_3mm)}. "
            )

        simple_answer += (
            f"Based on the available CT-derived information, the estimated aortic stenosis risk level is {risk_text}, "
            f"and the likelihood of severe aortic stenosis is {likelihood_text}. "
            "However, this system cannot diagnose aortic stenosis from CT facts alone. "
            "Echocardiography is still needed for confirmation."
        )
    else:
        simple_answer = (
            "In simple terms, this CT analysis did not detect aortic valve calcification in the current segmentation result. "
            "This does not replace a doctor's full interpretation of the CT scan, but this specific calcium-related finding was not identified."
        )

    add_qa(
        "patient_friendly_explanation",
        "Can you explain this CT result in simple terms?",
        simple_answer,
        answerable=True,
    )

    # 2. Meaning of calcification
    if calc_present:
        calc_meaning_answer = (
            "Aortic valve calcification means that a calcium-like deposit was detected near the aortic valve, "
            "which is the valve between the heart and the main artery. "
            "A small amount of calcification does not always mean a serious disease, but a larger calcium burden "
            "may be related to a higher risk of aortic valve narrowing."
        )
    else:
        calc_meaning_answer = (
            "Aortic valve calcification means a calcium-like deposit near the aortic valve. "
            "In this case, the system did not detect this finding in the current segmentation result."
        )

    add_qa(
        "patient_friendly_calcification_meaning",
        "What does aortic valve calcification mean?",
        calc_meaning_answer,
        answerable=True,
    )

    # 3. Is this serious?
    if not calc_present:
        serious_answer = (
            "Based on the current segmentation-derived facts, no aortic valve calcification was detected. "
            "This specific CT-derived finding does not suggest a high calcification-related risk. "
            "However, this system cannot rule out all heart diseases from this information alone."
        )
    elif risk_level == "low":
        serious_answer = (
            "Based on the current CT-derived calcium information, this result does not strongly suggest a high risk of severe aortic stenosis. "
            "Aortic valve calcification is present, but the estimated risk level is low. "
            "This does not mean the heart is completely normal; it only means that the current calcium-based analysis "
            "does not support severe aortic valve narrowing."
        )
    elif risk_level in ["increased", "high"]:
        serious_answer = (
            "Based on the current CT-derived calcium information, this result may suggest an increased risk of severe aortic valve narrowing. "
            "This is not a final diagnosis, but it should be discussed with a doctor. "
            "Echocardiography may be needed to confirm the valve condition."
        )
    else:
        serious_answer = (
            "The seriousness of this result cannot be reliably determined from the current facts alone. "
            "The system can describe calcification-related findings, but it cannot replace a doctor's diagnosis."
        )

    add_qa(
        "patient_friendly_seriousness",
        "Is this result serious?",
        serious_answer,
        answerable=True,
    )

    # 4. Should I worry?
    worry_answer = (
        "This result should be understood as a finding to discuss with a healthcare professional, not as a final diagnosis. "
    )

    if calc_present:
        worry_answer += (
            f"The system detected aortic valve calcification, and the estimated aortic stenosis risk level is {risk_text}. "
            "If you have symptoms such as chest discomfort, shortness of breath, dizziness, or fainting, you should discuss them with a doctor."
        )
    else:
        worry_answer += (
            "The system did not detect aortic valve calcification, but it cannot evaluate every possible heart condition."
        )

    add_qa(
        "patient_friendly_worry",
        "Should I worry about this calcification?",
        worry_answer,
        answerable=True,
    )

    # 5. Do I have heart disease?
    heart_disease_answer = (
        "This system cannot determine whether you have heart disease based on the current CT facts alone. "
    )

    if calc_present:
        heart_disease_answer += (
            f"It detected aortic valve calcification and provides a calcium-based aortic stenosis risk estimate, currently classified as {risk_text}. "
            "A final diagnosis requires a doctor's assessment, symptoms, medical history, and often additional tests."
        )
    else:
        heart_disease_answer += (
            "It only reports that no aortic valve calcification was detected in the current segmentation result. "
            "Other heart conditions cannot be ruled out from this limited information."
        )

    add_qa(
        "patient_friendly_diagnosis_safety",
        "Do I have heart disease?",
        heart_disease_answer,
        answerable=False,
    )

    # 6. Do I have aortic stenosis?
    as_answer = (
        "This system cannot definitively diagnose aortic stenosis from the current CT facts alone. "
    )

    if calc_present:
        as_answer += (
            f"It can only provide a risk-oriented interpretation based on aortic valve calcification. "
            f"The estimated risk level is {risk_text}, and severe aortic stenosis is classified as {likelihood_text}. "
            "Echocardiography information, such as valve blood-flow velocity, pressure gradient, and valve opening area, "
            "is required to confirm the diagnosis."
        )
    else:
        as_answer += (
            "No aortic valve calcification was detected, but aortic stenosis still cannot be fully assessed without echocardiography."
        )

    add_qa(
        "patient_friendly_aortic_stenosis_safety",
        "Do I have aortic stenosis?",
        as_answer,
        answerable=False,
    )

    # 7. Is my valve blocked?
    blocked_answer = (
        "The current facts cannot determine whether the valve is blocked. "
        "This system can describe aortic valve calcification and estimate calcium-related aortic stenosis risk, "
        "but valve blockage or narrowing requires echocardiographic assessment of blood flow and valve opening area."
    )

    add_qa(
        "patient_friendly_valve_blockage_safety",
        "Does this mean my valve is blocked?",
        blocked_answer,
        answerable=False,
    )

    # 8. Do I need another test?
    if calc_present:
        another_test_answer = (
            "This system cannot decide whether you personally need another test, but if aortic valve calcification is present, "
            "you can ask your doctor whether echocardiography is needed to evaluate valve opening and blood-flow speed. "
            f"In this case, the CT-derived aortic stenosis risk level is {risk_text}."
        )
    else:
        another_test_answer = (
            "This system did not detect aortic valve calcification, but it cannot decide whether you need another test. "
            "That decision should depend on your symptoms, medical history, and your doctor's interpretation of the full CT scan."
        )

    add_qa(
        "patient_friendly_next_test",
        "Do I need another test?",
        another_test_answer,
        answerable=True,
    )

    # 9. Ask doctor
    if calc_present:
        ask_doctor_answer = (
            "You may ask your doctor these questions: "
            "1) Does the aortic valve calcification need follow-up? "
            "2) Do I need echocardiography to check the valve opening and blood-flow speed? "
            "3) Is this finding related to symptoms such as chest discomfort, shortness of breath, dizziness, or fainting? "
            "4) How often should this finding be monitored? "
            f"The CT-derived aortic stenosis risk level is {risk_text}, but this is not a final diagnosis."
        )
    else:
        ask_doctor_answer = (
            "You may ask your doctor whether the full CT scan shows any other heart-related findings outside this system's segmentation targets. "
            "This system did not detect aortic valve calcification, but it cannot evaluate all possible heart conditions."
        )

    add_qa(
        "patient_friendly_next_steps",
        "What should I ask my doctor about this result?",
        ask_doctor_answer,
        answerable=True,
    )

    # 10. Symptoms
    symptom_answer = (
        "This system cannot determine whether symptoms such as chest pain or shortness of breath are caused by this CT finding. "
        "Symptoms require clinical evaluation by a doctor. "
    )

    if calc_present:
        symptom_answer += (
            "Aortic valve calcification may be relevant to valve disease risk, but symptom interpretation requires additional information, "
            "especially echocardiography and clinical history."
        )
    else:
        symptom_answer += (
            "No aortic valve calcification was detected, but this does not rule out other possible causes of symptoms."
        )

    add_qa(
        "patient_friendly_symptom_safety",
        "Can this result explain chest pain or shortness of breath?",
        symptom_answer,
        answerable=False,
    )

    return qa_list

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
    as_finding = get_aortic_stenosis_risk_finding(facts)
    agatston = as_finding.get("agatston_like", get_agatston_like_metrics(facts))
    as_risk = as_finding.get("risk_assessment", {})

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
            "aortic valve calcification is present, with a segmentation volume of approximately "
            f"{fmt_number(calc.get('volume_mm3'))} mm³ and a rule-based calcification burden of {severity_en}"
        )
    else:
        parts.append("no aortic valve calcification was segmented")

    if agatston.get("available", False):
        parts.append(
            "the Agatston-like aortic valve calcium score is approximately "
            f"{fmt_number(agatston.get('agatston_like_score_raw'))} raw and "
            f"{fmt_number(agatston.get('agatston_like_score_3mm_normalized'))} after 3-mm normalization"
        )
        parts.append(
            "the estimated aortic stenosis risk level is "
            f"{as_risk.get('risk_level', 'unknown')}, with severe AS likelihood "
            f"classified as {format_as_likelihood(as_risk.get('severe_aortic_stenosis_likelihood', 'unknown'))}"
        )
    else:
        parts.append("Agatston-like aortic stenosis risk assessment is unavailable")

    answer = (
        "Based on the segmentation-derived facts, "
        + "; ".join(parts)
        + ". These findings are derived from automatic segmentation. The aortic stenosis risk estimate does not replace echocardiographic evaluation."
    )

    return {
        "language": "en",
        "patient_id": patient_id,
        "category": "summary_report",
        "question": "Generate a structured summary based on the current facts.",
        "answer": answer,
        "answerable": True,
        "evidence": {
            "source": "structures + derived_metrics + diagnostic_findings",
            "value": {
                "myocardium": myocardium,
                "aortic_valve": valve,
                "aortic_valve_calcification": calc,
                "derived_metrics": metrics,
                "aortic_stenosis_risk": as_finding,
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

    # New disease-oriented QA based on Agatston-like AVC score
    qa_list.append(make_agatston_score_qa(facts))
    qa_list.append(make_aortic_stenosis_risk_qa(facts))
    qa_list.append(make_aortic_stenosis_diagnosis_safety_qa(facts))

    # Patient-friendly QA
    qa_list.extend(make_patient_friendly_qa_list(facts))

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