import argparse
import json
from pathlib import Path

import torch
from transformers import pipeline


SYSTEM_PROMPT = """
You are a dataset creator for an evidence-grounded cardiac CT
health consultation assistant.

Generate diverse question-and-answer pairs from the provided structured
facts and canonical answers.

Rules:
1. Statements about the patient must be supported by the structured facts.
2. Preserve all numerical values and units exactly.
3. Do not invent symptoms, medical history, diagnoses, echocardiography
   findings, treatment recommendations, or laboratory results.
4. A CT calcium-based aortic stenosis risk estimate is not a confirmed
   diagnosis of aortic stenosis.
5. Questions about diagnosis must explain what the CT facts suggest and
   what additional information is required.
6. Questions about medication, surgery, symptom causes, or unsupported
   diseases must be answered cautiously.
7. Write patient-friendly items in natural Traditional Chinese.
8. Do not expose raw JSON in the answers.
9. Return valid JSON only.

Generate:
- 2 technical questions
- 3 patient-friendly questions
- 2 clinical-confirmation questions
- 2 safety questions

Return this structure:
{
  "case_id": "...",
  "qa_pairs": [
    {
      "category": "technical | patient_friendly |
                   clinical_confirmation | safety",
      "language": "en | zh-TW",
      "question": "...",
      "answer": "...",
      "answerability": "fully_answerable |
                        partially_answerable |
                        requires_clinical_confirmation |
                        not_answerable",
      "facts_paths": ["..."]
    }
  ]
}
"""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_json(text):
    """Handle occasional markdown fences or surrounding text."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")

    return json.loads(text[start:end + 1])


def generate_case_qa(model, facts, canonical_qa):
    payload = {
        "case_id": facts.get("patient_id", "unknown"),
        "structured_facts": facts,
        "canonical_qa": canonical_qa,
    }

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]

    output = model(
        messages,
        max_new_tokens=2500,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
    )

    generated = output[0]["generated_text"]

    # Some pipeline versions return the complete message list.
    if isinstance(generated, list):
        generated_text = generated[-1]["content"]
    else:
        generated_text = generated

    return extract_json(generated_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts_path", required=True)
    parser.add_argument("--canonical_qa_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument(
        "--model_id",
        default="Qwen/Qwen2.5-3B-Instruct",
    )
    args = parser.parse_args()

    facts = load_json(args.facts_path)
    all_canonical_qa = load_json(args.canonical_qa_path)

    patient_id = facts.get("patient_id")

    canonical_qa = [
        item
        for item in all_canonical_qa
        if item.get("patient_id") == patient_id
    ]

    if not canonical_qa:
        raise ValueError(
            f"No canonical QA found for patient_id={patient_id}"
        )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    generator = pipeline(
        "text-generation",
        model=args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )

    result = generate_case_qa(
        model=generator,
        facts=facts,
        canonical_qa=canonical_qa,
    )

    save_json(result, args.out_path)
    print(f"[OK] Generated QA saved to: {args.out_path}")


if __name__ == "__main__":
    main()