import json, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM = (
    "You are a cardiac assistant. Answer in English. "
    "Use ONLY the provided FACTS. If the answer is not available in FACTS, "
    "reply exactly: Not available in provided facts."
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--facts_json", required=True)
    ap.add_argument("--question", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, cache_dir=r"D:\CardiacRate\hf_cache")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=r"D:\CardiacRate\hf_cache"
    )
    model = PeftModel.from_pretrained(model, args.lora_dir)
    model.eval()

    facts = json.load(open(args.facts_json, "r", encoding="utf-8"))
    facts_block = json.dumps(
        {"case_id": facts.get("case_id",""), "qc_flags": facts.get("qc_flags", []), "derived": facts.get("derived", {})},
        ensure_ascii=False, indent=2
    )

    prompt = (
        f"### SYSTEM\n{SYSTEM}\n\n"
        f"### FACTS\n{facts_block}\n\n"
        f"### QUESTION\n{args.question}\n\n"
        f"### ANSWER\n"
    )

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print(text.split("### ANSWER")[-1].strip())

if __name__ == "__main__":
    main()
