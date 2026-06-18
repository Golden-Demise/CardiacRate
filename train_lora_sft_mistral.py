import argparse
import json
import math
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def _to_text(value):
    """Convert dict/list facts into readable JSON text; keep strings unchanged."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def make_formatting_func(tokenizer):
    """Support common SFT jsonl formats:
    1) {"messages": [{"role": ..., "content": ...}, ...]}
    2) {"text": "..."}
    3) {"facts": ..., "question": ..., "answer": ...}
    4) {"instruction": ..., "input": ..., "output"/"response": ...}
    5) {"prompt": ..., "completion": ...}
    """

    def formatting_func(example):
        # Chat format: messages list
        if "messages" in example and example["messages"] is not None:
            return tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

        # Already formatted plain text
        if "text" in example and example["text"] is not None:
            return str(example["text"])

        # Your cardiac CT facts-QA format
        if all(k in example for k in ["facts", "question", "answer"]):
            facts = _to_text(example["facts"])
            question = _to_text(example["question"])
            answer = _to_text(example["answer"])
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an evidence-grounded cardiac CT health consultation assistant."
                        "Your role is to help users understand cardiac CT analysis results in clear and natural language. You do not replace a physician and must not make unsupported diagnoses or treatment decisions."
                        "Evidence rules:"
                        "1. Use the provided case-specific structured facts as the only source for statements about this patient."
                        "2. General medical education may be used only to explain medical terms or the usual meaning of a finding."
                        "3. Do not invent findings, symptoms, medical history, diagnoses, test results, or treatment recommendations."
                        "4. Preserve numerical values and units exactly as provided in the facts."
                        "5. Do not expose raw JSON unless the user explicitly asks for it."
                        "Answering strategy:"
                        "1. Answer in the same language as the user's question."
                        "2. Adapt the explanation to the user's apparent level of medical knowledge."
                        "3. For a simple factual question, answer directly and briefly."
                        "4. For an explanatory or risk-related question, when appropriate:"
                        "* state what was found;"
                        "* explain what it means in plain language;"
                        "* state what cannot be concluded;"
                        "* mention what additional clinical information may be needed."
                        "5. If the question is only partially supported, answer the supported portion and clearly identify the missing information."
                        "6. If the question cannot be answered from the available evidence, explain why rather than giving only a generic refusal."
                        "7. When discussing aortic stenosis, distinguish a CT calcium-based risk estimate from a confirmed diagnosis. A confirmed assessment generally requires clinical evaluation and echocardiographic information such as blood-flow velocity, mean pressure gradient, and aortic valve area."
                        "8. Do not decide whether the user needs medication, surgery, or another treatment."
                        "9. Use calm, patient-friendly wording. Avoid unnecessary technical terminology, but include technical measurements when they are relevant to the question."
                        "10. Do not repeatedly add the same warning when a brief limitation statement is sufficient."
                        "The goal is to be accurate, helpful, understandable, and appropriately cautious while remaining grounded in the provided evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Structured facts:\n{facts}\n\nQuestion:\n{question}",
                },
                {"role": "assistant", "content": answer},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        # Alpaca-like format
        if "instruction" in example:
            instruction = _to_text(example.get("instruction", ""))
            input_text = _to_text(example.get("input", ""))
            answer = _to_text(example.get("output", example.get("response", "")))
            user_content = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        # Prompt-completion format
        if "prompt" in example and "completion" in example:
            messages = [
                {"role": "user", "content": _to_text(example["prompt"])},
                {"role": "assistant", "content": _to_text(example["completion"])},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        raise ValueError(
            "Unsupported jsonl format. Expected one of: messages, text, "
            "facts/question/answer, instruction/input/output, or prompt/completion."
        )

    return formatting_func


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="e.g., mistralai/Mistral-7B-Instruct-v0.3, Qwen/Qwen2.5-3B-Instruct",
    )
    ap.add_argument("--train_jsonl", default="train.jsonl")
    ap.add_argument("--val_jsonl", default="val.jsonl")
    ap.add_argument("--out_dir", default=r"D:\heart_lora_mistral")
    ap.add_argument("--cache_dir", default=r"D:\CardiacRate\hf_cache")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--attn_only",
        action="store_true",
        help="Only apply LoRA to q/k/v/o projection layers. Use this if VRAM is tight.",
    )
    ap.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit QLoRA. Not recommended for 7B on 20GB VRAM unless you know it fits.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    cuda_ok = torch.cuda.is_available()
    print("is cuda available:", cuda_ok)

    bf16_ok = cuda_ok and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    if bf16_ok:
        compute_dtype = torch.bfloat16
    elif cuda_ok:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    use_4bit = cuda_ok and (not args.no_4bit)
    print("use 4-bit QLoRA:", use_4bit)
    print("compute dtype:", compute_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model_kwargs = dict(
        pretrained_model_name_or_path=args.model_name,
        device_map="auto" if cuda_ok else None,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    if use_4bit:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()

    if args.attn_only:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    ds_train = load_dataset("json", data_files=args.train_jsonl, split="train")
    ds_val = load_dataset("json", data_files=args.val_jsonl, split="train")

    steps_per_epoch = math.ceil(len(ds_train) / max(1, args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(10, int(0.03 * total_steps))

    sft_args = SFTConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        use_cpu=(not cuda_ok),
        report_to="none",
        max_length=args.max_seq_len,
        packing=False,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=lora_config,
        processing_class=tokenizer,
        formatting_func=make_formatting_func(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved LoRA adapter to: {args.out_dir}")


if __name__ == "__main__":
    main()
