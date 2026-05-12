import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import math, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="e.g., Qwen2.5-3B-Instruct")
    ap.add_argument("--train_jsonl", default="train.jsonl")
    ap.add_argument("--val_jsonl", default="val.jsonl")
    ap.add_argument("--out_dir", default="lora_out")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=r"D:\CardiacRate\hf_cache")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in fp16/bf16 (Ada 通常 bf16 OK；若你環境不穩就改 fp16)
    cuda_ok = torch.cuda.is_available()
    print("is cuda available：", cuda_ok)
    bf16_ok = cuda_ok and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

    # Ada 通常 bf16_ok=True；如果你環境抓不到GPU，cuda_ok=False，就會自動走 fp32
    if bf16_ok:
        dtype = torch.bfloat16
    elif cuda_ok:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=r"D:\CardiacRate\hf_cache"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # LoRA：通用 target modules（Qwen/Llama 都大致適用）
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    ds_train = load_dataset("json", data_files=args.train_jsonl, split="train")
    ds_val = load_dataset("json", data_files=args.val_jsonl, split="train")

    # ---- warmup_steps（取 3% steps，至少 10）
    steps_per_epoch = math.ceil(len(ds_train) / max(1, (args.batch_size * args.grad_accum)))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(10, int(0.03 * total_steps))

    sft_args = SFTConfig(
        output_dir=args.out_dir,              # 也建議放 D 槽：D:\heart_lora
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,            # ✅ 取代 warmup_ratio
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        use_cpu=(not cuda_ok),
        report_to="none",
        max_length=args.max_seq_len,          # 取代 max_seq_length / max_seq_len 舊寫法
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=lora_config,
        processing_class=tokenizer,           # ✅ 取代 tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved LoRA adapter to: {args.out_dir}")

if __name__ == "__main__":
    main()
