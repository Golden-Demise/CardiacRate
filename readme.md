```
CardiacRate\Scripts\activate.bat
```

```
python app_gradio.py --base_model Qwen/Qwen2.5-3B-Instruct --lora_dir D:\CardiacRate\lora\heart_lora --facts_dir D:\CardiacRate\dataset\facts --trust_remote_code
```

```
python app_gradio.py --base_model mistralai/Mistral-7B-Instruct-v0.3 --lora_dir D:\CardiacRate\lora\heart_lora_mistral_8 --facts_dir D:\CardiacRate\infer\8 --trust_remote_code
```

# Segmentation infer command for temp

```
Segmentation\infer.py

python Segmentation\infer.py --model_name unetcnx_a1 --checkpoint D:\CardiacRate\Segmentation\model\unetcnx_a1\best_model.pth --img_pth D:\CardiacRate\Segmentation\infer\ct\example.nii.gz --infer_dir D:\CardiacRate\Segmentation\infer\predict
```

```
python Segmentation\batch_infer.py --infer_script D:\CardiacRate\Segmentation\infer.py --input_dir D:\CardiacRate\dataset\ct --infer_dir D:\CardiacRate\dataset\predict --model_name unetcnx_a1 --checkpoint D:\CardiacRate\Segmentation\model\unetcnx_a1\best_model.pth
```

```
python Segmentation\eval_seg_metrics.py --pred_dir D:\CardiacRate\dataset\predict --gt_dir D:\CardiacRate\dataset\label --out_csv D:\CardiacRate\dataset\eval_seg_metrics.csv --summary_csv D:\CardiacRate\dataset\eval_seg_summary.csv --calci_min_vox 20
```

# make facts

```
python make_facts.py --mask_path D:\CardiacRate\dataset\label\patient0001_gt.nii.gz --image_path D:\CardiacRate\dataset\ct\patient0001.nii.gz --out_path D:\CardiacRate\patient0001.json
```

```
python batch_make_facts.py --image_dir D:\CardiacRate\dataset\ct --mask_dir D:\CardiacRate\dataset\label --out_dir D:\CardiacRate\dataset\facts3
```

# create QA

```
python build_reports_and_qa.py --facts_dir D:\CardiacRate\dataset\facts3 --out_path D:\CardiacRate\dataset\qa_dataset5_en.json
```

```
python prepare_sft_dataset.py --qa_json D:\CardiacRate\dataset\qa_dataset_en.json --out_train D:\CardiacRate\dataset\sft_train\sft_train7.jsonl --out_val D:\CardiacRate\dataset\sft_val\sft_val7.jsonl --val_ratio 0.1
```

以病人分割

```
python D:\CardiacRate\prepare_sft_dataset_by_patient.py --qa_json D:\CardiacRate\dataset\qa_dataset5_en_augmented_cleaned.json --out_train D:\CardiacRate\dataset\sft_train\sft_train9.jsonl --out_val D:\CardiacRate\dataset\sft_val\sft_val9.jsonl --out_test D:\CardiacRate\dataset\sft_test9.jsonl --split_summary D:\CardiacRate\dataset\sft_split\split_summary9.json --val_ratio 0.1 --test_ratio 0.1 --seed 42
```

# train lora

# Qwen

```
python train_lora_sft.py --model_name Qwen/Qwen2.5-3B-Instruct --train_jsonl D:\CardiacRate\dataset\sft_train.jsonl --val_jsonl D:\CardiacRate\dataset\sft_val.jsonl --out_dir D:\CardiacRate\heart_lora_2 --max_seq_len 1024 --batch_size 2 --grad_accum 8 --lr 2e-4 --epochs 3
```

# mistral

```
python train_lora_sft_mistral.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --train_jsonl D:\CardiacRate\dataset\sft_train\sft_train7.jsonl --val_jsonl D:\CardiacRate\dataset\sft_val\sft_val7.jsonl --out_dir D:\CardiacRate\lora\heart_lora_mistral_7 --max_seq_len 2048 --batch_size 1 --grad_accum 8 --epochs 3
```

# eval_qa.py

```
python eval_qa.py --base_model Qwen/Qwen2.5-3B-Instruct --lora_dir D:\CardiacRate\heart_lora_5 --facts_dir D:\CardiacRate\dataset\facts_test --qa_json D:\CardiacRate\dataset\qa_dataset4_en.json --out_json D:\CardiacRate\dataset\eval_results_5.json --out_csv D:\CardiacRate\dataset\eval_results_5.csv --max_samples 30 --max_new_tokens 128 --temperature 0.0
```

```
python eval_qa.py --base_model mistralai/Mistral-7B-Instruct-v0.3 --lora_dir D:\CardiacRate\heart_lora_mistral_5 --facts_dir D:\CardiacRate\dataset\facts3 --qa_json D:\CardiacRate\dataset\qa_dataset4_en.json --out_json D:\CardiacRate\dataset\eval_results_4.json --out_csv D:\CardiacRate\dataset\eval_results_5.csv --max_samples 30 --max_new_tokens 128 --temperature 0.0
```

```
python eval_qa.py --base_model mistralai/Mistral-7B-Instruct-v0.3 --lora_dir D:\CardiacRate\lora\heart_lora_mistral_7_1 --facts_dir D:\CardiacRate\dataset\facts3 --qa_json D:\CardiacRate\dataset\qa_dataset5_en_augmented.json --out_json D:\CardiacRate\eval\mistral_7_1_results.json --out_csv D:\CardiacRate\eval\mistral_7_1_results.csv --max_samples 0 --max_new_tokens 256 --temperature 0
```

```
python eval_qa.py --base_model mistralai/Mistral-7B-Instruct-v0.3 --lora_dir D:\CardiacRate\lora\heart_lora_mistral_7_2 --facts_dir D:\CardiacRate\dataset\facts_new --qa_json D:\CardiacRate\dataset\qa_dataset5_en_augmented_cleaned.json --split_summary D:\CardiacRate\dataset\sft_split\split_summary7_2.json --max_samples 0
```

python eval_qa.py --base_model mistralai/Mistral-7B-Instruct-v0.3 --lora_dir D:\CardiacRate\lora\heart_lora_mistral_8 --facts_dir D:\CardiacRate\dataset\facts3 --qa_json D:\CardiacRate\dataset\qa_dataset5_en_augmented_cleaned.json --split_summary D:\CardiacRate\dataset\sft_split\split_summary8.json --max_samples 0

# rephrase the question with mistral

# test

```
python augment_canonical_qa_with_mistral.py --qa_json D:\CardiacRate\dataset\qa_dataset5_en.json --out_json D:\CardiacRate\dataset\qa_dataset5_en_augmented_test_v3.json --variants_per_question 2 --max_templates 5 --max_rounds 8
```

# formal run

```
python D:\CardiacRate\augment_canonical_qa_with_mistral.py --qa_json D:\CardiacRate\dataset\qa_dataset5_en.json --out_json D:\CardiacRate\dataset\qa_dataset5_en_augmented.json --variants_per_question 5 --max_rounds 8
```

# out qa

```
python build_question_capability_catalog.py --input D:\CardiacRate\dataset\qa_dataset5_en_augmented_cleaned.json --output_json D:\CardiacRate\dataset\question_capability_catalog.json --output_md D:\CardiacRate\dataset\question_capability_catalog.md
```
