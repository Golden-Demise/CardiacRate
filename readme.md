```
CardiacRate\Scripts\activate.bat
```

```
python app_gradio.py --base_model Qwen/Qwen2.5-3B-Instruct --lora_dir D:\CardiacRate\heart_lora --facts_dir D:\CardiacRate\dataset\facts --trust_remote_code
python app_gradio.py --base_model mistralai/Mistral-7B-Instruct-v0.3 --lora_dir D:\CardiacRate\heart_lora_mistral_1 --facts_dir D:\CardiacRate\dataset\facts --trust_remote_code
```

# Segmentation infer command for temp

```
Segmentation\infer.py

python Segmentation\infer.py --model_name unetcnx_a1 --checkpoint D:\CardiacRate\Segmentation\model\unetcnx_a1\best_model.pth --img_pth D:\CardiacRate\Segmentation\infer\ct\example.nii.gz --infer_dir D:\CardiacRate\Segmentation\infer\predict
```

# create QA

```
python build_reports_and_qa.py --facts_dir D:\CardiacRate\dataset\facts_test --out_path D:\CardiacRate\dataset\qa_dataset_en.json
```

```
python prepare_sft_dataset.py --qa_json D:\CardiacRate\dataset\qa_dataset_en.json --out_train D:\CardiacRate\dataset\sft_train.jsonl --out_val D:\CardiacRate\dataset\sft_val.jsonl --val_ratio
0.1
```

```
Question：
1. Which anatomical structures were segmented in this CT case?
2. What is the volume of the myocardium?
3. What is the volume of the aortic valve?
4. What is the volume of the aortic valve calcification?
5. In which z-slices does the myocardium appear?
6. In which z-slices does the aortic valve appear?
7. In which z-slices does the aortic valve calcification appear?
8. Is aortic valve calcification present in this case?
9. What is the volume of the aortic valve calcification?
10. What is the calcification-to-aortic-valve volume ratio?
11. What is the estimated severity of aortic valve calcification in this case?
12. Can this system determine coronary artery stenosis?
13. Can this system estimate the ejection fraction?
14. Can this system assess cardiac function?
15. Generate a structured summary based on the current facts.
```

# train lora

```
python train_lora_sft.py --model_name Qwen/Qwen2.5-3B-Instruct --train_jsonl D:\CardiacRate\dataset\sft_train.jsonl --val_jsonl D:\CardiacRate\dataset\sft_val.jsonl --out_dir D:\CardiacRate\heart_lora_2 --max_seq_len 1024 --batch_size 2 --grad_accum 8 --lr 2e-4 --epochs 3
```

# mistral

```
python train_lora_sft_mistral.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --train_jsonl D:\CardiacRate\dataset\sft_train.jsonl --val_jsonl D:\CardiacRate\dataset\sft_val.jsonl --out_dir D:\heart_lora_mistral_1 --max_seq_len 2048 --batch_size 1 --grad_accum 8 --epochs 3
```
