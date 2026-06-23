# rebuild qa with mistral

python build_qa_test_re.py
|
V
merge_generated_qa.py

python merge_generated_qa.py --qa_dir D:\CardiacRate\dataset\generated_qa\test3 --facts_dir D:\CardiacRate\dataset\facts3 --out_json D:\CardiacRate\dataset\qa_dataset_generated.json
|
V
prepare_sft_dataset_by_patient.py

python prepare_sft_dataset_by_patient.py --qa_json D:\CardiacRate\dataset\qa_dataset_generated.json --out_train D:\CardiacRate\dataset\sft_train6.jsonl --out_val D:\CardiacRate\dataset\sft_val6.jsonl --split_summary D:\CardiacRate\dataset\sft\split_summary.json --val_ratio 0.1 --seed 42
