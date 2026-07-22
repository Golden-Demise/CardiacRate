# 基於結構化影像事實之心臟 CT 健康諮詢輔助系統
# Evidence-Grounded LLM-Based Cardiac CT Health Consultation Assistant
<br>
<div align="center">
    <img src="media/system preview.png" width="100%">
</div>
<br><br>

## Abstract

Large language models (LLMs) can explain complex information in natural language, but may generate unsupported responses without evidence constraints. Cardiac computed tomography (CT) provides high-resolution three-dimensional images, but its results often include specialized terminology, segmentation masks, and quantitative measurements that are difficult for general users to understand. This study proposes a retrieval-augmented cardiac CT health consultation system that uses structured facts from image analysis as evidence for LLM responses. The system segments the myocardium, aortic valve, and aortic valve calcification, then computes structural volumes, spatial ranges, and calcification statistics from the masks and CT intensity values. During consultation, relevant facts are retrieved based on the user's question, and a small curated knowledge base is additionally searched via TF-IDF retrieval to explain medical terminology without overriding the case facts. When facts are insufficient, the system states that the requested information cannot be determined, reducing unsupported speculation and improving traceability. Experiments used 100 cardiac CT volumes from Cheng Hsin General Hospital, achieving Dice scores of 0.9303 and 0.7078 for myocardium and aortic valve segmentation; question-answering evaluation showed that structured facts with LoRA fine-tuning improved evidence use and refusal behavior.


## Applications

Based on this pipeline, the system focuses on the following applications:

* **Structured Fact Extraction**: converting raw segmentation masks into a traceable, auditable `facts.json` per patient, with QC flags to catch shape mismatches, unreasonable volumes, or abnormal scores before any downstream conclusion is drawn.
* **Aortic Valve Calcification & Stenosis Risk Assessment**: reporting calcification presence/volume and a rule-based aortic stenosis *risk* level (not a diagnosis), always paired with the evidence and limitations behind it.
* **Evidence-Grounded QA / Consultation**: answering technical, patient-friendly, and safety-constrained questions about a specific CT case while refusing questions (surgery, medication, symptom causation, definitive diagnosis) that the available evidence cannot support.
* **Bilingual Gradio Demo**: an interactive demo that uploads a CT, runs segmentation, generates facts/summary, and answers chat questions with Chinese quick-question shortcuts.

### Pipeline
<br>
<div align="center">
    <img src="media/system architecture.png" width="100%">
</div>
<br><br>

## Setup

### Create environment

Project uses Python 3.10 on Windows with an NVIDIA GPU (developed on RTX 4000 Ada, 20 GB VRAM).

```bash
# 1. Install Python 3.10.7, then create the venv inside the project root
cd D:\CardiacRate
py -3.10 -m venv CardiacRate
CardiacRate\Scripts\activate.bat
```

```bash
# 2. Check the max CUDA version your driver supports
nvidia-smi
```

```bash
# 3. Install torch first, matching your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4+
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU only
```

```bash
# 4. Install remaining dependencies
pip install -r requirements.txt
```

```bash
# 5. Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

See `SETUP_環境重建指南.md` for full notes and caveats.

## Dataset

The primary dataset is 100 cardiac CT volumes (`patient0001.nii.gz` – `patient0100.nii.gz`) with segmentation labels:

```text
0 = background
1 = myocardium
2 = aortic valve
3 = aortic valve calcification
```

## Usage

### 1. Segmentation inference

```bash
python Segmentation\infer.py --model_name unetcnx_a1 --checkpoint D:\CardiacRate\Segmentation\model\unetcnx_a1\best_model.pth --img_pth D:\CardiacRate\Segmentation\infer\ct\example.nii.gz --infer_dir D:\CardiacRate\Segmentation\infer\predict
```

Batch inference over a whole CT directory:

```bash
python Segmentation\batch_infer.py --infer_script D:\CardiacRate\Segmentation\infer.py --input_dir D:\CardiacRate\dataset\ct --infer_dir D:\CardiacRate\dataset\predict --model_name unetcnx_a1 --checkpoint D:\CardiacRate\Segmentation\model\unetcnx_a1\best_model.pth
```

Evaluate segmentation quality (Dice / HD95 / etc.):

```bash
python Segmentation\eval_seg_metrics.py --pred_dir D:\CardiacRate\dataset\predict --gt_dir D:\CardiacRate\dataset\label --out_csv D:\CardiacRate\dataset\eval_seg_metrics.csv --summary_csv D:\CardiacRate\dataset\eval_seg_summary.csv --calci_min_vox 20
```

### 2. Structured fact generation

Single case:

```bash
python make_facts.py --mask_path D:\CardiacRate\dataset\label\patient0001_gt.nii.gz --image_path D:\CardiacRate\dataset\ct\patient0001.nii.gz --out_path D:\CardiacRate\patient0001.json
```

Batch over the dataset:

```bash
python batch_make_facts.py --image_dir D:\CardiacRate\dataset\ct --mask_dir D:\CardiacRate\dataset\label --out_dir D:\CardiacRate\dataset\facts3
```

### 3. Canonical QA generation

```bash
python build_reports_and_qa.py --facts_dir D:\CardiacRate\dataset\facts3 --out_path D:\CardiacRate\dataset\qa_dataset5_en.json
```

### 4. LLM-based QA augmentation (rephrasing with Mistral)

```bash
python D:\CardiacRate\augment_canonical_qa_with_mistral.py --qa_json D:\CardiacRate\dataset\qa_dataset5_en.json --out_json D:\CardiacRate\dataset\qa_dataset5_en_augmented.json --variants_per_question 5 --max_rounds 8
```

### 5. SFT dataset preparation

Split by QA pair:

```bash
python prepare_sft_dataset.py --qa_json D:\CardiacRate\dataset\qa_dataset_en.json --out_train D:\CardiacRate\dataset\sft_train\sft_train7.jsonl --out_val D:\CardiacRate\dataset\sft_val\sft_val7.jsonl --val_ratio 0.1
```

Split by patient (recommended, avoids leakage between train/val/test):

```bash
python D:\CardiacRate\prepare_sft_dataset_by_patient.py --qa_json D:\CardiacRate\dataset\qa_dataset5_en_augmented_cleaned.json --out_train D:\CardiacRate\dataset\sft_train\sft_train9.jsonl --out_val D:\CardiacRate\dataset\sft_val\sft_val9.jsonl --out_test D:\CardiacRate\dataset\sft_test9.jsonl --split_summary D:\CardiacRate\dataset\sft_split\split_summary9.json --val_ratio 0.1 --test_ratio 0.1 --seed 42
```

### 6. LoRA fine-tuning

Qwen2.5-3B-Instruct:

```bash
python train_lora_sft.py --model_name Qwen/Qwen2.5-3B-Instruct --train_jsonl D:\CardiacRate\dataset\sft_train.jsonl --val_jsonl D:\CardiacRate\dataset\sft_val.jsonl --out_dir D:\CardiacRate\heart_lora_2 --max_seq_len 1024 --batch_size 2 --grad_accum 8 --lr 2e-4 --epochs 3
```

Mistral-7B-Instruct-v0.3:

```bash
python train_lora_sft_mistral.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --train_jsonl D:\CardiacRate\dataset\sft_train\sft_train7.jsonl --val_jsonl D:\CardiacRate\dataset\sft_val\sft_val7.jsonl --out_dir D:\CardiacRate\lora\heart_lora_mistral_7 --max_seq_len 2048 --batch_size 1 --grad_accum 8 --epochs 3
```

### 7. Evaluation

```bash
python eval_qa.py --base_model mistralai/Mistral-7B-Instruct-v0.3 --lora_dir D:\CardiacRate\lora\heart_lora_mistral_8 --facts_dir D:\CardiacRate\dataset\facts3 --qa_json D:\CardiacRate\dataset\qa_dataset5_en_augmented_cleaned.json --split_summary D:\CardiacRate\dataset\sft_split\split_summary8.json --max_samples 0
```

### 8. Gradio demo

```bash
python app_gradio.py --base_model Qwen/Qwen2.5-3B-Instruct --lora_dir D:\CardiacRate\lora\heart_lora --facts_dir D:\CardiacRate\dataset\facts --trust_remote_code
```

4-bit quantized (lower VRAM):

```bash
python app_gradio.py --base_model mistralai/Mistral-7B-Instruct-v0.3 --lora_dir D:\CardiacRate\lora\heart_lora_mistral_9 --facts_dir D:\CardiacRate\infer\9 --trust_remote_code --load_in_4bit
```


## Acknowledgement

* Segmentation backbone: `unetcnx_a1`
* Base LLMs used for QA generation, augmentation, and fine-tuning: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

## Licenses

### Main License
This project is licensed under the MIT License - see [LICENSE_MIT.md](LICENSE_MIT.md) for details.

### Additional Licenses
Some components may be distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see [LICENSE.CC_BY_NC_SA_4.0.md](LICENSE.CC_BY_NC_SA_4.0.md) for details.
