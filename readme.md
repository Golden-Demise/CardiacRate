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

Full runnable commands for every stage (segmentation inference, fact generation, QA generation/augmentation, LoRA training, evaluation, and the Gradio demo) are kept in [`exceution_temp.md`](exceution_temp.md).

Quick start — Gradio demo:

```bash
python app_gradio.py --base_model Qwen/Qwen2.5-3B-Instruct --lora_dir D:\CardiacRate\lora\heart_lora --facts_dir D:\CardiacRate\dataset\facts --trust_remote_code
```


## Acknowledgement

* Segmentation backbone: `unetcnx_a1`
* Base LLMs used for QA generation, augmentation, and fine-tuning: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

## Licenses

### Main License
This project is licensed under the MIT License - see [LICENSE_MIT.md](LICENSE_MIT.md) for details.

### Additional Licenses
Some components may be distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see [LICENSE.CC_BY_NC_SA_4.0.md](LICENSE.CC_BY_NC_SA_4.0.md) for details.
