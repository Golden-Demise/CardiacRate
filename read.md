```
CardiacRate\Scripts\activate.bat
python train_lora_sft.py --model_name Qwen/Qwen2.5-3B-Instruct --out_dir heart_lora
```

```
python app_gradio.py --base_model Qwen/Qwen2.5-3B-Instruct --lora_dir D:\CardiacRate\heart_lora --facts_dir D:\CardiacRate\dataset\facts --trust_remote_code
```
