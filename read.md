```
CardiacRate\Scripts\activate.bat
python train_lora_sft.py --model_name Qwen/Qwen2.5-3B-Instruct --out_dir heart_lora
```

```
python app_gradio.py --base_model Qwen/Qwen2.5-3B-Instruct --lora_dir D:\CardiacRate\heart_lora --facts_dir D:\CardiacRate\dataset\facts --trust_remote_code
```

# Segmentation infer command for temp

```
Segmentation\infer.py

python Segmentation\infer.py --model_name unetcnx_a1 --checkpoint D:\CardiacRate\Segmentation\model\unetcnx_a1\best_model.pth --img_pth D:\CardiacRate\Segmentation\infer\ct\example.nii.gz --infer_dir D:\CardiacRate\Segmentation\infer\predict
```
