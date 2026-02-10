import os
import time
import importlib
from pathlib import Path

import torch

import numpy as np

from monai.data import decollate_batch
from monai.transforms import (
    EnsureChannelFirst,
    SqueezeDimd,
    KeepLargestConnectedComponent,
)


import matplotlib.pyplot as plt

from monai.data import NibabelWriter

def save_img(img, img_meta_dict, pth):
    writer = NibabelWriter()
    writer.set_data_array(EnsureChannelFirst(channel_dim="no_channel")(img))
    writer.set_metadata(img_meta_dict)
    writer.write(pth, verbose=True)

def infer(model, data, model_inferer, device):
    model.eval()
    with torch.no_grad():
        output = model_inferer(data['image'].to(device))
        output = torch.argmax(output, dim=1)
    return output


def get_filename(data):
    if "pred_meta_dict" in data and "filename_or_obj" in data["pred_meta_dict"]:
        fn = data["pred_meta_dict"]["filename_or_obj"]

    elif "image" in data and hasattr(data["image"], "meta") and "filename_or_obj" in data["image"].meta:
        fn = data["image"].meta["filename_or_obj"]

    else:
        raise KeyError(f"Cannot find filename. keys={list(data.keys())}")

    if isinstance(fn, (list, tuple)):
        fn = fn[0]

    return Path(str(fn)).name

def make_predict_name(filename: str, suffix="_predict"):
    p = Path(filename)

    if p.name.endswith(".nii.gz"):
        stem = p.name[:-len(".nii.gz")]
        return f"{stem}{suffix}.nii.gz"

    return f"{p.stem}{suffix}{p.suffix}"


def get_label_transform(data_name, keys=['label']):
    transform = importlib.import_module(f'transforms.{data_name}_transform')
    get_lbl_transform = getattr(transform, 'get_label_transform', None)
    return get_lbl_transform(keys)


def run_infering(
        model,
        data,
        model_inferer,
        post_transform,
        args
    ):
    ret_dict = {}
    data['pred'] = infer(model, data, model_inferer, args.device)
    
    # post process transform
    if args.infer_post_process:
        print('use post process infer')
        applied_labels = np.unique(data['pred'].flatten())[1:]
        data['pred'] = KeepLargestConnectedComponent(applied_labels=applied_labels)(data['pred'])
    
    # eval infer tta
    if 'label' in data.keys():
        # post label transform 
        sqz_transform = SqueezeDimd(keys=['label'])
        data = sqz_transform(data)
    
    # post transform
    data = post_transform(data)
    
    # eval infer origin
    if 'label' in data.keys():
        # get orginal label
        lbl_dict = {'label': data['label_meta_dict']['filename_or_obj']}
        label_loader = get_label_transform("chgh", keys=['label'])
        lbl_data = label_loader(lbl_dict)
        
        data['label'] = lbl_data['label']
        data['label_meta_dict'] = lbl_data['label']
    
    # save pred result
    print(data.keys())
    filename = get_filename(data)
    out_name = make_predict_name(filename)
    infer_img_pth = os.path.join(args.infer_dir, out_name)

    save_img(
      data['pred'], 
      data['pred_meta_dict'], 
      infer_img_pth
    )
        
    return ret_dict
