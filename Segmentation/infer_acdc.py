import os
from functools import partial
import torch
from networks.network import network
from inferer import run_infering
import importlib

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    Orientationd,
    ToNumpyd,
)
from monailabel.transform.post import Restored
import argparse

def is_deep_sup(checkpoint):
    for key in list(checkpoint["state_dict"].keys()):
        if 'ds' in key:
            return True
    return False

def get_infer_data(data_dict, args):
    keys = data_dict.keys()
    data_name = "acdc"
    transform = importlib.import_module(f'transforms.{data_name}_transform')
    get_inf_transform = getattr(transform, 'get_inf_transform', None)
    inf_transform = get_inf_transform(keys, args)
    data = inf_transform(data_dict)
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="Segmentation model id, e.g. unet3d")
    ap.add_argument("--device", default="cuda", type=str, help="c")
    ap.add_argument("--img_pth", required=True, default=None, help="target img for infer")
    ap.add_argument("--checkpoint", required=True, help="Segmentaion model pth put here")
    ap.add_argument("--infer_dir", default="./infers", type=str, help="directory to save the eval result")
    ap.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    ap.add_argument("--out_channels", default=4, type=int, help="number of output channels")
    ap.add_argument("--a_min", default=-42, type=float, help="a_min in ScaleIntensityRanged")
    ap.add_argument("--a_max", default=423, type=float, help="a_max in ScaleIntensityRanged")
    ap.add_argument("--space_x", default=0.7, type=float, help="spacing in x direction")
    ap.add_argument("--space_y", default=0.7, type=float, help="spacing in y direction")
    ap.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
    ap.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    ap.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    ap.add_argument("--roi_z", default=32, type=int, help="roi size in z direction")
    ap.add_argument("--infer_post_process", default=True, action="store_true", help="infer post process")

    args = ap.parse_args()

    os.makedirs(args.infer_dir, exist_ok=True)

    # device
    if torch.cuda.is_available():
        print("cuda is available")
        args.device = torch.device("cuda")
    else:
        print("cuda is not available")
        args.device = torch.device("cpu")
    
    model = network(args.model_name, args)

        # check point
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        
        if is_deep_sup(checkpoint) and args.model_name != 'cotr':
            # load check point epoch and best acc
            print("Tag 'ds (deeply supervised)' found in state dict - fixing!")
            for key in list(checkpoint["state_dict"].keys()):
                if 'ds' in key:
                    checkpoint["state_dict"].pop(key) 
        
        # load model
        model.load_state_dict(checkpoint["state_dict"])
        
        print(
          "=> loaded checkpoint '{}')"\
          .format(args.checkpoint)
        )
    
    keys = ['pred']
    post_transform = Compose([
        Orientationd(keys=keys, axcodes='LPS'),
        ToNumpyd(keys=keys),
        Restored(keys=keys, ref_image="image")
    ])
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=2,
        predictor=model,
        overlap=0.25,
    )

    data_dicts = [{'image': args.img_pth}]

    for data_dict in data_dicts:
        print('infer data:', data_dict)
      
        # load infer data
        data = get_infer_data(data_dict, args)

        # infer
        run_infering(
            model,
            data,
            model_inferer,
            post_transform,
            args
        )


if __name__ == "__main__":
    main()