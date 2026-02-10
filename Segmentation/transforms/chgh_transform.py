from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord
)

def get_train_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min, 
                a_max=args.a_max,
                b_min=0.0, 
                b_max=1.0,
                clip=True,
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.1,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.1,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.1,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.1,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.1,
            ),
            ToTensord(keys=["image", "label"])
        ]
    )


def get_val_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True
            ),
            ToTensord(keys=["image", "label"])
        ]
    )


def get_inf_transform(keys, args):
    if len(keys) == 2:
        # image and label
        mode = ("bilinear", "nearest")
    elif len(keys) == 3:
        # image and mutiple label
        mode = ("bilinear", "nearest", "nearest")
    else:
        # image
        mode = ("bilinear")
        
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=mode,
            ),
            ScaleIntensityRanged(
                keys=['image'],
                a_min=args.a_min, 
                a_max=args.a_max,
                b_min=0.0, 
                b_max=1.0,
                clip=True,
                allow_missing_keys=True
            ),
            EnsureChannelFirstd(keys=keys),
            ToTensord(keys=keys)
        ]
    )


def get_label_transform(keys=["label"]):
    return Compose(
        LoadImaged(keys=keys)
    )