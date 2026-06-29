import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandFlipd,
    NormalizeIntensityd,
    ToTensord,
    MapTransform,
)

# --- 終極防呆防線：確保所有進模型的資料都是 3D ---
class Force3DSpatiald(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue
            val = d[key]
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val)
            
            # 如果空間維度是 2D [C, H, W]，強加 Z 軸變成 [C, H, W, 1]
            if val.ndim == 3:
                d[key] = val.unsqueeze(-1)
            # 如果連 Channel 都沒有 [H, W]，直接補成 [1, H, W, 1]
            elif val.ndim == 2:
                d[key] = val.unsqueeze(0).unsqueeze(-1)
            else:
                d[key] = val
        return d
# -----------------------------------------------

def get_train_transform(args):
    roi_size = (args.roi_x, args.roi_y, args.roi_z) if hasattr(args, 'roi_x') else (96, 96, 32)

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        
        # 讀取後立刻確保是 3D
        Force3DSpatiald(keys=["image", "label"]),
        
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 5.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode="constant"),
        
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=0.5,
            spatial_size=roi_size,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1)
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        ToTensord(keys=["image", "label"]),
    ])

def get_val_transform(args):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        Force3DSpatiald(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 5.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])

def get_inf_transform(keys, args):
    keys_list = list(keys)
    # 動態判定 mode：影像用 bilinear，標籤(若有)必須用 nearest
    mode_list = ["bilinear" if k == "image" else "nearest" for k in keys_list]
    # 確保只對 image 做常態化，絕對不能對 label 做
    norm_keys = [k for k in keys_list if k == "image"]

    return Compose([
        LoadImaged(keys=keys_list),
        EnsureChannelFirstd(keys=keys_list, channel_dim="no_channel"),
        
        # 關鍵防線：推論時如果讀到 2D 影像，強制塞入 Z 軸
        Force3DSpatiald(keys=keys_list),
        
        Spacingd(keys=keys_list, pixdim=(1.5, 1.5, 5.0), mode=mode_list),
        Orientationd(keys=keys_list, axcodes="RAS"),
        NormalizeIntensityd(keys=norm_keys, nonzero=True, channel_wise=True),
        ToTensord(keys=keys_list),
    ])
def get_label_transform(keys=["label"]):
    """
    用於推論階段讀取原始 Ground Truth 標籤，以進行最終指標計算。
    只做最基本的讀取與通道擴充，不進行任何空間縮放，保持原始形狀。
    """
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
    ])