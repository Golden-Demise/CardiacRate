from monai.networks.nets import SwinUNETR, UNETR, UNet, AttentionUnet
from networks.unetcnx.unetcnx_a1 import UNETCNX_A1

def network(model_name, args):
    print(f'model: {model_name}')
    if model_name == 'unet3d':
        return UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(64, 128, 256, 256),
            strides=(2, 2, 2),
            num_res_units=0,
            act='RELU',
            norm='BATCH'
        ).to(args.device)

    elif model_name == 'attention_unet':
        return AttentionUnet(
          spatial_dims=3,
          in_channels=args.in_channels,
          out_channels=args.out_channels,
          channels=(32, 64, 128, 256),
          strides=(2, 2, 2),
        ).to(args.device)

    elif model_name == 'unetr':
        return UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(args.device)

    elif model_name == 'swinunetr':
        return SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=48,
            use_checkpoint=True,
        ).to(args.device)
    # -----------------------------------------------------------------------------------------------------
    # unetcnx exp netowrks
    # -----------------------------------------------------------------------------------------------------
    elif model_name == 'unetcnx_a1':
        return UNETCNX_A1(
            in_channels=args.in_channels, #1
            out_channels=args.out_channels, #4
            patch_size=4,
            kernel_size=7,
            exp_rate=4, 
            feature_size=48,
            depths=[3, 3, 9, 3],
            drop_path_rate=1,
            use_init_weights=False, 
            is_conv_stem=False,
            skip_encoder_name=None,
            deep_sup=False,
            first_feature_size_half=False
          ).to(args.device)
    else:
        raise ValueError(f'not found model name: {model_name}')

