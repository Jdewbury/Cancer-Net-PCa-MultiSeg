from typing import List

import monai
from torch import nn


def get_model(
    model_name: str,
    modalities: List[str] = ["cdis"],
    init_filters: int = 32,
    input_size: tuple = (128, 128),
) -> nn.Module:
    """Retrieves and initializes select model architecture.

    Args:
        model_name: name of desired model to load
        modalities: which modalities are being used
        init_filters: number of output channels in initial conv layer
        input_size: size of input image

    Returns:
        Initialization of model
    """
    modality_channels = {"dwi": 3, "adc": 1, "cdis": 1}
    in_channels = sum([modality_channels[m] for m in modalities])

    if model_name == "segresnet":
        return monai.networks.nets.SegResNet(
            spatial_dims=2,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=1,
            dropout_prob=0.2,
        )
    elif model_name == "unet":
        return monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif model_name == "unetplusplus":
        return monai.networks.nets.BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
        )
    elif model_name == "unetr":
        return monai.networks.nets.UNETR(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            img_size=input_size,
        )
    elif model_name == "swinunetr":
        return monai.networks.nets.SwinUNETR(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
        )
    elif model_name == "attentionunet":
        return monai.networks.nets.AttentionUnet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
    else:
        raise ValueError(
            f"Invalid model name: {model_name}"
            "Choose from 'segresnet', 'unet', 'unetplusplus', 'unetr', 'swinunetr', 'attentionunet', or 'mambaunet'."
        )
