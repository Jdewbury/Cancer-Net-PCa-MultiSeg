import torch
from monai.metrics import DiceMetric
from utils.dataset import CancerNetPCa


def evaluate(
    model: torch.nn.Module,
    dataloader: CancerNetPCa,
    dice_metric: DiceMetric,
    device: torch.device,
) -> float:
    model.to(device)
    model.eval()

    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            pred_volume = torch.zeros_like(masks)

            for slice_idx in range(inputs.shape[-1]):
                input_slice = inputs[..., slice_idx].squeeze(-1)
                output = model(input_slice)

                pred_slice = (torch.sigmoid(output) > 0.5).float()
                pred_volume[..., slice_idx] = pred_slice

            # compute Dice score
            dice_metric(y_pred=pred_volume, y=masks)

    dice_score = dice_metric.aggregate().item()

    return dice_score
