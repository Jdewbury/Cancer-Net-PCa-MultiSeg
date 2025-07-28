import torch
from monai.metrics import DiceMetric

from cancernet_pca.data import CancerNetPCa


def evaluate(
    model: torch.nn.Module,
    dataloader: CancerNetPCa,
    dice_metric: DiceMetric,
    device: torch.device,
    threshold: float = 0.5,
) -> float:
    model.to(device)
    model.eval()

    dice_metric.reset()

    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            input_slices = inputs.squeeze(0).permute(3, 0, 1, 2)
            outputs = model(input_slices)

            pred_volume = torch.sigmoid(outputs).permute(1, 2, 3, 0).unsqueeze(0)
            pred_binary = (pred_volume > threshold).float()

            # compute Dice score
            dice_metric(y_pred=pred_binary, y=masks)

    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()

    return dice_score
