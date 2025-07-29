import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader

from cancernet_pca_seg.data.dataset import CancerNetPCa
from cancernet_pca_seg.utils.io import make_dir, save_json


def set_all_seeds(seed):
    """Set all seeds to make results reproducible.

    Args:
        seed: desired seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def find_optimal_threshold(
    model: torch.nn.Module,
    dataloader: CancerNetPCa,
    dice_metric: DiceMetric,
    device: torch.device,
) -> Tuple:
    model.eval()

    thresholds = np.arange(0.1, 1, 0.01)
    best_dice = 0
    best_threshold = 0.5

    all_predictions = []
    all_masks = []

    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            b, c, h, w, d = inputs.shape

            input_slices = inputs.squeeze(0).permute(3, 0, 1, 2)
            outputs = model(input_slices)

            pred_volume = torch.sigmoid(outputs).permute(1, 2, 3, 0).unsqueeze(0)

            all_predictions.append(pred_volume)
            all_masks.append(masks)

    for t in thresholds:
        dice_metric.reset()

        for pred_volume, masks in zip(all_predictions, all_masks):
            pred_binary = (pred_volume > t).float()
            dice_metric(y_pred=pred_binary, y=masks)

        dice_score = dice_metric.aggregate().item()

        if dice_score > best_dice:
            best_dice = dice_score
            best_threshold = t

    return best_dice, best_threshold


def train_step(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0

    for inputs, masks in train_loader:
        optimizer.zero_grad()

        input_volume = inputs.to(device)
        mask_volume = masks.to(device)

        b, c, h, w, d = inputs.shape

        input_slices = input_volume.squeeze(0).permute(3, 0, 1, 2)
        mask_slices = mask_volume.squeeze(0).permute(3, 0, 1, 2)

        output = model(input_slices)
        loss = loss_function(output, mask_slices)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def run_training(
    output_dir: Path,
    dataset: CancerNetPCa,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    loss_function: torch.nn,
    dice_metric: DiceMetric,
    epochs: int = 200,
    early_stopping_patience: int = 15,
    min_improvement: float = 0.0005,
) -> Tuple:
    training_dir = make_dir(output_dir / "train")

    model.to(device)
    all_train_losses = []
    all_val_dice = []
    all_optimal_thresholds = []

    early_stopping_counter = 0
    best_val_dice = 0.0
    best_val_epoch = 0
    best_threshold = 0.5
    best_model_path = training_dir / "best_val_model.pth"

    for epoch_idx in range(epochs):
        epoch_loss = train_step(
            model=model,
            train_loader=dataset.train,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )
        all_train_losses.append(epoch_loss)

        val_dice, optimal_threshold = find_optimal_threshold(
            model=model,
            dataloader=dataset.val,
            dice_metric=dice_metric,
            device=device,
        )

        all_val_dice.append(val_dice)
        all_optimal_thresholds.append(optimal_threshold)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_dice)
            current_lr = optimizer.param_groups[0]["lr"]
        elif scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        if val_dice > best_val_dice:
            # only reset patience if the improvement is above the minimum threshold
            if (val_dice - best_val_dice) > min_improvement:
                early_stopping_counter = 0

            best_val_dice = val_dice
            best_val_epoch = epoch_idx
            best_threshold = optimal_threshold

            torch.save(
                {
                    "epoch": best_val_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": best_val_dice,
                    "optimal_threshold": best_threshold,
                },
                best_model_path,
            )
        else:
            early_stopping_counter += 1

        print(
            f"epoch {epoch_idx} train_loss: {epoch_loss:.4f} "
            f"val_dice:{val_dice:.4f} threshold: {optimal_threshold:.3f} "
            f"best_val_epoch: {best_val_epoch} learning_rate: {current_lr:.6f}"
        )

        if early_stopping_counter >= early_stopping_patience:
            break

    results_path = training_dir / "results.json"
    results = {
        "train_loss": all_train_losses,
        "val_dice": all_val_dice,
        "optimal_thresholds": all_optimal_thresholds,
        "best_val_epoch": best_val_epoch,
        "best_val_dice": best_val_dice,
        "best_threshold": best_threshold,
    }
    save_json(results_path, results)

    return best_val_dice, best_model_path, best_threshold
