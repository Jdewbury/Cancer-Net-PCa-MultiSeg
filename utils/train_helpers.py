from pathlib import Path
import torch
from utils.dataset import CancerNetPCa
from monai.metrics import DiceMetric
from utils.eval_helpers import evaluate
from utils.data_helpers import save_json
from typing import Tuple


def train(
    output_dir: Path,
    dataset: CancerNetPCa,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    loss_function: torch.nn,
    dice_metric: DiceMetric,
    epochs: int = 200,
    early_stopping_patience: int = 15,
    min_improvement: float = 0.0005,
) -> Tuple[float, Path]:
    training_dir = output_dir / "train"
    training_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    all_train_losses = []
    all_val_dice = []

    early_stopping_counter = 0
    best_val_dice = 0.0
    best_val_epoch = 0
    best_model_path = training_dir / "best_val_model.pth"

    for epoch_idx in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_slices = 0

        for inputs, masks in dataset.train:
            input_volume = inputs.to(device)
            mask_volume = masks.to(device)

            b, c, h, w, d = inputs.shape

            for slice_idx in range(d):
                optimizer.zero_grad()

                input_slice = input_volume[..., slice_idx].squeeze(-1)
                mask_slice = mask_volume[..., slice_idx].squeeze(-1)

                output = model(input_slice)
                loss = loss_function(output, mask_slice)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_slices += 1

        total_loss = epoch_loss / total_slices
        all_train_losses.append(total_loss)

        val_dice = evaluate(model, dataset.val, dice_metric, device)
        all_val_dice.append(val_dice)

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
            torch.save(
                {
                    "epoch": best_val_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": best_val_dice,
                },
                best_model_path,
            )
        else:
            early_stopping_counter += 1

        print(
            f"epoch {epoch_idx} train_loss: {total_loss} val_dice:{val_dice} best_val_epoch: {best_val_epoch} learning_rate: {current_lr}"
        )

        if early_stopping_counter >= early_stopping_patience:
            break

    results_path = training_dir / "results.json"
    results = {
        "train_loss": all_train_losses,
        "val_dice": all_val_dice,
        "best_val_epoch": best_val_epoch,
    }
    save_json(results_path, results)

    return best_val_dice, best_model_path
