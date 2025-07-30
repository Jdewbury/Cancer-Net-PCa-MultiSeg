from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader

from cancernet_pca_seg import Config
from cancernet_pca_seg.models import get_model


def load_fold_models(
    experiment_dir: Path,
    config: Config,
    device: torch.device,
) -> List:

    fold_models = []
    for fold_dir in sorted(experiment_dir.glob("fold_*")):
        fold_num = int(fold_dir.name.split("_")[1])
        model_path = fold_dir / "train" / "best_val_model.pth"

        if not model_path.exists():
            continue

        model = get_model(
            model_name=config.model,
            modalities=config.modalities,
            init_filters=config.init_filters,
            input_size=config.target_size,
        )

        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        fold_models.append(
            {
                "fold": fold_num,
                "model": model,
                "threshold": checkpoint["optimal_threshold"],
                "val_dice": checkpoint["val_dice"],
            }
        )

    return fold_models


def run_ensembling(
    models: List,
    dataloader: DataLoader,
    device: torch.device,
    dice_metric: DiceMetric,
) -> Tuple:

    ensemble_dice_scores = []
    val_dices = np.array([m["val_dice"] for m in models])
    weights = val_dices / val_dices.sum()

    thresholds = np.array([m["threshold"] for m in models])
    weighted_threshold = np.sum(weights * thresholds)

    for inputs, masks in dataloader:
        inputs = inputs.to(device)
        masks = masks.to(device)

        raw_predictions = []
        for model_dict in models:
            with torch.no_grad():
                b, c, h, w, d = inputs.shape
                input_slices = inputs.squeeze(0).permute(3, 0, 1, 2)
                outputs = model_dict["model"](input_slices)
                pred_volume = torch.sigmoid(outputs).permute(1, 2, 3, 0).unsqueeze(0)

                raw_predictions.append(pred_volume)

        # weighted soft ensembling
        weighted_predictions = []
        for i, pred in enumerate(raw_predictions):
            weighted_predictions.append(weights[i] * pred)

        ensemble_pred = torch.sum(torch.stack(weighted_predictions), dim=0)
        final_pred = (ensemble_pred > weighted_threshold).float()

        dice_metric.reset()
        dice_metric(y_pred=final_pred, y=masks)
        dice_score = dice_metric.aggregate().item()
        ensemble_dice_scores.append(dice_score)

    overall_ensemble_dice = np.mean(ensemble_dice_scores)

    results = {
        "sample_test_dice": ensemble_dice_scores,
        "ensemble_test_dice": overall_ensemble_dice,
        "ensemble_threshold": weighted_threshold,
        "num_folds": len(models),
    }

    return results
