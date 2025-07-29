#!/usr/bin/env python3
"""
Fixed visualization script for pre-trained model predictions
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from cancernet_pca import CancerNetPCa, Config
from cancernet_pca.models import get_model
from cancernet_pca.utils.io import load_json


def load_trained_model(model_path: Path, config: dict):
    """Load a trained model from checkpoint"""

    model = get_model(
        model_name=config["model"],
        modalities=config["modalities"],
        init_filters=config["init_filters"],
        input_size=config["target_size"],
    )

    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint["optimal_threshold"]


def predict_volume(model, input_volume, device, threshold=0.5):
    """Run inference on a single volume"""
    model.eval()
    model.to(device)

    print(f"Input to predict_volume: {input_volume.shape}")

    with torch.no_grad():
        input_volume = input_volume.to(device)

        # Ensure we have the right shape [batch, channels, height, width, depth]
        while input_volume.dim() > 5:
            input_volume = input_volume.squeeze(0)
            print(f"Squeezed to: {input_volume.shape}")

        if input_volume.dim() == 4:  # [c, h, w, d]
            input_volume = input_volume.unsqueeze(0)  # Add batch -> [1, c, h, w, d]
            print(f"Added batch dimension: {input_volume.shape}")

        b, c, h, w, d = input_volume.shape
        print(f"Final input shape: b={b}, c={c}, h={h}, w={w}, d={d}")

        # Process slices (same as in training)
        input_slices = input_volume.squeeze(0).permute(3, 0, 1, 2)  # [d, c, h, w]
        print(f"Input slices shape: {input_slices.shape}")

        outputs = model(input_slices)
        print(f"Model outputs shape: {outputs.shape}")

        pred_volume = (
            torch.sigmoid(outputs).permute(1, 2, 3, 0).unsqueeze(0)
        )  # [1, h, w, d] -> [1, 1, h, w, d]
        print(f"Prediction volume shape: {pred_volume.shape}")

        pred_binary = (pred_volume > threshold).float()

    return pred_volume.cpu(), pred_binary.cpu()


def visualize_predictions(
    img_volume,
    mask_volume,
    pred_volume,
    pred_binary,
    slice_indices=None,
    save_path=None,
):
    """Visualize predictions across multiple slices"""

    print(
        f"Visualize - img: {img_volume.shape}, mask: {mask_volume.shape}, pred: {pred_volume.shape}"
    )

    # Ensure all tensors have consistent dimensions
    # Remove extra dimensions and get to [h, w, d]
    while img_volume.dim() > 4:
        img_volume = img_volume.squeeze(0)
    while mask_volume.dim() > 4:
        mask_volume = mask_volume.squeeze(0)
    while pred_volume.dim() > 4:
        pred_volume = pred_volume.squeeze(0)
    while pred_binary.dim() > 4:
        pred_binary = pred_binary.squeeze(0)

    # Get dimensions - should be [c, h, w, d] or [h, w, d]
    if img_volume.dim() == 4:  # [c, h, w, d]
        img_for_viz = img_volume[0]  # Take first channel [h, w, d]
    else:  # [h, w, d]
        img_for_viz = img_volume

    if mask_volume.dim() == 4:  # [c, h, w, d]
        mask_for_viz = mask_volume[0]  # Take first channel [h, w, d]
    else:  # [h, w, d]
        mask_for_viz = mask_volume

    if pred_volume.dim() == 4:  # [c, h, w, d]
        pred_for_viz = pred_volume[0]  # Take first channel [h, w, d]
    else:  # [h, w, d]
        pred_for_viz = pred_volume

    if pred_binary.dim() == 4:  # [c, h, w, d]
        pred_binary_viz = pred_binary[0]  # Take first channel [h, w, d]
    else:  # [h, w, d]
        pred_binary_viz = pred_binary

    print(f"After processing - img: {img_for_viz.shape}, mask: {mask_for_viz.shape}")

    if slice_indices is None:
        # Show slices from different parts of the volume
        d = img_for_viz.shape[-1]
        slice_indices = [max(0, d // 4), d // 2, min(d - 1, 3 * d // 4)]

    n_slices = len(slice_indices)
    fig, axes = plt.subplots(4, n_slices, figsize=(5 * n_slices, 16))

    if n_slices == 1:
        axes = axes.reshape(-1, 1)

    for i, slice_idx in enumerate(slice_indices):
        # Get individual slices [h, w]
        img_slice = img_for_viz[:, :, slice_idx].numpy()
        mask_slice = mask_for_viz[:, :, slice_idx].numpy()
        pred_slice = pred_for_viz[:, :, slice_idx].numpy()
        pred_binary_slice = pred_binary_viz[:, :, slice_idx].numpy()

        # Image
        axes[0, i].imshow(img_slice, cmap="gray")
        axes[0, i].set_title(f"Image - Slice {slice_idx}")
        axes[0, i].axis("off")

        # Ground truth mask
        axes[1, i].imshow(mask_slice, cmap="Reds", alpha=0.8)
        axes[1, i].set_title("Ground Truth")
        axes[1, i].axis("off")

        # Prediction probability
        im = axes[2, i].imshow(pred_slice, cmap="Blues", vmin=0, vmax=1)
        axes[2, i].set_title("Prediction Probability")
        axes[2, i].axis("off")
        plt.colorbar(im, ax=axes[2, i], fraction=0.046)

        # Overlay comparison
        axes[3, i].imshow(img_slice, cmap="gray")
        axes[3, i].imshow(mask_slice, cmap="Reds", alpha=0.4, label="GT")
        axes[3, i].imshow(pred_binary_slice, cmap="Blues", alpha=0.4, label="Pred")
        axes[3, i].set_title("Overlay (Red=GT, Blue=Pred)")
        axes[3, i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()


def compute_dice_score(pred, target):
    """Compute Dice score between prediction and target"""
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)

    return dice.item()


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory (contains experimental_params.json)",
    )
    parser.add_argument(
        "--patient_idx",
        type=int,
        default=0,
        help="Index of patient to visualize (from test set)",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save visualizations"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load experiment config
    experiment_dir = Path(args.experiment_dir)
    config_path = experiment_dir / "experimental_params.json"
    config = load_json(config_path)

    print(f"Loaded config: {config['experiment_name']}")
    print(f"Model: {config['model']}, Modalities: {config['modalities']}")

    # Load model
    model_path = Path(args.model_path)
    model, optimal_threshold = load_trained_model(model_path, config)
    print(f"Loaded model with optimal threshold: {optimal_threshold:.3f}")

    # Create dataset (use same fold as the model for consistency)
    fold_idx = int(model_path.parent.parent.name.split("_")[1]) - 1  # fold_1 -> idx 0
    dataset = CancerNetPCa(
        img_dirs=config["img_dirs"],
        mask_dir=config["mask_dir"],
        modalities=config["modalities"],
        num_folds=config["num_folds"],
        fold_idx=fold_idx,
        test_split=config["test_split"],
        batch_size=1,  # Single sample
        lesion_mask=config["use_lesion_mask"],
        seed=config["seed"],
    )

    # Get test sample directly from dataset (not dataloader)
    if args.patient_idx >= len(dataset.test_dataset):
        print(
            f"Patient index {args.patient_idx} out of range. Max: {len(dataset.test_dataset)-1}"
        )
        return

    img_volume, mask_volume = dataset.test_dataset[args.patient_idx]

    print(f"Processing patient {args.patient_idx}")
    print(f"Original img shape: {img_volume.shape}")
    print(f"Original mask shape: {mask_volume.shape}")

    # Run prediction
    pred_volume, pred_binary = predict_volume(
        model, img_volume, device, optimal_threshold
    )

    # Compute metrics
    dice_score = compute_dice_score(pred_binary, mask_volume)
    print(f"Dice score: {dice_score:.4f}")

    # Visualize
    save_path = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"patient_{args.patient_idx}_predictions.png"

    visualize_predictions(
        img_volume, mask_volume, pred_volume, pred_binary, save_path=save_path
    )


if __name__ == "__main__":
    main()

# Example usage:
# poetry run python scripts/visualize_model.py \
#   --model_path /home/jarett/Cancer-Net-PCa-Seg/results/benchmark/lesion/cdis/swinunetr/cdis-swinunetr/fold_1/train/best_val_model.pth \
#   --experiment_dir /home/jarett/Cancer-Net-PCa-Seg/results/benchmark/lesion/cdis/swinunetr/cdis-swinunetr \
#   --patient_idx 0 \
#   --save_dir visualizations/
