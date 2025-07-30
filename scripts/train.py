from pathlib import Path

import torch
from monai.metrics import DiceMetric

from cancernet_pca_seg import CancerNetPCa, Config
from cancernet_pca_seg.evaluation import evaluate
from cancernet_pca_seg.models import get_model
from cancernet_pca_seg.training import (
    get_optimizer,
    get_scheduler,
    run_training,
    set_all_seeds,
)
from cancernet_pca_seg.utils.io import make_dir, save_json


def train_and_evaluate_folds():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = Config()
    config.update_from_args()
    config_dict = config.to_dict()

    set_all_seeds(config.seed)

    experiment_dir = make_dir(Path(config.output_dir) / config.experiment_name)

    save_json(experiment_dir / "experimental_params.json", config_dict)

    for fold_idx in range(config.num_folds):
        fold_dir = experiment_dir / f"fold_{fold_idx+1}"

        dataset = CancerNetPCa(
            img_dirs=config.img_dirs,
            mask_dir=config.mask_dir,
            modalities=config.modalities,
            num_folds=config.num_folds,
            fold_idx=fold_idx,
            test_split=config.test_split,
            batch_size=config.batch_size,
            lesion_mask=config.use_lesion_mask,
            seed=config.seed,
        )

        model = get_model(
            model_name=config.model,
            modalities=config.modalities,
            init_filters=config.init_filters,
            input_size=config.target_size,
        )

        optimizer = get_optimizer(
            optimizer_name=config.optimizer,
            model=model,
            learning_rate=config.learning_rate,
        )

        scheduler = get_scheduler(
            scheduler_name=config.scheduler,
            optimizer=optimizer,
            epochs=config.epochs,
            lr_step=config.lr_step,
            lr_patience=config.lr_patience,
        )

        # evaluation metrics
        loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean")
        dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )

        best_val_dice, best_model_path, best_threshold = run_training(
            output_dir=fold_dir,
            dataset=dataset,
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            dice_metric=dice_metric,
            epochs=config.epochs,
            early_stopping_patience=config.early_stopping_patience,
            min_improvement=config.min_improvement,
        )

        print("Training complete")

        # evaluate on test set
        if best_model_path.exists() and best_val_dice != 0.0:
            print("Evaluating on test set")
            test_dir = make_dir(fold_dir / "inference")

            checkpoint = torch.load(best_model_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

            test_dice = evaluate(
                model=model,
                dataloader=dataset.test,
                dice_metric=dice_metric,
                device=device,
                threshold=best_threshold,
            )
            results = {"test_dice": test_dice}
            save_json(test_dir / "test_results.json", results)

            print(f"Saved test inference results to {test_dir}")


if __name__ == "__main__":
    train_and_evaluate_folds()
