import argparse
import torch
from monai.metrics import DiceMetric
from utils.dataset import CancerNetPCa
from models.model_factory import get_model
from utils.optimizer import get_optimizer, get_scheduler
from utils.config import Config
from pathlib import Path
from utils.eval_helpers import evaluate
from utils.train_helpers import train
from utils.data_helpers import save_json, set_all_seeds


def parse_args():
    config = Config()
    config_dict = config.to_dict()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dirs",
        nargs="+",
        type=str,
        help=f"Directory containing image data. Pass multiple directories for more than one modality. Default: {config.img_dirs}",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        help=f"Directory containing mask data. Default: {config.mask_dir}",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        type=str,
        choices=["cdis", "dwi", "adc"],
        help=f"One or more image modalities to evaluate. Default: {config.modalities}",
    )
    parser.add_argument(
        "--target_size",
        type=tuple,
        help=f"Target size of input image into model. Default: {config.target_size}",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help=f"Batch size for the training and validation loops. Default: {config.batch_size}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help=f"Total number of training epochs. Default: {config.epochs}",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help=f"Initial learning rate for training. Default: {config.learning_rate}",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        help=f"Number of K folds to evaluate over. Default: {config.num_folds}",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        help=f"Percent allocation of dataset to the test set. Default: {config.test_split}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=f"Seed to use for splitting dataset. Default: {config.seed}",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        help=f"Number of epochs without improvement for early stopping. Default: {config.early_stopping_patience}",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["segresnet", "unet", "swinunetr", "attentionunet", "mambaunet"],
        help=f"Model architecture to be used for training. Default: {config.model}",
    )
    parser.add_argument(
        "--init_filters",
        type=int,
        help=f"Number of filters for model. Default: {config.init_filters}",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd"],
        help=f"Optimizer to use for training. Default: {config.optimizer}",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=[None, "step", "cosine", "plateau"],
        help=f"Learning rate scheduler to use. Default: {config.scheduler}",
    )
    parser.add_argument(
        "--lr_step",
        type=float,
        help=f"Learning rate step size. Default: {config.lr_step}",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        help=f"Learning rate patience before reduction. Default: {config.lr_patience}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help=f"Output directory to save training results. Default: {config.output_dir}",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help=f"Name of experiment. Default: {config.experiment_name}",
    )

    args = parser.parse_args()
    args_dict = vars(args)

    # update default config values with passed results
    parsed_args = {k: v for k, v in args_dict.items() if v is not None}
    if parsed_args.get("use_lesion_mask") is not None:
        parsed_args["use_lesion_mask"] = parsed_args["use_lesion_mask"] == "True"

    config_dict.update(parsed_args)

    return config_dict


def run_training_and_evaluation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = parse_args()
    set_all_seeds(config["seed"])

    experiment_dir = Path(config["output_dir"]) / config["experiment_name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    save_json(experiment_dir / "experimental_params.json", config)

    for fold_idx in range(config["num_folds"]):
        fold_dir = experiment_dir / f"fold_{fold_idx+1}"

        dataset = CancerNetPCa(
            img_dirs=config["img_dirs"],
            mask_dir=config["mask_dir"],
            modalities=config["modalities"],
            num_folds=config["num_folds"],
            fold_idx=fold_idx,
            test_split=config["test_split"],
            batch_size=config["batch_size"],
            lesion_mask=True,
            seed=config["seed"],
        )

        model = get_model(
            model_name=config["model"],
            modalities=config["modalities"],
            init_filters=config["init_filters"],
            input_size=config["target_size"],
        )
        optimizer = get_optimizer(
            optimizer_name=config["optimizer"],
            model=model,
            learning_rate=config["learning_rate"],
        )
        scheduler = get_scheduler(
            scheduler_name=config["scheduler"],
            optimizer=optimizer,
            epochs=config["epochs"],
            lr_step=config["lr_step"],
            lr_patience=config["lr_patience"],
        )
        # evaluation metrics
        loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean")
        dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )

        best_val_dice, best_model_path = train(
            output_dir=fold_dir,
            dataset=dataset,
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            dice_metric=dice_metric,
            epochs=config["epochs"],
            early_stopping_patience=config["early_stopping_patience"],
        )

        if best_val_dice != 0.0:
            # evaluate on test set
            test_dir = fold_dir / "inference"
            test_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = torch.load(best_model_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

            test_dice = evaluate(model, dataset.test, dice_metric, device)
            results = {"test_dice": test_dice}
            save_json(test_dir / "test_results.json", results)


if __name__ == "__main__":
    run_training_and_evaluation()
