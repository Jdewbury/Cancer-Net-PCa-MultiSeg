from pathlib import Path

import torch
from monai.metrics import DiceMetric

from cancernet_pca_seg import CancerNetPCa, Config
from cancernet_pca_seg.evaluation import load_fold_models, run_ensembling
from cancernet_pca_seg.training import set_all_seeds
from cancernet_pca_seg.utils.io import load_json, make_dir, save_json


def ensemble_inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = Config()
    config.update_from_args()

    experiment_dir = Path(config.experiment_dir)
    exp_config_dict = load_json(experiment_dir / "experimental_params.json")

    for k, v in exp_config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)

    set_all_seeds(config.seed)

    fold_models = load_fold_models(experiment_dir, config, device)
    print(f"Loaded {len(fold_models)} models")

    dataset = CancerNetPCa(
        img_dirs=config.img_dirs,
        mask_dir=config.mask_dir,
        modalities=config.modalities,
        num_folds=config.num_folds,
        fold_idx=0,
        test_split=config.test_split,
        batch_size=config.batch_size,
        lesion_mask=config.use_lesion_mask,
        seed=config.seed,
    )

    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )

    print(f"Running ensemble inference on {len(dataset.test)} samples")
    results = run_ensembling(
        models=fold_models,
        dataloader=dataset.test,
        device=device,
        dice_metric=dice_metric,
    )

    ensemble_dir = make_dir(experiment_dir / "ensemble")
    save_json(ensemble_dir / "ensemble_results.json", results)

    print(f"Saved ensemble inference results to {ensemble_dir}")


if __name__ == "__main__":
    ensemble_inference()
