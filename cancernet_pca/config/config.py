import argparse
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class Config:
    # dataset args
    img_dirs: List[str] = field(default_factory=lambda: ["data/images"])
    mask_dir: str = "data_2"
    modalities: List[str] = field(default_factory=lambda: ["cdis"])
    target_size: Tuple[int, int] = (128, 128)
    use_lesion_mask: bool = True

    # training args
    batch_size: int = 1
    epochs: int = 200
    learning_rate: float = 0.001
    num_folds: int = 5
    test_split: float = 0.1
    seed: int = 42
    early_stopping_patience: int = 50
    min_improvement: float = 0.001

    # model args
    model: str = "unet"
    init_filters: int = 32

    # optimizer args
    optimizer: str = "adamw"
    scheduler: str = "plateau"
    lr_step: float = 0.5
    lr_patience: int = 15

    # output args
    output_dir: str = "results"
    experiment_name: str = "cancer-net-pca-seg"

    def update_from_args(self) -> None:
        parser = argparse.ArgumentParser(description="CancerNet-PCa Segmentation")

        # dataset args
        parser.add_argument(
            "--img_dirs",
            nargs="+",
            type=str,
            help=f"Directory containing image data. Pass multiple directories for more than one modality. Default: {self.img_dirs}",
        )
        parser.add_argument(
            "--mask_dir",
            type=str,
            help=f"Directory containing mask data. Default: {self.mask_dir}",
        )
        parser.add_argument(
            "--modalities",
            nargs="+",
            type=str,
            choices=["cdis", "dwi", "adc"],
            help=f"One or more image modalities to evaluate. Default: {self.modalities}",
        )
        parser.add_argument(
            "--target_size_h",
            type=int,
            help=f"Target height of input image. Default: {self.target_size[0]}",
        )
        parser.add_argument(
            "--target_size_w",
            type=int,
            help=f"Target width of input image. Default: {self.target_size[1]}",
        )
        parser.add_argument(
            "--use_lesion_mask",
            type=str,
            choices=["true", "false"],
            help=f"Whether to use lesion mask (true) or prostate mask (false). Default: {self.use_lesion_mask}",
        )

        # training args
        parser.add_argument(
            "--batch_size",
            type=int,
            help=f"Batch size for the training and validation loops. Default: {self.batch_size}",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            help=f"Total number of training epochs. Default: {self.epochs}",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            help=f"Initial learning rate for training. Default: {self.learning_rate}",
        )
        parser.add_argument(
            "--num_folds",
            type=int,
            help=f"Number of K folds to evaluate over. Default: {self.num_folds}",
        )
        parser.add_argument(
            "--test_split",
            type=float,
            help=f"Percent allocation of dataset to the test set. Default: {self.test_split}",
        )
        parser.add_argument(
            "--seed",
            type=int,
            help=f"Seed to use for splitting dataset. Default: {self.seed}",
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            help=f"Number of epochs without improvement for early stopping. Default: {self.early_stopping_patience}",
        )
        parser.add_argument(
            "--min_improvement",
            type=float,
            help=f"Minimum dice improvement to reset patience counter. Default: {self.min_improvement}",
        )

        # model args
        parser.add_argument(
            "--model",
            type=str,
            choices=["segresnet", "unet", "swinunetr", "attentionunet", "mambaunet"],
            help=f"Model architecture to be used for training. Default: {self.model}",
        )
        parser.add_argument(
            "--init_filters",
            type=int,
            help=f"Number of filters for model. Default: {self.init_filters}",
        )

        # optimizer args
        parser.add_argument(
            "--optimizer",
            type=str,
            choices=["adam", "adamw", "sgd"],
            help=f"Optimizer to use for training. Default: {self.optimizer}",
        )
        parser.add_argument(
            "--scheduler",
            type=str,
            choices=[None, "step", "cosine", "plateau"],
            help=f"Learning rate scheduler to use. Default: {self.scheduler}",
        )
        parser.add_argument(
            "--lr_step",
            type=float,
            help=f"Learning rate step size. Default: {self.lr_step}",
        )
        parser.add_argument(
            "--lr_patience",
            type=int,
            help=f"Learning rate patience before reduction. Default: {self.lr_patience}",
        )

        # output args
        parser.add_argument(
            "--output_dir",
            type=str,
            help=f"Output directory to save training results. Default: {self.output_dir}",
        )
        parser.add_argument(
            "--experiment_name",
            type=str,
            help=f"Name of experiment. Default: {self.experiment_name}",
        )

        args = parser.parse_args()

        for k, v in vars(args).items():
            if (
                v is not None
                and hasattr(self, k)
                and k not in ["use_lesion_mask", "target_size_h", "target_size_w"]
            ):
                setattr(self, k, v)

        if args.use_lesion_mask is not None:
            self.use_lesion_mask = args.use_lesion_mask == "true"

        if args.target_size_h is not None or args.target_size_w is not None:
            curr_h, curr_w = self.target_size
            new_h = args.target_size_h if args.target_size_h is not None else curr_h
            new_w = args.target_size_w if args.target_size_w is not None else curr_w
            self.target_size = (new_h, new_w)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
