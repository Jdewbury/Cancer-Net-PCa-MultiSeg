from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class Config:
    # dataset params
    img_dirs: List[str] = field(default_factory=lambda: ["data/images"])
    mask_dir: str = "data_2"
    modalities: List[str] = field(default_factory=lambda: ["cdis"])
    use_lesion_mask: bool = True

    # training params
    batch_size: int = 1
    epochs: int = 200
    learning_rate: float = 0.001
    num_folds: int = 5
    test_split: float = 0.1
    seed: int = 42
    early_stopping_patience: int = 15

    # model params
    model: str = "unet"
    init_filters: int = 32

    # optimizer params
    optimizer: str = "adam"
    scheduler: str = "step"
    lr_step: float = 0.1

    # output params
    output_dir: str = "results"
    experiment_name: str = "cancer-net-pca-seg"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
