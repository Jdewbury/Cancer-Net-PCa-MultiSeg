import torch
from torch import nn


def get_optimizer(
    optimizer_name: str, model: nn.Module, learning_rate: float = 0.001
) -> torch.optim:
    """Retrieves and initializes select optimizer.

    Args:
        optimizer_name: name of desired optimizer to load
        model: The model whose parameters will be optimized
        learning_rate: learning rate of optimizer

    Returns:
        Initialization of optimizer
    """
    if optimizer_name == "adam":
        print("Using Adam optimizer")
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )
    elif optimizer_name == "adamw":
        print("Using AdamW optimizer")
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )
    elif optimizer_name == "sgd":
        print("Using SGD optimizer")
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(
            f"Invalid optimizer name: {optimizer_name}. Choose from 'adam', 'adamw', or 'sgd'."
        )


def get_scheduler(
    scheduler_name: str, optimizer: torch.optim, epochs: int, lr_step: float = 0.1
) -> torch.optim.lr_scheduler:
    """Retrieves and initializes select learning rate scheduler.

    Args:
        scheduler_name: name of scheduler to initialize
        optimizer: The optimizer whose learning rate will be scheduled
        epochs: number of epochs used during training
        lr_step: step size of learning rate

    Returns:
        Initialization of learning rate scheduler, or None if no scheduler is specified
    """
    if scheduler_name == "step":
        print("Using StepLR")
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=(epochs // 4), gamma=lr_step
        )
    elif scheduler_name == "cosine":
        print("Using CosineAnnealingLR")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(epochs // 4), eta_min=0
        )
    else:
        print("No LR scheduler")
        return None
