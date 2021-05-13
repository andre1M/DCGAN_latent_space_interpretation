from modules.global_vars import OUTPUT_ROOT

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import dual_annealing
from omegaconf import DictConfig
from torch import nn
import numpy as np
import torchvision
import torch


def objective(
        guess: float,
        sample: torch.Tensor,
        model: nn.Module,
        criterion: nn.Module
) -> np.array:
    new_sample = model(guess)
    return criterion(new_sample, sample).numpy()


def solve(
        cfg: DictConfig,
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        run_tag: str
) -> None:
    """
    Simulated annealing iterative solver.

    :param cfg:
    :param model:
    :param criterion:
    :param dataloader:
    :param run_tag:
    :return:
    """

    writer = SummaryWriter(OUTPUT_ROOT)
    samples = next(iter(dataloader))
    # [-3, 3] covers 99.7% of normally distributed values
    bounds = np.ones(size=(cfg.h_dim, 2)) * np.array([-3, 3])

    for i, s in enumerate(samples):
        optimizer = dual_annealing(
            objective, bounds, (s, model, criterion), seed=cfg.seed
        )
        fake_image = model(torch.from_numpy(optimizer.x))
        sample_grid = torchvision.utils.make_grid(
            tensor=[s, fake_image],
            nrow=1,
            normalize=True
        )
        writer.add_image(
            tag=f"{cfg.model.name}/{run_tag}/REAL->GENERATED",
            img_tensor=sample_grid,
            global_step=i
        )

