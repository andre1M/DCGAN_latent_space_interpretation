from modules.global_vars import OUTPUT_ROOT

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
from tqdm.auto import tqdm
from torch import nn
import pyswarms as ps
import numpy as np
import torchvision
import torch


def objective(
        guess: np.ndarray,
        kwargs
) -> np.array:
    with torch.no_grad():
        new_sample = kwargs["model"](
            torch.from_numpy(guess).view(-1, kwargs["h_dim"], 1, 1).float()
        )
    loss = kwargs["criterion"](
        new_sample.squeeze(1),
        kwargs["sample"].repeat(kwargs["num_particles"], 1, 1)
    ).detach().numpy()
    return loss.sum((-1, -2))


def solve(
        cfg: DictConfig,
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        run_tag: str
) -> None:
    """
    Particle swarm iterative solver.

    :param cfg:
    :param model:
    :param criterion:
    :param dataloader:
    :param run_tag:
    :return:
    """

    writer = SummaryWriter(OUTPUT_ROOT)
    samples = next(iter(dataloader))[0]
    options = {"c1": cfg.model.c1, "c2": cfg.model.c2, "w": cfg.model.w}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=cfg.model.num_particles,
        dimensions=cfg.h_dim,
        options=options
    )

    for i, s in enumerate(tqdm(samples)):
        s = s * 2 - 1
        best_cost, best_pos = optimizer.optimize(
            objective_func=objective,
            iters=1000,
            verbose=False,
            kwargs={
                "sample": s,
                "model": model,
                "criterion": criterion,
                "h_dim": cfg.h_dim,
                "num_particles": cfg.model.num_particles
            },
        )
        with torch.no_grad():
            fake_image = model(
                torch.from_numpy(best_pos).view(1, -1, 1, 1).float()
            )
        sample_grid = torchvision.utils.make_grid(
            tensor=[s, fake_image.squeeze(0)],
            nrow=1,
            normalize=True
        )
        writer.add_image(
            tag=f"{cfg.model.name}/{run_tag}/REAL->GENERATED",
            img_tensor=sample_grid,
            global_step=i
        )
