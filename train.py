from modules.global_vars import ITER_METHODS
from modules.dataset import get_dataloader
from modules.utils import get_optimizer, get_criterion, get_run_tag, \
    get_model, get_train_func, get_solve_func

from omegaconf import DictConfig
import hydra
import torch


@hydra.main(config_path="config", config_name="config")
def run(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    dataloader = get_dataloader(cfg)
    model = get_model(cfg)
    optimizer = get_optimizer(cfg.model, model)
    criterion = get_criterion(cfg.model)
    run_tag = get_run_tag(cfg.model)
    if cfg.model in ITER_METHODS:
        solve = get_solve_func(cfg.model)
        solve(cfg, model, dataloader, run_tag)
    train = get_train_func(cfg.model)
    _ = train(
        cfg=cfg,
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        run_tag=run_tag
    )
    print()
    print("--- Training is finished! ---")


if __name__ == "__main__":
    run()
