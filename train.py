from modules.dataset import get_dataloader
from modules.utils import get_optimizer, get_criterion, get_run_tag, \
    get_model, get_train_func

from omegaconf import DictConfig
import hydra
import torch


@hydra.main(config_path="config", config_name="config")
def run(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    dataloader = get_dataloader(cfg.batch_size)
    model = get_model(cfg)
    optimizer = get_optimizer(cfg.model, model)
    criterion = get_criterion(cfg.model)
    run_tag = get_run_tag(cfg.model)
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