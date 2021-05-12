from modules.global_vars import DEVICE, CHECKPOINTS
from modules.models.alt_dcgan import Generator, Discriminator
from modules.models.encoder import Encoder
from modules.train_routines import train_gan, train_encoder

from omegaconf import DictConfig
from torch import nn, optim
import torch

from typing import Dict, Union, Callable


def load_weights(model: nn.Module, cfg: DictConfig) -> nn.Module:
    checkpoint_path = CHECKPOINTS / f"{cfg.model.dec_checkpoint}.pth"
    model.load_state_dict(
        torch.load(checkpoint_path)["generator"]["model_state_dict"]
    )
    return model


def get_optimizer(
        cfg: DictConfig,
        model: Union[nn.Module, Dict[str, nn.Module]]
) -> Union[optim.Optimizer, Dict[str, optim.Optimizer]]:
    # unique encoder part
    if cfg.name == "encoder":
        if cfg.optimizer == "adam":
            optimizer = optim.Adam(
                params=model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=(cfg.beta1, cfg.beta2)
            )
        else:
            raise NotImplementedError
        return optimizer

    if cfg.optimizer == "adam":
        if type(model) == dict:
            optimizer = dict()
            for key in model:
                if key == cfg.gen_key:
                    optimizer[key] = optim.Adam(
                        params=model[key].parameters(),
                        lr=cfg.lr * cfg.gen_lr_scale,
                        weight_decay=cfg.weight_decay,
                        betas=(cfg.beta1, cfg.beta2),
                    )
                optimizer[key] = optim.Adam(
                    params=model[key].parameters(),
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    betas=(cfg.beta1, cfg.beta2),
                )
        elif type(model) == nn.Module:
            optimizer = optim.Adam(
                params=model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=(cfg.beta1, cfg.beta2)
            )
        else:
            raise TypeError
    else:
        raise NotImplementedError

    return optimizer


def get_criterion(cfg: DictConfig) -> nn.Module:
    if cfg.criterion == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif cfg.criterion == "MSELoss":
        return nn.MSELoss(reduction="sum")
    else:
        raise NotImplementedError


def get_run_tag(cfg: DictConfig) -> str:
    return f"{cfg.name}/{cfg.optimizer}_lr={cfg.lr}_wd={cfg.weight_decay}" \
           f"_betas=({cfg.beta1},{cfg.beta2})"


def get_model(cfg: DictConfig) -> Union[nn.Module, Dict[str, nn.Module]]:
    if cfg.model.name == "DCGAN":
        model = {
            cfg.model.gen_key: Generator(cfg.h_dim, cfg.model.expansion, 1).to(DEVICE),
            cfg.model.dis_key: Discriminator(1, cfg.model.compression).to(DEVICE)
        }
        for key in model:
            model[key].init_weights()
    elif cfg.model.name == "encoder":
        model = Encoder(
            in_channels=1,
            out_channels=1,
            expansion=cfg.model.expansion,
            compression=cfg.model.compression,
            n_layers=cfg.model.num_layers,
            h_dim=cfg.h_dim
        )
        model.decoder = load_weights(model.decoder, cfg)
    else:
        raise NotImplementedError

    return model


def get_train_func(cfg: DictConfig) -> Callable:
    if cfg.name == "DCGAN":
        return train_gan
    elif cfg.name == "encoder":
        return train_encoder
    else:
        raise NotImplementedError
