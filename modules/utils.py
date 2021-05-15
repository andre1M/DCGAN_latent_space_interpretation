from modules.global_vars import DEVICE, CHECKPOINTS
from modules.models.dcgan_v3 import Generator, Discriminator
from modules.models.encoder_v2 import GeneratorEnc, DiscriminatorEnc
from modules.train_routines import train_gan, train_encoder_v2
from modules.models.sim_annealing import solve as sa_solve

from omegaconf import DictConfig
from torch import nn, optim
import omegaconf
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
    # if cfg.name == "encoder":
    #     if cfg.optimizer == "adam":
    #         optimizer = optim.Adam(
    #             params=model.parameters(),
    #             lr=cfg.lr,
    #             weight_decay=cfg.weight_decay,
    #             betas=(cfg.beta1, cfg.beta2)
    #         )
    #     else:
    #         raise NotImplementedError
    #     return optimizer

    if cfg.optimizer == "adam":
        if type(model) == dict:
            optimizer = dict()
            for key in model:
                if key == cfg.gen_key:
                    if cfg.name == "encoder":
                        optimizer[key] = optim.Adam(
                            params=model[key].encoder.parameters(),
                            lr=cfg.lr * cfg.gen_lr_scale,
                            weight_decay=cfg.weight_decay,
                            betas=(cfg.beta1, cfg.beta2),
                        )
                        continue
                    else:
                        optimizer[key] = optim.Adam(
                            params=model[key].parameters(),
                            lr=cfg.lr * cfg.gen_lr_scale,
                            weight_decay=cfg.weight_decay,
                            betas=(cfg.beta1, cfg.beta2),
                        )
                        continue
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
    try:
        tag = f"{cfg.name}/{cfg.optimizer}_lr={cfg.lr}_wd={cfg.weight_decay}" \
              f"_betas=({cfg.beta1},{cfg.beta2})"
    except omegaconf.errors.ConfigAttributeError:
        tag = f"{cfg.name}/{cfg.criterion}"
    return tag


def get_model(cfg: DictConfig) -> Union[nn.Module, Dict[str, nn.Module]]:
    if cfg.model.name == "DCGAN":
        model = {
            cfg.model.gen_key: Generator(cfg.h_dim, cfg.model.expansion, 1).to(DEVICE),
            cfg.model.dis_key: Discriminator(1, cfg.model.compression).to(DEVICE)
        }
        for key in model:
            model[key].init_weights()
    elif cfg.model.name == "encoder":
        model = {
            cfg.model.gen_key: GeneratorEnc(
                1, 1, cfg.model.compression, cfg.model.expansion, cfg.model.num_layers, cfg.h_dim
            ),
            cfg.model.dis_key: DiscriminatorEnc(2, cfg.model.compression).to(DEVICE)
        }
        # model = Encoder(
        #     in_channels=1,
        #     out_channels=1,
        #     expansion=cfg.model.expansion,
        #     compression=cfg.model.compression,
        #     n_layers=cfg.model.num_layers,
        #     h_dim=cfg.h_dim
        # )
        model[cfg.model.gen_key].decoder = load_weights(
            model[cfg.model.gen_key].decoder,
            cfg
        )
        model[cfg.model.gen_key].decoder.freeze_weights()
    else:
        model = Generator(cfg.h_dim, cfg.model.expansion, 1).to(DEVICE)
        model = load_weights(model, cfg)
        model.eval()

    return model


def get_train_func(cfg: DictConfig) -> Callable:
    if cfg.name == "DCGAN":
        return train_gan
    elif cfg.name == "encoder":
        return train_encoder_v2
    else:
        raise NotImplementedError


def get_solve_func(cfg: DictConfig) -> Callable:
    if cfg.name == "simulated_annealing":
        return sa_solve
    else:
        raise NotImplementedError
