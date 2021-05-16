from modules.global_vars import DEVICE, CHECKPOINTS, ITER_METHODS
from modules.models.dcgan_v3 import Generator, Discriminator
from modules.models.encoder_v1 import EncoderDecoder
from modules.models.encoder_v2 import GeneratorEnc, DiscriminatorEnc
from modules.train_routines import train_gan, train_encoder_v1, train_encoder_v2
from modules.models.sim_annealing import solve as sa_solve
from modules.models.particle_swarm import solve as ps_solve

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
    if cfg.name == "DCGAN":
        optimizer = dict()

        if cfg.optimizer == "adam":
            for key in model:
                if key == cfg.gen_key:
                    optimizer[key] = optim.Adam(
                        params=model[key].parameters(),
                        lr=cfg.lr * cfg.gen_lr_scale,
                        weight_decay=cfg.weight_decay,
                        betas=(cfg.beta1, cfg.beta2)
                    )
                    continue
                optimizer[key] = optim.Adam(
                    params=model[key].parameters(),
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    betas=(cfg.beta1, cfg.beta2)
                )

        else:
            raise NotImplementedError

    elif cfg.name == "encoder":

        if cfg.version == "v1":

            if cfg.optimizer == "adam":
                optimizer = optim.Adam(
                    params=model.encoder.parameters(),
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    betas=(cfg.beta1, cfg.beta2)
                )

            else:
                raise NotImplementedError

        elif cfg.version == "v2":
            optimizer = dict()

            if cfg.optimizer == "adam":
                for key in model:
                    if key == "generator":
                        optimizer[key] = optim.Adam(
                            params=model[key].encoder.parameters(),
                            lr=cfg.lr * cfg.gen_lr_scale,
                            weight_decay=cfg.weight_decay,
                            betas=(cfg.beta1, cfg.beta2)
                        )
                        continue
                    optimizer[key] = optim.Adam(
                        params=model[key].parameters(),
                        lr=cfg.lr,
                        weight_decay=cfg.weight_decay,
                        betas=(cfg.beta1, cfg.beta2)
                    )

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return optimizer


def get_criterion(cfg: DictConfig) -> nn.Module:
    if cfg.criterion == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif cfg.criterion == "MSELoss":
        if cfg.name == "encoder":
            if cfg.version == "v1":
                return nn.MSELoss(reduction="sum")
            elif cfg.version == "v2":
                raise NotImplementedError
        elif cfg.name == "simulated_annealing":
            return nn.MSELoss(reduction="sum")
        elif cfg.name == "particle_swarm":
            return nn.MSELoss(reduction="none")
        else:
            print("Using MSELoos with default arguments")
            return nn.MSELoss()
    else:
        raise NotImplementedError


def get_run_tag(cfg: DictConfig) -> str:
    if cfg.name == "simulated_annealing":
        tag = f"{cfg.name}/{cfg.criterion}"
    elif cfg.name == "particle_swarm":
        tag = f"{cfg.name}/{cfg.criterion}_c1={cfg.c1}_c2={cfg.c2}_w={cfg.w}_" \
              f"num_particles={cfg.num_particles}"
    elif cfg.name == "encoder":
        tag = f"{cfg.name}_{cfg.version}/{cfg.optimizer}_lr={cfg.lr}_wd=" \
              f"{cfg.weight_decay}_betas=({cfg.beta1},{cfg.beta2})"
    elif cfg.name == "DCGAN":
        tag = f"{cfg.name}/{cfg.optimizer}_lr={cfg.lr}_wd=" \
              f"{cfg.weight_decay}_betas=({cfg.beta1},{cfg.beta2})"
    else:
        raise NotImplementedError

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

        if cfg.model.version == "v1":
            model = EncoderDecoder(
                1, 1, cfg.model.compression, cfg.model.expansion, cfg.model.num_layers, cfg.h_dim
            )
            model.decoder = load_weights(model.decoder, cfg)
            model.freeze_decoder()

        elif cfg.model.version == "v2":
            model = {
                cfg.model.gen_key: GeneratorEnc(
                    1, 1, cfg.model.compression, cfg.model.expansion, cfg.model.num_layers, cfg.h_dim
                ),
                cfg.model.dis_key: DiscriminatorEnc(2, cfg.model.compression).to(DEVICE)
            }
            model[cfg.model.gen_key].decoder = load_weights(
                model[cfg.model.gen_key].decoder,
                cfg
            )
            model[cfg.model.gen_key].decoder.freeze_weights()
        else:
            raise NotImplementedError

    elif cfg.model.name in ITER_METHODS:
        model = Generator(cfg.h_dim, cfg.model.expansion, 1).to(DEVICE)
        model = load_weights(model, cfg)
        model.freeze_weights()
        model.eval()

    else:
        raise NotImplementedError

    return model


def get_train_func(cfg: DictConfig) -> Callable:
    if cfg.name == "DCGAN":
        return train_gan
    elif cfg.name == "encoder":
        if cfg.version == "v1":
            return train_encoder_v1
        elif cfg.version == "v2":
            return train_encoder_v2
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_solve_func(cfg: DictConfig) -> Callable:
    if cfg.name == "simulated_annealing":
        return sa_solve
    elif cfg.name == "particle_swarm":
        return ps_solve
    else:
        raise NotImplementedError
