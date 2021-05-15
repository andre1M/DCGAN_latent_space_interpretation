from modules.global_vars import DEVICE, OUTPUT_ROOT, CHECKPOINTS

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from torch import nn, optim
from tqdm.auto import tqdm
import torchvision
import torch

from typing import Dict, Optional, Tuple, Union


def real_loss(
        d_out: torch.Tensor,
        criterion: nn.Module,
        smooth: bool = False
) -> torch.Tensor:
    """
    Loss function for the real images.

    :param d_out:
    :param criterion: loss function (must operate on logits)
    :param smooth:
    :return:
    """

    # label smoothing
    if smooth:
        labels = torch.ones_like(d_out.squeeze()) * 0.9
    else:
        labels = torch.ones_like(d_out.squeeze())

    # calculate loss
    labels.to(DEVICE)
    loss = criterion(d_out.squeeze(), labels)

    return loss


def fake_loss(
        d_out: torch.Tensor,
        criterion: nn.Module,
        smooth: bool = False
) -> torch.Tensor:
    """
    Loss function for the fake images.

    :param d_out:
    :param criterion: loss function (must operate on logits)
    :param smooth:
    :return:
    """

    labels = torch.zeros_like(d_out.squeeze())
    if smooth:
        labels += 0.1

    # calculate loss
    labels.to(DEVICE)
    loss = criterion(d_out.squeeze(), labels)

    return loss


def save_model(
        model: Union[nn.Module, Dict[str, nn.Module]],
        epoch: int,
        model_name: str
) -> None:
    """
    Save model's parameters.

    :param model:
    :param epoch:
    :param model_name:
    :return:
    """

    if type(model) == dict:
        checkpoint = dict()
        for key in model:
            checkpoint[key] = dict(
                epoch=epoch,
                model_state_dict=model[key].state_dict()
            )
        torch.save(checkpoint, CHECKPOINTS / f"{model_name}.pth")
    else:
        checkpoint = model.state_dict()

    torch.save(checkpoint, CHECKPOINTS / f"{model_name}.pth")


def generate_noise(size: Tuple[int, int]) -> torch.Tensor:
    noise = torch.normal(
        mean=0,
        std=1,
        size=size,
        device=DEVICE
    )
    noise = noise.unsqueeze(-1)
    return noise.unsqueeze(-1)


def train_gan(
        cfg: DictConfig,
        model: Dict[str, nn.Module],
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: Dict[str, optim.Optimizer],
        run_tag: str,
        scheduler: Optional[object] = None,
        restart: Optional[int] = None
) -> Dict[str, nn.Module]:
    """
    Basic train routine for GAN.

    :param cfg:
    :param model:
    :param dataloader:
    :param criterion:
    :param optimizer:
    :param run_tag:
    :param scheduler:
    :param restart: epoch to start from
    :return:
    """

    writer = SummaryWriter(OUTPUT_ROOT)

    fixed_h = generate_noise(size=(cfg.model.h_sample_size, cfg.h_dim))

    # ensure model are in train mode
    model[cfg.model.gen_key].train()
    model[cfg.model.dis_key].train()

    iter_count = 0

    for epoch in range(cfg.num_epochs):
        d_running_loss = 0.0
        g_running_loss = 0.0

        print("Epoch {}/{}".format(epoch + 1, cfg.num_epochs))
        print('-' * 15)

        for real_img, _ in tqdm(dataloader):

            # ============================================ #
            #            TRAIN THE DISCRIMINATOR           #
            # ============================================ #
            # 1. Real images
            # cheap trick but it helps a lot
            real_img = real_img * 2 - 1
            d_real_out = model[cfg.model.dis_key](real_img.to(DEVICE))
            d_real_loss = real_loss(d_real_out, criterion, True)

            # 2. Generate fake images
            h = generate_noise(size=(cfg.batch_size, cfg.model.h_dim))
            fake_images = model[cfg.model.gen_key](h).to(DEVICE)

            # Compute the discriminator loss on fake images and total loss
            d_fake_out = model[cfg.model.dis_key](fake_images)
            d_fake_loss = fake_loss(d_fake_out, criterion, True)
            d_loss = d_real_loss + d_fake_loss

            # 3. Backpropagation
            optimizer[cfg.model.dis_key].zero_grad()
            d_loss.backward()
            optimizer[cfg.model.dis_key].step()

            # ========================================= #
            #            TRAIN THE GENERATOR            #
            # ========================================= #

            # 1. Generate fake images
            h = generate_noise(size=(cfg.batch_size, cfg.model.h_dim))
            fake_images = model[cfg.model.gen_key](h).to(DEVICE)

            # 2. Compute the discriminator loss on fake images
            # with flipped labels
            d_fake_out = model[cfg.model.dis_key](fake_images).to(DEVICE)
            # use real loss to flip labels
            g_loss = real_loss(d_fake_out, criterion, True)

            # 3. Backpropagation
            optimizer[cfg.model.gen_key].zero_grad()
            g_loss.backward()
            optimizer[cfg.model.gen_key].step()

            # ========================================= #
            #            COMPUTE STATISTICS             #
            # ========================================= #
            d_running_loss += d_loss.item() * cfg.batch_size
            g_running_loss += g_loss.item() * cfg.batch_size

            iter_count += 1
            writer.add_scalars(
                main_tag=f"Loss_{cfg.model.name}/{run_tag}",
                tag_scalar_dict={"discriminator_real": d_real_loss.item(),
                                 "discriminator_fake": d_fake_loss.item(),
                                 "generator": g_loss.item()},
                global_step=iter_count
            )

        # compute epoch statistics and save to the tensorboard
        # noinspection PyTypeChecker
        d_epoch_loss = d_running_loss / len(dataloader.dataset)
        # noinspection PyTypeChecker
        g_epoch_loss = g_running_loss / len(dataloader.dataset)
        writer.add_scalars(
            main_tag=f"Mean_epoch_loss_{cfg.model.name}/{run_tag}",
            tag_scalar_dict={"discriminator": d_epoch_loss,
                             "generator": g_epoch_loss},
            global_step=epoch + 1
        )

        # generate sample fake images and save to the tensorboard
        model[cfg.model.gen_key].eval()
        with torch.no_grad():
            samples = model[cfg.model.gen_key](fixed_h)
        model[cfg.model.gen_key].train()

        sample_grid = torchvision.utils.make_grid(
            tensor=samples,
            nrow=cfg.model.h_sample_size // 6,
            normalize=True
        )
        writer.add_image(
            tag=f"{cfg.model.name}/Fake_image_samples",
            img_tensor=sample_grid,
            global_step=epoch + 1
        )

        save_model(model, epoch, cfg.model.name)

    writer.close()

    return model


def train_encoder_v1(
        cfg: DictConfig,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        run_tag: str,
        scheduler: Optional[object] = None,
        restart: Optional[int] = None
) -> nn.Module:
    """
    Basic train routine for GAN.

    :param cfg:
    :param model:
    :param dataloader:
    :param criterion:
    :param optimizer:
    :param run_tag:
    :param scheduler:
    :param restart:
    :return:
    """

    writer = SummaryWriter(OUTPUT_ROOT)

    fixed_img = next(iter(dataloader))[0]

    sample_grid = torchvision.utils.make_grid(
        tensor=fixed_img,
        nrow=cfg.batch_size // 8,
        normalize=True
    )
    writer.add_image(
        tag="Encoder/Real_image_samples",
        img_tensor=sample_grid,
        global_step=1
    )

    del sample_grid

    # ensure model are in proper mode
    model.train()

    iter_count = 0

    for epoch in range(cfg.num_epochs):
        running_loss = 0.0

        print("Epoch {}/{}".format(epoch + 1, cfg.num_epochs))
        print('-' * 15)

        for real_img, _ in tqdm(dataloader):

            # 1. Real images to vector
            # cheap trick but it helps a lot
            real_img = real_img * 2 - 1
            fake_images = model(real_img.to(DEVICE))

            # 2.Compute loss
            loss = criterion(real_img, fake_images)

            # 3. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ========================================= #
            #            COMPUTE STATISTICS             #
            # ========================================= #
            running_loss += loss.item() * cfg.batch_size * \
                real_img.size(-1) * real_img.size(-2)

            iter_count += 1
            writer.add_scalar(
                tag=f"Loss_{cfg.model.name}/{run_tag}",
                scalar_value=loss.item(),
                global_step=iter_count
            )

        # compute epoch statistics and save to the tensorboard
        # noinspection PyTypeChecker
        epoch_loss = running_loss / len(dataloader.dataset)
        writer.add_scalar(
            tag=f"Mean_epoch_loss_{cfg.model.name}/{run_tag}",
            scalar_value=epoch_loss,
            global_step=epoch + 1
        )

        # generate sample fake images and save to the tensorboard
        model.eval()
        with torch.no_grad():
            fake_image_samples = model(fixed_img)
        model.train()

        sample_grid = torchvision.utils.make_grid(
            tensor=fake_image_samples,
            nrow=cfg.batch_size // 8,
            normalize=True
        )
        writer.add_image(
            tag="Encoder/Fake_image_samples",
            img_tensor=sample_grid,
            global_step=epoch + 1
        )

        save_model(model, epoch, cfg.model.name)

    writer.close()

    return model


def train_encoder_v2(
        cfg: DictConfig,
        model: Dict[str, nn.Module],
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: Dict[str, optim.Optimizer],
        run_tag: str,
        scheduler: Optional[object] = None,
        restart: Optional[int] = None
) -> Dict[str, nn.Module]:
    """
    Basic train routine for GAN.

    :param cfg:
    :param model:
    :param dataloader:
    :param criterion:
    :param optimizer:
    :param run_tag:
    :param scheduler:
    :param restart:
    :return:
    """

    writer = SummaryWriter(OUTPUT_ROOT)

    fixed_img = next(iter(dataloader))[0]

    sample_grid = torchvision.utils.make_grid(
        tensor=fixed_img,
        nrow=cfg.batch_size // 8,
        normalize=True
    )
    writer.add_image(
        tag="Encoder/Real_image_samples",
        img_tensor=sample_grid,
        global_step=1
    )

    del sample_grid

    criterion_l1 = nn.L1Loss()

    # ensure model are in proper mode
    model[cfg.model.gen_key].train()
    model[cfg.model.dis_key].train()

    iter_count = 0

    for epoch in range(cfg.num_epochs):
        d_running_loss = 0.0
        g_running_loss = 0.0

        print("Epoch {}/{}".format(epoch + 1, cfg.num_epochs))
        print('-' * 15)

        for real_img, _ in tqdm(dataloader):

            # ============================================ #
            #            TRAIN THE DISCRIMINATOR           #
            # ============================================ #
            # model[cfg.model.dis_key].requires_grad_(True)
            # 1.Generate pairs of real-fake and real-real images
            # cheap trick but it helps a lot
            real_img = real_img * 2 - 1
            real_img = real_img.to(DEVICE)
            fake_images = model[cfg.model.gen_key](real_img).to(DEVICE)
            fake_pairs = torch.cat((real_img, fake_images), 1)
            real_pairs = torch.cat((real_img, real_img), 1)

            # 2. Make predictions
            d_fake_out = model[cfg.model.dis_key](fake_pairs)
            d_real_out = model[cfg.model.dis_key](real_pairs)

            # 3. Compute loss
            d_real_loss = real_loss(d_fake_out, criterion, True)
            d_fake_loss = fake_loss(d_real_out, criterion, True)
            d_loss = (d_real_loss + d_fake_loss) / 2

            # 4. Backpropagation
            optimizer[cfg.model.dis_key].zero_grad()
            d_loss.backward()
            optimizer[cfg.model.dis_key].step()

            # ========================================= #
            #            TRAIN THE GENERATOR            #
            # ========================================= #
            # model[cfg.model.dis_key].requires_grad_(False)
            # 1. Generate fake images and make real-fake pairs
            fake_images = model[cfg.model.gen_key](real_img).to(DEVICE)
            fake_pairs = torch.cat((real_img, fake_images), 1)

            # 2. Make predictions and compute loss
            d_fake_out = model[cfg.model.dis_key](fake_pairs.detach())
            g_loss_l1 = criterion_l1(real_img, fake_images)
            g_loss = real_loss(d_fake_out, criterion, True) + \
                g_loss_l1 * cfg.model.lambda_l1

            # 3. Backpropagation
            optimizer[cfg.model.gen_key].zero_grad()
            g_loss.backward()
            optimizer[cfg.model.gen_key].step()

            # ========================================= #
            #            COMPUTE STATISTICS             #
            # ========================================= #

            d_running_loss += d_loss.item() * cfg.batch_size
            g_running_loss += g_loss.item() * cfg.batch_size

            iter_count += 1
            writer.add_scalars(
                main_tag=f"Loss_{cfg.model.name}/{run_tag}",
                tag_scalar_dict={"discriminator_real": d_real_loss.item(),
                                 "discriminator_fake": d_fake_loss.item(),
                                 "generator": g_loss.item()},
                global_step=iter_count
            )

        # compute epoch statistics and save to the tensorboard
        # noinspection PyTypeChecker
        d_epoch_loss = d_running_loss / len(dataloader.dataset)
        # noinspection PyTypeChecker
        g_epoch_loss = g_running_loss / len(dataloader.dataset)
        writer.add_scalars(
            main_tag=f"Mean_epoch_loss_{cfg.model.name}/{run_tag}",
            tag_scalar_dict={"discriminator": d_epoch_loss,
                             "generator": g_epoch_loss},
            global_step=epoch + 1
        )

        # generate sample fake images and save to the tensorboard
        model[cfg.model.gen_key].eval()
        with torch.no_grad():
            fake_image_samples = model[cfg.model.gen_key](fixed_img)
        model[cfg.model.gen_key].train()

        sample_grid = torchvision.utils.make_grid(
            tensor=fake_image_samples,
            nrow=cfg.batch_size // 8,
            normalize=True
        )
        writer.add_image(
            tag="Encoder/Fake_image_samples",
            img_tensor=sample_grid,
            global_step=epoch + 1
        )

        save_model(model, epoch, cfg.model.name)

    writer.close()

    return model
