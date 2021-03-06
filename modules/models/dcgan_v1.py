from torch import nn
import torch

from typing import Union, Tuple


def _conv(
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 2,
        padding: Union[int, Tuple[int, int]] = 1,
        batch_norm: bool = False
) -> nn.Module:
    """
    Basic convolutional building block.

    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param batch_norm:
    :return:
    """

    layers = list()
    layers.append(nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    ))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def _transposed_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 2,
        padding: Union[int, Tuple[int, int]] = 1,
        batch_norm: bool = False
) -> nn.Module:
    """
    Basic transposed-convolutional building block.

    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param batch_norm:
    :return:
    """

    layers = list()
    layers.append(nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    ))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


# Source: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Introduced in https://arxiv.org/pdf/1511.06434.pdf
def _weights_init(m: nn.Module) -> None:
    module_name = m.__class__.__name__
    if module_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif module_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(
            self,
            input_dim: int,
            compression: int,
            out_channels: int
    ) -> None:
        super().__init__()

        self.t_conv1 = _transposed_conv(
            in_channels=input_dim,
            out_channels=compression * 8,
            kernel_size=4,
            stride=1,
            padding=0,
            batch_norm=True
        )
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.t_conv2 = _transposed_conv(
            in_channels=compression * 8,
            out_channels=compression * 4,
            kernel_size=4,
            batch_norm=True
        )

        self.t_conv3 = _transposed_conv(
            in_channels=compression * 4,
            out_channels=compression * 2,
            kernel_size=4,
            batch_norm=True
        )

        self.t_conv4 = _transposed_conv(
            in_channels=compression * 2,
            out_channels=compression,
            kernel_size=4,
            batch_norm=True
        )

        self.t_conv5 = _transposed_conv(
            in_channels=compression,
            out_channels=out_channels,
            kernel_size=4,
            batch_norm=False
        )
        self.tanh = nn.Tanh()

        self.compression = compression

    def init_weights(self) -> None:
        self.apply(_weights_init)

    def forward(self, x) -> torch.Tensor:
        out = self.t_conv1(x)
        out = self.leaky_relu(out)
        out = self.t_conv2(out)
        out = self.leaky_relu(out)
        out = self.t_conv3(out)
        out = self.leaky_relu(out)
        out = self.t_conv4(out)
        out = self.leaky_relu(out)
        out = self.t_conv5(out)
        out = self.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, expansion: int) -> None:
        super().__init__()

        self.conv1 = _conv(
            in_channels=in_channels,
            out_channels=expansion,
            kernel_size=4,
            batch_norm=False
        )
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = _conv(
            in_channels=expansion,
            out_channels=expansion * 2,
            kernel_size=4,
            batch_norm=True
        )

        self.conv3 = _conv(
            in_channels=expansion * 2,
            out_channels=expansion * 4,
            kernel_size=4,
            batch_norm=True
        )

        self.conv4 = _conv(
            in_channels=expansion * 4,
            out_channels=expansion * 8,
            kernel_size=4,
            batch_norm=True
        )

        self.conv5 = _conv(
            in_channels=expansion * 8,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            batch_norm=False
        )

    def init_weights(self) -> None:
        self.apply(_weights_init)

    def forward(self, x) -> torch.Tensor:
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        out = self.conv3(out)
        out = self.leaky_relu(out)
        out = self.conv4(out)
        out = self.leaky_relu(out)
        out = self.conv5(out)

        return out
