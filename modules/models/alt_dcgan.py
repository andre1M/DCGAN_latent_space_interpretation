from modules.models.dcgan import _conv, _transposed_conv, _weights_init

from torch import nn
import torch


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
        self.relu = nn.ReLU(inplace=True)

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
            kernel_size=1,
            stride=1,
            padding=0,
            batch_norm=False
        )
        self.tanh = nn.Tanh()

    def init_weights(self) -> None:
        self.apply(_weights_init)

    def forward(self, x) -> torch.Tensor:
        out = self.t_conv1(x)
        out = self.relu(out)
        out = self.t_conv2(out)
        out = self.relu(out)
        out = self.t_conv3(out)
        out = self.relu(out)
        out = self.t_conv4(out)
        out = self.relu(out)
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
            out_channels=1,
            kernel_size=6,
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
        out = out.view(-1, 1)

        return out
