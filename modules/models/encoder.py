from modules.models.resnet import BasicBlock
from modules.models.alt_dcgan import _conv, Generator

from torch import nn
import torch


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def __repr__(self) -> str:
        return self.__class__.__name__


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion: int,
            compression: int,
            n_layers: int,
            h_dim: int
    ) -> None:
        super().__init__()

        self.conv1 = _conv(
            in_channels=in_channels,
            out_channels=expansion,
            kernel_size=4,
            batch_norm=True
        )
        self.relu = nn.ReLU(inplace=True)

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

        layers = list()
        last_activation = True
        for i in range(n_layers):
            if i + 1 == n_layers:
                last_activation = False
            layers.append(BasicBlock(
                in_channels=expansion * 4,
                out_channels=expansion * 4,
                stride=1,
                identity_downsample=Identity(),
                last_activation=last_activation,
                bias=False
            ))

        self.res_block = nn.Sequential(*layers)

        self.flatten_dim = expansion * 4 * 4 * 4
        self.fc = nn.Linear(self.flatten_dim, h_dim)

        self.decoder = Generator(h_dim, compression, out_channels)
        self.freeze_decoder()

    def freeze_decoder(self):
        for name, param in self.decoder.named_parameters():
            param.requires_grad = False

    def forward(self, x) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.res_block(out)
        out = out.view(-1, self.flatten_dim)
        out = self.fc(out)
        out = out.unsqueeze(-1)
        out = out.unsqueeze(-1)
        out = self.decoder(out)

        return out
