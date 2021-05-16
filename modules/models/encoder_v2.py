from modules.models.resnet import BasicBlock
from modules.models.dcgan_v3 import _conv, _weights_init, Generator, \
    Discriminator

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
            expansion: int,
            num_layers: int,
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

        if num_layers == 0:
            self.res_block = Identity()
        else:
            layers = list()
            last_activation = True
            for i in range(num_layers):
                if i == num_layers - 1:
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

        self.flatten_dim = expansion * 4 * 3 * 3
        self.fc = nn.Linear(self.flatten_dim, h_dim)

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

        return out


class Decoder(Generator):
    def __init__(
            self,
            input_dim: int,
            compression: int,
            out_channels: int
    ) -> None:
        super().__init__(input_dim, compression, out_channels)

    def freeze_weights(self):
        for name, param in self.named_parameters():
            param.requires_grad = False


class GeneratorEnc(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            compression,
            expansion,
            num_layers,
            h_dim
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            expansion=expansion,
            num_layers=num_layers,
            h_dim=h_dim
        )
        self.decoder = Decoder(
            input_dim=h_dim,
            compression=compression,
            out_channels=out_channels
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out.unsqueeze(-1)
        out = out.unsqueeze(-1)
        self.decoder.eval()
        out = self.decoder(out)

        return out


class DiscriminatorEnc(Discriminator):
    def __init__(self, in_channels: int, expansion: int) -> None:
        super().__init__(in_channels, expansion)
        self.init_weights()

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
