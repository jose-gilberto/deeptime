

from __future__ import annotations

from typing import Any, Tuple

import pytorch_lightning as pl
import torch
from torch import nn, optim

from deeptime.models.utils import Conv1dSamePadding, ConvBlock, UpSample


class ResNetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True
    ) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=1,
                bias=bias
            ) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias
                ),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


class ResNetAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        in_channels: int,
        in_features: int,
        mid_channels: int = 64,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()

        self.e = nn.Sequential(
            ResNetBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                bias=False,
            ),
            ResNetBlock(
                in_channels=mid_channels,
                out_channels=mid_channels * 2,
                bias=False,
            ),
            ResNetBlock(
                in_channels=mid_channels * 2,
                out_channels=mid_channels * 2,
                bias=False,
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=mid_channels * 2 * in_features,
                out_features=256,
                bias=False,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=128,
                bias=False,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=latent_dim,
                bias=False,
            )
        )

        self.d = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=128,
                bias=False,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=256,
                bias=False,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=mid_channels * 2 * in_features,
                bias=False,
            ),
            UpSample(
                out_channels=mid_channels * 2,
                series_length=in_features,
            ),
            ResNetBlock(
                in_channels=mid_channels * 2,
                out_channels=mid_channels * 2
            ),
            ResNetBlock(
                in_channels=mid_channels * 2,
                out_channels=mid_channels
            ),
            ResNetBlock(
                in_channels=mid_channels,
                out_channels=in_channels
            )
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.e(x)
        x_hat = self.d(z)
        return x_hat, z

    def _get_reconstruction_loss(self, batch: Any) -> torch.Tensor:
        x, _ = batch

        x_hat, _ = self(x)
        loss = nn.functional.mse_loss(x_hat, x)

        return loss

    def training_step(self, batch: Any) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, *args: Any, **kwargs: Any):
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any):
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=1e-4)
