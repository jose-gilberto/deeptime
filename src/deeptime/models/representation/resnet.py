

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
        learning_rate: float = 5e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.in_channels = in_channels
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.mid_channels = mid_channels

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
            nn.Tanh(),
            nn.Linear(
                in_features=256,
                out_features=128,
                bias=False,
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=128,
                out_features=latent_dim,
                bias=False,
            ),
            nn.Tanh(),
        )

        self.d = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=128,
                bias=False,
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=128,
                out_features=256,
                bias=False,
            ),
            nn.Tanh(),
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

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, *args: Any, **kwargs: Any):
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.learning_rate)


class ResNetVariationalAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        in_channels: int,
        in_features: int,
        mid_channels: int,
        latent_dim: int
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.in_features = in_features
        self.mid_channels = mid_channels
        self.latent_dim = latent_dim

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
                out_features=latent_dim * 2,
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

    def reparametrize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.e(x)
        x = x.view(-1, 2, self.latent_dim)

        mu = x[:, 0, :]
        log_var = x[:, 1, :]

        z = self.reparametrize(mu, log_var)

        x_hat = self.d(z)
        return x_hat, mu, log_var

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _ = batch

        x_hat, mu, log_var = self(x)

        recon_loss = nn.functional.mse_loss(x_hat, x, size_average=False)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=1e-4)
