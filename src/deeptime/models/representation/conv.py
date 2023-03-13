
from __future__ import annotations

from typing import Any, Tuple

import pytorch_lightning as pl
import torch
from torch import nn, optim

from deeptime.models.utils import Conv1dSamePadding  # GlobalAveragePooling,
from deeptime.models.utils import UpSample
from deeptime.nn.activations import SinL

activations_layers = {
    'relu': nn.ReLU,
    'swish': nn.SiLU,
    'tanh': nn.Tanh,
    'leakyrelu': nn.LeakyReLU,
    'sinl': SinL
}


class ConvAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        in_channels: int,
        in_features: int,
        latent_dim: int,
        activation: str = 'relu',
        optimizer: str = 'Adam',
        learning_rate: float = 1e-4,
        weight_decay: float = 0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.in_features = in_features

        self.latent_dim = latent_dim
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.activation = activation
        activation_kwargs = {}
        if self.activation == 'leakyrelu':
            activation_kwargs = {
                'negative_slope': 0.1,
            }

        self.e = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=128,
                kernel_size=8,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=128),
            # nn.Tanh(),
            activations_layers[self.activation](**activation_kwargs),
            Conv1dSamePadding(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=256),
            activations_layers[self.activation](**activation_kwargs),
            Conv1dSamePadding(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=128),
            activations_layers[self.activation](**activation_kwargs),

            # GlobalAveragePooling(),
            nn.Flatten(),

            nn.Linear(
                in_features=128 * in_features,
                out_features=256,
                bias=False
            ),
            activations_layers[self.activation](**activation_kwargs),
            nn.Linear(
                in_features=256,
                out_features=128,
                bias=False
            ),
            activations_layers[self.activation](**activation_kwargs),
            nn.Linear(
                in_features=128,
                out_features=latent_dim,
                bias=False
            ),
            activations_layers[self.activation](**activation_kwargs),
        )

        self.d = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=128,  # Must be equals to original series size
                bias=False
            ),
            activations_layers[self.activation](**activation_kwargs),
            nn.Linear(
                in_features=128,
                out_features=256,
                bias=False
            ),
            activations_layers[self.activation](**activation_kwargs),
            nn.Linear(
                in_features=256,
                out_features=128 * in_features,
                bias=False
            ),
            UpSample(
                out_channels=128,
                series_length=in_features
            ),
            Conv1dSamePadding(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                bias=False
            ),
            nn.BatchNorm1d(num_features=256),
            activations_layers[self.activation](**activation_kwargs),
            Conv1dSamePadding(
                in_channels=256,
                out_channels=128,
                kernel_size=5,
                bias=False
            ),
            nn.BatchNorm1d(num_features=128),
            activations_layers[self.activation](**activation_kwargs),
            Conv1dSamePadding(
                in_channels=128,
                out_channels=1,
                kernel_size=8,
                bias=False
            ),
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
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        x, _ = batch
        return self(x), x

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.learning_rate)


class ConvVariationalAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        in_channels: int,
        in_features: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.in_features = in_features
        self.latent_dim = latent_dim

        self.e = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=128,
                kernel_size=8,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=128),
            nn.Tanh(),
            Conv1dSamePadding(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=256),
            nn.Tanh(),
            Conv1dSamePadding(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=128),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(
                in_features=128 * in_features,
                out_features=256,
                bias=False
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=256,
                out_features=128,
                bias=False
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=128,
                out_features=latent_dim * 2,
                bias=False
            )
        )

        self.d = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=128,  # Must be equals to original series size
                bias=False
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=128,
                out_features=256,
                bias=False
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=256,
                out_features=128 * in_features,
                bias=False
            ),
            UpSample(
                out_channels=128,
                series_length=in_features
            ),
            Conv1dSamePadding(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                bias=False
            ),
            nn.BatchNorm1d(num_features=256),
            nn.Tanh(),
            Conv1dSamePadding(
                in_channels=256,
                out_channels=128,
                kernel_size=5,
                bias=False
            ),
            nn.BatchNorm1d(num_features=128),
            nn.Tanh(),
            Conv1dSamePadding(
                in_channels=128,
                out_channels=1,
                kernel_size=8,
                bias=False
            ),
        )

    def reparametrize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
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

        recon_loss = nn.functional.mse_loss(
            x_hat, x
        )
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        loss = recon_loss + kl_loss

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x_hat, mu, log_var = self(x)

        recon_loss = nn.functional.mse_loss(
            x_hat, x
        )
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        loss = recon_loss + kl_loss

        self.log('val_loss', loss, prog_bar=True)
        # self.log('val_recon_loss', recon_loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=1e-4)
