"""
Linear autoencoder module.
"""
from __future__ import annotations

from typing import Any, Tuple

import pytorch_lightning as pl
import torch
from torch import nn, optim

from deeptime.nn.activations import SinL

activations_layers = {
    'relu': nn.ReLU,
    'swish': nn.SiLU,
    'tanh': nn.Tanh,
    'leakyrelu': nn.LeakyReLU,
    'sinl': SinL
}


class LinearAutoEncoder(pl.LightningModule):
    """ Linear AutoEncoder.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: str = 'relu',
        optimizer: str = 'adam',
        learning_rate: float = 1e-4,
        weight_decay: float = 0,
    ) -> None:
        super().__init__()
        # Save Model Hyperparameters
        self.save_hyperparameters()

        activation_params = {}

        if activation == 'leakyrelu':
            activation_params = {
                'negative_slope': 0.1
            }

        # Encoder Architecture
        self.e = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=500,
                bias=False
            ),
            activations_layers[activation](**activation_params),
            nn.Linear(
                in_features=500,
                out_features=500,
                bias=False
            ),
            activations_layers[activation](**activation_params),
            nn.Linear(
                in_features=500,
                out_features=latent_dim,
                bias=False
            ),
            activations_layers[activation](**activation_params),
        )

        # Decoder Architecture
        self.d = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=500,
                bias=False
            ),
            activations_layers[activation](**activation_params),
            nn.Linear(
                in_features=500,
                out_features=500,
                bias=False
            ),
            activations_layers[activation](**activation_params),
            nn.Linear(
                in_features=500,
                out_features=input_dim,
                bias=False
            )
        )

        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.e(x)
        return self.d(z), z

    def _get_reconstruction_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x, _ = batch
        x = x.view(x.shape[0], -1)

        x_hat, z = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        return loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            return optimizer

        raise NotImplementedError()


class LinearVariationalAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        learning_rate: float = 5e-4,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.learning_rate = learning_rate

        # Encoder
        self.e = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=256, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=2*latent_dim, bias=False),
            nn.Tanh(),
        )

        # Decoder
        self.d = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=512, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=input_dim, bias=False)
        )

    def reparametrize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        """
        std = torch.exp(0.5 * log_var)  # Standard Deviation
        eps = torch.randn_like(std)  # randn_like as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the in space
        return sample

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.e(x)
        x = x.view(-1, 2, self.latent_dim)

        mu = x[:, 0, :]  # The first latent_size features as a mean
        log_var = x[:, 1, :]  # The other features values as variance

        z = self.reparametrize(mu, log_var)

        x_hat = self.d(z)
        return x_hat, mu, log_var

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x = x.view(x.shape[0], -1)

        x_hat, mu, log_var = self(x)

        bce_loss = nn.functional.mse_loss(x_hat, x, size_average=False)
        kl_loss = (
            -0.5 *
            (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1)
        ).mean(dim=0)

        loss = bce_loss + kl_loss
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Any,
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = batch
        x = x.view(x.shape[0], -1)

        x_hat, mu, log_var = self(x)

        bce_loss = nn.functional.mse_loss(x_hat, x, size_average=False)
        kl_loss = (
            -0.5 *
            (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1)
        ).mean(dim=0)

        loss = bce_loss + kl_loss
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer
