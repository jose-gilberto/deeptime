
from __future__ import annotations

from typing import Any, Tuple

import pytorch_lightning as pl
import torch
from torch import nn, optim

from deeptime.models.utils import Conv1dSamePadding  # GlobalAveragePooling,
from deeptime.models.utils import UpSample


class ConvAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        in_channels: int,
        in_features: int,
        latent_dim: int,
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

        self.e = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=128,
                kernel_size=8,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            Conv1dSamePadding(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            Conv1dSamePadding(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            # GlobalAveragePooling(),
            nn.Flatten(),

            nn.Linear(
                in_features=128 * in_features,
                out_features=256,
                bias=False
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=128,
                bias=False
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=latent_dim,
                bias=False
            )
        )

        self.d = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=128,  # Must be equals to original series size
                bias=False
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=256,
                bias=False
            ),
            nn.ReLU(),
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
            nn.ReLU(),
            Conv1dSamePadding(
                in_channels=256,
                out_channels=128,
                kernel_size=5,
                bias=False
            ),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
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

    def validation_step(self, *args: Any, **kwargs: Any):
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any):
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.learning_rate)
