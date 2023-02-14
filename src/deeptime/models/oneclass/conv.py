
from __future__ import annotations

import os
from typing import Any, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score
from torch import nn, optim

from deeptime.models.utils import Conv1dSamePadding  # GlobalAveragePooling,

# from deeptime.models.utils import UpSample


class ConvOCC(pl.LightningModule):

    def __init__(
        self,
        in_channels: int,
        in_features: int,
        latent_dim: int,
        optimizer: str = 'Adam',
        learning_rate: float = 1e-6,
        weight_decay: float = 0,
        radius: float = 0.
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.in_features = in_features

        self.latent_dim = latent_dim
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.R = torch.tensor(radius, device=self.device)

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

            # GlobalAveragePooling(),
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
                out_features=latent_dim,
                bias=False
            ),
            nn.Tanh()
        )

    def _init_center(
        self,
        data_loader,
        eps: float = 0.1
    ):
        n_samples = 0
        c = torch.zeros(self.latent_dim, device=self.device)

        self.eval()

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                z = self(x)

                n_samples += 1
                c += torch.sum(z, dim=0)

        c /= n_samples

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) * (c > 0)] = eps

        self.center = c

    def load_pretrained_weights(self, path: str):
        if not os.path.exists(path):
            raise RuntimeError()
        self.load_state_dict(torch.load(path), strict=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.e(x)
        return z

    def _get_oneclass_loss(
        self,
        batch: Any,
        return_representations: bool = False
    ):
        x, _ = batch

        z = self(x)

        self.center = self.center.to(self.device)
        self.R = self.R.to(self.device)

        # z dimensions = [n_instances, latent_dim]
        distance = torch.sum((z - self.center) ** 2, dim=1)
        scores = distance - self.R ** 2
        loss = torch.mean(torch.max(torch.exp(scores), scores + 1))

        if return_representations:
            return loss, z

        return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        z = self(x)
        self.center = self.center.to(self.device)
        distances = torch.sum((z - self.center) ** 2, dim=1)
        scores = distances - self.R ** 2

        return torch.tensor(
            [1 if score <= 0 else -1 for score in scores]
        ).to(self.device)

    # def _get_reconstruction_loss(self, batch: Any) -> torch.Tensor:
    #     x, _ = batch
    #     x_hat, _ = self(x)
    #     loss = nn.functional.mse_loss(x_hat, x)
    #     return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # recon_loss = self._get_reconstruction_loss(batch)
        occ_loss = self._get_oneclass_loss(batch)
        self.log('train_loss', occ_loss, prog_bar=True)
        return occ_loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # loss, z = self._get_oneclass_loss(batch, return_representations=True)
        x, labels = batch
        pred = self.predict(x)

        pred = np.array(pred.tolist())
        labels = np.array(labels.tolist())

        # self.log('val_loss', loss, prog_bar=True)
        self.log('test_f1_score', f1_score(labels, pred))
        return

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self._get_oneclass_loss(batch)

        x, labels = batch
        pred = self.predict(x)

        pred = np.array(pred.tolist())
        labels = np.array(labels.tolist())

        self.log('val_loss', loss)
        self.log('val_f1_score', f1_score(labels, pred), prog_bar=True)

        return loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.learning_rate)
