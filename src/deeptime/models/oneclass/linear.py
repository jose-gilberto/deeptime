from __future__ import annotations

import os
from typing import Any, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch import nn, optim

from deeptime.nn.activations import SinL

activations_layers = {
    'relu': nn.ReLU,
    'swish': nn.SiLU,
    'tanh': nn.Tanh,
    'leakyrelu': nn.LeakyReLU,
    'sinl': SinL,
}


class LinearOCC(pl.LightningModule):
    """ Linear AutoEncoder for One Class Classification with
    Time Series data.

    Args:
        input_dim (int): Input Time Series length.
        latent_dim (int): Latent dimension length.
        optim (str, optional): _description_. Defaults to 'Adam'.
        learning_rate (float, optional): _description_. Defaults to 1e-4.
        radius (float, optional): _description_. Defaults to 0.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: str = 'relu',
        optim: str = 'Adam',
        learning_rate: float = 1e-6,
        radius: float = 0
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        activation_params = {}

        if activation == 'leakyrelu':
            activation_params = {
                'negative_slope': 0.1
            }

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

        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.R = torch.tensor(radius, device=self.device)

        self.optimizer_name = optim
        self.learning_rate = learning_rate

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
        batch: Tuple[torch.Tensor, torch.Tensor],
        return_representations: bool = False
    ) -> torch.Tensor:
        x, _ = batch

        z = self(x)

        self.center = self.center.to(self.device)
        self.R = self.R.to(self.device)

        # z dimensions = [n_instances, latent_dim]
        distance = torch.sum((z - self.center) ** 2, dim=1)
        scores = distance - self.R ** 2
        # loss = torch.mean(torch.max(torch.exp(scores), scores + 1))
        loss = torch.mean(torch.max(torch.zeros_like(scores), scores))

        if return_representations:
            return loss, z

        return loss

    # def _get_reconstruction_loss(
    #     self,
    #     batch: Tuple[torch.Tensor, torch.Tensor]
    # ) -> torch.Tensor:
    #     x, _ = batch
    #     x = x.view(x.shape[0], -1)

    #     x_hat, z = self(x)
    #     loss = nn.functional.mse_loss(x_hat, x)

    #     return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        z = self(x)
        self.center = self.center.to(self.device)
        distances = torch.sum((z - self.center) ** 2, dim=1)
        scores = distances - self.R ** 2

        return torch.tensor(
            [1 if score <= 0 else -1 for score in scores]
        ).to(self.device)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # recon_loss = self._get_reconstruction_loss(batch)
        occ_loss = self._get_oneclass_loss(batch)

        self.log('train_loss', occ_loss, prog_bar=True)

        return occ_loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # loss, z = self._get_oneclass_loss(batch, return_representations=True)
        x, labels = batch
        pred = self.predict(x)

        pred = np.array(pred.tolist())
        labels = np.array(labels.tolist())

        # self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_score', f1_score(labels, pred), prog_bar=True)
        return

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, labels = batch
        pred = self.predict(x)

        # Convert to numpy array like structures
        pred = np.array(pred.tolist())
        labels = np.array(labels.tolist())

        self.log('test_accuracy', accuracy_score(labels, pred))
        self.log('test_f1_score', f1_score(labels, pred))
        self.log('test_recall_score', recall_score(labels, pred))
        self.log('test_precision_score', precision_score(labels, pred))

    # def predict_step(
    #     self,
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx: int = 0
    # ) -> Any:
    #     x, y = batch
    #     _, z = self(x)
    #     return z, y

    def configure_optimizers(self) -> optim.Optimizer:
        if self.optimizer_name == 'Adam':
            return optim.Adam(
                self.parameters(),
                lr=self.learning_rate
            )
        raise NotImplementedError()
