from __future__ import annotations

from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score
from torch import nn

from deeptime.nn.activations import LeakySineLU, LiSiLU, LiSin

activation_layers = {
    'relu': nn.ReLU,
    'lisin': LiSin,
    'lisilu': LiSiLU,
    'leakysinelu': LeakySineLU
}


class LinearClassifier(pl.LightningModule):

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        num_classes: int = 2,
        activation: str = 'relu',
        learning_rate: float = 1e-2
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=hidden_features
            ),
            activation_layers[activation](),
            nn.Linear(
                in_features=hidden_features,
                out_features=hidden_features
            ),
            activation_layers[activation](),
            nn.Linear(
                in_features=hidden_features,
                out_features=num_classes
            ),
            nn.Softmax()
        )

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)

        loss = self.criterion(y_pred, y)
        self.log(
            'train_loss',
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )

        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> None:
        x, y = batch
        y_ = np.array(torch.argmax(y, dim=1).tolist()) + 1

        y_pred = self(x)
        y_pred_ = np.array(torch.argmax(y_pred, dim=1).tolist()) + 1

        self.log('Accuracy Score', accuracy_score(y_, y_pred_))
