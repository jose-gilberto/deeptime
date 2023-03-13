from __future__ import annotations

from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch import nn

from deeptime.models.utils import Conv1dSamePadding
from deeptime.nn.activations import LeakySineLU, LiSiLU, LiSin

activation_layers = {
    'relu': nn.ReLU,
    'lisin': LiSin,
    'lisilu': LiSiLU,
    'leakysinelu': LeakySineLU
}


class ResNetConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: str = 'relu',
        bias: bool = True
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias
            ),
            nn.BatchNorm1d(num_features=out_channels),
            activation_layers[activation](),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResNetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = 'relu',
    ) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ResNetConvBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=1,
                bias=bias,
                activation=activation
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
                    bias=bias,
                ),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


class ResNetClassifier(pl.LightningModule):

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 32,
        num_classes: int = 2,
        activation: str = 'relu',
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.layers = nn.Sequential(*[
            ResNetBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation=activation,
            ),
            ResNetBlock(
                in_channels=mid_channels,
                out_channels=mid_channels * 2,
                activation=activation,
            ),
            ResNetBlock(
                in_channels=mid_channels * 2,
                out_channels=mid_channels * 2,
                activation=activation,
            ),
        ])

        self.final = nn.Sequential(
            nn.Linear(mid_channels * 2, num_classes),
            nn.Softmax()
        )

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 1e-3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.final(x.mean(dim=-1))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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

    def validation_step(
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
        # self.log('Precision Score', precision_score(y_, y_pred_))
        # self.log('Recall Score', recall_score(y_, y_pred_))
