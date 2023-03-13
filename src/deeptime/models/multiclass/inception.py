from __future__ import annotations

from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch import nn

from deeptime.nn.activations import SinL

# from torch.nn import functional as F




activation_layers = {
    'sinl': SinL,
    'relu': nn.ReLU,
}


def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes


def pass_through(x: torch.Tensor) -> torch.Tensor:
    return x


class Inception(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int] = [9, 19, 39],
        bottleneck_channels: int = 32,
        activation: str = 'relu',
        return_indices: bool = False
    ) -> None:
        super().__init__()

        self.return_indices = return_indices

        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = nn.Sequential()
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False
        )

        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False
        )

        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False
        )

        self.max_pool = nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
            return_indices=return_indices
        )

        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.batch_norm = nn.BatchNorm1d(num_features=4 * out_channels)
        self.activation = activation_layers[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_bottleneck = self.bottleneck(x)

        if self.return_indices:
            z_maxpool, indices = self.max_pool(x)
        else:
            z_maxpool = self.max_pool(x)

        z1 = self.conv_from_bottleneck_1(z_bottleneck)
        z2 = self.conv_from_bottleneck_2(z_bottleneck)
        z3 = self.conv_from_bottleneck_3(z_bottleneck)
        z4 = self.conv_from_maxpool(z_maxpool)

        z = torch.cat([z1, z2, z3, z4], dim=1)
        z = self.activation(self.batch_norm(z))

        if self.return_indices:
            return z, indices
        return z


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        kernel_sizes: Tuple[int] = [9, 19, 39],
        bottleneck_channels: int = 32,
        use_residual: bool = True,
        activation: str = 'relu',
        return_indices: bool = False
    ) -> None:
        super().__init__()

        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.activation_function = activation_layers[self.activation]()

        self.inception_1 = Inception(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )

        self.inception_2 = Inception(
            in_channels=4 * out_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )

        self.inception_3 = Inception(
            in_channels=4 * out_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )

        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(num_features=4 * out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.return_indices:
            z, i1 = self.inception_1(x)
            z, i2 = self.inception_2(z)
            z, i3 = self.inception_3(z)
        else:
            z = self.inception_1(x)
            z = self.inception_2(z)
            z = self.inception_3(z)

        if self.use_residual:
            z = z + self.residual(x)
            z = self.activation_function(z)

        if self.return_indices:
            return z, [i1, i2, i3]
        return z


class Flatten(nn.Module):
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.out_dim = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, self.out_dim)


class InceptionTimeClassifier(pl.LightningModule):

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        activation: str = 'relu',
        learning_rate: float = 1e-3
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            InceptionBlock(
                in_channels=in_channels,
                out_channels=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=activation
            ),
            InceptionBlock(
                in_channels=32 * 4,
                out_channels=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=activation
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=n_classes),
            nn.Softmax(),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        # y = y.type(torch.LongTensor).to(self.device)

        y_pred = self(x)
        # y_pred = y_pred.type(torch.LongTensor).to(self.device)

        loss = self.criterion(y_pred, y)
        self.log(
            'train_loss',
            loss, prog_bar=True,
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
        # y = y.type(torch.LongTensor).to(self.device)

        y_pred = self(x)
        # y_pred = y_pred.type(torch.LongTensor).to(self.device)

        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> None:
        x, y = batch
        y_pred = torch.argmax(self(x), dim=1)
        y = torch.argmax(y, dim=1)

        labels = np.array(y.tolist())
        preds = np.array(y_pred.tolist())

        self.log('acc_score', accuracy_score(labels, preds))
        self.log('recall_score', recall_score(labels, preds))
        self.log('precision_score', precision_score(labels, preds))

        return
