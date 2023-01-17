
from __future__ import annotations

from typing import Any, List, Tuple, Union, cast

import pytorch_lightning as pl
import torch
from torch import nn, optim

from deeptime.models.utils import Conv1dSamePadding, UpSample


class InceptionBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool,
        stride: int = 1,
        bottleneck_channels: int = 32,
        kernel_size: int = 41
    ) -> None:
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                bias=False
            )

        kernel_sizes = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = (
            bottleneck_channels if self.use_bottleneck else in_channels
        )
        channels = [start_channels] + [out_channels] * 3

        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=stride,
                bias=False
            ) for i in range(len(kernel_sizes))
        ])

        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_x = x

        if self.use_bottleneck:
            x = self.bottleneck(x)

        x = self.conv_layers(x)

        if self.use_residual:
            x = x + self.residual(org_x)

        return x


class InceptionAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        in_features: int,
        out_channels: Union[List[int], int],
        bottleneck_channels: Union[List[int], int],
        kernel_sizes: Union[List[int], int],
        use_residuals: Union[List[bool], bool, str] = 'default',
        latent_dim: int = 32,
    ) -> None:
        super().__init__()

        channels = (
            [in_channels] +
            cast(List[int], self._expand_to_blocks(out_channels, num_blocks))
        )

        bottleneck_channels = cast(
            List[int],
            self._expand_to_blocks(bottleneck_channels, num_blocks)
        )

        kernel_sizes = cast(
            List[int],
            self._expand_to_blocks(kernel_sizes, num_blocks)
        )

        if use_residuals == 'default':
            use_residuals = [
                True if i % 3 == 2 else False for i in range(num_blocks)
            ]
        use_residuals = cast(
            List[bool],
            self._expand_to_blocks(
                cast(Union[bool, List[bool]], use_residuals),
                num_blocks
            )
        )

        self.e_blocks = nn.Sequential(*[
            InceptionBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                residual=use_residuals[i],
                bottleneck_channels=bottleneck_channels[i],
                kernel_size=kernel_sizes[i]
            ) for i in range(num_blocks)
        ])

        self.e_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=in_features * out_channels,
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
                bias=False
            )
        )

        self.d_blocks = nn.Sequential(*[
            InceptionBlock(
                in_channels=channels[i + 1],
                out_channels=channels[i],
                residual=use_residuals[i],
                bottleneck_channels=bottleneck_channels[i],
                kernel_size=kernel_sizes[i]
            ) for i in range(num_blocks - 1, -1, -1)
        ])

        self.d_head = nn.Sequential(
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
                out_features=128 * in_features,
                bias=False,
            ),
            UpSample(out_channels=128, series_length=in_features),
        )

    @staticmethod
    def _expand_to_blocks(
        value: Union[int, bool, List[int], List[bool]],
        num_blocks: int
    ) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks

        return value

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.e_head(self.e_blocks(x))
        x_hat = self.d_blocks(self.d_head(z))
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

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=1e-4)
