from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv1d_same_padding(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: torch.Tensor,
    dilation: torch.Tensor,
    groups: torch.Tensor,
):
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = x.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)

    if padding % 2 != 0:
        x = F.pad(x, [0, 1])

    return F.conv1d(
        input=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding // 2,
        dilation=dilation,
        groups=groups
    )


class Conv1dSamePadding(nn.Conv1d):
    """ Implement the 'same' padding functionality from Tensorflow.

    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv1d_same_padding(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.dilation,
            self.groups
        )


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
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
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GlobalAveragePooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)


class UpSample(nn.Module):
    def __init__(self, out_channels: int, series_length: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.series_length = series_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(
            x.shape[0],
            self.out_channels,
            self.series_length
        )
