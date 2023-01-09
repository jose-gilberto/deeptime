from __future__ import annotations

import torch
from torch import nn


class LinearBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=output_dim,
                bias=False
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
