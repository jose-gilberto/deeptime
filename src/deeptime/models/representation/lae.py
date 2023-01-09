"""
Linear autoencoder module.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from deeptime.models.utils import LinearBlock


class LinearAutoEncoder(nn.Module):
    """ Linear AutoEncoder.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.e = nn.Sequential(
            nn.Dropout(p=0.1),
            LinearBlock(
                input_dim=input_dim,
                output_dim=500,
                dropout=0.2
            ),
            LinearBlock(
                input_dim=500,
                output_dim=500,
                dropout=0.2
            ),
            LinearBlock(
                input_dim=500,
                output_dim=latent_dim,
                dropout=0.3
            )
        )

        self.d = nn.Sequential(
            LinearBlock(
                input_dim=latent_dim,
                output_dim=500,
                dropout=0.3
            ),
            LinearBlock(
                input_dim=500,
                output_dim=500,
                dropout=0.2,
            ),
            LinearBlock(
                input_dim=500,
                output_dim=input_dim,
                dropout=0.0
            )
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.e(x)
        return self.d(z), z
