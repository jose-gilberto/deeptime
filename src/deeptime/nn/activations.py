from __future__ import annotations

import torch
from torch import nn


class SineLU(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.sin(x) ** 2) + x


# class LiSiLU(nn.Module):

#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.max(torch.zeros_like(x), torch.sin(x) + x)


class LeakySineLU(nn.Module):

    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(
            self.alpha * ((torch.sin(x) ** 2) + x),
            (torch.sin(x) ** 2) + x
        )
