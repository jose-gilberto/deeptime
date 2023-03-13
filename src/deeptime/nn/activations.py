from __future__ import annotations

import torch
from torch import nn


class LiSin(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + x


class LiSiLU(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), torch.sin(x) + x)


class LeakySineLU(nn.Module):

    def __init__(self, alpha: float = 0.3, beta: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        negative_idx = x < 0
        x[negative_idx] = self.beta * torch.sin(self.alpha * x) + x
        x[~negative_idx] = torch.sin(self.alpha * x) + x
        return x
