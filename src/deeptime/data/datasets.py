""" That module will define some dataset classes.

Each dataset will handle a specific case of loading data
into a format that allow us to use a PyTorch model.

The dataset will implement the structure defined at
torch.Dataset to be compatible with any PyTorch data flow.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """ Base Dataset class that only implements the basic
    structure of a dataset that will be used by other classes.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray = None) -> None:

        assert x.shape[0] == y.shape[0], 'The data and labels must ' \
            'have the same number of instances.'

        self.x = x
        self.y = y

    def __getitem__(
        self,
        index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.y:
            return (
                torch.from_numpy(self.x[index]).float(),
                torch.from_numpy(self.y[index]).float()
            )

        return torch.from_numpy(self.x[index]).float()

    def __len__(self) -> int:
        return len(self.x)


class UCRDataset(BaseDataset):

    def __init__(self, x: np.ndarray, y: np.ndarray = None) -> None:
        raise NotImplementedError


class UEADataset(BaseDataset):

    def __init__(self, x: np.ndarray, y: np.ndarray = None) -> None:
        raise NotImplementedError
