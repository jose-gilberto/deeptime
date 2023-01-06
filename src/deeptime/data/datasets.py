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
from sktime.datasets import load_UCR_UEA_dataset
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

    def __init__(
        self,
        name: str,
        labeled: bool = True,
        split: str = 'train'
    ) -> None:
        x, y = UCRDataset.load_ucr_dataset(name=name, split=split)

        if labeled is False:
            y = None

        super().__init__(x=x, y=y)

    @staticmethod
    def load_ucr_dataset(
        name: str,
        split: str = 'train',
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Download and load an UCR Dataset from the sktime library.
        The dataset will be transformed into a array-like format compatible
        with the torch.Dataset implementation.

        See:
            - https://www.timeseriesclassification.com/dataset.php

        Args:
            - name (str): The dataset name from the timeseriesclassification
                table. Compatible with the sktime method to load the data.
            - split (str): The data split that comes from sktime lib, must be
                equals to 'train' or 'test'. Defaults to 'train'.
            - verbose (bool, optional): If verbose its True, then the method
                will output some steps to provide some way of logging to the
                user. Defaults to False.

        Returns:
            - (Tuple[ndarray, ndarray]): Returns
                a tuple with four positions, each one represents respectively
                x, y.
        """
        # TODO: implement some way of passing a logger object to the method
        # and if verbose its true use that logger to output the messages
        if verbose:
            print(f'Loading {name}/{split} dataset from sktime lib...')

        x, y = load_UCR_UEA_dataset(name=name, split='train')

        # Since the features from the sktime are interpreted as a Series
        # inside of a dataframe we have to manually convert them
        y = np.array(y, dtype=np.int32)

        # Convert the sktime Series like object into a ndarray
        x_trasformed = np.array([
            value[0].tolist() for value in x.values
        ])

        # Now X features have a shape like (n_instances, n_features)
        # We need to convert them to a (n_instances, 1, n_features)
        x = np.expand_dims(x_trasformed, axis=1)

        return x, y


class UEADataset(BaseDataset):

    def __init__(self, x: np.ndarray, y: np.ndarray = None) -> None:
        raise NotImplementedError
