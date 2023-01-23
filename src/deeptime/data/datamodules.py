
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from sktime.datasets import load_UCR_UEA_dataset
from torch.utils.data import DataLoader, random_split

from deeptime.data.datasets import BaseDataset


class UcrDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = './',
        batch_size: int = 32,
        filter_label: Optional[int] = None,
        train_size: float = 0.8,
        val_size: float = 0.2
    ) -> None:
        super().__init__()

        # The dataset name has to be compatible with the
        # TSC table available at:
        # https://www.timeseriesclassification.com/dataset.php
        self.dataset_name = dataset_name

        self.data_dir = data_dir
        self.filter_label = filter_label

        # To see about the impact of batch_size in the training phase
        # you can access the article
        # https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e
        # Kevin Shen, 2018.
        self.batch_size = batch_size

        assert train_size + val_size == 1

        self.train_size = train_size
        self.val_size = val_size

    def prepare_data(self) -> None:
        """ Download the dataset from the sktime library and loads into a
        data structure compatible with the transformations that will occur.

        NOTE: Pytorch Lightning ensure that the prepare_data is called only
        within a single process on CPU, so we can safely add the download
        logic within.
        """
        x_train, y_train = load_UCR_UEA_dataset(
            name=self.dataset_name, split='train')
        x_test, y_test = load_UCR_UEA_dataset(
            name=self.dataset_name, split='test')

        # Since the features from sktime are interpreted as an object
        # we need to convert the labels to a int32 format.
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)

        # Convert the sktime pd.Series object into a ndarray to represent
        # each observed instance of a time series
        x_train_transformed = np.array([
            value[0].tolist() for value in x_train.values
        ])
        x_test_transformed = np.array([
            value[0].tolist() for value in x_test.values
        ])

        # Now X features have a shape like (n_instances, n_features)
        # we need to convert them to a (n_instances, 1, n_features) format
        x_train = np.expand_dims(
            x_train_transformed, axis=1)
        x_test = np.expand_dims(
            x_test_transformed, axis=1)

        if self.filter_label:
            # Filter only the positive examples used to train the
            # one class models
            x_train = x_train[y_train == self.filter_label]
            y_train = y_train[y_train == self.filter_label]

            # Replace all the other classes with a negative label (1)
            # and positive instances with label 0
            y_test = np.array(
                [0 if label == self.filter_label
                 else 1 for label in y_test]
            )

        dataset_dir = os.path.join(
            self.data_dir,
            self.dataset_name
        )

        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

        # Save train data
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[2])
        np.savetxt(os.path.join(dataset_dir, 'x_train.np'), x_train)
        np.savetxt(os.path.join(dataset_dir, 'y_train.np'), y_train)
        # Save test data
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[2])
        np.savetxt(os.path.join(dataset_dir, 'x_test.np'), x_test)
        np.savetxt(os.path.join(dataset_dir, 'y_test.np'), y_test)

    def setup(self, stage: str) -> None:
        dataset_dir = os.path.join(
            self.data_dir,
            self.dataset_name
        )

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit':
            x_train = np.loadtxt(
                os.path.join(dataset_dir, 'x_train.np'),
                dtype=np.float64
            )
            x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
            y_train = np.loadtxt(
                os.path.join(dataset_dir, 'y_train.np'),
                dtype=np.int32
            )
            # Instanciate the full dataset
            full_dataset = BaseDataset(x=x_train, y=y_train)

            train_size = int(len(full_dataset) * self.train_size)
            val_size = len(full_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

        # Assign test dataset for use in dataloaders
        if stage == 'test':
            x_test = np.loadtxt(
                os.path.join(dataset_dir, 'x_test.np'),
                dtype=np.float64
            )
            x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
            y_test = np.loadtxt(
                os.path.join(dataset_dir, 'y_test.np'),
                dtype=np.int32
            )

            self.test_dataset = BaseDataset(x=x_test, y=y_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def teardown(self, stage: str) -> None:
        ...
