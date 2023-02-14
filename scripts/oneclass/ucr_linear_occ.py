from __future__ import annotations

import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import RichProgressBar  # EarlyStopping,
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from sktime.datasets import load_UCR_UEA_dataset
# from torch import nn
from torch.utils.data import DataLoader

# from deeptime.data.datamodules import UcrDataModule
from deeptime.data import BaseDataset
from deeptime.models.oneclass.linear import LaeOCC

# import pandas as pd





# Ignore all PyTorch UserWarnings
warnings.filterwarnings('ignore')

progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description='green_yellow',
        progress_bar='green1',
        progress_bar_finished='green1',
        progress_bar_pulse='#6206E0',
        batch_progress='green_yellow',
        time='grey82',
        processing_speed='grey82',
        metrics='grey82',
    )
)

DATASETS = [
    'Yoga', 'WormsTwoClass', 'Wine', 'Wafer', 'TwoLeadECG',
    'Strawberry', 'SemgHandGenderCh2', 'BeetleFly', 'BirdChicken', 'Computers',
    'DistalPhalanxOutlineCorrect',  'Earthquakes', 'ECG200', 'ECGFiveDays',
    'FordA', 'FordB'
]

for dataset in DATASETS:
    print()
    print('#' * 50)
    print(f'Experiment with {dataset} dataset...')
    print('#' * 50)
    print()

    # data_dir = 'C:\\Users\\medei\\Desktop\\Gilberto\\' +\
    #     'Projetos\\deeptime\\docs\\datasets'

    x_train, y_train = load_UCR_UEA_dataset(name=dataset, split='train')
    x_test, y_test = load_UCR_UEA_dataset(name=dataset, split='test')
    sequence_length = x_train.values[0][0].shape[0]

    x_train_transformed = np.array([
        value[0].tolist() for value in x_train.values
    ])
    y_train = np.array(y_train, dtype=np.int32)

    x_test_transformed = np.array([
        value[0].tolist() for value in x_test.values
    ])
    y_test = np.array(y_test, dtype=np.int32)

    x_train = np.expand_dims(x_train_transformed, axis=1)
    x_test = np.expand_dims(x_test_transformed, axis=1)

    for label in np.unique(y_train):
        x_train = x_train[y_train == label]
        y_train = y_train[y_train == label]

        y_test = np.array([
            0 if y == label else 1 for y in y_test
        ])

        train_dataset = BaseDataset(x=x_train, y=y_train)
        train_loader = DataLoader(train_dataset, batch_size=32)

        model = LaeOCC(
            input_dim=sequence_length,
            latent_dim=2,
            radius=.35
        )

        trainer = pl.Trainer(
            max_epochs=1000,
            accelerator='gpu',
            devices=-1,
            callbacks=[
                # EarlyStopping(
                #     monitor='train_loss',
                #     mode='min',
                #     patience=50,
                #     min_delta=0.1
                # ),
                progress_bar
            ]
        )

        trainer.fit(model, train_dataloaders=train_loader)

        dataset_dir = os.path.join('../../pretrain/oneclass/', dataset)
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

        label_dir = os.path.join(dataset_dir, str(label))
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        model_dir = os.path.join(label_dir, 'laeocc.pth')
        torch.save(model.state_dict(), model_dir)
        break
    break
