from __future__ import annotations

import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from sktime.datasets import load_UCR_UEA_dataset
from torch.utils.data import DataLoader

from deeptime.data import BaseDataset
from deeptime.models.representation import ConvAutoEncoder

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

LATENT_DIM = 32

DATASETS = [
    'Yoga',
    # 'WormsTwoClass',
    # 'Wine',
    # 'Wafer',
    # 'TwoLeadECG',
    # 'Strawberry',
    # 'SemgHandGenderCh2',
    # 'BeetleFly',
    # 'BirdChicken',
    # 'Computers',
    # 'DistalPhalanxOutlineCorrect',
    # 'Earthquakes',
    # 'ECG200',
    # 'ECGFiveDays',
    # 'FordA',
    # 'FordB',
    # 'HandOutlines',
    # 'ItalyPowerDemand',
    # 'MiddlePhalanxOutlineCorrect',
    # 'Chinatown',
    # 'FreezerRegularTrain',
    # 'FreezerSmallTrain',
    # 'GunPointAgeSpan',
    # 'GunPointMaleVersusFemale',
    # 'GunPointOldVersusYoung',
    # 'PowerCons',
    # 'Coffee',
    # 'Ham',
    # 'Herring',
    # 'Lightning2',
    # 'MoteStrain',
    # 'PhalangesOutlinesCorrect',
    # 'ProximalPhalanxOutlineCorrect',
    # 'ShapeletSim',
    # 'SonyAIBORobotSurface1',
    # 'SonyAIBORobotSurface2',
    # 'ToeSegmentation1',
    # 'ToeSegmentation2',
    # 'HouseTwenty'
]

for dataset in DATASETS:
    print()
    print('#' * 50)
    print(f'Experiment with {dataset} dataset...')
    print('#' * 50)
    print()

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

    train_dataset = BaseDataset(x=x_train, y=y_train)
    train_loader = DataLoader(train_dataset, batch_size=32)

    test_dataset = BaseDataset(x=x_test, y=y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = ConvAutoEncoder(
        in_channels=1,
        in_features=sequence_length,
        latent_dim=LATENT_DIM,
    )

    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator='gpu',
        devices=-1,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=50,
                min_delta=0.1
            ),
            progress_bar
        ]
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )

    model_weight_dir = os.path.join(
        '../../pretrain/representation/',
        dataset
    )

    if not os.path.exists(model_weight_dir):
        os.mkdir(model_weight_dir)

    torch.save(
        model.state_dict(),
        os.path.join(model_weight_dir, f'conv_autoencoder-l{LATENT_DIM}.pt')
    )
