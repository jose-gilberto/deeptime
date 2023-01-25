from __future__ import annotations

import os

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from torch import nn

from deeptime.data.datamodules import UcrDataModule
from deeptime.models.representation import LinearAutoEncoder

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
    'FordA', 'FordB', 'HandOutlines', 'ItalyPowerDemand',
    'MiddlePhalanxOutlineCorrect', 'Chinatown', 'FreezerRegularTrain',
    'FreezerSmallTrain', 'GunPointAgeSpan', 'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung', 'PowerCons', 'Coffee', 'Ham', 'Herring',
    'Lightning2', 'MoteStrain', 'PhalangesOutlinesCorrect',
    'ProximalPhalanxOutlineCorrect', 'ShapeletSim', 'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2', 'ToeSegmentation1', 'ToeSegmentation2',
    'HouseTwenty'
]

RECON_LOSSES = []

for dataset in DATASETS:
    print()
    print('#' * 50)
    print(f'Experiment with {dataset} dataset...')
    print('#' * 50)
    print()

    data_dir = 'C:\\Users\\medei\\Desktop\\Gilberto\\' +\
        'Projetos\\deeptime\\docs\\datasets'

    data_module = UcrDataModule(dataset_name=dataset, data_dir=data_dir)

    model = LinearAutoEncoder(
        input_dim=data_module.sequence_length,
        latent_dim=32
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        devices=-1,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=50,
                min_delta=0.001
            ),
            progress_bar
        ]
    )

    trainer.fit(model=model, datamodule=data_module)

    preds = trainer.predict(model, dataloaders=data_module.test_dataloader())
    recon_loss = 0.

    for batch in preds:
        (x_hat, z), x = batch
        recon_loss += nn.functional.mse_loss(x_hat, x).item()

    RECON_LOSSES.append(recon_loss)

    # Save the model weights
    model_weight_dir = os.path.join('../../pretrain/representation/', dataset)

    if not os.path.exists(model_weight_dir):
        os.mkdir(model_weight_dir)

    torch.save(
        model.state_dict(),
        os.path.join(model_weight_dir, 'linear_autoencoder.pt')
    )

metrics = pd.DataFrame({
    'dataset': DATASETS,
    'distance': RECON_LOSSES
})

metrics.to_csv('./ucr_linear_autoencoder.metrics.csv', index=False)
