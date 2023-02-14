from __future__ import annotations

import os
import warnings

# import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from sktime.datasets import load_UCR_UEA_dataset
# from torch import nn
from torch.utils.data import DataLoader

from deeptime.data import BaseDataset
from deeptime.models.representation import LinearAutoEncoder

# import matplotlib.pyplot as plt

# from tqdm import tqdm




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

RECON_LOSSES = []

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

    model = LinearAutoEncoder(
        input_dim=sequence_length,
        latent_dim=32,
        # log_first=True
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        devices=-1,
        callbacks=[
            progress_bar,
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ]
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )

    # preds = trainer.predict(model, dataloaders=data_module.test_dataloader())

    # recon_loss = 0.
    # n_samples = 0.

    # for batch in preds:
    #     (x_hat, z), x = batch
    #     n_samples += x_hat.shape[0]
    #     recon_loss += nn.functional.mse_loss(x_hat, x).item()

    # RECON_LOSSES.append(recon_loss / n_samples)

    # Saving the plots to make the gif

    # for i, x in tqdm(enumerate(model.xs)):
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(list(range(sequence_length)), x, color='blue')
    #     plt.plot(list(range(sequence_length)), model.xhats[i], color='red')
    #     plt.savefig(f'./logs/yoga.{i}.png')
    #     plt.close()

    # Save the model weights
    model_weight_dir = os.path.join('../../pretrain/representation/', dataset)

    if not os.path.exists(model_weight_dir):
        os.mkdir(model_weight_dir)

    torch.save(
        model.state_dict(),
        os.path.join(model_weight_dir, 'linear_autoencoder.pt')
    )

# metrics = pd.DataFrame({
#     'dataset': DATASETS,
#     'distance': RECON_LOSSES
# })

# metrics.to_csv('./ucr_linear_autoencoder.metrics.csv', index=False)
