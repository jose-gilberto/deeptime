from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from sktime.datasets import load_UCR_UEA_dataset
from torch.utils.data import DataLoader

from deeptime.data import BaseDataset
from deeptime.models.representation import LinearVariationalAutoEncoder

warnings.filterwarnings('ignore')


class Plot2DCallback(pl.Callback):

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        plt.figure(figsize=(10, 10))

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        colors = []

        representations = []
        for x, _ in trainer.train_dataloader:
            x = x.view(x.shape[0], -1)
            _, z, _ = pl_module(x.to('cuda'))
            representations.extend(z.tolist())

        for _ in representations:
            colors.append('lightseagreen')

        test_representations = []
        for x, _ in trainer.val_dataloaders[0]:
            x = x.view(x.shape[0], -1)
            _, z, _ = pl_module(x.to('cuda'))
            test_representations.extend(z.tolist())

        for _ in test_representations:
            colors.append('darkslategray')

        representations.extend(test_representations)
        representations = np.array(representations)

        plt.scatter(representations[:, 0], representations[:, 1], c=colors)

        plt.savefig(os.path.join(self.path, f'{trainer.current_epoch}.png'))
        plt.close()


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
LATENT_DIM = 2
LABEL = 1
DATASET = 'Yoga'
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

print('#' * 50)
print(f'Experiment with {DATASET} dataset...')
print('#' * 50)
print()

x_train, y_train = load_UCR_UEA_dataset(name=DATASET, split='train')
x_test, y_test = load_UCR_UEA_dataset(name=DATASET, split='test')

sequence_length = x_train.values[0][0].shape[0]

y_test = np.array(y_test, dtype=np.int32)
y_train = np.array(y_train, dtype=np.int32)

# Filter only the class of interest
x_train = x_train[y_train == LABEL]
y_train = y_train[y_train == LABEL]

x_train_transformed = np.array([
    value[0].tolist() for value in x_train.values
])

x_test_transformed = np.array([
    value[0].tolist() for value in x_test.values
])

x_train = np.expand_dims(x_train_transformed, axis=1)
x_test = np.expand_dims(x_test_transformed, axis=1)

train_dataset = BaseDataset(x=x_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=32)

test_dataset = BaseDataset(x=x_test, y=y_test)
test_loader = DataLoader(test_dataset, batch_size=32)

model = LinearVariationalAutoEncoder(
    input_dim=sequence_length,
    latent_dim=LATENT_DIM
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
        ),
        progress_bar,
        Plot2DCallback(path='./plots/linear_variational_autoencoder/')
    ]
)

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=test_loader
)

model_weight_dir = os.path.join(
    '../../pretrain/representation/',
    DATASET
)

if not os.path.exists(model_weight_dir):
    os.mkdir(model_weight_dir)

torch.save(
    model.state_dict(),
    os.path.join(
        model_weight_dir,
        f'linear_variational_autoencoder-dim={LATENT_DIM}-l={LABEL}.pt'
    )
)
