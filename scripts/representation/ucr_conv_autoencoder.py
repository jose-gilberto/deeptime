from __future__ import annotations

import os
import warnings

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

ACTIVATION = 'sinl'
DATASET = 'ECGFiveDays'
LABEL = 1
LATENT_DIM = 32


print('#' * 50)
print(f'Experiment with {DATASET} dataset...')
print('#' * 50)
print()

# Loading data
x_train, y_train = load_UCR_UEA_dataset(name=DATASET, split='train')
x_test, y_test = load_UCR_UEA_dataset(name=DATASET, split='test')

sequence_length = x_train.values[0][0].shape[0]

y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)

# Filter only the class of interest
x_train = x_train[y_train == LABEL]
y_train = y_train[y_train == LABEL]

# Transform features to the correct format
x_train_transformed = np.array([
    value[0].tolist() for value in x_train.values
])
x_test_transformed = np.array([
    value[0].tolist() for value in x_test.values
])

# Expand the 3th dim to convolutional neural networks
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
    activation=ACTIVATION
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
        ),
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
        f'conv_autoencoder-dim={LATENT_DIM}-l={LABEL}.pt'
    )
)
