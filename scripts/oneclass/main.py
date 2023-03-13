from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSVDD(pl.LightningModule):
    def __init__(self, nu=0.1, lr=1e-4):
        super().__init__()
        self.nu = nu
        self.lr = lr
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, padding=2),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True)
        )
        self.z = None
        self.c = None

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def deep_svdd_loss(self, x, z, c):
        rho = torch.abs(F.cosine_similarity(x - c, z - c))
        loss = torch.mean(torch.square(1 - rho))
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.unsqueeze(1)
        z = self.z
        c = self.c

        if z is None or c is None:
            z = self.encoder(x).detach()
            self.z = z
            c = z.mean(dim=0)
            self.c = c

        loss = self.deep_svdd_loss(self(x), z, c)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict(self, x):
        x = x.unsqueeze(1)
        z = self.z
        c = self.c
        nu = self.nu

        if z is None or c is None:
            raise ValueError('Model not yet trained')

        rho = torch.abs(F.cosine_similarity(self(x) - c, z - c))
        dist = torch.square(1 - rho)
        scores = torch.where(dist < nu, dist, nu * torch.ones_like(dist))
        return scores
