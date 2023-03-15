from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def plot_distributions(
    val_dicts: Dict,
    color: str = 'C0',
    xlabel: str = None,
    stat: str = 'count',
    use_kde: bool = True
) -> plt.Figure:
    columns = len(val_dicts)
    fig, ax = plt.subplots(1, columns, figsize=(columns * 5, 4))
    fig_index = 0
    for key in sorted(val_dicts.keys()):
        key_ax = ax[fig_index % columns]
        sns.histplot(
            val_dicts[key],
            ax=key_ax,
            color=color,
            bins=50,
            stat=stat,
            kde=use_kde and (
                (val_dicts[key].max() - val_dicts[key].min()) >
                1e-8
            ),
        )  # Only plot kde if there is variance
        hidden_dim_str = (
            r'(%i $\to$ %i)' %
            (val_dicts[key].shape[1], val_dicts[key].shape[0])
            if len(val_dicts[key].shape) > 1 else ''
        )
        key_ax.set_title(f'{key} {hidden_dim_str}')
        if xlabel is not None:
            key_ax.set_xlabel(xlabel)
        fig_index += 1
    fig.subplots_adjust(wspace=0.4)
    return fig


def visualize_weight_distribution(
    model: torch.nn.Module,
    color: str = 'C0',
) -> None:
    weights = {}
    for name, param in model.named_parameters():
        if name.endswith('.bias'):
            continue
        key_name = f'Layer {name.split(".")[1]}'
        weights[key_name] = param.detach().view(-1).cpu().numpy()

    fig = plot_distributions(weights, color=color, xlabel='Weight vals')
    fig.suptitle('Weight distribution', fontsize=14, y=1.05)
    plt.show()
    plt.close()


def visualize_gradients(
    model: nn.Module,
    train_dataset: Dataset,
    device: torch.device = torch.device('cpu'),
    color: str = 'C0',
    print_variance: bool = False,
) -> None:
    model.eval()
    small_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
    sequences, labels = next(iter(small_loader))
    sequences, labels = sequences.to(device), labels.to(device)

    model.zero_grad()
    preds = model(sequences)
    loss = torch.nn.functional.cross_entropy(preds, labels)
    loss.backward()

    grads = {
        name: params.grad.view(-1).cpu().clone().numpy()
        for name, params in model.named_parameters()
        if 'weight' in name
    }

    model.zero_grad()

    fig = plot_distributions(grads, color=color, xlabel='Grad magnitude')
    fig.suptitle('Gradient distribution', fontsize=14, y=1.05)
    plt.show()
    plt.close()

    if print_variance:
        for key in sorted(grads.keys()):
            print(f'{key} - Variance: {np.var(grads[key])}')


def visualize_activations(
    model: nn.Module,
    train_dataset: Dataset,
    device: torch.device = torch.device('cpu'),
    color: str = 'C0',
    print_variance: bool = False
) -> None:
    model.eval()
    small_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
    sequences, labels = next(iter(small_loader))
    sequences, labels = sequences.to(device), labels.to(device)

    feats = sequences.view(sequences.shape[0], -1)
    activations = {}
    with torch.no_grad():
        for layer_index, layer in enumerate(model.layers):
            feats = layer(feats)
            if isinstance(layer, nn.Linear):
                activations[f"Layer {layer_index}"] = (
                    feats.view(-1).detach().cpu().numpy()
                )

    fig = plot_distributions(
        activations,
        color=color,
        stat="density",
        xlabel="Activation vals"
    )
    fig.suptitle("Activation distribution", fontsize=14, y=1.05)
    plt.show()
    plt.close()

    if print_variance:
        for key in sorted(activations.keys()):
            print(f"{key} - Variance: {np.var(activations[key])}")
