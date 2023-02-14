from __future__ import annotations

import torch
from torch import nn


def one_class_loss(
    z: torch.Tensor,
    c: torch.Tensor,
    r: torch.Tensor,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    alpha_recon: float = 1.,
    alpha_ps: float = 1.,
    use_ps: bool = True
) -> torch.Tensor:
    """_summary_

    Args:
        z (torch.Tensor): _description_
        c (torch.Tensor): _description_
        r (torch.Tensor): _description_
        x (torch.Tensor): _description_
        x_hat (torch.Tensor): _description_
        alpha_recon (float, optional): _description_. Defaults to 1..
        alpha_ps (float, optional): _description_. Defaults to 1..
        use_ps (bool, optional): _description_. Defaults to True.

    Returns:
        torch.Tensor: _description_
    """
    if use_ps:
        return (
            svdd_loss(z, c, r) +
            positive_sample_loss(z, c, r, alpha_ps) +
            nn.functional.mse_loss(x, x_hat) * alpha_recon
        )

    return svdd_loss(z, c, r) + nn.functional.mse_loss(x, x_hat) * alpha_recon


def svdd_loss(
    z: torch.Tensor,
    c: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """_summary_

    Args:
        z (torch.Tensor): _description_
        c (torch.Tensor): _description_
        r (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    distances = torch.sum((z - c) ** 2, dim=1)
    scores = distances - r ** 2

    l_svdd = torch.max(torch.zeros_like(scores), scores)

    return torch.mean(l_svdd)


def positive_sample_loss(
    z: torch.Tensor,
    c: torch.Tensor,
    r: torch.Tensor,
    alpha: float = 1.,
) -> torch.Tensor:
    """Applies the modified Exponential Linear Unit function, element-wise.
    The result is an modified version of ELU calculus that implies in:

    $$
    L_{ps} =
    $$

    Args:
        - z (torch.Tensor): The representation that comes from an NN Model.
        - c (torch.Tensor): The torch array-like representing the center
            of the hypersphere.
        - r (torch.Tensor): The torch array-like with the radius value for
            the hypersphere.
        - alpha (float, optional): The ratio number that will be multiplied
            by the exp(x) in representations that are inside the hypersphere.
            Defaults to 1.

    NOTE: test the impact of that loss in a backprop, since the weights can
    be too close to 0, the network can overfit if the epochs goes too far.
    """
    distances = torch.sum((z - c) ** 2, dim=1)
    # A negative score implies that we have an output, z, inside
    # of the hypersphere. A bigger negative number implies that the
    # distance from the point to the center is bigger, in that case
    # the point is more closer to the hypersphere center (c)
    scores = distances - r ** 2
    # When we have a negative number we only replace them with a zero
    l_ps = torch.max(torch.zeros_like(scores), scores)
    # Now we have a mask for all 0 values in the loss for positive samples
    mask = l_ps == 0

    l_ps[~mask] = scores + 1
    l_ps[mask] = torch.exp(scores) * alpha

    return torch.mean(l_ps)
