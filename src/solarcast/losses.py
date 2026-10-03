"""Loss functions. Each one encodes a different view of which errors hurt most."""
from __future__ import annotations

import torch


def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 weights: torch.Tensor | None = None) -> torch.Tensor:
    """MSE with an optional weight per horizon (e.g. care more about near days)."""
    err = (pred - target) ** 2
    if weights is not None:
        err = err * weights
    return err.mean()


def pseudo_huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Quadratic for small errors, linear for large ones: less sensitive to outliers."""
    err = pred - target
    return (delta ** 2 * (torch.sqrt(1 + (err / delta) ** 2) - 1)).mean()


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, q: float = 0.5) -> torch.Tensor:
    """Pinball loss. q > 0.5 punishes under-prediction more, q < 0.5 over-prediction."""
    err = target - pred
    return torch.maximum(q * err, (q - 1) * err).mean()
