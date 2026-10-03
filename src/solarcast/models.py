"""Neural network definitions only. Training loops live in train.py."""
from __future__ import annotations

import torch
from torch import nn

N_OTHER = 7   # solar_t, day_sin, day_cos and four rolling statistics
N_LAGS = 7    # solar_t_minus_1 ... solar_t_minus_7


class MultiOutputMLP(nn.Module):
    """One network predicts all horizons at once."""

    def __init__(self, n_features: int = 14, n_outputs: int = 7,
                 hidden: tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers, width = [], n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionMLP(nn.Module):
    """Self-attention over the seven lag values, then an MLP.

    The lags are treated as a short sequence so the network can learn which
    past days matter most; the remaining features bypass the attention block.
    """

    def __init__(self, n_outputs: int = 7, attention_dim: int = 16,
                 hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        self.dk = attention_dim
        self.embed = nn.Linear(1, attention_dim)
        self.w_q = nn.Linear(attention_dim, attention_dim)
        self.w_k = nn.Linear(attention_dim, attention_dim)
        self.w_v = nn.Linear(attention_dim, attention_dim)
        layers, width = [], attention_dim + N_OTHER
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_outputs))
        self.mlp = nn.Sequential(*layers)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        other, lags = x[:, :N_OTHER], x[:, N_OTHER:]
        emb = self.embed(lags.unsqueeze(-1))                       # (B, 7, dk)
        scores = self.w_q(emb) @ self.w_k(emb).transpose(-2, -1) / self.dk ** 0.5
        attn = torch.softmax(scores, dim=-1)                       # (B, 7, 7)
        self.last_attention = attn.detach()
        pooled = (attn @ self.w_v(emb)).mean(dim=1)                # (B, dk)
        return self.mlp(torch.cat([pooled, other], dim=-1))


def build_model(name: str, mcfg: dict, n_features: int = 14, n_outputs: int = 7) -> nn.Module:
    if name == "mlp":
        return MultiOutputMLP(n_features, n_outputs, tuple(mcfg["hidden"]), mcfg["dropout"])
    if name == "attention":
        return AttentionMLP(n_outputs, mcfg["attention_dim"], dropout=mcfg["dropout"])
    raise ValueError(f"Unknown architecture: {name}")
