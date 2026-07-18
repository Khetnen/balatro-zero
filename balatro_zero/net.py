"""Policy/value network: flat features + center-key embeddings.

Heads:
  - policy:   logits over the flat Discrete(500) action slots
  - win:      P(run is won), sigmoid
  - progress: predicted best run progress (state.progress), sigmoid

Joker/consumable/market identity enters through a shared learned embedding
table over the ~300 center keys, pooled (masked mean + max) per group and
fused with the flat-feature torso. This is the v3 fix for the v1/v2
plateau: with identity only as scalar features, the net could not rank
which joker/shop purchase is good, capping play at the no-economy ceiling.

Search optimizes a blend of the two scalar heads (combined_value); early
in training P(win) is ~0 everywhere, so the progress head supplies the
gradient — a value-space curriculum instead of reward shaping.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from balatro_zero.state import MAX_ACTIONS, N_EMBED, OBS_DIM, Obs, stack_obs

WIN_WEIGHT = 0.5
PROGRESS_WEIGHT = 0.5

EMBED_DIM = 32


def _pool(emb: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Masked mean+max pool over the slot dimension -> [B, 2*E]."""
    mask = (ids > 0).unsqueeze(-1).float()          # [B, S, 1]
    count = mask.sum(dim=1).clamp(min=1.0)          # [B, 1]
    mean = (emb * mask).sum(dim=1) / count
    neg = torch.where(mask > 0, emb, torch.full_like(emb, -1e9))
    mx = neg.max(dim=1).values
    mx = torch.where(count > 0.5, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = MAX_ACTIONS,
                 hidden: int = 512, depth: int = 3, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        self.torso = nn.Sequential(*layers)

        self.embed = nn.Embedding(N_EMBED, embed_dim, padding_idx=0)
        pooled = 2 * embed_dim * 3  # mean+max for jokers / consumables / market
        self.fuse = nn.Sequential(
            nn.Linear(hidden + pooled, hidden), nn.LayerNorm(hidden), nn.ReLU()
        )

        self.policy = nn.Linear(hidden, n_actions)
        self.win = nn.Linear(hidden, 1)
        self.progress = nn.Linear(hidden, 1)

    def forward(
        self,
        flat: torch.Tensor,
        joker_ids: torch.Tensor,
        consumable_ids: torch.Tensor,
        market_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.torso(flat)
        pooled = torch.cat(
            [
                _pool(self.embed(joker_ids), joker_ids),
                _pool(self.embed(consumable_ids), consumable_ids),
                _pool(self.embed(market_ids), market_ids),
            ],
            dim=-1,
        )
        h = self.fuse(torch.cat([h, pooled], dim=-1))
        return (
            self.policy(h),
            torch.sigmoid(self.win(h)).squeeze(-1),
            torch.sigmoid(self.progress(h)).squeeze(-1),
        )


def combined_value(p_win: np.ndarray, prog: np.ndarray) -> np.ndarray:
    """Scalar value in [0,1] that search maximizes."""
    return WIN_WEIGHT * p_win + PROGRESS_WEIGHT * prog


@torch.no_grad()
def evaluate(
    net: PolicyValueNet, obs: Obs | list[Obs], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Batch-evaluate observations -> (logits [B, A], value [B])."""
    batch = stack_obs([obs] if isinstance(obs, Obs) else obs)
    flat, jid, cid, mid = (torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in batch)
    logits, p_win, prog = net(flat.float(), jid, cid, mid)
    return (
        logits.cpu().numpy(),
        combined_value(p_win.cpu().numpy(), prog.cpu().numpy()),
    )
