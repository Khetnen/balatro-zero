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
    """Masked mean+max pool over the slot dimension -> [B, 2*E].

    The empty-group guard MUST test the true count, not the clamped one.
    It previously read ``count > 0.5`` after ``count`` had already been
    clamped to a minimum of 1.0, so the condition was always true, the
    zeroing was dead code, and the -1e9 max-pool sentinel survived into
    the fused vector for any group with no items.

    That is not a corner case: consumables and the market are BOTH empty
    at every mid-blind decision, so a 1e9-magnitude component reached the
    fuse layer, and LayerNorm then squashed every real feature to noise.
    Measured on an untrained v3 net, three unrelated joker boards with
    three different flat-feature vectors returned the identical value to
    ten decimal places -- the network was blind, not merely weak. This
    sat underneath the whole v1/v2/v3 plateau.
    """
    mask = (ids > 0).unsqueeze(-1).float()          # [B, S, 1]
    present = mask.sum(dim=1)                       # [B, 1] TRUE count
    count = present.clamp(min=1.0)
    mean = (emb * mask).sum(dim=1) / count
    neg = torch.where(mask > 0, emb, torch.full_like(emb, -1e9))
    mx = neg.max(dim=1).values
    mx = torch.where(present > 0.5, mx, torch.zeros_like(mx))
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


class JokerEncoder(nn.Module):
    """Order-aware encoder over the joker board.

    ``_pool`` is permutation-invariant, so the v3 network literally
    cannot represent "Blueprint is immediately left of Photograph" --
    and Blueprint copies THE JOKER TO ITS RIGHT (card.lua:2321), so
    board order is a first-class game mechanic, not presentation. The
    same blindness applies to every pairwise synergy: Blueprint x Trio
    and Blueprint x Photograph produce the identical pooled vector.

    Each slot becomes a token of [ID embedding | descriptor | position],
    self-attended so a token sees its neighbours, and only then pooled.
    Pooling after attention is fine: the tokens are contextualised.
    """

    def __init__(self, embed: nn.Embedding, desc: torch.Tensor,
                 n_slots: int, d_model: int = 64, heads: int = 4,
                 layers: int = 2) -> None:
        super().__init__()
        self.embed = embed
        self.register_buffer("desc", desc, persistent=False)
        self.proj = nn.Linear(embed.embedding_dim + desc.shape[1], d_model)
        self.pos = nn.Embedding(n_slots, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=2 * d_model,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.attn = nn.TransformerEncoder(enc, num_layers=layers)
        self.out_dim = 2 * d_model

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, s = ids.shape
        tok = torch.cat([self.embed(ids), self.desc[ids]], dim=-1)
        h = self.proj(tok) + self.pos.weight[:s].unsqueeze(0)

        pad = ids == 0
        # A fully padded row (no jokers yet -- every run's first shop)
        # would make attention softmax over nothing and yield NaN. Keep
        # slot 0 visible for those rows, then zero the pooled result.
        empty = pad.all(dim=1)
        pad = pad.clone()
        pad[empty, 0] = False
        h = self.attn(h, src_key_padding_mask=pad)

        keep = (~pad).unsqueeze(-1).float()
        count = keep.sum(dim=1).clamp(min=1.0)
        mean = (h * keep).sum(dim=1) / count
        mx = torch.where(keep > 0, h, torch.full_like(h, -1e9)).max(dim=1).values
        pooled = torch.cat([mean, mx], dim=-1)
        return torch.where(empty.unsqueeze(-1), torch.zeros_like(pooled), pooled)


class PolicyValueNetV4(PolicyValueNet):
    """v3 network with the joker board self-attended instead of pooled.

    Consumables and the market keep mean+max: their order carries no
    mechanics. Only the joker board has adjacency and pairwise effects.

    Subclasses v3 so the torso/heads and their init stay identical and
    the only variable under test is the joker encoder.
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = MAX_ACTIONS,
                 hidden: int = 512, depth: int = 3, embed_dim: int = EMBED_DIM,
                 d_model: int = 64) -> None:
        super().__init__(obs_dim, n_actions, hidden, depth, embed_dim)
        from balatro_zero.joker_features import descriptor_table
        from balatro_zero.state import N_JOKER_SLOTS

        desc = torch.from_numpy(descriptor_table())
        self.jokers = JokerEncoder(self.embed, desc, N_JOKER_SLOTS, d_model)
        pooled = self.jokers.out_dim + 2 * 2 * embed_dim  # jokers + cons + market
        self.fuse = nn.Sequential(
            nn.Linear(hidden + pooled, hidden), nn.LayerNorm(hidden), nn.ReLU()
        )

    def forward(self, flat, joker_ids, consumable_ids, market_ids):
        h = self.torso(flat)
        pooled = torch.cat(
            [
                self.jokers(joker_ids),
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


def load_net(ckpt: str, device: str = "cpu") -> PolicyValueNet:
    """Build whichever architecture a checkpoint was trained with.

    The v0-v10 checkpoints are difficulty-ladder rungs, so adding the
    attention encoder must not strand them. Sniff the state dict rather
    than storing a flag, which old checkpoints would not have.
    """
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    net = PolicyValueNetV4() if any(k.startswith("jokers.") for k in sd) \
        else PolicyValueNet()
    net.load_state_dict(sd)
    net.eval()
    return net


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
