"""Factored policy targets: candidate sets, collation, and the loss.

The positional Discrete(500) head is supervised by slot index — slot i
means "the i-th enumerated legal action of that state", which means
nothing across states. The factored (V5) head is supervised by CONTENT:
a training target is search's improved policy over the root's candidate
set, stored as compact per-candidate descriptors:

    action type (uint8) | entity slot (int8, -1 = none) | card bitmask

The loss recomposes each candidate's log-probability from the three
factor heads exactly as ``net.action_logit`` does at inference —
normalised type and entity log-softmaxes plus a Bernoulli set
log-probability over the live hand slots — then takes cross-entropy
between search's weights and the softmax over the candidate set. The
softmax-over-candidates is what makes a one-hot guided target
meaningful: it says "this action, NOT the others that were legal".

``scripts/factored_loss_probe.py`` is the falsifiable gate: the batched
path here must match a per-action ``action_logit`` reference to float
tolerance, including the edge cases (entity slot out of range, card
index beyond the live hand, hands wider than the head).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from balatro_zero.net import MAX_ENTITY_SLOTS, N_HAND_SLOTS


@dataclass
class CandidateSet:
    """One decision's policy target: search's weights over its candidates."""

    types: np.ndarray       # uint8 [K] action type per candidate
    entities: np.ndarray    # int8  [K] entity slot, -1 = no entity factor
    card_masks: np.ndarray  # uint32[K] bit h set = hand slot h selected
    has_cards: np.ndarray   # bool  [K] candidate carries a card target
    pi: np.ndarray          # float32 [K] improved-policy weight (sums to 1)
    n_hand: int             # live hand slots at the root


def encode_candidates(
    actions: Sequence[Any], pi: np.ndarray, n_hand: int
) -> CandidateSet:
    """Compress a root's action list + improved policy into a CandidateSet.

    Mirrors ``net.action_logit``'s conventions exactly, so training and
    inference score identically: an entity slot outside the head's range
    contributes nothing (-1 here, bounds check there); a selected card
    index beyond ``min(n_hand, N_HAND_SLOTS)`` is ignored (bit dropped
    here, membership loop bounded there). A MacroPlan is scored by its
    opening move, like ``search._head_action``.
    """
    k = len(actions)
    types = np.empty(k, dtype=np.uint8)
    entities = np.full(k, -1, dtype=np.int8)
    card_masks = np.zeros(k, dtype=np.uint32)
    has_cards = np.zeros(k, dtype=bool)
    for i, a in enumerate(actions):
        head = a.seq[0] if hasattr(a, "seq") else a
        types[i] = int(head.action_type)
        e = head.entity_target
        if e is not None and 0 <= int(e) < MAX_ENTITY_SLOTS:
            entities[i] = int(e)
        tgt = head.card_target
        if tgt is not None and len(tgt) > 0:
            has_cards[i] = True
            m = 0
            for c in tgt:
                if 0 <= int(c) < N_HAND_SLOTS:
                    m |= 1 << int(c)
            card_masks[i] = m
    w = np.asarray(pi[:k], dtype=np.float32)
    return CandidateSet(types, entities, card_masks, has_cards, w, int(n_hand))


@dataclass
class FactoredBatch:
    """Collated candidate sets as flat tensors (one row per candidate)."""

    b_idx: torch.Tensor     # int64 [N] sample index of each candidate
    types: torch.Tensor     # int64 [N]
    entities: torch.Tensor  # int64 [N], -1 = none
    in_mask: torch.Tensor   # float [N, N_HAND_SLOTS] selected live slots
    out_mask: torch.Tensor  # float [N, N_HAND_SLOTS] unselected live slots
    weights: torch.Tensor   # float [N]
    n_samples: int


def collate_candidate_sets(
    sets: Sequence[CandidateSet], device: torch.device
) -> FactoredBatch:
    b_idx = np.concatenate(
        [np.full(len(cs.pi), i, dtype=np.int64) for i, cs in enumerate(sets)]
    )
    types = np.concatenate([cs.types for cs in sets]).astype(np.int64)
    entities = np.concatenate([cs.entities for cs in sets]).astype(np.int64)
    weights = np.concatenate([cs.pi for cs in sets]).astype(np.float32)
    masks = np.concatenate([cs.card_masks for cs in sets]).astype(np.int64)
    has = np.concatenate([cs.has_cards for cs in sets])
    live = np.concatenate(
        [np.full(len(cs.pi), min(int(cs.n_hand), N_HAND_SLOTS), dtype=np.int64)
         for cs in sets]
    )

    slots = np.arange(N_HAND_SLOTS, dtype=np.int64)[None, :]
    bits = ((masks[:, None] >> slots) & 1).astype(np.float32)
    live_m = (slots < live[:, None]).astype(np.float32)
    hc = has[:, None].astype(np.float32)
    in_mask = bits * live_m * hc
    out_mask = (1.0 - bits) * live_m * hc

    t = torch.from_numpy
    return FactoredBatch(
        b_idx=t(b_idx).to(device),
        types=t(types).to(device),
        entities=t(entities).to(device),
        in_mask=t(in_mask).to(device),
        out_mask=t(out_mask).to(device),
        weights=t(weights).to(device),
        n_samples=len(sets),
    )


def factored_policy_loss(
    type_lg: torch.Tensor,
    ent_lg: torch.Tensor,
    card_lg: torch.Tensor,
    fb: FactoredBatch,
) -> torch.Tensor:
    """Cross-entropy over each sample's candidate set, averaged over samples.

    Per candidate: score = type log-softmax + entity log-softmax (when it
    has an entity) + Bernoulli set log-prob over live hand slots (when it
    has cards) — the ``action_logit`` composition, batched. Then a
    segment log-softmax normalises each sample's candidates against each
    other, and the loss is -sum(pi * logp).
    """
    lt = F.log_softmax(type_lg, dim=-1)
    le = F.log_softmax(ent_lg, dim=-1)
    lp_in = F.logsigmoid(card_lg)
    lp_out = F.logsigmoid(-card_lg)

    b = fb.b_idx
    s = lt[b, fb.types]
    has_ent = fb.entities >= 0
    zero = torch.zeros((), device=s.device, dtype=s.dtype)
    s = s + torch.where(has_ent, le[b, fb.entities.clamp(min=0)], zero)
    s = s + (fb.in_mask * lp_in[b] + fb.out_mask * lp_out[b]).sum(dim=-1)

    # Segment log-softmax over each sample's candidates (max-shifted for
    # stability; the shift constant carries no gradient, as usual).
    n = fb.n_samples
    m = torch.zeros(n, device=s.device, dtype=s.dtype)
    m.index_reduce_(0, b, s.detach(), "amax", include_self=False)
    ex = torch.exp(s - m[b])
    denom = torch.zeros(n, device=s.device, dtype=s.dtype).index_add_(0, b, ex)
    logp = s - (m + torch.log(denom))[b]
    return -(fb.weights * logp).sum() / n
