"""Raw-state helpers for search: clone, enumerate, step, observe.

Operates on the engine's game-state dict directly, bypassing the Gymnasium
layer — search pays only for what it uses. Cloning dominates search cost;
a pickle round-trip is ~2x faster than copy.deepcopy on these dicts
(0.84ms vs 1.81ms on a late-game state, Ryzen 9 3900X).

The flat action enumeration is a standalone port of
BalatroGymnasiumEnv._enumerate_actions so the policy head stays compatible
with jackdaw's Discrete(500) convention: slot i = i-th enumerated legal
FactoredAction of the current state.
"""

from __future__ import annotations

import pickle
from itertools import combinations
from typing import Any

import numpy as np

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.game import step as _engine_step
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.action_space import (
    ActionType,
    factored_to_engine_action,
    get_action_mask,
    get_consumable_target_info,
)
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_spec import FactoredAction
from jackdaw.env.observation import NUM_CENTER_KEYS, center_key_id, encode_observation

MAX_ACTIONS = 500
CARD_COMBO_BUDGET = 200

_SPEC = balatro_game_spec()
_ENTITY_INFO: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.feature_dim) for et in _SPEC.entity_types
]
_SIMPLE_TYPES = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if not at.needs_entity_target and not at.needs_card_select
)
_ENTITY_ONLY_TYPES = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if at.needs_entity_target and not at.needs_card_select
)
_CARD_ONLY_TYPES = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if at.needs_card_select and not at.needs_entity_target
)

# Pure-rearrangement actions are excluded from search entirely: they are
# free, infinitely repeatable, and (hand order especially) gameplay-null,
# so under root noise they become a stall trap — observed in run v0 as
# games ballooning to ~270 moves at ante 1 with zero progress. Joker order
# does matter in edge cases (trigger order); revisit if/when the agent is
# good enough for that to bind.
_EXCLUDED_TYPES = frozenset(
    int(t)
    for t in (
        ActionType.SwapJokersLeft,
        ActionType.SwapJokersRight,
        ActionType.SwapHandLeft,
        ActionType.SwapHandRight,
        ActionType.SortHandRank,
        ActionType.SortHandSuit,
    )
)

OBS_DIM: int = (
    _SPEC.global_feature_dim
    + sum(mc * fd for _, mc, fd in _ENTITY_INFO)
    + len(_ENTITY_INFO)
)

# Combo subsampling must be deterministic per process so a node enumerated
# twice sees the same action list (targets/indices stay aligned).
_combo_rng = np.random.default_rng(0)


def new_run(seed: str, back_key: str = "b_red", stake: int = 1) -> dict[str, Any]:
    """Fresh game state at blind select, mirroring DirectAdapter.reset."""
    gs = initialize_run(back_key, stake, seed)
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Small"
    return gs


def clone(gs: dict[str, Any]) -> dict[str, Any]:
    return pickle.loads(pickle.dumps(gs, protocol=5))


def step_factored(gs: dict[str, Any], action: FactoredAction) -> None:
    _engine_step(gs, factored_to_engine_action(action, gs))


def is_terminal(gs: dict[str, Any]) -> bool:
    return gs.get("phase") == GamePhase.GAME_OVER


def won(gs: dict[str, Any]) -> bool:
    return bool(gs.get("won", False))


def ante(gs: dict[str, Any]) -> int:
    return gs.get("round_resets", {}).get("ante", 1)


def in_unresolved_blind(gs: dict[str, Any]) -> bool:
    """True while a selected blind is still undecided (or was lost)."""
    return gs.get("phase") in (GamePhase.SELECTING_HAND, GamePhase.GAME_OVER)


def blinds_beaten(gs: dict[str, Any]) -> int:
    """Blinds actually defeated so far.

    ``gs["round"]`` counts blinds *started*: the engine bumps it in the
    select-blind callback (game.py:183, vanilla button_callbacks.lua:2533),
    not at defeat — so the in-progress or just-lost blind must be backed
    out.  Skipped blinds are never started and so never counted, matching
    the pre-fix counter this replaces.
    """
    return gs.get("round", 0) - (1 if in_unresolved_blind(gs) else 0)


def progress(gs: dict[str, Any]) -> float:
    """Run progress in [0,1]: blinds beaten out of 24, plus fractional chip
    progress toward the current blind. A won run is 1.0.

    This is the cold-start signal: identical all-zero outcome targets (no
    game beats blind 1 early in training) give the value heads nothing to
    rank states by; chip fractions differentiate games from the first
    self-play batch onward.
    """
    if won(gs):
        return 1.0
    frac = 0.0
    if in_unresolved_blind(gs):
        blind = gs.get("blind")
        target = getattr(blind, "chips", 0) if blind is not None else 0
        if target > 0:
            frac = min(gs.get("chips", 0) / target, 0.999)
    return min((blinds_beaten(gs) + frac) / 24.0, 1.0)


def obs_vector(gs: dict[str, Any]) -> np.ndarray:
    """Flatten jackdaw's dict observation into one float32 vector."""
    o = encode_observation(gs).to_game_observation()
    parts: list[np.ndarray] = [o.global_context.astype(np.float32).ravel()]
    counts: list[float] = []
    for name, max_count, feat_dim in _ENTITY_INFO:
        arr = o.entities.get(name)
        padded = np.zeros((max_count, feat_dim), dtype=np.float32)
        if arr is not None and arr.shape[0] > 0:
            n = min(arr.shape[0], max_count)
            padded[:n] = arr[:n]
            counts.append(n / max_count)
        else:
            counts.append(0.0)
        parts.append(padded.ravel())
    parts.append(np.asarray(counts, dtype=np.float32))
    return np.concatenate(parts)


# --- Structured observation (flat features + center-id slots) -------------
# Joker/consumable/shop identity as learnable embeddings: the flat feature
# encoding collapses 150 jokers onto scalar features, which capped shop
# judgment (run v1/v2 plateau at the no-economy ceiling, prog ~0.10).

N_EMBED = NUM_CENTER_KEYS + 1  # 0 = pad/unknown
N_JOKER_SLOTS = 12
N_CONSUMABLE_SLOTS = 4
N_MARKET_SLOTS = 12  # shop cards + vouchers + boosters + open pack contents


from dataclasses import dataclass  # noqa: E402


@dataclass
class Obs:
    flat: np.ndarray            # float32 [OBS_DIM]
    joker_ids: np.ndarray       # int64 [N_JOKER_SLOTS]
    consumable_ids: np.ndarray  # int64 [N_CONSUMABLE_SLOTS]
    market_ids: np.ndarray      # int64 [N_MARKET_SLOTS]


def _pad_ids(keys: list[str], n: int) -> np.ndarray:
    arr = np.zeros(n, dtype=np.int64)
    for i, k in enumerate(keys[:n]):
        arr[i] = center_key_id(k)
    return arr


def observe(gs: dict[str, Any]) -> Obs:
    def keys_of(items: list[Any]) -> list[str]:
        return [getattr(c, "center_key", "") for c in items]

    market = (
        keys_of(gs.get("shop_cards", []))
        + keys_of(gs.get("shop_vouchers", []))
        + keys_of(gs.get("shop_boosters", []))
        + keys_of(gs.get("pack_cards", []))
    )
    return Obs(
        flat=obs_vector(gs),
        joker_ids=_pad_ids(keys_of(gs.get("jokers", [])), N_JOKER_SLOTS),
        consumable_ids=_pad_ids(keys_of(gs.get("consumables", [])), N_CONSUMABLE_SLOTS),
        market_ids=_pad_ids(market, N_MARKET_SLOTS),
    )


def stack_obs(obs_list: list[Obs]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([o.flat for o in obs_list]),
        np.stack([o.joker_ids for o in obs_list]),
        np.stack([o.consumable_ids for o in obs_list]),
        np.stack([o.market_ids for o in obs_list]),
    )


def _card_combos(
    legal_cards: np.ndarray, min_select: int, max_select: int
) -> list[tuple[int, ...]]:
    upper = min(len(legal_cards), max_select)
    lower = min(min_select, upper)
    result: list[tuple[int, ...]] = []
    for k in range(lower, upper + 1):
        result.extend(tuple(int(c) for c in combo) for combo in combinations(legal_cards, k))
    return result


def _subsample(items: list[Any], budget: int) -> list[Any]:
    if len(items) <= budget:
        return items
    indices = _combo_rng.choice(len(items), size=budget, replace=False)
    return [items[i] for i in sorted(indices)]


def legal_factored(gs: dict[str, Any]) -> list[FactoredAction]:
    """Enumerate legal FactoredActions (flat Discrete(500) convention)."""
    mask = get_action_mask(gs)
    actions: list[FactoredAction] = []
    type_mask = mask.type_mask

    for t in range(len(type_mask)):
        if not type_mask[t] or t in _EXCLUDED_TYPES:
            continue
        if t in _SIMPLE_TYPES:
            actions.append(FactoredAction(action_type=t))
        elif t in _ENTITY_ONLY_TYPES:
            if t in mask.entity_masks:
                for idx in np.nonzero(mask.entity_masks[t])[0]:
                    actions.append(FactoredAction(action_type=t, entity_target=int(idx)))
        elif t in _CARD_ONLY_TYPES:
            legal_cards = np.nonzero(mask.card_mask)[0]
            combos = _subsample(
                _card_combos(legal_cards, mask.min_card_select, mask.max_card_select),
                CARD_COMBO_BUDGET,
            )
            for combo in combos:
                actions.append(FactoredAction(action_type=t, card_target=combo))
        elif t == ActionType.UseConsumable:
            if t in mask.entity_masks:
                raw_consumables = gs.get("consumables", [])
                for c_idx in np.nonzero(mask.entity_masks[t])[0]:
                    c_idx_int = int(c_idx)
                    if c_idx_int >= len(raw_consumables):
                        continue
                    card = raw_consumables[c_idx_int]
                    min_cards, max_cards, needs = get_consumable_target_info(card)
                    if needs:
                        legal_cards = np.nonzero(mask.card_mask)[0]
                        combos = _subsample(
                            _card_combos(legal_cards, min_cards, max_cards),
                            CARD_COMBO_BUDGET,
                        )
                        for combo in combos:
                            actions.append(
                                FactoredAction(
                                    action_type=t,
                                    entity_target=c_idx_int,
                                    card_target=combo,
                                )
                            )
                    else:
                        actions.append(FactoredAction(action_type=t, entity_target=c_idx_int))

    return actions[:MAX_ACTIONS]
