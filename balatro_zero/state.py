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

import json
import pickle
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

import jackdaw.engine as _jd_engine
from jackdaw.engine.actions import GamePhase
from jackdaw.engine.data.blind_scaling import get_blind_target
from jackdaw.engine.game import step as _engine_step
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.action_space import (
    ActionType,
    factored_to_engine_action,
    get_action_mask,
    get_consumable_target_info,
)
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_spec import FactoredAction
from jackdaw.env.observation import (
    _ENHANCEMENT_IDX,
    _RANK_IDX,
    _SUIT_IDX,
    _TAG_IDX,
    NUM_CENTER_KEYS,
    NUM_RANKS,
    NUM_TAGS,
    center_key_id,
    encode_observation,
)

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

OBS_DIM_LEGACY: int = (
    _SPEC.global_feature_dim
    + sum(mc * fd for _, mc, fd in _ENTITY_INFO)
    + len(_ENTITY_INFO)
)

# Layout of the per-slot hand rows inside the flat vector (obs_vector
# concatenates global context, then each entity area's padded rows in
# _ENTITY_INFO order). The V6 card pointer reads hand content from here.
assert _ENTITY_INFO[0][0] == "hand_card", _ENTITY_INFO[0]
HAND_FLAT_OFFSET: int = _SPEC.global_feature_dim
HAND_FLAT_ROWS: int = _ENTITY_INFO[0][1]   # 8
HAND_FLAT_DIM: int = _ENTITY_INFO[0][2]    # 15

# --- v14 observability block (2026-08-14) ---------------------------------
# Public state the player can see that the obs never carried, appended
# STRICTLY AFTER the legacy layout so every pre-v14 offset stays valid and
# any old checkpoint reads flat[:OBS_DIM_LEGACY] unchanged (evaluate_* and
# train slice by the net's own in_features). All of it is observable
# history / concrete public objects, so determinize() is untouched.
#
#   [boss]   one-hot of this ante's boss (round_resets.blind_choices["Boss"];
#            the old v[26] blind-key scalar was identically ZERO — centers.json
#            holds no bl_* keys — so every net v1-v13 was boss-blind)
#   [tags]   the Small/Big SKIP-TAG OFFERS (blind_tags; the obs only ever
#            had tags already OWNED — skips were chosen blind to the reward)
#   [deck]   undrawn-deck suit x rank counts (/4, capped) + enhancement
#            fractions (the live multiset is run-modified, NOT inferable
#            from the discard histogram + 8 visible hand rows)
#   [hand]   hand rows 9-16 (legacy flat truncates at 8; with hand-size
#            vouchers the agent chose among invisible cards)
#   [market] one content row PER MARKET SLOT, packed in observe()'s market
#            order (shop cards | vouchers | boosters | pack), so a pointer
#            head can BIND cost/edition/rank content to the slot it scores
#            — pack playing cards all embed c_base, the documented V6
#            residual. Row = [is_pack_card | shop-item or playing-card
#            features, zero-padded].
# Gate: scripts/observability_probe.py (zero-movement controls per piece).

_BLINDS_PATH = Path(_jd_engine.__file__).resolve().parent / "data" / "blinds.json"
with open(_BLINDS_PATH) as _f:
    BLIND_KEYS: tuple[str, ...] = tuple(sorted(json.load(_f)))
_BLIND_KEY_IDX: dict[str, int] = {k: i for i, k in enumerate(BLIND_KEYS)}
N_BLIND_KEYS: int = len(BLIND_KEYS)  # 30

N_DECK_HIST: int = len(_SUIT_IDX) * NUM_RANKS       # 52
N_DECK_ENH: int = max(_ENHANCEMENT_IDX.values())    # 8 (m_bonus..m_lucky)
EXTRA_HAND_ROWS: int = 8                            # hand slots 9-16
MARKET_FEAT_DIM: int = 1 + HAND_FLAT_DIM            # flag + widest row (15)

V14_BOSS_OFFSET: int = OBS_DIM_LEGACY
V14_TAG_OFFSET: int = V14_BOSS_OFFSET + N_BLIND_KEYS
V14_DECK_OFFSET: int = V14_TAG_OFFSET + 2 * NUM_TAGS
EXTRA_HAND_OFFSET: int = V14_DECK_OFFSET + N_DECK_HIST + N_DECK_ENH
MARKET_FEAT_OFFSET: int = EXTRA_HAND_OFFSET + EXTRA_HAND_ROWS * HAND_FLAT_DIM
OBS_DIM: int = MARKET_FEAT_OFFSET + 12 * MARKET_FEAT_DIM  # 12 = N_MARKET_SLOTS


def _v14_block(gs: dict[str, Any], entities: dict[str, Any]) -> np.ndarray:
    """The appended public-state features; offsets local to the block."""
    v = np.zeros(OBS_DIM - OBS_DIM_LEGACY, dtype=np.float32)
    rr = gs.get("round_resets", {})

    bi = _BLIND_KEY_IDX.get(rr.get("blind_choices", {}).get("Boss", ""))
    if bi is not None:
        v[bi] = 1.0

    tags = rr.get("blind_tags", {})
    tbase = N_BLIND_KEYS
    for k, bname in enumerate(("Small", "Big")):
        ti = _TAG_IDX.get(tags.get(bname, ""))
        if ti is not None:
            v[tbase + k * NUM_TAGS + ti] = 1.0

    dbase = tbase + 2 * NUM_TAGS
    deck: list[Any] = gs.get("deck") or []
    n_deck = max(len(deck), 1)
    for card in deck:
        b = getattr(card, "base", None)
        if b is not None:
            suit = b.suit.value if hasattr(b.suit, "value") else str(b.suit)
            rank = b.rank.value if hasattr(b.rank, "value") else str(b.rank)
            si = _SUIT_IDX.get(suit)
            ri = _RANK_IDX.get(rank)
            if si is not None and ri is not None:
                v[dbase + si * NUM_RANKS + ri] += 0.25
        ei = _ENHANCEMENT_IDX.get(getattr(card, "center_key", ""), 0)
        if ei > 0:
            v[dbase + N_DECK_HIST + ei - 1] += 1.0 / n_deck
    np.minimum(v[dbase:dbase + N_DECK_HIST], 1.0,
               out=v[dbase:dbase + N_DECK_HIST])

    hbase = dbase + N_DECK_HIST + N_DECK_ENH
    hand_arr = entities.get("hand_card")
    if hand_arr is not None and hand_arr.shape[0] > HAND_FLAT_ROWS:
        extra = hand_arr[HAND_FLAT_ROWS:HAND_FLAT_ROWS + EXTRA_HAND_ROWS]
        v[hbase:hbase + extra.shape[0] * HAND_FLAT_DIM] = (
            np.asarray(extra, dtype=np.float32).ravel()
        )

    mbase = hbase + EXTRA_HAND_ROWS * HAND_FLAT_DIM
    shop_arr = entities.get("shop_item")
    pack_arr = entities.get("pack_card")
    n_shop = (len(gs.get("shop_cards", [])) + len(gs.get("shop_vouchers", []))
              + len(gs.get("shop_boosters", [])))
    n_pack = len(gs.get("pack_cards", []))
    for k in range(min(n_shop + n_pack, N_MARKET_SLOTS)):
        row = mbase + k * MARKET_FEAT_DIM
        if k < n_shop:
            if shop_arr is not None and k < shop_arr.shape[0]:
                r = shop_arr[k]
                v[row + 1:row + 1 + r.shape[0]] = r
        else:
            v[row] = 1.0
            p = k - n_shop
            if pack_arr is not None and p < pack_arr.shape[0]:
                r = pack_arr[p]
                v[row + 1:row + 1 + r.shape[0]] = r
    return v

# Combo subsampling must be deterministic per process so a node enumerated
# twice sees the same action list (targets/indices stay aligned).
_combo_rng = np.random.default_rng(0)


# v13 progress annotations, maintained by step_factored on every step.
# Both are OBSERVABLE HISTORY (the player knows their deepest ante and
# best single-blind score), so determinize() correctly leaves them alone.
_BZ_BEST = "_bz_best_blind_chips"   # C*: max chips scored in any one blind
_BZ_FRONTIER = "_bz_frontier_ante"  # deepest ante reached (Hieroglyph-proof)


def new_run(seed: str, back_key: str = "b_red", stake: int = 1) -> dict[str, Any]:
    """Fresh game state at blind select, mirroring DirectAdapter.reset."""
    gs = initialize_run(back_key, stake, seed)
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Small"
    gs[_BZ_BEST] = 0
    gs[_BZ_FRONTIER] = 1
    return gs


def clone(gs: dict[str, Any]) -> dict[str, Any]:
    return pickle.loads(pickle.dumps(gs, protocol=5))


# Vanilla seed alphabet: digits 1-9, letters A-N and P-Z (no 0 or O),
# matching rng.generate_starting_seed.
_FRESH_SEED_CHARS = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"


def determinize(gs: dict[str, Any], rng: np.random.Generator) -> None:
    """Re-randomize everything the player has not observed, in place.

    A rollout clone carries the run's PRNG streams, so stepping it replays
    this run's TRUE future (draws, shop rolls, pack contents) — naive
    search is clairvoyant. Applied to a clone, this turns it into a sample
    from the honest distribution instead:

      * the PRNG is replaced wholesale under a fresh random seed — every
        stream lazily re-initializes from it on next touch, so all future
        rolls (shops, packs, next-ante bosses, probability jokers) become
        fresh samples in one stroke;
      * the undrawn deck is reshuffled — deck ORDER is the one piece of
        future that is materialized in state rather than rolled on demand.
        The multiset is untouched (deck composition is public information).

    Everything already observed — hand, jokers, the current shop/pack
    contents, this ante's boss and tags — is concrete objects in the state
    dict and is not touched by either step.

    Known residuals (documented, deliberately unfixed): the current ante's
    shop voucher key is pre-rolled at ante start, so a rollout entering a
    not-yet-visited shop sees the true one; boss-flipped face-down hand
    cards are materialized. Both are single items with small stakes.

    Strictly, the honest posterior is "futures from seeds consistent with
    the observed history," not "a fresh seed" — but the hash streams are
    near-independent, and fresh-seed sampling is what determinization
    means in the MCTS literature.
    """
    fresh = "".join(
        _FRESH_SEED_CHARS[i]
        for i in rng.integers(0, len(_FRESH_SEED_CHARS), size=8)
    )
    gs["rng"] = PseudoRandom(fresh)
    deck = gs.get("deck") or []
    if len(deck) > 1:
        order = rng.permutation(len(deck))
        deck[:] = [deck[i] for i in order]


def step_factored(gs: dict[str, Any], action: FactoredAction) -> None:
    _engine_step(gs, factored_to_engine_action(action, gs))
    # Maintain the v13 progress annotations. Chips retain the blind total
    # from the clearing step until the next blind starts (start_round is
    # what zeroes them), so the post-step max captures every blind's
    # final score, including failed boss attempts.
    c = gs.get("chips", 0)
    if c > gs.get(_BZ_BEST, 0):
        gs[_BZ_BEST] = c
    a = gs.get("round_resets", {}).get("ante", 1)
    if a > gs.get(_BZ_FRONTIER, 1):
        gs[_BZ_FRONTIER] = a


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


def _standard_boss_req(gs: dict[str, Any], ante_n: int) -> int:
    """The ante's effect-free boss bar: 2x base, stake/deck scaled.

    Boss chip requirements trade off against boss EFFECTS (The Needle is
    1x base because it allows one hand), so chips scored under normal
    conditions must be measured against the effect-free standard bar —
    the actual requirement is only honest ON the boss blind itself,
    where the chips are being scored under the effect.
    """
    scaling = gs.get("modifiers", {}).get("scaling", 1)
    ante_scaling = gs.get("starting_params", {}).get("ante_scaling", 1.0)
    return get_blind_target(ante_n, "Boss", scaling, ante_scaling)


def _frontier(gs: dict[str, Any]) -> int:
    return max(gs.get(_BZ_FRONTIER, 1), ante(gs))


def progress(gs: dict[str, Any]) -> float:
    """Run progress in [0,1] (v13, 2026-08-13): ante-anchored frontier
    position plus demonstrated chips against the frontier's boss bar.

        progress = (f - 1 + frac) / 8,   f = deepest ante reached

        frac = min(c  / A_f, .999)  on the frontier boss blind (live)
               min(c  / S_f, .999)  on any other active blind (live)
               min(C* / S_f, .999)  between blinds (high-water)

    where c = chips this blind, C* = best single-blind chips this run
    (absolute — percentages are computed at read time, so denominator
    changes re-price automatically), S_f = the frontier's standard boss
    bar and A_f the actual one.

    Design properties (settled with the user; details in project memory):
    the shaped signal is NEUTRAL to skips and to ante-1 vouchers — the
    integer moves only at new-frontier boss kills, so no optional action
    can farm or forfeit shaped progress; those tradeoffs live in the
    value function. Raw live chips within a blind keep hand-root search
    deltas (the v1 zero-variance and v4 flat-leaf lessons); the C* read
    between blinds keeps econ-rollout realization at the round-advance
    leaf. Numbers are NOT comparable to pre-v13 progress (blinds/24).
    """
    if won(gs):
        return 1.0
    f = _frontier(gs)
    s_f = _standard_boss_req(gs, f)
    blind = gs.get("blind")
    if gs.get("phase") == GamePhase.SELECTING_HAND and blind is not None:
        if ante(gs) == f and getattr(blind, "boss", False):
            denom = getattr(blind, "chips", 0) or s_f
        else:
            denom = s_f
        frac = min(gs.get("chips", 0) / denom, 0.999) if denom > 0 else 0.0
    else:
        frac = min(gs.get(_BZ_BEST, 0) / s_f, 0.999) if s_f > 0 else 0.0
    return min((f - 1 + frac) / 8.0, 1.0)


def progress_cap_read(gs: dict[str, Any]) -> float:
    """Between-blind read with the live blind folded into the store.

    For econ-rollout leaves that hit the step cap MID-blind: the root (a
    shop state) reads C*/S_f, so a capped mid-blind leaf reading live
    c/S_f would show a phantom progress LOSS that systematically punishes
    multi-purchase shop lines (more buys = more rollout steps = capped
    more often). Valuing such leaves at max(C*, c)/S_f removes the
    regime artifact; within-blind discrimination is not the job at econ
    roots, where the candidates being ranked are shop actions.
    """
    if won(gs):
        return 1.0
    f = _frontier(gs)
    s_f = _standard_boss_req(gs, f)
    best = max(gs.get(_BZ_BEST, 0), gs.get("chips", 0))
    frac = min(best / s_f, 0.999) if s_f > 0 else 0.0
    return min((f - 1 + frac) / 8.0, 1.0)


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
    parts.append(_v14_block(gs, o.entities))
    return np.concatenate(parts)


# --- Structured observation (flat features + center-id slots) -------------
# Joker/consumable/shop identity as learnable embeddings: the flat feature
# encoding collapses 150 jokers onto scalar features, which capped shop
# judgment (run v1/v2 plateau at the no-economy ceiling, prog ~0.10).

N_EMBED = NUM_CENTER_KEYS + 1  # 0 = pad/unknown
N_JOKER_SLOTS = 12
N_CONSUMABLE_SLOTS = 4
N_MARKET_SLOTS = 12  # shop cards + vouchers + boosters + open pack contents
assert OBS_DIM == MARKET_FEAT_OFFSET + N_MARKET_SLOTS * MARKET_FEAT_DIM


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
    # One int conversion per card, not one per card per combo — combos of
    # ~8 cards number in the hundreds and this runs at every rollout step.
    cards = [int(c) for c in legal_cards]
    result: list[tuple[int, ...]] = []
    for k in range(lower, upper + 1):
        result.extend(combinations(cards, k))
    return result


def _subsample(
    items: list[Any], budget: int, rng: np.random.Generator | None = None
) -> list[Any]:
    if len(items) <= budget:
        return items
    # The module-global fallback makes trajectories CALL-PATTERN-SENSITIVE
    # (any earlier enumeration in the process advances the stream) — the
    # documented scripted-probe-nondeterminism class. Callers that own a
    # per-game generator (search, selfplay, eval) pass it here so their
    # games are pure functions of (weights, seed, rng seed, config).
    gen = rng if rng is not None else _combo_rng
    indices = gen.choice(len(items), size=budget, replace=False)
    return [items[i] for i in sorted(indices)]


def _forced_card_index(gs: dict[str, Any]) -> int | None:
    """Cerulean Bell's auto-selected card, or None.

    The engine rejects any play/discard that omits it (game.py
    _require_forced_card, vanilla blind.lua:572-87 — the card cannot be
    deselected in the real UI). jackdaw's action mask cannot express a
    must-include constraint, so the enumerator must filter combos here;
    before this, every rollout that reached a Cerulean Bell boss crashed
    the game on an IllegalActionError (first seen v11, 2026-08-12 — the
    engine check landed 2026-07-26, after the last self-play run).
    """
    for i, c in enumerate(gs.get("hand", [])):
        ability = getattr(c, "ability", None)
        if isinstance(ability, dict) and ability.get("forced_selection"):
            return i
    return None


def legal_factored(
    gs: dict[str, Any], rng: np.random.Generator | None = None
) -> list[FactoredAction]:
    """Enumerate legal FactoredActions (flat Discrete(500) convention).

    ``rng`` feeds combo subsampling; None falls back to the module-global
    stream (legacy behavior — context-sensitive, see _subsample).
    """
    mask = get_action_mask(gs)
    actions: list[FactoredAction] = []
    type_mask = mask.type_mask
    forced = _forced_card_index(gs)

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
            combos = _card_combos(legal_cards, mask.min_card_select, mask.max_card_select)
            if forced is not None:
                # Play/Discard must include the forced card; combos without
                # it are engine-illegal. Keep the unfiltered list only in
                # the pathological case where the forced card is not even
                # selectable (never observed; behaves as before the filter).
                kept = [c for c in combos if forced in c]
                if kept:
                    combos = kept
            combos = _subsample(combos, CARD_COMBO_BUDGET, rng)
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
                            rng,
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
