"""Gold probe: clairvoyant per-blind beam search + peek economy.

The upper-anchor difficulty probe. Hand play is a beam search over
play/discard/tarot SEQUENCES on engine clones — within a blind the sim's
draws are deterministic, so the search sees the true future ("what would
near-optimal play extract from this seed"). Economy phases reuse the
scripted router with clairvoyant peeking forced on.

Cost: ~1-5s per blind, ~0.5-2 min per deep game — usable at low K for the
top rung of the difficulty ladder, not for training-scale rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from jackdaw.env.action_space import ActionType
from jackdaw.env.game_spec import FactoredAction

from balatro_zero.router import (
    ECON_PHASES,
    _suit_of,
    _rank_id,
    flags_override,
    key_of,
    scripted_econ_action,
    targeted_tarot_action,
)
from balatro_zero.state import (
    ante,
    blinds_beaten,
    clone,
    is_terminal,
    legal_factored,
    new_run,
    progress,
    step_factored,
    won,
)

MAX_MOVES = 600
BEAM_WIDTH = 5
MAX_DEPTH = 12
N_PLAY_KEEP = 6

# Leaf score tiers (round cleared >> alive >> dead).
WON_RUN = 1e9
CLEARED = 1e6
DEAD = -1e6


@dataclass
class _Node:
    gs: dict
    seq: list[FactoredAction] = field(default_factory=list)
    score: float = 0.0


def _leaf_score(gs, start_beaten: int) -> tuple[bool, float]:
    """(is_terminal_for_blind, score). Higher = better."""
    if won(gs):
        return True, WON_RUN
    if is_terminal(gs):
        return True, DEAD + gs.get("chips", 0)
    if blinds_beaten(gs) > start_beaten:  # blind defeated (pre-cash-out)
        cr = gs.get("current_round", {})
        return True, CLEARED + cr.get("hands_left", 0) * 1e3 + gs.get("dollars", 0)
    return False, float(gs.get("chips", 0))


def _rank_groups(hand) -> list[tuple[int, ...]]:
    """Hand indices grouped by rank, biggest group first."""
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(hand):
        groups.setdefault(_rank_id(c), []).append(i)
    return sorted((tuple(g) for g in groups.values()), key=len, reverse=True)


def _dedup(combos) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    out = []
    for c in combos:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _play_candidates(gs) -> list[FactoredAction]:
    hand = gs.get("hand", [])
    n = len(hand)
    if n == 0:
        return []
    combos: list[tuple[int, ...]] = []
    if n >= 5:
        combos.extend(combinations(range(n), 5))
    else:
        combos.append(tuple(range(n)))
    # Sub-five plays.  The generator used to emit ONLY 5-card and 1-card
    # plays, so it could not even represent a short hand that outscores
    # the five-card version — which is precisely what Half Joker (+20
    # Mult at <=3 cards) and the hand-type x-mults reward.  Observed: a
    # 5-card trips for 34,020 offered where the same trips played as 3
    # cards scored ~109,620.
    groups = _rank_groups(hand)
    combos.extend(g for g in groups if 2 <= len(g) <= 4)
    # Two rank groups together: two pair, and the 4-5 card sets that
    # short full houses live in.
    multi = [g for g in groups if len(g) >= 2]
    for a, b in combinations(multi, 2):
        merged = tuple(sorted(a + b))
        if len(merged) <= 5:
            combos.append(merged)
    # Generic high-card finishers for flat-scoring boards.
    by_rank = sorted(range(n), key=lambda i: -_rank_id(hand[i]))
    combos.extend(tuple(sorted(by_rank[:k])) for k in (2, 3) if n >= k)
    # Small "finisher" plays: cheap ways to close an almost-dead blind.
    combos.extend((i,) for i in range(min(n, 4)))
    return [
        FactoredAction(action_type=int(ActionType.PlayHand), card_target=c)
        for c in _dedup(combos)
    ]


def _discard_candidates(gs) -> list[FactoredAction]:
    hand = gs.get("hand", [])
    cr = gs.get("current_round", {})
    if not hand or cr.get("discards_left", 0) <= 0:
        return []
    suits: dict = {}
    for i, c in enumerate(hand):
        suits.setdefault(_suit_of(c), []).append(i)
    top_suit = max(suits, key=lambda s: len(suits[s]))
    off = sorted(
        (i for i, c in enumerate(hand) if _suit_of(c) != top_suit),
        key=lambda i: _rank_id(hand[i]),
    )
    worst = sorted(range(len(hand)), key=lambda i: _rank_id(hand[i]))
    cands = []
    for k in (len(off), 3, 2):
        if 0 < k <= len(off):
            cands.append(tuple(sorted(off[: min(k, 5)])))
    cands.append(tuple(sorted(worst[:2])))

    # Rank-oriented digs.  Every candidate above is defined relative to
    # the most common SUIT, so the beam could only ever dig toward a
    # flush; on a pairs/trips board it discarded the wrong cards and
    # then reported the blind as unwinnable.  Observed: projected DIES
    # at 5,678/11,000 where one rank-oriented discard reached 40,185.
    groups = _rank_groups(hand)
    singles = sorted(
        (g[0] for g in groups if len(g) == 1), key=lambda i: _rank_id(hand[i])
    )
    # Pitch the unpaired cards, keeping the pairs/trips to build on.
    for k in (5, 3, 2):
        if len(singles) >= k:
            cands.append(tuple(sorted(singles[:k])))
    if singles:
        cands.append(tuple(sorted(singles[: min(len(singles), 5)])))

    # Straight-oriented: keep the densest five-rank window, pitch outside it.
    ranks = [_rank_id(c) for c in hand]
    if ranks:
        best_lo, best_keep = None, -1
        for lo in range(min(ranks), max(ranks) + 1):
            keep = len({r for r in ranks if lo <= r <= lo + 4})
            if keep > best_keep:
                best_lo, best_keep = lo, keep
        if best_lo is not None and best_keep >= 3:
            outside = [
                i for i, r in enumerate(ranks) if not best_lo <= r <= best_lo + 4
            ]
            if outside:
                cands.append(tuple(sorted(outside[:5])))

    return [
        FactoredAction(action_type=int(ActionType.Discard), card_target=c)
        for c in _dedup(cands)
    ]


def _filter_plays(gs, plays: list[FactoredAction], constraints: dict) -> list[FactoredAction]:
    """Drop plays whose hand type violates veto/require constraints.

    Classification uses the engine evaluator (respects Four Fingers /
    Shortcut / Smeared). Returns [] if nothing survives — the caller
    falls back to unconstrained plays, so constraints can never wedge
    a blind."""
    veto = set(constraints.get("veto") or ())
    require = set(constraints.get("require") or ())
    if not veto and not require:
        return plays
    from jackdaw.engine.hand_eval import evaluate_hand

    hand = gs.get("hand", [])
    jokers = gs.get("jokers", [])
    out = []
    for a in plays:
        cards = [hand[i] for i in a.card_target if i < len(hand)]
        ht = evaluate_hand(cards, jokers).detected_hand
        if ht in veto or (require and ht not in require):
            continue
        out.append(a)
    return out


def _expand(
    node: _Node, start_beaten: int, constraints: dict | None = None
) -> list[tuple[FactoredAction, dict]]:
    gs = node.gs
    succ: list[tuple[float, FactoredAction, dict]] = []
    # Plays: simulate all candidates, keep the best few.
    plays: list[tuple[float, FactoredAction, dict]] = []
    candidates = _play_candidates(gs)
    if constraints:
        candidates = _filter_plays(gs, candidates, constraints) or candidates
    for a in candidates:
        sim = clone(gs)
        try:
            step_factored(sim, a)
        except Exception:  # noqa: BLE001
            continue
        _, s = _leaf_score(sim, start_beaten)
        plays.append((s, a, sim))
    plays.sort(key=lambda t: -t[0])
    succ.extend(plays[:N_PLAY_KEEP])
    # Discards and tarot: few candidates, keep all legal.
    extras = _discard_candidates(gs)
    if (t := targeted_tarot_action(gs)) is not None:
        extras.append(t)
    for a in extras:
        sim = clone(gs)
        try:
            step_factored(sim, a)
        except Exception:  # noqa: BLE001
            continue
        _, s = _leaf_score(sim, start_beaten)
        succ.append((s, a, sim))
    return [(a, sim) for _, a, sim in succ]


def plan_blind(gs, constraints: dict | None = None) -> list[FactoredAction]:
    """Beam-search the current blind; return the best action sequence.

    *constraints*: optional {"veto": [hand types], "require": [hand types]}
    steering which hand types the beam may play (falls back to
    unconstrained when nothing legal survives)."""
    start_beaten = blinds_beaten(gs)
    frontier = [_Node(clone(gs))]
    best_leaf: _Node | None = None
    best_alive: _Node | None = None

    for _ in range(MAX_DEPTH):
        children: list[_Node] = []
        for node in frontier:
            for action, sim in _expand(node, start_beaten, constraints):
                terminal, score = _leaf_score(sim, start_beaten)
                child = _Node(sim, node.seq + [action], score)
                if terminal:
                    if best_leaf is None or score > best_leaf.score:
                        best_leaf = child
                else:
                    children.append(child)
                    if best_alive is None or score > best_alive.score:
                        best_alive = child
        if best_leaf is not None and best_leaf.score >= CLEARED:
            break  # a clearing line exists; deeper search only refines it
        children.sort(key=lambda n: -n.score)
        frontier = children[:BEAM_WIDTH]
        if not frontier:
            break

    pick = best_leaf or best_alive
    return pick.seq if pick else []


ECON_ROLLOUT_HORIZON = 3   # blinds to simulate when valuing an econ action
ECON_ROLLOUT_MOVES = 250


def _fast_hand_action(gs) -> FactoredAction | None:
    """Feature-only hand policy for rollouts: no clone enumeration.

    Play the flush if held; else discard toward it; else dump high cards.
    Weaker than the beam, but rollout values are only compared to each
    other, so relative ordering is what matters.
    """
    hand = gs.get("hand", [])
    if not hand:
        return None
    by_suit: dict = {}
    for i, c in enumerate(hand):
        by_suit.setdefault(_suit_of(c), []).append(i)
    top = max(by_suit.values(), key=len)
    top_sorted = sorted(top, key=lambda i: -_rank_id(hand[i]))
    cr = gs.get("current_round", {})
    if len(top) >= 5:
        return FactoredAction(
            action_type=int(ActionType.PlayHand),
            card_target=tuple(sorted(top_sorted[:5])),
        )
    if cr.get("discards_left", 0) > 0 and len(top) >= 3:
        off = [i for i in range(len(hand)) if i not in top][:5]
        if off:
            return FactoredAction(
                action_type=int(ActionType.Discard), card_target=tuple(sorted(off))
            )
    best5 = sorted(range(len(hand)), key=lambda i: -_rank_id(hand[i]))[:5]
    return FactoredAction(
        action_type=int(ActionType.PlayHand), card_target=tuple(sorted(best5))
    )


def _rollout_value(gs) -> float:
    """Cheap rollout: fast play until HORIZON blinds clear; realized progress."""
    sim = clone(gs)
    stop_beaten = blinds_beaten(sim) + ECON_ROLLOUT_HORIZON
    moves = 0
    with flags_override(peek=False):  # nested peeking explodes rollout cost
        while (
            not is_terminal(sim)
            and not won(sim)
            and blinds_beaten(sim) < stop_beaten
            and moves < ECON_ROLLOUT_MOVES
        ):
            phase = sim.get("phase")
            a = scripted_econ_action(sim) if phase in ECON_PHASES else _fast_hand_action(sim)
            if a is None:
                legal = legal_factored(sim)
                if not legal:
                    break
                a = legal[0]
            try:
                step_factored(sim, a)
            except Exception:  # noqa: BLE001
                legal = legal_factored(sim)
                if not legal:
                    break
                step_factored(sim, legal[0])
            moves += 1
    # Tiebreak on board strength, NOT dollars: at saturated progress a cash
    # tiebreak rewards hoarding, and build payoffs land past the horizon.
    from balatro_zero.router import joker_tier

    board = sum(joker_tier(key_of(c)) for c in sim.get("jokers", []))
    return progress(sim) + 0.004 * board + 1e-5 * sim.get("dollars", 0)


def _econ_candidates(gs) -> list[FactoredAction]:
    """Plausible econ actions worth valuing by rollout (small, typed set)."""
    legal = legal_factored(gs)
    # SkipBlind is deliberately absent: a 3-blind rollout can't price the
    # forfeited blind money (compounds past the horizon) and reliably
    # over-values skips — measured as ante-1 deaths on skipped blinds.
    keep_types = {
        int(ActionType.BuyCard), int(ActionType.RedeemVoucher),
        int(ActionType.OpenBooster), int(ActionType.Reroll),
        int(ActionType.NextRound), int(ActionType.SelectBlind),
        int(ActionType.SkipPack),
        int(ActionType.PickPackCard), int(ActionType.CashOut),
    }
    cands = [a for a in legal if a.action_type in keep_types]
    # One sell candidate: the worst joker (board-upgrade enabler).
    jokers = gs.get("jokers", [])
    if len(jokers) >= 5:
        from balatro_zero.router import joker_tier

        worst_i = min(range(len(jokers)), key=lambda i: joker_tier(key_of(jokers[i])))
        for a in legal:
            if a.action_type == ActionType.SellJoker and a.entity_target == worst_i:
                cands.append(a)
                break
    return cands[:12]


def econ_action_by_rollout(gs) -> FactoredAction | None:
    """Pick the econ action whose two-blind rollout realizes the most progress.

    Only shop and blind-select decisions are worth the rollout cost; pack
    picks stay scripted (rank-based, and rarely pivotal at pack prices).
    """
    from jackdaw.engine.actions import GamePhase

    # Immediate no-brainer uses (Soul, Jupiter, money tarots) first.
    scripted = scripted_econ_action(gs)
    if scripted is not None and scripted.action_type == ActionType.UseConsumable:
        return scripted
    if gs.get("phase") == GamePhase.PACK_OPENING:
        return scripted
    cands = _econ_candidates(gs)
    if not cands:
        return scripted
    if len(cands) == 1:
        return cands[0]
    cands = cands[:8]
    best, best_v = None, -1.0
    for a in cands:
        sim = clone(gs)
        try:
            step_factored(sim, a)
        except Exception:  # noqa: BLE001
            continue
        v = _rollout_value(sim)
        if v > best_v:
            best, best_v = a, v
    return best or scripted


def gold_game(seed: str, econ_eps: float = 0.0, rng=None, plan: dict | None = None) -> dict:
    """Play one full clairvoyant game; returns outcome stats."""
    import numpy as np

    from balatro_zero.router import plan_override

    if rng is None:
        rng = np.random.default_rng(0)
    with flags_override(peek=True, skip_tags=False), plan_override(plan):
        gs = new_run(seed)
        moves = 0
        max_ante = ante(gs)
        while not is_terminal(gs) and not won(gs) and moves < MAX_MOVES:
            phase = gs.get("phase")
            if phase in ECON_PHASES:
                action = None
                if econ_eps > 0 and rng.random() < econ_eps:
                    legal = legal_factored(gs)
                    action = legal[int(rng.integers(len(legal)))] if legal else None
                if action is None:
                    action = econ_action_by_rollout(gs)
                if action is None:
                    legal = legal_factored(gs)
                    if not legal:
                        break
                    action = legal[0]
                try:
                    step_factored(gs, action)
                except Exception:  # noqa: BLE001
                    legal = legal_factored(gs)
                    if not legal:
                        break
                    step_factored(gs, legal[0])
                moves += 1
            else:
                seq = plan_blind(gs)
                if not seq:
                    legal = legal_factored(gs)
                    if not legal:
                        break
                    seq = [legal[0]]
                for a in seq:
                    if is_terminal(gs) or won(gs):
                        break
                    try:
                        step_factored(gs, a)
                    except Exception:  # noqa: BLE001 — plan diverged: replan
                        break
                    moves += 1
            max_ante = max(max_ante, ante(gs))
        return {
            "won": won(gs),
            "max_ante": max_ante,
            "progress": round(max(progress(gs), 0.0), 4),
            "moves": moves,
            "board": [key_of(c) for c in gs.get("jokers", [])],
            "dollars": gs.get("dollars", 0),
        }
