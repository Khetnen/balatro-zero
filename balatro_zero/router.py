"""Scripted router policy: flush-committed hand play + priority-rule economy.

Extracted from scripts/route_harvest.py so self-play workers (spawned
processes) can import it for guided-economy games. The engine itself is the
evaluator for hand plays; the economy encodes the community macro-strategy
(Soul grabs, xmult buys, Jupiter stacking, rerolls, junk-selling).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from jackdaw.engine.actions import GamePhase
from jackdaw.env.action_space import ActionType
from jackdaw.env.game_spec import FactoredAction

from balatro_zero.state import clone, legal_factored, progress, step_factored

# Multiplicative / build-defining jokers worth routing a run around.
XMULT = {
    "j_blueprint", "j_brainstorm", "j_madness", "j_hologram", "j_vampire",
    "j_constellation", "j_obelisk", "j_lucky_cat", "j_steel_joker",
    "j_glass_joker", "j_campfire", "j_throwback", "j_card_sharp", "j_baron",
    "j_mime", "j_photograph", "j_baseball", "j_ancient", "j_ramen",
    "j_castle", "j_wee", "j_stencil", "j_loyalty_card", "j_cavendish",
    "j_dusk", "j_sock_and_buskin", "j_hack", "j_fibonacci", "j_steel",
}
# Linear-but-growing engines; decent support pieces.
SCALER = {
    "j_ride_the_bus", "j_green_joker", "j_red_card", "j_spare_trousers",
    "j_fortune_teller", "j_flash", "j_rocket", "j_bull", "j_bootstraps",
    "j_supernova", "j_runner", "j_ice_cream", "j_constellation",
}
# Score-dead legendaries (utility-only) rank below scoring xmult jokers.
LEGENDARY_SCORING = {"j_triboulet", "j_yorick", "j_canio", "j_perkeo"}

PLANET_FLUSH = "c_jupiter"
PLANETS = {
    "c_mercury", "c_venus", "c_earth", "c_mars", "c_jupiter",
    "c_saturn", "c_uranus", "c_neptune", "c_pluto",
}
# Planet discipline: the build is Flush — only Jupiter (and Black Hole,
# which levels everything) gets USED; off-build planets are sold for tempo.
USE_PLANETS = {PLANET_FLUSH, "c_black_hole"}
ECON_TAROTS = {"c_hermit", "c_temperance"}
# Tarots used with card targets during hand phases (the enhancement pillar).
TARGETED_TAROTS = {"c_chariot", "c_justice", "c_death", "c_hanged_man"}
PRIORITY_VOUCHERS = {"v_telescope", "v_observatory"}
PACK_PREF = ("p_spectral", "p_arcana", "p_buffoon", "p_celestial", "p_standard")
ECON_PHASES = {GamePhase.SHOP, GamePhase.BLIND_SELECT, GamePhase.PACK_OPENING}
# Tags worth skipping the SMALL blind for (free jokers/packs/editions/cash).
GOOD_SKIP_TAGS = {
    "tag_investment", "tag_coupon", "tag_negative", "tag_foil", "tag_holo",
    "tag_polychrome", "tag_charm", "tag_meteor", "tag_buffoon",
    "tag_ethereal", "tag_rare",
}


# Feature flags for ablation (set from env ROUTER_FLAGS=csv of names to enable;
# default all on — the current best config is whatever benchmarks best).
import os as _os

_flags_env = _os.environ.get("ROUTER_FLAGS")
# Ablation vs 32-seed RBASE baseline (2026-07-17): baseline 0.307; planet
# +0.003, interest +0.006, hermit +0.004 (each within noise, theoretically
# sound -> on); skip_tags -0.058 (forfeits a shop visit early -> OFF).
FLAGS = {
    "planet_discipline": True,
    "hermit_gate": True,
    "interest": True,
    "skip_tags": False,
    "telescope": True,        # prioritize Telescope/Observatory; hold Jupiters under Observatory
    "tarot_targeting": True,  # use Chariot/Justice/Death/Hanged Man on the flush suit
    "desperation": True,      # no xmult by ante 4: drop the reserve, reroll hard
    "peek": False,            # clairvoyant probe: clone-simulate pack opens and
                              # rerolls before committing (exact — same streams)
}


def _pick_rank(k: str) -> int:
    """Value of a pack-openable item to the flush build."""
    if k in PLAN["priority_buys"]:
        return 50
    if k in PLAN["avoid_buys"]:
        return 1
    return (
        100 if k == "c_soul"
        else 60 if k == "c_black_hole"
        else 50 if k in XMULT
        else 30 if k in SCALER
        else 25 if k == PLANET_FLUSH
        else (12 if FLAGS["tarot_targeting"] else 1) if k in TARGETED_TAROTS
        else (4 if FLAGS["planet_discipline"] else 18) if k in PLANETS
        else 10 if k in ECON_TAROTS
        else 1
    )
if _flags_env is not None:
    _on = {f.strip() for f in _flags_env.split(",") if f.strip()}
    FLAGS = {k: (k in _on) for k in FLAGS}


from contextlib import contextmanager as _contextmanager

# Per-seed build plan (set via plan_override): seed-specific purchase
# priorities authored by an LLM from a scouted shop/pack future.
PLAN: dict = {"priority_buys": (), "avoid_buys": ()}


@_contextmanager
def plan_override(plan: dict | None):
    old = dict(PLAN)
    if plan:
        PLAN.update({
            "priority_buys": tuple(plan.get("priority_buys", ())),
            "avoid_buys": tuple(plan.get("avoid_buys", ())),
        })
    try:
        yield
    finally:
        PLAN.clear()
        PLAN.update(old)


@_contextmanager
def flags_override(**kw):
    """Temporarily override FLAGS (e.g. the gold probe forcing peek=True)."""
    old = dict(FLAGS)
    FLAGS.update(kw)
    try:
        yield
    finally:
        FLAGS.clear()
        FLAGS.update(old)


def interest_reserve(gs) -> int:
    """Dollars to protect for interest ($1 per $5 held, capped at $25).

    Ramps in with ante: early tempo beats interest; from ante ~6 the full
    $25 floor is worth holding for non-critical purchases.
    """
    return min(25, 5 * (ante_of(gs) - 1))


def ante_of(gs) -> int:
    return gs.get("round_resets", {}).get("ante", 1)


def key_of(card) -> str:
    return getattr(card, "center_key", "")


def joker_tier(k: str) -> int:
    if k in LEGENDARY_SCORING:
        return 4
    if k in XMULT:
        return 3
    if k in SCALER:
        return 2
    if k.startswith("j_"):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Hand play: engine-exact greedy + flush-committed discards
# ---------------------------------------------------------------------------


def best_card_action(gs, action_type: int) -> tuple[FactoredAction | None, float]:
    """Try every card combo for `action_type` on clones; return (best, gain)."""
    hand = gs.get("hand", [])
    n = len(hand)
    base_prog = progress(gs)
    best, best_gain = None, -1.0
    for k in range(1, min(5, n) + 1):
        for combo in combinations(range(n), k):
            a = FactoredAction(action_type=action_type, card_target=combo)
            sim = clone(gs)
            try:
                step_factored(sim, a)
            except Exception:  # noqa: BLE001 — illegal under boss rule etc.
                continue
            gain = progress(sim) - base_prog
            if gain > best_gain:
                best, best_gain = a, gain
    return best, best_gain


def _suit_of(card):
    return getattr(getattr(card, "base", None), "suit", None)


def _rank_id(card) -> int:
    return getattr(getattr(card, "base", None), "id", 0)


def targeted_tarot_action(gs) -> FactoredAction | None:
    """Fire a held enhancement tarot at the flush suit (hand phase only)."""
    hand = gs.get("hand", [])
    cons = gs.get("consumables", [])
    if not hand or not cons:
        return None
    suits = Counter(s for c in hand if (s := _suit_of(c)) is not None)
    if not suits:
        return None
    top_suit, top_n = suits.most_common(1)[0]
    suit_idx = sorted(
        (i for i, c in enumerate(hand) if _suit_of(c) == top_suit),
        key=lambda i: _rank_id(hand[i]),
    )
    off_idx = sorted(
        (i for i, c in enumerate(hand) if _suit_of(c) != top_suit),
        key=lambda i: _rank_id(hand[i]),
    )
    for ci, c in enumerate(cons):
        k = key_of(c)
        if k not in TARGETED_TAROTS:
            continue
        target: tuple[int, ...] | None = None
        if k in ("c_chariot", "c_justice") and top_n >= 3:
            target = (suit_idx[-1],)  # steel/glass the best flush-suit card
        elif k == "c_hanged_man" and off_idx:
            target = tuple(off_idx[:2])  # destroy the worst off-suit junk
        elif k == "c_death" and len(suit_idx) >= 2:
            weak, strong = suit_idx[0], suit_idx[-1]
            if weak < strong:  # left card becomes the right card
                target = (weak, strong)
        if target is None:
            continue
        a = FactoredAction(
            action_type=int(ActionType.UseConsumable),
            entity_target=ci,
            card_target=tuple(sorted(target)),
        )
        sim = clone(gs)
        try:
            step_factored(sim, a)
            return a
        except Exception:  # noqa: BLE001 — not legal here (boss rule etc.)
            continue
    return None


def scripted_hand_action(gs) -> FactoredAction | None:
    hand = gs.get("hand", [])
    if not hand:
        return None
    if FLAGS["tarot_targeting"] and (a := targeted_tarot_action(gs)):
        return a
    play, play_gain = best_card_action(gs, int(ActionType.PlayHand))
    if play is None:
        return None

    round_state = gs.get("current_round", {})
    discards_left = round_state.get("discards_left", 0)
    hands_left = round_state.get("hands_left", 0)

    blind = gs.get("blind")
    target = getattr(blind, "chips", 0) if blind is not None else 0
    need_frac = 0.0
    if target > 0:
        need_frac = max(0.0, (target - gs.get("chips", 0)) / target) / 24.0

    suits = Counter(getattr(c.base, "suit", None) for c in hand if getattr(c, "base", None))
    top_suit, top_n = (suits.most_common(1) or [(None, 0)])[0]

    # Discard toward a flush when: we can afford it, a flush is brewing but
    # not ready, and the best play alone doesn't cover what we still need.
    if (
        discards_left > 0
        and hands_left >= 1
        and top_suit is not None
        and 3 <= top_n <= 4
        and play_gain < need_frac
    ):
        off = tuple(
            i for i, c in enumerate(hand)
            if getattr(getattr(c, "base", None), "suit", None) != top_suit
        )[:5]
        if off:
            a = FactoredAction(action_type=int(ActionType.Discard), card_target=off)
            sim = clone(gs)
            try:
                step_factored(sim, a)
                return a
            except Exception:  # noqa: BLE001
                pass
    return play


# ---------------------------------------------------------------------------
# Economy script (shop / blind select / packs)
# ---------------------------------------------------------------------------


def scripted_econ_action(gs, legal: list[FactoredAction] | None = None) -> FactoredAction | None:
    if legal is None:
        legal = legal_factored(gs)
    by_type: dict[int, list] = defaultdict(list)
    for a in legal:
        by_type[a.action_type].append(a)

    jokers = gs.get("jokers", [])
    joker_keys = [key_of(c) for c in jokers]
    cons = gs.get("consumables", [])
    phase = gs.get("phase")

    def use_matching(keys: set[str]):
        for a in by_type[ActionType.UseConsumable]:
            if a.entity_target is not None and a.entity_target < len(cons):
                if key_of(cons[a.entity_target]) in keys and a.card_target is None:
                    return a
        return None

    if len(jokers) < 5 and (a := use_matching({"c_soul"})):
        return a
    # Flush discipline: only Jupiter / Black Hole get used; Hermit only when
    # doubling real money (it doubles up to $20 — at $4 it's a waste).
    # Under Observatory, held Jupiters give x1.5 mult each — hold, don't use.
    use_planets = USE_PLANETS if FLAGS["planet_discipline"] else PLANETS
    if FLAGS["telescope"] and gs.get("used_vouchers", {}).get("v_observatory"):
        use_planets = use_planets - {PLANET_FLUSH}
    if (a := use_matching(use_planets)):
        return a
    if not FLAGS["hermit_gate"] or gs.get("dollars", 0) >= 10:
        if (a := use_matching({"c_hermit"})):
            return a
    if (a := use_matching({"c_temperance"})):
        return a

    if phase == GamePhase.BLIND_SELECT:
        # Skip the small blind when its tag is a free-value tag.
        if FLAGS["skip_tags"] and by_type[ActionType.SkipBlind] \
                and gs.get("blind_on_deck") == "Small":
            tag = gs.get("round_resets", {}).get("blind_tags", {}).get("Small")
            if tag in GOOD_SKIP_TAGS:
                return by_type[ActionType.SkipBlind][0]
        if by_type[ActionType.SelectBlind]:
            return by_type[ActionType.SelectBlind][0]
        return None

    if phase == GamePhase.SHOP:
        dollars = gs.get("dollars", 0)
        reserve = interest_reserve(gs) if FLAGS["interest"] else 0
        desperate = (
            FLAGS["desperation"]
            and ante_of(gs) >= 3
            and not any(k in XMULT for k in joker_keys)
        )
        if desperate:
            reserve = 0
        # Telescope line: buy on sight, ignoring the reserve — it converts
        # every future Celestial pack into Jupiters (then Observatory turns
        # held Jupiters into x1.5 each).
        if FLAGS["telescope"]:
            vouchers = gs.get("shop_vouchers", [])
            for a in by_type[ActionType.RedeemVoucher]:
                if a.entity_target is not None and a.entity_target < len(vouchers):
                    if key_of(vouchers[a.entity_target]) in PRIORITY_VOUCHERS:
                        return a
        if FLAGS["planet_discipline"]:
            # Sell held off-build consumables (frees slots, feeds interest).
            # Keep at most ONE targeted tarot: a full rack of them blocks
            # Soul pickups (observed on god seeds — the whole point missed).
            keep = USE_PLANETS | {"c_soul", "c_hermit", "c_temperance"}
            tarots_kept = 0
            for a in by_type[ActionType.SellConsumable]:
                if a.entity_target is None or a.entity_target >= len(cons):
                    continue
                k = key_of(cons[a.entity_target])
                if k in keep:
                    continue
                if FLAGS["tarot_targeting"] and k in TARGETED_TAROTS and tarots_kept < 1:
                    tarots_kept += 1
                    continue
                return a
        elif len(cons) >= 2 and not any(key_of(c) == "c_soul" for c in cons):
            if by_type[ActionType.SellConsumable]:
                return by_type[ActionType.SellConsumable][0]
        shop_cards = gs.get("shop_cards", [])
        # Buy the best build piece on offer, gated by the interest reserve:
        # build-critical pieces (soul/xmult) spend freely, support dips $10
        # under the reserve, filler respects it fully.
        best, best_rank = None, 0
        for a in by_type[ActionType.BuyCard]:
            if a.entity_target is None or a.entity_target >= len(shop_cards):
                continue
            card = shop_cards[a.entity_target]
            k = key_of(card)
            cost = getattr(card, "cost", 0)
            rank = (
                6 if k == "c_soul"
                else 5 if k in XMULT
                else 4 if k in SCALER
                else 3 if k == PLANET_FLUSH
                else 2 if k.startswith("j_") and len(jokers) < 4
                else 1 if k in ECON_TAROTS
                else 0
            )
            if k in PLAN["priority_buys"]:
                rank = max(rank, 5)
            elif k in PLAN["avoid_buys"]:
                rank = 0
            if rank in (3, 4) and dollars - cost < reserve - 10:
                rank = 0
            elif rank in (1, 2) and dollars - cost < reserve:
                rank = 0
            if rank > best_rank:
                best, best_rank = a, rank
        if best is not None:
            return best
        # Board full of junk while shop offers an upgrade: sell the worst.
        if len(jokers) >= 5 and by_type[ActionType.SellJoker]:
            offer = max((joker_tier(key_of(c)) for c in shop_cards), default=0)
            worst_i = min(range(len(joker_keys)), key=lambda i: joker_tier(joker_keys[i]))
            if offer > joker_tier(joker_keys[worst_i]):
                for a in by_type[ActionType.SellJoker]:
                    if a.entity_target == worst_i:
                        return a
        boosters = gs.get("shop_boosters", [])
        if FLAGS["peek"]:
            # Clairvoyant pack choice: simulate each open on a clone (streams
            # are deterministic, so the peek IS the real contents); open the
            # best pack only if it holds something worth picking.
            best_pack, best_val = None, 0
            for a in by_type[ActionType.OpenBooster]:
                if a.entity_target is None or a.entity_target >= len(boosters):
                    continue
                if dollars - getattr(boosters[a.entity_target], "cost", 0) < reserve - 10:
                    continue
                sim = clone(gs)
                try:
                    step_factored(sim, a)
                except Exception:  # noqa: BLE001
                    continue
                val = max((_pick_rank(key_of(c)) for c in sim.get("pack_cards", [])), default=0)
                if val > best_val:
                    best_pack, best_val = a, val
            surplus = dollars - reserve > 12
            if best_pack is not None and (best_val >= 12 or (surplus and best_val >= 4)):
                return best_pack
        else:
            for pref in PACK_PREF:
                for a in by_type[ActionType.OpenBooster]:
                    if a.entity_target is not None and a.entity_target < len(boosters):
                        booster = boosters[a.entity_target]
                        if key_of(booster).startswith(pref):
                            if dollars - getattr(booster, "cost", 0) >= reserve:
                                return a
        if by_type[ActionType.RedeemVoucher] and dollars >= max(18, reserve + 10):
            return by_type[ActionType.RedeemVoucher][0]
        if FLAGS["peek"] and by_type[ActionType.Reroll]:
            # Clairvoyant reroll: CHAIN the lookahead — walk up to 4 rerolls
            # deep on a clone (streams are deterministic) and pay for the
            # first real reroll if a build-critical piece is reachable
            # within budget. Single-step peeking deadlocks: a bad next shop
            # never changes, so the router would refuse rerolls forever.
            budget = dollars - (0 if desperate else max(reserve - 10, 0))
            sim = clone(gs)
            spent = 0.0
            for _ in range(4):
                acts = [a for a in legal_factored(sim) if a.action_type == ActionType.Reroll]
                if not acts:
                    break
                d0 = sim.get("dollars", 0)
                try:
                    step_factored(sim, acts[0])
                except Exception:  # noqa: BLE001
                    break
                spent += d0 - sim.get("dollars", 0)
                if spent > budget:
                    break
                nxt = [key_of(c) for c in sim.get("shop_cards", [])]
                if any(k in XMULT or k == "c_soul" for k in nxt):
                    return by_type[ActionType.Reroll][0]
            # Nothing reachable: desperation still burns for tempo.
            if desperate and dollars >= 5:
                return by_type[ActionType.Reroll][0]
        # Shop offers nothing build-relevant: buy another look while rich.
        # Desperate (no xmult, ante 3+): burn everything hunting one.
        elif by_type[ActionType.Reroll] and (desperate or dollars >= reserve + 8):
            return by_type[ActionType.Reroll][0]
        if by_type[ActionType.NextRound]:
            return by_type[ActionType.NextRound][0]
        return None

    if phase == GamePhase.PACK_OPENING:
        pack_cards = gs.get("pack_cards", [])
        best, best_rank = None, 0
        for a in by_type[ActionType.PickPackCard]:
            if a.entity_target is None or a.entity_target >= len(pack_cards):
                continue
            rank = _pick_rank(key_of(pack_cards[a.entity_target]))
            if rank > best_rank:
                best, best_rank = a, rank
        if best_rank >= 10:
            return best
        if by_type[ActionType.SkipPack]:
            return by_type[ActionType.SkipPack][0]
        return best  # junk pick beats being stuck

    return None
