"""Interactive run driver: Claude (or a human) makes every econ decision;
the beam plays hands. State persists to disk between invocations.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/interactive_run.py --new SEED
    uv run --no-sync python scripts/interactive_run.py --act N

Each call advances the game to the next econ decision point (hand phases
auto-played by plan_blind, cash-out automatic), prints a state summary and
numbered options, then saves state and exits.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from jackdaw.engine.actions import GamePhase
from jackdaw.env.action_space import ActionType

from balatro_zero.goldprobe import plan_blind
from balatro_zero.router import flags_override, key_of
from balatro_zero.state import (
    ante,
    is_terminal,
    legal_factored,
    new_run,
    progress,
    step_factored,
    won,
)

STATE = Path("runs/interactive/state.pkl")  # default; override with --state

DECISION_TYPES = {
    int(ActionType.BuyCard), int(ActionType.RedeemVoucher),
    int(ActionType.OpenBooster), int(ActionType.Reroll),
    int(ActionType.NextRound), int(ActionType.SelectBlind),
    int(ActionType.SkipBlind), int(ActionType.SkipPack),
    int(ActionType.PickPackCard), int(ActionType.SellJoker),
    int(ActionType.SellConsumable), int(ActionType.UseConsumable),
}

HANDS = ["Flush", "Straight", "Two Pair", "Pair", "Three of a Kind",
         "Full House", "Four of a Kind", "High Card", "Straight Flush", "Flush House"]


def describe(gs, a) -> str:
    t = ActionType(a.action_type)
    def key_at(coll, i):
        items = gs.get(coll, [])
        if i is None or i >= len(items):
            return "?"
        c = items[i]
        return f"{key_of(c)} ${getattr(c, 'cost', '?')}"
    if t == ActionType.BuyCard:
        return f"BUY {key_at('shop_cards', a.entity_target)}"
    if t == ActionType.RedeemVoucher:
        return f"VOUCHER {key_at('shop_vouchers', a.entity_target)}"
    if t == ActionType.OpenBooster:
        return f"OPEN {key_at('shop_boosters', a.entity_target)}"
    if t == ActionType.PickPackCard:
        items = gs.get("pack_cards", [])
        i = a.entity_target
        extra = f" targets={a.card_target}" if a.card_target else ""
        return f"PICK {key_of(items[i]) if i is not None and i < len(items) else '?'}{extra}"
    if t == ActionType.SellJoker:
        return f"SELL {key_at('jokers', a.entity_target)}"
    if t == ActionType.SellConsumable:
        return f"SELLC {key_at('consumables', a.entity_target)}"
    if t == ActionType.UseConsumable:
        items = gs.get("consumables", [])
        i = a.entity_target
        extra = f" targets={a.card_target}" if a.card_target else ""
        return f"USE {key_of(items[i]) if i is not None and i < len(items) else '?'}{extra}"
    if t == ActionType.SkipBlind:
        tag = gs.get("round_resets", {}).get("blind_tags", {}).get(
            gs.get("blind_on_deck", ""), "?")
        return f"SKIP BLIND (tag: {tag})"
    return t.name


def summary(gs) -> str:
    rr = gs.get("round_resets", {})
    lines = []
    lines.append(
        f"ante {ante(gs)} | blind on deck: {gs.get('blind_on_deck')} "
        f"| round {gs.get('round', 0)} | prog {progress(gs):.3f} | ${gs.get('dollars', 0)}"
    )
    choices = rr.get("blind_choices", {})
    if choices:
        lines.append(f"blinds this ante: {choices}")
    tags = rr.get("blind_tags", {})
    if tags:
        lines.append(f"skip tags: {tags}")
    lines.append(
        "board: " + (", ".join(key_of(c) for c in gs.get("jokers", [])) or "(empty)")
    )
    lines.append(
        "consumables: " + (", ".join(key_of(c) for c in gs.get("consumables", [])) or "(none)")
    )
    hl = gs.get("hand_levels")
    if hl is not None:
        lvls = []
        for h in HANDS:
            try:
                chips, mult = hl.get(h)
                lvls.append(f"{h} {chips}x{mult}")
            except Exception:  # noqa: BLE001
                continue
        lines.append("hand values: " + " | ".join(lvls[:8]))
    cards = gs.get("playing_cards") or gs.get("deck", [])
    suits: dict = {}
    for c in cards:
        s = getattr(getattr(c, "base", None), "suit", None)
        suits[s] = suits.get(s, 0) + 1
    lines.append(f"full deck ({len(cards)}): " + ", ".join(f"{k}:{v}" for k, v in suits.items()))
    if gs.get("phase") == GamePhase.SHOP:
        lines.append(
            "shop: cards=[" + ", ".join(f"{key_of(c)} ${getattr(c, 'cost', '?')}" for c in gs.get("shop_cards", []))
            + "] vouchers=[" + ", ".join(f"{key_of(c)} ${getattr(c, 'cost', '?')}" for c in gs.get("shop_vouchers", []))
            + "] packs=[" + ", ".join(f"{key_of(c)} ${getattr(c, 'cost', '?')}" for c in gs.get("shop_boosters", []))
            + "]"
        )
    if gs.get("phase") == GamePhase.PACK_OPENING:
        lines.append("pack contents: " + ", ".join(key_of(c) for c in gs.get("pack_cards", [])))
    return "\n".join(lines)


def advance(gs) -> list:
    """Auto-play until an econ decision point; return decision options."""
    guard = 0
    with flags_override(peek=True, skip_tags=False):
        while not is_terminal(gs) and not won(gs) and guard < 1200:
            guard += 1
            phase = gs.get("phase")
            if phase == GamePhase.SELECTING_HAND:
                seq = plan_blind(gs)
                if not seq:
                    legal = legal_factored(gs)
                    if not legal:
                        return []
                    seq = [legal[0]]
                for a in seq:
                    if is_terminal(gs) or won(gs):
                        break
                    try:
                        step_factored(gs, a)
                    except Exception:  # noqa: BLE001
                        break
                continue
            if phase == GamePhase.ROUND_EVAL:
                legal = legal_factored(gs)
                cash = [a for a in legal if a.action_type == ActionType.CashOut]
                if cash:
                    step_factored(gs, cash[0])
                    continue
            legal = legal_factored(gs)
            opts = [a for a in legal if a.action_type in DECISION_TYPES]
            if opts:
                return opts
            if not legal:
                return []
            step_factored(gs, legal[0])
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=str, default=None)
    ap.add_argument("--act", type=int, default=None)
    ap.add_argument("--state", type=str, default=None)
    args = ap.parse_args()

    global STATE
    if args.state:
        STATE = Path(args.state)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    if args.new:
        gs = new_run(args.new)
        moves = []
    else:
        gs, moves = pickle.loads(STATE.read_bytes())
        opts = pickle.loads(Path(str(STATE) + ".opts").read_bytes())
        if args.act is None or not (0 <= args.act < len(opts)):
            print(f"need --act 0..{len(opts) - 1}")
            return
        a = opts[args.act]
        moves.append(describe(gs, a))
        try:
            step_factored(gs, a)
        except Exception as e:  # noqa: BLE001
            print(f"ILLEGAL ({e}); state unchanged")

    opts = advance(gs)
    print(summary(gs))
    if won(gs):
        print(f"\n*** WON after {len(moves)} decisions ***")
    elif is_terminal(gs):
        blind = gs.get("blind")
        target = getattr(blind, "chips", 0) if blind else 0
        print(f"\n*** GAME OVER (ante {ante(gs)}, {gs.get('chips', 0)}/{target}) "
              f"after {len(moves)} decisions ***")
    elif not opts:
        print("\n*** no decisions available (stuck) ***")
    else:
        print("\noptions:")
        for i, a in enumerate(opts):
            print(f"  [{i}] {describe(gs, a)}")
    STATE.write_bytes(pickle.dumps((gs, moves), protocol=5))
    Path(str(STATE) + ".opts").write_bytes(pickle.dumps(opts, protocol=5))


if __name__ == "__main__":
    main()
