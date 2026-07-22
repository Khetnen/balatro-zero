"""Interactive run driver: an agent (Claude, an LLM, or a human) makes the
strategy decisions; the beam handles residual chip extraction. State
persists to disk between invocations.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/interactive_run.py --new SEED
    uv run --no-sync python scripts/interactive_run.py --act "BUY j_obelisk"
    uv run --no-sync python scripts/interactive_run.py --act "play Kh Qh Jh Th 9h"

Every blind stops at the first hand (and after each non-auto step) so the
agent sees the draw before deciding. At a HAND stop --act accepts:
    pass                     beam takes its single best step
    auto                     beam finishes the rest of this blind
    play  <cards>            pin an exact play, e.g.  play Kh Qh 7s
    discard <cards>          pin an exact discard
    use <consumable> [on <cards>]   e.g.  use c_sixth_sense on 6d
    copy <card> onto <card>  Death macro: free swaps set direction, then fires
    order jokers <k1, k2..>  free joker reorder macro (also works in shops)
    veto <hand types>        beam must not play these (e.g. veto Flush)
    require <hand types>     beam plays only these
    clear constraints        drop veto/require
Cards are rank+suit tokens (Kh, Th, 9c; # suffix picks duplicates: Kh#2).
Constraints persist across blinds until cleared.

At ECON stops --act accepts an option description substring (or index,
optionally guarded with --expect SUBSTR), exactly as before; "order
jokers" also works there. A failed or ambiguous --act applies NOTHING
and reprints the options.
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.hand_eval import evaluate_hand
from jackdaw.env.action_space import ActionType

from balatro_zero.goldprobe import plan_blind
from balatro_zero.router import flags_override, key_of, targeted_tarot_action
from balatro_zero.state import (
    ante,
    is_terminal,
    legal_factored,
    new_run,
    progress,
    step_factored,
    won,
)
from jackdaw.env.game_spec import FactoredAction

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

HAND_TYPES = {h.lower(): h for h in [
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush",
    "Full House", "Four of a Kind", "Straight Flush", "Five of a Kind",
    "Flush House", "Flush Five",
]}

RANK_CHAR = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J", "10": "T"}

CTL0 = {"auto_round": -1, "veto": [], "require": []}


# ---------------------------------------------------------------------------
# Card labels & token parsing
# ---------------------------------------------------------------------------


def card_label(c) -> str:
    """Short label like Kh / Ts / 9c, annotated with enhancement/seal."""
    base = getattr(c, "base", None)
    if base is None:
        return key_of(c)
    rank = str(getattr(base.rank, "value", base.rank))
    suit = str(getattr(base.suit, "value", base.suit))
    lbl = RANK_CHAR.get(rank, rank[:1]) + suit[:1].lower()
    tags = []
    name = (getattr(c, "ability", None) or {}).get("name", "")
    if name and name != "Default Base":
        tags.append(name.replace(" Card", "").lower())
    seal = getattr(c, "seal", None)
    if seal:
        tags.append(f"{seal}seal".lower())
    ed = getattr(c, "edition", None)
    if isinstance(ed, dict) and ed.get("type"):
        tags.append(str(ed["type"]))
    return f"{lbl}[{'/'.join(tags)}]" if tags else lbl


def hand_labels(gs) -> list[str]:
    """Labels for the current hand, disambiguating duplicates with #n."""
    labels = [card_label(c) for c in gs.get("hand", [])]
    seen: dict[str, int] = {}
    out = []
    for lbl in labels:
        plain = lbl.split("[")[0]
        seen[plain] = seen.get(plain, 0) + 1
        out.append(lbl if seen[plain] == 1 else
                   lbl.replace(plain, f"{plain}#{seen[plain]}", 1))
    return out


def parse_card_tokens(gs, text: str) -> list[int] | None:
    """Map space/comma-separated card tokens to hand indices, or None.

    Tokens are plain labels (Kh, 10s/Ts) or #-disambiguated (Kh#2); a
    plain token with duplicates takes the leftmost not yet used."""
    labels = [l.split("[")[0].lower() for l in hand_labels(gs)]
    out: list[int] = []
    for tok in re.split(r"[\s,]+", text.strip()):
        if not tok:
            continue
        t = tok.lower().replace("10", "t", 1)
        matches = [i for i, l in enumerate(labels)
                   if (l == t or l.split("#")[0] == t) and i not in out]
        if not matches:
            print(f"no card {tok!r} in hand ({' '.join(hand_labels(gs))})")
            return None
        out.append(matches[0])
    return out or None


# ---------------------------------------------------------------------------
# Describe / summary
# ---------------------------------------------------------------------------


def describe(gs, a) -> str:
    t = ActionType(a.action_type)
    def key_at(coll, i):
        items = gs.get(coll, [])
        if i is None or i >= len(items):
            return "?"
        c = items[i]
        return f"{key_of(c)} ${getattr(c, 'cost', '?')}"
    def sell_at(coll, i):
        items = gs.get(coll, [])
        if i is None or i >= len(items):
            return "?"
        c = items[i]
        return f"{key_of(c)} ${getattr(c, 'sell_cost', '?')}"
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
        return f"SELL {sell_at('jokers', a.entity_target)}"
    if t == ActionType.SellConsumable:
        return f"SELLC {sell_at('consumables', a.entity_target)}"
    if t == ActionType.UseConsumable:
        items = gs.get("consumables", [])
        i = a.entity_target
        extra = ""
        if a.card_target:
            hl = hand_labels(gs)
            extra = " on " + " ".join(
                hl[j] if j < len(hl) else str(j) for j in a.card_target)
        return f"USE {key_of(items[i]) if i is not None and i < len(items) else '?'}{extra}"
    if t == ActionType.SkipBlind:
        tag = gs.get("round_resets", {}).get("blind_tags", {}).get(
            gs.get("blind_on_deck", ""), "?")
        return f"SKIP BLIND (tag: {tag})"
    if t == ActionType.PlayHand:
        return f"PLAY {_combo_desc(gs, a.card_target)}"
    if t == ActionType.Discard:
        hl = hand_labels(gs)
        return "DISCARD " + " ".join(hl[i] for i in a.card_target if i < len(hl))
    if t in (ActionType.SwapJokersLeft, ActionType.SwapJokersRight):
        return f"{t.name} {a.entity_target}"
    return t.name


def _combo_desc(gs, combo) -> str:
    hand = gs.get("hand", [])
    hl = hand_labels(gs)
    cards = [hand[i] for i in combo if i < len(hand)]
    ht = evaluate_hand(cards, gs.get("jokers", [])).detected_hand
    return " ".join(hl[i] for i in combo if i < len(hl)) + f" ({ht})"


def summary(gs, ctl) -> str:
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
    def _card_label(c) -> str:
        ed = getattr(c, "edition", None)
        ed_s = ""
        if ed:
            if isinstance(ed, dict):
                ed_s = "/".join(k for k, v in ed.items() if v and k != "type")
                ed_s = ed_s or str(ed.get("type", ""))
            else:
                ed_s = str(ed)
        return f"{key_of(c)}[{ed_s}]" if ed_s else key_of(c)

    lines.append(
        "board: " + (", ".join(_card_label(c) for c in gs.get("jokers", [])) or "(empty)")
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
    if gs.get("phase") == GamePhase.SELECTING_HAND:
        cr = gs.get("current_round", {})
        blind = gs.get("blind")
        target = getattr(blind, "chips", 0) if blind else 0
        lines.append(
            f"blind: {getattr(blind, 'name', '?')} {gs.get('chips', 0)}/{target}"
            f" | hands {cr.get('hands_left', 0)} discards {cr.get('discards_left', 0)}"
        )
        lines.append("hand: " + " ".join(hand_labels(gs)))
    if ctl.get("veto") or ctl.get("require"):
        lines.append(f"constraints: veto={ctl.get('veto')} require={ctl.get('require')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Advance
# ---------------------------------------------------------------------------


def _constraints(ctl) -> dict | None:
    if ctl.get("veto") or ctl.get("require"):
        return {"veto": ctl["veto"], "require": ctl["require"]}
    return None


def _hand_options(gs, ctl) -> list[dict]:
    seq = plan_blind(gs, _constraints(ctl))
    first = seq[0] if seq else None
    if first is None:
        legal = legal_factored(gs)
        first = legal[0] if legal else None
    opts = []
    if first is not None:
        opts.append({"kind": "pass", "action": first,
                     "desc": f"PASS - beam: {describe(gs, first)}"})
    opts.append({"kind": "auto", "action": None,
                 "desc": "AUTO - beam finishes this blind"})
    for a in legal_factored(gs):
        if a.action_type == ActionType.UseConsumable:
            opts.append({"kind": "action", "action": a, "desc": describe(gs, a)})
    return opts


def advance(gs, ctl) -> tuple[str, list]:
    """Play forward to the next decision point.

    Returns (kind, opts) with kind 'hand' | 'econ' | 'end'; opts entries
    are {"kind": "pass"|"auto"|"action", "action": FactoredAction|None,
    "desc": str}."""
    guard = 0
    with flags_override(peek=True, skip_tags=False):
        while not is_terminal(gs) and not won(gs) and guard < 2000:
            guard += 1
            phase = gs.get("phase")
            if phase == GamePhase.SELECTING_HAND:
                if ctl.get("auto_round") == gs.get("round", 0):
                    # Auto mode: fire held enhancement tarots (cross-blind
                    # payoff the beam never chooses), then execute the beam
                    # plan; outer loop replans if the blind continues.
                    fired = 0
                    while fired < 2:
                        t = targeted_tarot_action(gs)
                        if t is None:
                            break
                        try:
                            step_factored(gs, t)
                            fired += 1
                        except Exception:  # noqa: BLE001
                            break
                    seq = plan_blind(gs, _constraints(ctl))
                    if not seq:
                        legal = legal_factored(gs)
                        if not legal:
                            return "end", []
                        seq = [legal[0]]
                    for a in seq:
                        if is_terminal(gs) or won(gs):
                            break
                        try:
                            step_factored(gs, a)
                        except Exception:  # noqa: BLE001
                            break
                    continue
                return "hand", _hand_options(gs, ctl)
            if phase == GamePhase.ROUND_EVAL:
                legal = legal_factored(gs)
                cash = [a for a in legal if a.action_type == ActionType.CashOut]
                if cash:
                    step_factored(gs, cash[0])
                    continue
            legal = legal_factored(gs)
            opts = [a for a in legal if a.action_type in DECISION_TYPES]
            if opts:
                return "econ", [
                    {"kind": "action", "action": a, "desc": describe(gs, a)}
                    for a in opts
                ]
            if not legal:
                return "end", []
            step_factored(gs, legal[0])
    return "end", []


# ---------------------------------------------------------------------------
# Command handling
# ---------------------------------------------------------------------------


def _find_consumable(gs, token: str) -> int | None:
    matches = [i for i, c in enumerate(gs.get("consumables", []))
               if token.lower() in key_of(c).lower()]
    if len(matches) != 1:
        print(f"consumable {token!r}: {len(matches)} matches "
              f"({[key_of(c) for c in gs.get('consumables', [])]})")
        return None
    return matches[0]


def _parse_hand_types(text: str) -> list[str] | None:
    out = []
    for part in re.split(r"[,;]+", text):
        p = part.strip().lower()
        if not p:
            continue
        if p not in HAND_TYPES:
            print(f"unknown hand type {part.strip()!r}; known: "
                  + ", ".join(sorted(set(HAND_TYPES.values()))))
            return None
        out.append(HAND_TYPES[p])
    return out


def _order_jokers(gs, text: str) -> bool:
    """Bring named jokers to the front in the given order via free swaps."""
    jokers = gs.get("jokers", [])
    refs: list = []
    for part in re.split(r"[,;]+", text):
        tok = part.strip().lower()
        if not tok:
            continue
        matches = [j for j in jokers
                   if tok in key_of(j).lower()
                   and not any(j is r for r in refs)]
        if not matches:
            print(f"no joker matches {part.strip()!r} "
                  f"(board: {[key_of(j) for j in jokers]})")
            return False
        refs.append(matches[0])
    for pos, ref in enumerate(refs):
        cur = next(i for i, j in enumerate(jokers) if j is ref)
        while cur > pos:
            step_factored(gs, FactoredAction(
                action_type=int(ActionType.SwapJokersLeft), entity_target=cur))
            cur -= 1
    return True


def _death_macro(gs, src_tok: str, dst_tok: str) -> FactoredAction | None:
    """copy A onto B: free-swap A rightmost of the pair, return the USE."""
    di = next((i for i, c in enumerate(gs.get("consumables", []))
               if key_of(c) == "c_death"), None)
    if di is None:
        print("no c_death held")
        return None
    src = parse_card_tokens(gs, src_tok)
    dst = parse_card_tokens(gs, dst_tok)
    if not src or not dst or src[0] == dst[0]:
        print("copy needs two distinct hand cards")
        return None
    hand = gs.get("hand", [])
    a_card, b_card = hand[src[0]], hand[dst[0]]
    while hand.index(a_card) < hand.index(b_card):
        step_factored(gs, FactoredAction(
            action_type=int(ActionType.SwapHandRight),
            entity_target=hand.index(a_card)))
    targets = tuple(sorted((hand.index(a_card), hand.index(b_card))))
    return FactoredAction(
        action_type=int(ActionType.UseConsumable),
        entity_target=di, card_target=targets)


def resolve_hand_act(gs, ctl, opts: list[dict], act: str):
    """Return (FactoredAction|('auto')|None, changed_ctl: bool)."""
    a = act.strip()
    low = a.lower()
    if low == "pass":
        entry = opts[0] if opts and opts[0]["kind"] == "pass" else None
        return (entry["action"] if entry else None), False
    if low == "auto":
        return "auto", False
    if m := re.match(r"^play\s+(.+)$", a, re.I):
        idxs = parse_card_tokens(gs, m.group(1))
        if idxs is None:
            return None, False
        return FactoredAction(action_type=int(ActionType.PlayHand),
                              card_target=tuple(sorted(idxs))), False
    if m := re.match(r"^discard\s+(.+)$", a, re.I):
        idxs = parse_card_tokens(gs, m.group(1))
        if idxs is None:
            return None, False
        return FactoredAction(action_type=int(ActionType.Discard),
                              card_target=tuple(sorted(idxs))), False
    if m := re.match(r"^use\s+(\S+)(?:\s+on\s+(.+))?$", a, re.I):
        ci = _find_consumable(gs, m.group(1))
        if ci is None:
            return None, False
        targets = None
        if m.group(2):
            idxs = parse_card_tokens(gs, m.group(2))
            if idxs is None:
                return None, False
            targets = tuple(sorted(idxs))
        return FactoredAction(action_type=int(ActionType.UseConsumable),
                              entity_target=ci, card_target=targets), False
    if m := re.match(r"^copy\s+(\S+)\s+onto\s+(\S+)$", a, re.I):
        act_or_none = _death_macro(gs, m.group(1), m.group(2))
        return act_or_none, False
    if m := re.match(r"^(veto|require)\s+(.+)$", a, re.I):
        hts = _parse_hand_types(m.group(2))
        if hts is None:
            return None, False
        key = m.group(1).lower()
        ctl[key] = sorted(set(ctl.get(key, [])) | set(hts))
        print(f"{key} -> {ctl[key]}")
        return None, True
    if low in ("clear constraints", "clear"):
        ctl["veto"], ctl["require"] = [], []
        print("constraints cleared")
        return None, True
    if m := re.match(r"^order\s+jokers?\s+(.+)$", a, re.I):
        return ("order", m.group(1)), False
    # fall back: index / description match against enumerated options
    i = resolve_act([o["desc"] for o in opts], act, None)
    if i is None:
        return None, False
    if opts[i]["kind"] == "auto":
        return "auto", False
    return opts[i]["action"], False


def resolve_act(descs: list[str], act: str | None, expect: str | None) -> int | None:
    """Map --act (index or description substring) to an option index.

    Returns None (and prints why + the live options) instead of guessing:
    a failed resolution must leave the state untouched."""
    def show(msg: str) -> None:
        print(msg)
        for i, d in enumerate(descs):
            print(f"  [{i}] {d}")
    try:
        i = int(act) if act is not None else -1
    except ValueError:
        hits = [i for i, d in enumerate(descs) if act.lower() in d.lower()]
        if not hits:
            show(f"no option matches {act!r}; options:")
            return None
        if len({descs[i] for i in hits}) > 1:
            show(f"ambiguous {act!r} ("
                 + ", ".join(f"[{i}]" for i in hits) + "); options:")
            return None
        return hits[0]
    if not (0 <= i < len(descs)):
        show(f"need --act 0..{len(descs) - 1} or a description substring; options:")
        return None
    if expect and expect.lower() not in descs[i].lower():
        show(f"--expect {expect!r} does not match [{i}] {descs[i]!r}; nothing applied:")
        return None
    return i


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def apply_act(gs, ctl, kind: str, opts: list[dict], act: str,
              expect: str | None) -> str | None:
    """Apply one --act. Returns a move-log entry, or None if nothing ran."""
    if kind == "hand":
        resolved, ctl_changed = resolve_hand_act(gs, ctl, opts, act)
        if ctl_changed:
            return ""  # constraint update: no game action, reprint state
        if resolved is None:
            return None
        if resolved == "auto":
            ctl["auto_round"] = gs.get("round", 0)
            return "AUTO"
        if isinstance(resolved, tuple) and resolved[0] == "order":
            return "ORDER" if _order_jokers(gs, resolved[1]) else None
        try:
            entry = describe(gs, resolved)
            step_factored(gs, resolved)
            return entry
        except Exception as e:  # noqa: BLE001
            print(f"ILLEGAL ({e}); state unchanged")
            return None
    # econ stop
    if m := re.match(r"^order\s+jokers?\s+(.+)$", act.strip(), re.I):
        return "ORDER" if _order_jokers(gs, m.group(1)) else None
    i = resolve_act([o["desc"] for o in opts], act, expect)
    if i is None:
        return None
    a = opts[i]["action"]
    try:
        entry = describe(gs, a)
        step_factored(gs, a)
        return entry
    except Exception as e:  # noqa: BLE001
        print(f"ILLEGAL ({e}); state unchanged")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=str, default=None)
    ap.add_argument("--act", type=str, default=None)
    ap.add_argument("--expect", type=str, default=None)
    ap.add_argument("--state", type=str, default=None)
    args = ap.parse_args()

    global STATE
    if args.state:
        STATE = Path(args.state)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    if args.new:
        gs = new_run(args.new)
        moves: list[str] = []
        ctl = dict(CTL0)
    else:
        payload = pickle.loads(STATE.read_bytes())
        gs, moves, ctl = payload if len(payload) == 3 else (*payload, dict(CTL0))
        kind, opts = pickle.loads(Path(str(STATE) + ".opts").read_bytes())
        if args.act is None:
            print(f"need --act (stop kind: {kind})")
            return
        entry = apply_act(gs, ctl, kind, opts, args.act, args.expect)
        if entry:
            moves.append(entry)

    kind, opts = advance(gs, ctl)
    print(summary(gs, ctl))
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
        print(f"\n[{kind.upper()} stop] options:")
        for i, o in enumerate(opts):
            print(f"  [{i}] {o['desc']}")
        if kind == "hand":
            print("  (free-form: play/discard <cards>, use <cons> [on <cards>], "
                  "copy A onto B, order jokers .., veto/require <hand types>, clear)")
    STATE.write_bytes(pickle.dumps((gs, moves, ctl), protocol=5))
    Path(str(STATE) + ".opts").write_bytes(pickle.dumps((kind, opts), protocol=5))


if __name__ == "__main__":
    main()
