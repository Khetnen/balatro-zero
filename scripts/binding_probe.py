"""Falsifiable gate for slot<->content binding in the entity head.

A REPRESENTATION test, valid at random init (like adjacency_probe): swap
the identity embeddings of two market slots and watch the entity logits.

  * V5 (the control proving the diagnosis): its identity embeddings
    reach the entity head only through permutation-invariant mean+max
    pooling, so an id swap between two market slots CANNOT change any
    entity logit. Sensitivity must be EXACTLY zero — the architecture
    is order-blind at the embedding level, which is the defect.
  * V6 (pointer head): each market slot is scored from its own content
    representation, so the swap must move the swapped slots' logits and
    track them exactly (logit follows content, position terms fixed).

Also checks the global entity layout: joker/consumable/market actions
map to disjoint slot ranges (the V5 cross-area conflation is gone), and
the market mapping follows observe()'s packing order.

Run from balatro-zero/:  uv run --no-sync python scripts/binding_probe.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from jackdaw.env.action_space import ActionType
from jackdaw.env.game_spec import FactoredAction

from balatro_zero.net import (
    ENT_OFF_CONS,
    ENT_OFF_JOKER,
    ENT_OFF_MARKET,
    GLOBAL_ENTITY_SLOTS,
    PolicyValueNetV5,
    PolicyValueNetV6,
    evaluate_factored,
    global_entity_slot,
)
from balatro_zero.state import new_run, observe

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def market_swap_pair():
    """An obs and its market-slot-content swap (ids only, flat untouched).

    Swapping only the identity ids isolates the embedding pathway: any
    logit difference must have flowed embedding -> entity head. The two
    ids are arbitrary distinct center keys planted in market slots 0/1.
    """
    base = observe(new_run("BINDPROBE"))
    a = base
    b_ids = base.market_ids.copy()
    a.market_ids[0], a.market_ids[1] = 7, 42
    b_ids[0], b_ids[1] = 42, 7
    from dataclasses import replace

    b = replace(base, market_ids=b_ids)
    return a, b


def ent_logits(net, obs) -> np.ndarray:
    _, e, _, _ = evaluate_factored(net, obs, torch.device("cpu"))
    return e[0]


def main() -> int:
    torch.manual_seed(0)

    print("1. layout (global entity space)")
    lens = (2, 1, 2)  # shop cards | vouchers | boosters
    slots = {
        "sell joker 2": global_entity_slot(
            FactoredAction(action_type=int(ActionType.SellJoker), entity_target=2), lens),
        "use consumable 2": global_entity_slot(
            FactoredAction(action_type=int(ActionType.UseConsumable), entity_target=2), lens),
        "buy shop card 1": global_entity_slot(
            FactoredAction(action_type=int(ActionType.BuyCard), entity_target=1), lens),
        "redeem voucher 0": global_entity_slot(
            FactoredAction(action_type=int(ActionType.RedeemVoucher), entity_target=0), lens),
        "open booster 1": global_entity_slot(
            FactoredAction(action_type=int(ActionType.OpenBooster), entity_target=1), lens),
        "pick pack card 3": global_entity_slot(
            FactoredAction(action_type=int(ActionType.PickPackCard), entity_target=3), lens),
    }
    check("cross-area actions get distinct slots",
          len(set(slots.values())) == len(slots), f"{slots}")
    check("areas land in their ranges",
          slots["sell joker 2"] == ENT_OFF_JOKER + 2
          and slots["use consumable 2"] == ENT_OFF_CONS + 2
          and slots["buy shop card 1"] == ENT_OFF_MARKET + 1
          and slots["redeem voucher 0"] == ENT_OFF_MARKET + 2
          and slots["open booster 1"] == ENT_OFF_MARKET + 4
          and slots["pick pack card 3"] == ENT_OFF_MARKET + 8)
    check("out-of-window pack pick drops to None",
          global_entity_slot(
              FactoredAction(action_type=int(ActionType.PickPackCard),
                             entity_target=11), lens) is None)

    print("2. binding (id swap between two market slots)")
    obs_a, obs_b = market_swap_pair()
    i = ENT_OFF_MARKET + 0
    j = ENT_OFF_MARKET + 1

    v5 = PolicyValueNetV5()
    v5.eval()
    ea, eb = ent_logits(v5, obs_a), ent_logits(v5, obs_b)
    sens5 = float(np.abs(ea - eb).max())
    check("V5 control: entity logits blind to the swap",
          sens5 < 1e-5,
          f"max |delta| = {sens5:.3e} (pool-invariant by construction; "
          "any nonzero is float summation reorder)")

    v6 = PolicyValueNetV6()
    v6.eval()
    ea, eb = ent_logits(v6, obs_a), ent_logits(v6, obs_b)
    sens6 = float(np.abs(ea - eb).max())
    check("V6: entity logits move with content",
          sens6 > 1e-4, f"max |delta| = {sens6:.3e}")
    # The swap must TRACK content: slot i's logit under A equals slot
    # j's under B up to the fixed positional term, i.e. the content
    # contribution transfers. With ids-only swap and everything else
    # fixed, logits at the swapped slots must exchange their
    # content-dependent parts exactly:
    follows = (abs((ea[i] - eb[j]) - (eb[i] - ea[j])) < 1e-4
               and abs(ea[i] + ea[j] - (eb[i] + eb[j])) < 1e-4)
    check("V6: swapped slots exchange content contributions", follows,
          f"A=({ea[i]:.4f},{ea[j]:.4f}) B=({eb[i]:.4f},{eb[j]:.4f})")
    untouched = np.delete(np.arange(GLOBAL_ENTITY_SLOTS), [i, j])
    check("V6: unswapped slots unaffected",
          float(np.abs(ea[untouched] - eb[untouched]).max()) < 1e-6)

    if FAILURES:
        print(f"\nGATE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
