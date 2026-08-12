"""Falsifiable gate for forced-card (Cerulean Bell) enumeration.

The engine rejects any play/discard omitting the forced card
(game.py _require_forced_card, added 2026-07-26 for bug #67), but the
action mask cannot express must-include, so legal_factored must filter
combos itself. Before the filter, every rollout reaching a Cerulean
Bell boss crashed its game (first seen in v11, game SP50W5G4).

Checks (exit non-zero on failure):
  1. With hand[k] flagged forced_selection (the exact flag the engine
     reads), every enumerated Play/Discard combo contains k.
  2. Stepping every such action on clones raises no IllegalActionError.
  3. Control: without the flag, combos omitting k exist (the filter is
     not simply always-on).

Run from balatro-zero/:  uv run --no-sync python scripts/forced_card_probe.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from jackdaw.env.action_space import ActionType

from balatro_zero.state import clone, legal_factored, new_run, step_factored

CARD_TYPES = (int(ActionType.PlayHand), int(ActionType.Discard))
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def hand_state(seed: str):
    gs = new_run(seed)
    sel = [a for a in legal_factored(gs) if a.action_type == ActionType.SelectBlind]
    step_factored(gs, sel[0])
    return gs


def card_actions(gs):
    return [a for a in legal_factored(gs)
            if int(a.action_type) in CARD_TYPES and a.card_target]


def main() -> int:
    gs = hand_state("BELLPROBE")
    k = 2
    gs["hand"][k].ability["forced_selection"] = True

    acts = card_actions(gs)
    check("play/discard combos all include the forced card",
          len(acts) > 0 and all(k in a.card_target for a in acts),
          f"{len(acts)} actions")

    bad = 0
    for a in acts:
        sim = clone(gs)
        try:
            step_factored(sim, a)
        except Exception as e:  # noqa: BLE001
            bad += 1
            first = first if bad > 1 else f"{a.card_target}: {e}"
    check("every enumerated action steps cleanly", bad == 0,
          f"{bad} rejected" + (f" (first: {first})" if bad else ""))

    ctrl = hand_state("BELLPROBE")
    ctrl_acts = card_actions(ctrl)
    check("control: unforced hand enumerates combos omitting the card",
          any(k not in a.card_target for a in ctrl_acts),
          f"{len(ctrl_acts)} actions")

    if FAILURES:
        print(f"\nGATE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
