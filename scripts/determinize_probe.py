"""Falsifiable gate for state.determinize — honest search futures.

Four checks on a mid-blind state, exit non-zero on any failure:

  1. PRESERVATION — determinize leaves everything the player has observed
     untouched: hand, jokers, dollars, chips, round, phase, blind target,
     and the deck MULTISET. The deck ORDER must change (that is the point).
  2. HONESTY — the same discard applied to N determinized clones draws
     different replacement cards across clones, while N clairvoyant clones
     all draw the SAME cards (the control that proves the diagnosis).
  3. REPRODUCIBILITY — the same numpy seed produces the same determinized
     future (honest != irreproducible; futures are pseudorandom from the
     search rng, just independent of the run's true future).
  4. INTEGRATION — gumbel_search runs determinized end-to-end on a fresh
     net, hand root and econ root both, and the source state is unmutated.

Run from balatro-zero/:  uv run --no-sync python scripts/determinize_probe.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from balatro_zero.net import PolicyValueNet
from balatro_zero.search import gumbel_search
from balatro_zero.state import (
    clone,
    determinize,
    legal_factored,
    new_run,
    step_factored,
)
from jackdaw.env.action_space import ActionType

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def card_key(c) -> tuple:
    """Identity of a physical card: unique sort_id when present."""
    sid = getattr(c, "sort_id", None)
    if sid is not None:
        return (sid,)
    base = getattr(c, "base", None)
    return (getattr(base, "suit", None), getattr(base, "id", None))


def deck_keys(gs) -> list[tuple]:
    return [card_key(c) for c in gs.get("deck", [])]


def hand_keys(gs) -> list[tuple]:
    return [card_key(c) for c in gs.get("hand", [])]


def mid_blind_state(seed: str):
    """Fresh run stepped into the small blind (hand dealt)."""
    gs = new_run(seed)
    legal = legal_factored(gs)
    sel = [a for a in legal if a.action_type == ActionType.SelectBlind]
    assert sel, "no SelectBlind at run start"
    step_factored(gs, sel[0])
    assert gs.get("hand"), "no hand after blind select"
    return gs


def pick_discard(gs, n_cards: int = 4):
    legal = legal_factored(gs)
    discards = [
        a for a in legal
        if a.action_type == ActionType.Discard
        and a.card_target is not None
        and len(a.card_target) >= n_cards
    ]
    assert discards, "no discard actions enumerated"
    # Largest discard = most replacement draws = sharpest honesty signal.
    return max(discards, key=lambda a: len(a.card_target))


def main() -> int:
    gs = mid_blind_state("DETPROBE1")

    # ---- 1. preservation -------------------------------------------------
    print("1. preservation")
    rng = np.random.default_rng(7)
    det = clone(gs)
    determinize(det, rng)
    check("hand untouched", hand_keys(det) == hand_keys(gs))
    check("jokers untouched",
          [getattr(j, "center_key", None) for j in det.get("jokers", [])]
          == [getattr(j, "center_key", None) for j in gs.get("jokers", [])])
    for field in ("dollars", "chips", "round", "phase"):
        check(f"{field} untouched", det.get(field) == gs.get(field))
    check("blind target untouched",
          getattr(det.get("blind"), "chips", None)
          == getattr(gs.get("blind"), "chips", None))
    check("deck multiset preserved",
          sorted(deck_keys(det)) == sorted(deck_keys(gs)),
          f"{len(det.get('deck', []))} cards")
    check("deck order changed", deck_keys(det) != deck_keys(gs))
    check("rng seed replaced", det["rng"].seed_str != gs["rng"].seed_str)

    # ---- 2. honesty ------------------------------------------------------
    print("2. honesty (same discard, different futures)")
    action = pick_discard(gs)
    n = 8

    def drawn_after(sim) -> tuple:
        before = set(hand_keys(sim))
        step_factored(sim, action)
        return tuple(sorted(set(hand_keys(sim)) - before))

    rng = np.random.default_rng(11)
    det_draws = set()
    for _ in range(n):
        sim = clone(gs)
        determinize(sim, rng)
        det_draws.add(drawn_after(sim))
    clair_draws = {drawn_after(clone(gs)) for _ in range(n)}
    check("clairvoyant clones all draw the true future (control)",
          len(clair_draws) == 1)
    check("determinized clones sample different futures",
          len(det_draws) > 1, f"{len(det_draws)}/{n} distinct draw sets")

    # ---- 3. reproducibility ----------------------------------------------
    print("3. reproducibility")
    outcomes = []
    for _ in range(2):
        rng = np.random.default_rng(23)
        sim = clone(gs)
        determinize(sim, rng)
        outcomes.append((drawn_after(sim), deck_keys(sim)))
    check("same np seed -> same future", outcomes[0] == outcomes[1])

    # ---- 4. integration ---------------------------------------------------
    print("4. integration (gumbel_search determinized)")
    torch.manual_seed(0)
    net = PolicyValueNet()
    net.eval()
    device = torch.device("cpu")
    src_deck_before = deck_keys(gs)
    src_rng_before = gs["rng"].seed_str

    t0 = time.perf_counter()
    res_hand = gumbel_search(gs, net, device, n_sims=8, k_max=4, depth=1,
                             rng=np.random.default_rng(3), determinize=True)
    dt_hand = time.perf_counter() - t0
    check("hand root returns a result", res_hand is not None,
          f"{dt_hand*1e3:.0f} ms at sims=8")

    econ = new_run("DETPROBE2")  # blind_select IS an econ root
    t0 = time.perf_counter()
    res_econ = gumbel_search(econ, net, device, n_sims=8, k_max=4, depth=1,
                             rng=np.random.default_rng(4), determinize=True)
    dt_econ = time.perf_counter() - t0
    check("econ root returns a result", res_econ is not None,
          f"{dt_econ*1e3:.0f} ms at sims=8 (blind-horizon rollouts)")

    check("source state never mutated by determinized search",
          deck_keys(gs) == src_deck_before
          and gs["rng"].seed_str == src_rng_before)

    # determinize() cost, isolated
    sims = [clone(gs) for _ in range(200)]
    rng = np.random.default_rng(5)
    t0 = time.perf_counter()
    for s in sims:
        determinize(s, rng)
    per = (time.perf_counter() - t0) / len(sims)
    print(f"  [info] determinize cost: {per*1e6:.0f} us/clone "
          f"(clone itself ~900 us)")

    if FAILURES:
        print(f"\nGATE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
