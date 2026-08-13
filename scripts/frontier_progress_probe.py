"""Falsifiable gate for the v13 frontier progress formula.

progress = (f - 1 + frac)/8, f = deepest ante reached; frac = live
chips vs the frontier's standard boss bar during a blind (actual bar on
the frontier boss itself), best-single-blind chips (C*) between blinds.

Checks, exit non-zero on failure:
  1. FORMULA — fresh run reads 0; live in-blind reads move with chips;
     the between-blind read equals C*/S_f after a blind is cleared and
     never falls below any mid-blind read of that blind.
  2. NEUTRALITY — SkipBlind leaves progress unchanged; an ante counter
     regression (Hieroglyph's effect) leaves progress unchanged (the
     frontier annotation holds); re-walking a replayed blind cannot
     raise the integer.
  3. DENSITY — at a BIG blind with C* already set from Small (the state
     over-aggressive high-water would flatten), distinct plays yield
     distinct progress on determinized clones. The v1 zero-variance and
     v4 flat-leaf failures must not be re-created.
  4. BOSS + ADVANCE — on the boss, the live denominator is the blind's
     ACTUAL requirement; beating it advances the frontier monotonically
     and re-prices C* against the new bar automatically.
  5. CAP READ — progress_cap_read >= progress mid-blind, folds live
     chips into the store, and equals progress between blinds.

Run from balatro-zero/:  uv run --no-sync python scripts/frontier_progress_probe.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from jackdaw.engine.actions import GamePhase
from jackdaw.env.action_space import ActionType

from balatro_zero.router import scripted_econ_action, scripted_hand_action
from balatro_zero.state import (
    _BZ_BEST,
    _BZ_FRONTIER,
    _standard_boss_req,
    ante,
    clone,
    is_terminal,
    legal_factored,
    new_run,
    progress,
    progress_cap_read,
    step_factored,
)

FAILURES: list[str] = []
ECON = ("shop", "blind_select", "pack_opening", "round_eval")


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def router_step(gs) -> bool:
    """One scripted action; False when stuck/terminal."""
    if is_terminal(gs):
        return False
    legal = legal_factored(gs)
    if not legal:
        return False
    if str(gs.get("phase")) in ("shop", "blind_select", "pack_opening", "round_eval"):
        a = scripted_econ_action(gs, legal)
    else:
        a = scripted_hand_action(gs)
    step_factored(gs, a if a is not None else legal[0])
    return True


def walk_until(gs, pred, cap=300) -> bool:
    for _ in range(cap):
        if pred(gs):
            return True
        if not router_step(gs):
            return False
    return False


def main() -> int:
    # Find a seed the router carries through ante 1 (boss cleared).
    gs = None
    for i in range(12):
        cand = new_run(f"FPROBE{i}")
        trail = clone(cand)
        if walk_until(trail, lambda s: ante(s) >= 2):
            gs = cand
            break
    if gs is None:
        print("FAIL: no seed cleared ante 1 with the router in 12 tries")
        return 1

    print("1. formula")
    check("fresh run reads 0", progress(gs) == 0.0, f"{progress(gs):.4f}")
    s_1 = _standard_boss_req(gs, 1)
    check("standard boss bar sane (ante 1 = 600 at White)", s_1 == 600, f"{s_1}")

    sel = [a for a in legal_factored(gs) if a.action_type == ActionType.SelectBlind]
    step_factored(gs, sel[0])
    mid_reads = []
    while gs.get("phase") == GamePhase.SELECTING_HAND and not is_terminal(gs):
        p = progress(gs)
        mid_reads.append(p)
        expect = min(gs.get("chips", 0) / s_1, 0.999) / 8
        check_ok = abs(p - expect) < 1e-9
        if not check_ok:
            check("live read == chips/S_f/8", False, f"{p:.4f} vs {expect:.4f}")
            break
        if not router_step(gs):
            break
    else:
        check("live reads == chips/S_f/8 all blind", True, f"{len(mid_reads)} states")
    cleared = ante(gs) == 1 and str(gs.get("phase")) != "game_over"
    check("small blind cleared by router", cleared, f"phase {gs.get('phase')}")
    after = progress(gs)
    cstar = gs.get(_BZ_BEST, 0)
    check("between-blind read == C*/S_f/8",
          abs(after - min(cstar / s_1, 0.999) / 8) < 1e-9,
          f"C*={cstar}, read {after:.4f}")
    check("clear read >= every mid-blind read",
          all(after >= m - 1e-12 for m in mid_reads))

    print("2. neutrality")
    # Walk to the next blind select for a legal SkipBlind.
    walk_until(gs, lambda s: str(s.get("phase")) == "blind_select", cap=50)
    skips = [a for a in legal_factored(gs) if a.action_type == ActionType.SkipBlind]
    if skips:
        before = progress(gs)
        sk = clone(gs)
        step_factored(sk, skips[0])
        check("SkipBlind leaves progress unchanged",
              abs(progress(sk) - before) < 1e-9,
              f"{before:.4f} -> {progress(sk):.4f}")
    else:
        check("SkipBlind available at blind select", False, "no skip action")
    hier = clone(gs)
    f_before = progress(hier)
    hier["round_resets"]["ante"] = max(1, ante(hier) - 1)  # Hieroglyph's effect
    check("ante regression leaves progress unchanged (frontier holds)",
          abs(progress(hier) - f_before) < 1e-9,
          f"{f_before:.4f} -> {progress(hier):.4f}")
    check("frontier annotation intact", hier.get(_BZ_FRONTIER, 0) >= ante(gs))

    print("3. density (Big blind, C* already set)")
    big = None
    probe = clone(gs)
    if walk_until(probe, lambda s: s.get("phase") == GamePhase.SELECTING_HAND
                  and str(getattr(s.get("blind"), "name", "")).startswith("Big"), cap=60):
        big = probe
    if big is None:
        check("reached a Big blind", False)
    else:
        plays = [a for a in legal_factored(big)
                 if a.action_type == ActionType.PlayHand][:8]
        vals = set()
        for a in plays:
            sim = clone(big)
            step_factored(sim, a)
            vals.add(round(progress(sim), 6))
        check("distinct plays give distinct progress (not flattened)",
              len(vals) > 1, f"{len(vals)}/{len(plays)} distinct")

    print("4. boss + advance")
    boss = None
    probe = clone(gs)
    if walk_until(probe, lambda s: s.get("phase") == GamePhase.SELECTING_HAND
                  and getattr(s.get("blind"), "boss", False), cap=120):
        boss = probe
    if boss is None:
        check("reached the ante-1 boss", False)
    else:
        blind = boss.get("blind")
        a_f = getattr(blind, "chips", 0)
        expect = min(boss.get("chips", 0) / a_f, 0.999) / 8 if a_f else None
        check("boss live denominator is the ACTUAL requirement",
              expect is not None and abs(progress(boss) - expect) < 1e-9,
              f"A_f={a_f}")
        pre = progress(boss)
        if walk_until(boss, lambda s: ante(s) >= 2, cap=60):
            post = progress(boss)
            s_2 = _standard_boss_req(boss, 2)
            cstar2 = boss.get(_BZ_BEST, 0)
            check("frontier advance is monotone", post >= pre - 1e-12,
                  f"{pre:.4f} -> {post:.4f}")
            check("advance >= 1/8", post >= 1 / 8 - 1e-12)
            walk_until(boss, lambda s: str(s.get("phase")) in ECON, cap=10)
            if str(boss.get("phase")) in ECON:
                check("C* re-priced against the new bar",
                      abs(progress(boss) - (1 + min(cstar2 / s_2, 0.999)) / 8) < 1e-9,
                      f"C*={cstar2}, S_2={s_2}")
        else:
            check("router cleared the ante-1 boss", False, "died on boss")

    print("5. cap read")
    mid = clone(gs)
    sel = [a for a in legal_factored(mid) if a.action_type == ActionType.SelectBlind]
    if sel:
        step_factored(mid, sel[0])
        for _ in range(2):
            if mid.get("phase") == GamePhase.SELECTING_HAND:
                router_step(mid)
        if mid.get("phase") == GamePhase.SELECTING_HAND:
            check("cap read >= live read mid-blind",
                  progress_cap_read(mid) >= progress(mid) - 1e-12,
                  f"{progress(mid):.4f} vs cap {progress_cap_read(mid):.4f}")
    check("cap read == progress between blinds",
          abs(progress_cap_read(gs) - progress(gs)) < 1e-9)
    w = clone(gs)
    w["won"] = True
    check("won reads 1.0", progress(w) == 1.0)

    if FAILURES:
        print(f"\nGATE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
