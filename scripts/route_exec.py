"""Execute a pre-authored decision route (Claude's clairvoyant run plan).

Route = ordered list of match strings. At each econ decision point, if the
next route entry matches an option's description, take it and advance the
pointer; otherwise take the default action (SelectBlind > NextRound >
SkipPack > option 0) WITHOUT advancing, unless the entry is marked "!" as
consume-anyway. Hand play: gold beam. Logs every decision.

Usage: uv run --no-sync python scripts/route_exec.py SEED ROUTE_JSON
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "scripts")
from interactive_run import advance, describe, summary  # noqa: E402

from balatro_zero.state import ante, is_terminal, new_run, progress, step_factored, won  # noqa: E402
from jackdaw.env.action_space import ActionType  # noqa: E402

seed, route_path = sys.argv[1], sys.argv[2]
route: list[str] = json.load(open(route_path, encoding="utf-8"))

gs = new_run(seed)
ptr = 0
n_dec = 0
while True:
    opts = advance(gs)
    if won(gs) or is_terminal(gs) or not opts:
        break
    import os
    if os.environ.get("ROUTE_VERBOSE") and gs.get("shop_cards") is not None:
        from balatro_zero.router import key_of as _k
        print(f"      $${gs.get('dollars', 0)} shop=["
              + ", ".join(f"{_k(c)}${getattr(c, 'cost', '?')}" for c in gs.get("shop_cards", []))
              + "] v=[" + ", ".join(_k(c) for c in gs.get("shop_vouchers", []))
              + "] p=[" + ", ".join(_k(c) for c in gs.get("shop_boosters", []))
              + "] pack=[" + ", ".join(_k(c) for c in gs.get("pack_cards", [])) + "]",
              flush=True)
    # Match within a small lookahead window so minor drift skips at most a
    # couple of entries instead of wedging the pointer or burning the route.
    WINDOW = 3
    action = None
    for j in range(ptr, min(ptr + WINDOW, len(route))):
        hit = next((a for a in opts if route[j] in describe(gs, a)), None)
        if hit is not None:
            for k in range(ptr, j):
                print(f"      (dropped unmatched route entry: {route[k]})", flush=True)
            action = hit
            ptr = j + 1
            print(f"[{n_dec:3d}] ROUTE  {describe(gs, action)}", flush=True)
            break
    if action is None:
        for want in (ActionType.SelectBlind, ActionType.NextRound, ActionType.SkipPack):
            action = next((a for a in opts if a.action_type == want), None)
            if action is not None:
                break
        if action is None:
            action = opts[0]
        print(f"[{n_dec:3d}] default {describe(gs, action)}", flush=True)
    try:
        step_factored(gs, action)
    except Exception as e:  # noqa: BLE001
        print(f"step failed: {e}")
        break
    n_dec += 1

print("\n" + summary(gs))
tag = "WON" if won(gs) else ("GAME OVER" if is_terminal(gs) else "STUCK")
print(f"*** {tag}: ante {ante(gs)} prog {progress(gs):.3f} after {n_dec} decisions "
      f"(route used {ptr}/{len(route)}) ***")
