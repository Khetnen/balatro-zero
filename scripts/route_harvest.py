"""Route god seeds with a fully scripted policy, harvesting full-state
snapshots from deep/winning trajectories for curriculum seeding.

Macro strategy (the community-standard flush build):
- Economy: grab The Soul instantly, buy xmult/scaling jokers, open packs by
  build relevance, level Flush with planets (Jupiter first), sell junk.
- Hand play: engine-exact greedy — enumerate legal plays on cloned states
  and take the one with the best realized progress; when the best play is
  weak and a flush is brewing, discard off-suit instead.

No neural net involved: the engine is its own evaluator.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/route_harvest.py [SEEDS_JSON] [OUT_PKL] [TOP_K]
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations

from balatro_zero.router import (  # noqa: F401 — re-exported for god_scan
    ECON_PHASES,
    SCALER,
    XMULT,
    key_of,
    scripted_econ_action,
    scripted_hand_action,
)
from balatro_zero.state import (
    ante,
    is_terminal,
    legal_factored,
    new_run,
    progress,
    step_factored,
    won,
)

MAX_MOVES = 500
SNAP_MIN_ANTE = 2
SNAPS_PER_GAME = 8


# ---------------------------------------------------------------------------
# Game loop + harvest
# ---------------------------------------------------------------------------


def route_game(
    seed: str, record_demos: bool = False
) -> tuple[dict, list[bytes], list[tuple]]:
    """Returns (stats, snapshots, demo_samples).

    Demo samples are (Obs, pi_onehot, z_win, z_togo) tuples matching the
    self-play sample format, recorded only for steps whose chosen action
    appears in the Discrete(500) enumeration (combo subsampling can miss
    large hand combos; those steps are skipped).
    """
    import numpy as np

    from balatro_zero.state import MAX_ACTIONS, observe

    gs = new_run(seed)
    snapshots: list[bytes] = []
    positions: list[tuple] = []  # (obs, action_idx, progress@root)
    last_snap_ante = ante(gs)
    moves = 0
    max_progress = progress(gs)
    while not is_terminal(gs) and moves < MAX_MOVES:
        phase = gs.get("phase")
        action = scripted_econ_action(gs) if phase in ECON_PHASES else scripted_hand_action(gs)
        fallback = action is None
        if action is None:
            legal = legal_factored(gs)
            if not legal:
                break
            action = legal[0]
        appended = False
        if record_demos and not fallback:
            legal = legal_factored(gs)
            try:
                idx = legal.index(action)
            except ValueError:
                idx = -1  # subsampled out of the enumeration: skip
            if idx >= 0:
                positions.append((observe(gs), idx, progress(gs)))
                appended = True
        try:
            step_factored(gs, action)
        except Exception:  # noqa: BLE001 — scripted combo edge case: pick any legal
            if appended:
                positions.pop()  # recorded action never executed
            legal = legal_factored(gs)
            if not legal:
                break
            step_factored(gs, legal[0])
        moves += 1
        max_progress = max(max_progress, progress(gs))
        a = ante(gs)
        if a >= SNAP_MIN_ANTE and a > last_snap_ante and len(snapshots) < SNAPS_PER_GAME:
            snapshots.append(pickle.dumps(gs, protocol=5))
            last_snap_ante = a
    w = won(gs)
    final_best = 1.0 if w else max(max_progress, progress(gs))
    demos = []
    if record_demos:
        z_win = 1.0 if w else 0.0
        for obs, idx, prog_t in positions:
            pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
            pi[idx] = 1.0
            demos.append((obs, pi, z_win, float(np.clip(final_best - prog_t, 0.0, 1.0))))
    stats = {
        "seed": seed,
        "won": w,
        "ante": ante(gs),
        "progress": round(progress(gs), 3),
        "moves": moves,
        "board": [key_of(c) for c in gs.get("jokers", [])],
        "dollars": gs.get("dollars", 0),
    }
    return stats, snapshots, demos


def main() -> None:
    seeds_json = sys.argv[1] if len(sys.argv) > 1 else "runs/god_seeds.json"
    out_pkl = sys.argv[2] if len(sys.argv) > 2 else "runs/god_pool.pkl"
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    demos_pkl = sys.argv[4] if len(sys.argv) > 4 else None

    with open(seeds_json, encoding="utf-8") as f:
        candidates = json.load(f)[:top_k]

    pool: list[bytes] = []
    all_demos: list[tuple] = []
    results = []
    for i, cand in enumerate(candidates):
        t0 = time.perf_counter()
        try:
            stats, snaps, demos = route_game(cand["seed"], record_demos=demos_pkl is not None)
        except Exception as e:  # noqa: BLE001 — engine bug on a deep path: skip seed
            print(f"[{i}] {cand['seed']} CRASHED: {type(e).__name__}: {e}", flush=True)
            continue
        pool.extend(snaps)
        if stats["ante"] >= 3:  # demo quality gate: only games that went deep
            all_demos.extend(demos)
        results.append(stats)
        print(
            f"[{i}] {stats['seed']}: won={stats['won']} ante={stats['ante']} "
            f"prog={stats['progress']} moves={stats['moves']} ${stats['dollars']} "
            f"snaps={len(snaps)} ({time.perf_counter() - t0:.0f}s)\n"
            f"    board: {stats['board']}",
            flush=True,
        )

    with open(out_pkl, "wb") as f:
        pickle.dump(pool, f, protocol=5)
    if demos_pkl is not None:
        with open(demos_pkl, "wb") as f:
            pickle.dump(all_demos, f, protocol=5)
        print(f"{len(all_demos)} demo samples -> {demos_pkl}")
    n_won = sum(r["won"] for r in results)
    n_deep = sum(r["ante"] >= 4 for r in results)
    print(f"\n{len(pool)} snapshots -> {out_pkl} | {n_won} wins, {n_deep} games at ante>=4 of {len(results)}")


if __name__ == "__main__":
    main()
