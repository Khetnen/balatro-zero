"""Closeout probe: can an RL checkpoint FINISH winning runs from their own
mid-run states?

The diagnostic the v16/v17 fork hinges on (plan settled 2026-08-18). The
189 LLM-won games prove a winning trajectory exists through every one of
these states; the demonstrators converted 100% of them by construction.
If the best RL rung cannot convert even ante-6/7 states -- boards already
built, economy already banked, a few blinds from the win -- then reaching
such states is not the binding constraint, capacity/search is, and no
demo-mixing scheme fixes that. If it converts well from late states and
decays as the start moves earlier, backward-chained curriculum has a
gradient to climb and v17 is live.

Start states come from scripts/demo_replay.py (snapshots.pkl: verified
bit-exact replays of the winning games, state at the first decision stop
of each ante). Games are played exactly like training eval: greedy (no
root noise), honest determinized search, ladder config sims/depth.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/closeout_probe.py \
        --snapshots runs/demo_replay/snapshots.pkl \
        --ckpt runs/v13/ckpt_0200.pt --antes 5 6 7 \
        --per-ante 60 --rollouts 2 --workers 14 \
        --out runs/closeout/v13_it200.jsonl

Output: one jsonl row per game (resumable: existing keys are skipped),
plus a per-ante summary printed at the end.
"""
from __future__ import annotations

import argparse
import json
import pickle
import zlib
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch


def probe_task(ckpt: str, snap: bytes, key: str, sims: int, depth: int,
               max_moves: int) -> dict:
    """One greedy game from a snapshot (spawn-safe pool task)."""
    from balatro_zero.selfplay import SelfPlayConfig, _cached_net, play_game
    from balatro_zero.state import ante, progress

    torch.set_num_threads(1)
    net = _cached_net(ckpt)
    if not getattr(net, "_warmed", False):
        # The FIRST forward through the joker TransformerEncoder takes a
        # different kernel path than every later one, and the float
        # jitter is enough to flip knife-edge deep-state games (measured
        # 2026-08-18: same key, same rng — cold 62 moves vs warm 56;
        # post-warm repeats are bit-identical). One dummy forward makes
        # every game a warm-path pure function of (ckpt, key).
        with torch.no_grad():
            net(torch.zeros(1, net.torso[0].in_features),
                torch.zeros(1, 12, dtype=torch.int64),
                torch.zeros(1, 4, dtype=torch.int64),
                torch.zeros(1, 12, dtype=torch.int64))
        net._warmed = True
    cfg = SelfPlayConfig(
        sims=sims, k_max=8, depth=depth, determinize=True,
        curriculum_frac=0.0, guided_frac=0.0, max_moves=max_moves,
    )
    start = pickle.loads(snap)
    row = {"key": key, "start_ante": ante(start),
           "start_progress": round(progress(start), 4)}
    rng = np.random.default_rng(zlib.crc32(f"CLOSEOUT|{key}".encode()))
    try:
        _, st, _ = play_game(
            net, torch.device("cpu"), "UNUSED", cfg, rng,
            root_noise=False, start_state=start,
        )
    except Exception as e:  # noqa: BLE001 -- one bad game must not kill the probe
        row.update({"error": repr(e)})
        return row
    row.update({
        "won": st.won, "final_ante": st.max_ante, "moves": st.moves,
        "progress": round(st.progress, 4),
    })
    return row


def _probe_star(args: tuple) -> dict:
    """imap_unordered passes one positional arg; unpack to probe_task."""
    return probe_task(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="runs/demo_replay/snapshots.pkl")
    ap.add_argument("--ckpt", default="runs/v13/ckpt_0200.pt",
                    help="net to probe (default: the panel-best RL rung)")
    ap.add_argument("--antes", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--per-ante", type=int, default=60,
                    help="seeds sampled per ante (deterministic, rng 123)")
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--max-moves", type=int, default=400)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    snaps: dict[str, dict[int, bytes]] = pickle.loads(
        Path(args.snapshots).read_bytes())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}

    rng = np.random.default_rng(123)
    seeds_sorted = sorted(snaps)
    tasks = []
    for a in args.antes:
        having = [s for s in seeds_sorted if a in snaps[s]]
        pick = (having if len(having) <= args.per_ante else
                [having[i] for i in sorted(rng.choice(
                    len(having), size=args.per_ante, replace=False))])
        for seed in pick:
            for r in range(args.rollouts):
                key = f"{seed}|a{a}|r{r}"
                if key in done:
                    continue
                tasks.append((args.ckpt, snaps[seed][a], key,
                              args.sims, args.depth, args.max_moves))
    print(f"{len(tasks)} games to run ({len(done)} already done)", flush=True)
    if not tasks:
        summarize(out)
        return

    n = 0
    with get_context("spawn").Pool(args.workers) as pool:
        with open(out, "a", encoding="utf-8") as f:
            # imap_unordered streams rows as games finish (starmap would
            # buffer the whole probe before the first write, so a crash
            # would lose everything and progress would be invisible).
            for row in pool.imap_unordered(_probe_star, tasks, chunksize=1):
                f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
                if n % 25 == 0:
                    print(f"{n}/{len(tasks)} done", flush=True)
    summarize(out)


def summarize(out: Path) -> None:
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    by_ante: dict[int, list[dict]] = {}
    for r in rows:
        if "error" in r:
            continue
        by_ante.setdefault(r["start_ante"], []).append(r)
    errs = sum(1 for r in rows if "error" in r)
    print(f"\n== {out} ({len(rows)} games, {errs} errors) ==")
    for a in sorted(by_ante):
        g = by_ante[a]
        wins = sum(r["won"] for r in g)
        mean_fa = sum(r["final_ante"] for r in g) / len(g)
        adv = sum(r["final_ante"] > a for r in g)
        print(f"ante {a}: {len(g)} games | WINS {wins} ({wins/len(g):.1%}) "
              f"| mean final ante {mean_fa:.2f} "
              f"| advanced-at-all {adv} ({adv/len(g):.1%})")


if __name__ == "__main__":
    main()
