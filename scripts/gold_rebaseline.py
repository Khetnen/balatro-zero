"""Deterministic gold-probe reference on the current engine.

Two fixed seed sets, one deterministic game each (econ_eps=0, rng seed 0),
so the numbers are a stable engine fingerprint: rerun after engine changes
and compare means directly.

  RBASE-32  seeds "RBASE0".."RBASE31" (the LLM-plan A/B baseline panel)
  GOD-25    first 25 seeds of runs/god_seeds.json (static-feature rich seeds)

References:
  jackdaw b55f3a2 (pre-fix engine, 2026-07):
    RBASE mean prog 0.516 mean ante 4.34  ante>=4 23/32  wins 0/32
    GOD   mean prog 0.486 mean ante 4.04  ante>=4 19/25  wins 0/25
  jackdaw dc6eb32 (70 bugs fixed, 2026-07-27), probe blind-clear detection
  restored after the round-counter fix (engine bug #1) broke it:
    RBASE mean prog 0.467 mean ante 3.91  ante>=4 22/32  wins 0/32
    GOD   mean prog 0.512 mean ante 4.20  ante>=4 18/25  wins 0/25

STALE AS OF 2026-07-28 -- both reference blocks above predate two changes
that move these numbers BY DESIGN, so do not read a delta against them as
a regression until the anchor is recomputed:
  * goldprobe candidate generators widened -- _play_candidates now emits
    2/3/4-card plays (it only ever emitted 5-card and 1-card, so it could
    not represent a short hand outscoring the five-card one) and
    _discard_candidates gained rank- and straight-oriented digs (every
    candidate used to be defined relative to the most common SUIT).  The
    probe is strictly less blind, so gold should go UP.
  * engine bug #73 -- Spectral packs were masked skip-only for every
    agent, so no gold game has ever picked one.

A collapse here (mean ante ~1.7, runs dying in ante 1) means the probe has
gone blind to some engine state change, not that the engine got harder --
that is exactly how the round-counter drift was caught.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/gold_rebaseline.py [--workers 12]
        [--out runs/gold_rebaseline_v2.json]
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np


def _run(seed: str) -> dict:
    from balatro_zero.goldprobe import gold_game

    g = gold_game(seed)
    g["seed"] = seed
    return g


def summarize(tag: str, rows: list[dict]) -> dict:
    prog = [r["progress"] for r in rows]
    ante = [r["max_ante"] for r in rows]
    out = {
        "tag": tag,
        "n": len(rows),
        "mean_progress": round(float(np.mean(prog)), 4),
        "mean_ante": round(float(np.mean(ante)), 3),
        "ante_ge_4": sum(1 for a in ante if a >= 4),
        "wins": sum(1 for r in rows if r["won"]),
        "rows": rows,
    }
    print(f"== {tag}: mean prog {out['mean_progress']:.3f} "
          f"mean ante {out['mean_ante']:.2f} "
          f"ante>=4 {out['ante_ge_4']}/{out['n']} "
          f"WINS {out['wins']}/{out['n']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=str, default="runs/gold_rebaseline_v2.json")
    args = ap.parse_args()

    rbase = [f"RBASE{i}" for i in range(32)]
    god = [d["seed"] for d in
           json.loads(Path("runs/god_seeds.json").read_text(encoding="utf-8"))][:25]

    t0 = time.perf_counter()
    ctx = get_context("spawn")
    result = {}
    with ctx.Pool(args.workers) as pool:
        for tag, seeds in (("RBASE", rbase), ("GOD", god)):
            rows = []
            for r in pool.imap(_run, seeds):
                print(f"[{tag}] {r['seed']}: won={r['won']} ante={r['max_ante']} "
                      f"prog={r['progress']:.3f} ${r['dollars']} board={r['board']}",
                      flush=True)
                rows.append(r)
            rows.sort(key=lambda r: seeds.index(r["seed"]))
            result[tag] = summarize(tag, rows)

    print(f"({(time.perf_counter() - t0) / 60:.1f}m)")
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
