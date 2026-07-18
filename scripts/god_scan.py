"""Scan seeds for god-tier early queues (Immolate-style, but dynamic).

For each seed: play ante 1's small+big blind with a chip cheat, enter each
shop, open every booster pack, and score what the seed offers in its first
two shops. High scores = early legendaries (The Soul), xmult engines, or
scaling jokers — seeds where a mediocre policy can assemble a winning board.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/god_scan.py [N_SEEDS] [OUT_JSON]
"""
from __future__ import annotations

import json
import random
import sys
import time

from jackdaw.bridge.backend import RPCError, SimBackend

from balatro_zero.router import SCALER, XMULT  # shared joker tiers

SOUL = {"c_soul"}

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def keys_of(items) -> list[str]:
    return [getattr(c, "center_key", "?") for c in items]


def clear_blind_and_shop(sim: SimBackend) -> None:
    sim.handle("select", {})
    sim.handle("set", {"chips": 100000})
    sim.handle("play", {"cards": [0, 1, 2, 3, 4]})
    sim.handle("cash_out", {})


def open_all_packs(sim: SimBackend, found: list[str]) -> None:
    gs = sim._gs
    n = len(gs.get("shop_boosters", []))
    for _ in range(n):
        if not gs.get("shop_boosters"):
            break
        sim.handle("set", {"dollars": 100})
        try:
            sim.handle("buy", {"pack": 0})
        except RPCError:
            break
        found.extend(keys_of(gs.get("pack_cards", [])))
        for method, params in (
            ("skip", {}),
            ("pack", {"card": 0}),
            ("pack", {"cards": [0]}),
        ):
            try:
                sim.handle(method, params)
                break
            except RPCError:
                continue
        else:
            return  # stuck in pack phase; caller sees partial results


def probe_seed(seed: str) -> tuple[float, dict]:
    sim = SimBackend()
    sim.handle("start", {"deck": "RED", "stake": "WHITE", "seed": seed})
    gs = sim._gs
    shop_items: list[str] = []
    pack_items: list[str] = []
    for _shop in range(2):  # shops after small and big blind
        clear_blind_and_shop(sim)
        shop_items.extend(keys_of(gs.get("shop_cards", [])))
        shop_items.extend(keys_of(gs.get("shop_vouchers", [])))
        open_all_packs(sim, pack_items)
        try:
            sim.handle("next_round", {})
        except RPCError:
            break  # stuck (e.g. unskippable pack) — score what we saw

    seen = shop_items + pack_items
    score = (
        100.0 * sum(k in SOUL for k in seen)
        + 12.0 * sum(k in XMULT for k in seen)
        + 4.0 * sum(k in SCALER for k in seen)
    )
    detail = {
        "seed": seed,
        "score": score,
        "souls": [k for k in seen if k in SOUL],
        "xmult": [k for k in seen if k in XMULT],
        "scalers": [k for k in seen if k in SCALER],
        "shop": shop_items,
        "packs": pack_items,
    }
    return score, detail


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out_path = sys.argv[2] if len(sys.argv) > 2 else "runs/god_seeds.json"
    rng = random.Random(1234)

    results = []
    errors = 0
    t0 = time.perf_counter()
    for i in range(n_seeds):
        seed = "".join(rng.choices(ALPHABET, k=8))
        try:
            score, detail = probe_seed(seed)
        except Exception as e:  # engine edge cases: log, keep scanning
            errors += 1
            if errors <= 5:
                print(f"  [err] {seed}: {type(e).__name__}: {e}", flush=True)
            continue
        results.append(detail)
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"  {i + 1}/{n_seeds} ({rate:.0f} seeds/s, {errors} errors)", flush=True)

    results.sort(key=lambda d: -d["score"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results[:50], f, indent=1)

    dt = time.perf_counter() - t0
    print(f"\nscanned {len(results)} seeds in {dt:.1f}s ({len(results)/dt:.0f}/s), {errors} errors")
    print(f"top 10 (full top-50 -> {out_path}):")
    for d in results[:10]:
        print(f"  {d['seed']}  score {d['score']:5.0f}  souls={d['souls']} xmult={d['xmult']} scalers={d['scalers']}")


if __name__ == "__main__":
    main()
