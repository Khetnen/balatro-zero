"""A/B: gold probe with LLM seed plans vs plain gold, same seeds.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/gold_ab.py runs/plans_rbase.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from balatro_zero.goldprobe import gold_game

plans_path = sys.argv[1] if len(sys.argv) > 1 else "runs/plans_rbase.json"
plans = json.load(open(plans_path, encoding="utf-8"))

rows = []
t0 = time.perf_counter()
for seed, plan in plans.items():
    a = gold_game(seed)
    b = gold_game(seed, plan=plan)
    rows.append((seed, a, b))
    mark = "==" if abs(b["progress"] - a["progress"]) < 1e-9 else (
        "PLAN+" if b["progress"] > a["progress"] else "plan-"
    )
    print(f"{seed:8s} plain a{a['max_ante']} {a['progress']:.3f}"
          f"{' WIN' if a['won'] else ''}  |  plan a{b['max_ante']} {b['progress']:.3f}"
          f"{' WIN' if b['won'] else ''}  {mark}", flush=True)

pa = [a["progress"] for _, a, _ in rows]
pb = [b["progress"] for _, _, b in rows]
wa = sum(a["won"] for _, a, _ in rows)
wb = sum(b["won"] for _, _, b in rows)
better = sum(1 for _, a, b in rows if b["progress"] > a["progress"] + 1e-9)
worse = sum(1 for _, a, b in rows if b["progress"] < a["progress"] - 1e-9)
print(f"\nplain: mean {np.mean(pa):.3f}  wins {wa}/{len(rows)}")
print(f"plan:  mean {np.mean(pb):.3f}  wins {wb}/{len(rows)}")
print(f"per-seed: plan better on {better}, worse on {worse}, tied {len(rows) - better - worse}")
print(f"({(time.perf_counter() - t0) / 60:.0f}m)")
