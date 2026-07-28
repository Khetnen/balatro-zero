"""Paired v1 vs v2 panel comparison (works on a partially written v2 file).

Compares only records present in BOTH panels, keyed by (probe, seed,
rollout) — the rollout rng is derived from that triple, so aggregate
deltas are attributable to the engine + probe changes.

READ THE PER-SEED COLUMNS WITH CARE.  Two things make individual cells
untrustworthy as "this seed got harder":

* engine fixes shift RNG stream consumption, so a seed's post-fix run is
  a different sample rather than the same run under corrected rules; and
* the scripted probes are not reproducible cell-by-cell when replayed in
  one process (router churns ~40% of cells against itself; gold is stable).

The distribution, the theta ladder and the rank correlations transfer;
single-cell "worse/better" counts are churn as much as signal.

Usage (from balatro-zero/): panel_compare.py [v1.jsonl] [v2.jsonl]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

V1 = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/difficulty/panel.jsonl")
V2 = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/difficulty/panel_v2.jsonl")

PROBE_ORDER = ["chip_greedy", "router_e30", "router_e15", "router",
               "v6_s16", "v10_s16", "v10_s64", "gold"]


def load(p: Path) -> tuple[dict, int]:
    out, errs = {}, 0
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn final line while the sweep is still appending
        if "error" in r:
            errs += 1
            continue
        out[(r["probe"], r["seed"], r["rollout"])] = r
    return out, errs


a, ea = load(V1)
b, eb = load(V2)
shared = sorted(set(a) & set(b))
print(f"v1 {len(a)} records ({ea} errors) | v2 {len(b)} records ({eb} errors) "
      f"| paired {len(shared)}\n")

rows = defaultdict(list)
for k in shared:
    rows[k[0]].append((a[k], b[k]))

print(f"{'probe':12s} {'n':>6s}  {'v1 prog':>8s} {'v2 prog':>8s} {'delta':>7s}  "
      f"{'v1 ante':>7s} {'v2 ante':>7s}  {'v1 win':>6s} {'v2 win':>6s}  "
      f"{'worse':>5s} {'better':>6s}")
for name in PROBE_ORDER + [p for p in rows if p not in PROBE_ORDER]:
    r = rows.get(name)
    if not r:
        continue
    p1 = np.array([x["progress"] for x, _ in r])
    p2 = np.array([y["progress"] for _, y in r])
    a1 = np.array([x["max_ante"] for x, _ in r])
    a2 = np.array([y["max_ante"] for _, y in r])
    w1 = sum(x["won"] or x["max_ante"] >= 9 for x, _ in r)
    w2 = sum(y["won"] or y["max_ante"] >= 9 for _, y in r)
    print(f"{name:12s} {len(r):6d}  {p1.mean():8.3f} {p2.mean():8.3f} "
          f"{p2.mean() - p1.mean():+7.3f}  {a1.mean():7.2f} {a2.mean():7.2f}  "
          f"{w1:6d} {w2:6d}  {(p2 < p1 - 1e-9).sum():5d} {(p2 > p1 + 1e-9).sum():6d}")

seeds_done = defaultdict(set)
for pr, sd, k in b:
    seeds_done[pr].add(sd)
print("\nv2 coverage:", {p: len(s) for p, s in sorted(seeds_done.items())})
