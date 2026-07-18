"""Per-seed difficulty curves from a difficulty_eval panel.

Implements the function-valued difficulty definition: each seed gets a
curve of expected outcome vs probe skill, where a probe's skill theta is
its mean progress over the whole panel (self-calibrating: no external
scale needed).

Per-seed summaries:
  auc      — mean outcome across the ladder (overall generosity)
  slope    — d(outcome)/d(theta) least-squares (discrimination: how much
             this seed rewards skill; ~0 = outcome fixed regardless of skill)
  floor    — weakest-probe outcome (early-game brutality)
  headroom — strongest-probe outcome minus floor

Usage (from balatro-zero/):
    uv run --no-sync python scripts/difficulty_curves.py \
        [runs/difficulty/panel.jsonl] [runs/difficulty/curves.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    panel = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/difficulty/panel.jsonl")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/difficulty/curves.json")

    by_probe: dict[str, list[float]] = defaultdict(list)
    by_seed_probe: dict[tuple[str, str], list[dict]] = defaultdict(list)
    n_err = 0
    for line in panel.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "error" in r:
            n_err += 1
            continue
        by_probe[r["probe"]].append(r["progress"])
        by_seed_probe[(r["seed"], r["probe"])].append(r)

    theta = {p: float(np.mean(v)) for p, v in by_probe.items()}
    ladder = sorted(theta, key=theta.get)
    print(f"panel: {sum(len(v) for v in by_probe.values())} records "
          f"({n_err} errors) | probe ladder (theta = panel mean progress):")
    for p in ladder:
        wins = sum(r["won"] for rs in by_seed_probe.values() for r in rs if r["probe"] == p)
        print(f"  {p:14s} theta={theta[p]:.3f}  n={len(by_probe[p])}  wins={wins}")

    seeds = sorted({s for s, _ in by_seed_probe})
    thetas = np.array([theta[p] for p in ladder])
    curves = {}
    for seed in seeds:
        means, antes, wr, ns = [], [], [], []
        for p in ladder:
            rs = by_seed_probe.get((seed, p), [])
            if not rs:
                means.append(None)
                antes.append(None)
                wr.append(None)
                ns.append(0)
                continue
            means.append(float(np.mean([r["progress"] for r in rs])))
            antes.append(float(np.mean([r["max_ante"] for r in rs])))
            wr.append(float(np.mean([r["won"] for r in rs])))
            ns.append(len(rs))
        ok = [i for i, m in enumerate(means) if m is not None]
        y = np.array([means[i] for i in ok])
        x = thetas[ok]
        slope = float(np.polyfit(x, y, 1)[0]) if len(ok) >= 2 and np.ptp(x) > 0 else 0.0
        curves[seed] = {
            "theta": [round(float(t), 4) for t in thetas],
            "probes": ladder,
            "mean_progress": [None if m is None else round(m, 4) for m in means],
            "mean_ante": [None if a is None else round(a, 3) for a in antes],
            "win_rate": [None if w is None else round(w, 3) for w in wr],
            "n": ns,
            "auc": round(float(y.mean()), 4),
            "slope": round(slope, 3),
            "floor": round(float(y[0]), 4),
            "headroom": round(float(y[-1] - y[0]), 4),
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"theta": theta, "curves": curves}, indent=1),
                   encoding="utf-8")

    aucs = {s: c["auc"] for s, c in curves.items()}
    slopes = {s: c["slope"] for s, c in curves.items()}
    print(f"\n{len(seeds)} seeds -> {out}")
    print(f"auc:   mean {np.mean(list(aucs.values())):.3f}  "
          f"spread [{min(aucs.values()):.3f}, {max(aucs.values()):.3f}]")
    print(f"slope: mean {np.mean(list(slopes.values())):.3f}  "
          f"spread [{min(slopes.values()):.3f}, {max(slopes.values()):.3f}]")

    def show(title, items):
        print(f"\n{title}:")
        for s in items:
            c = curves[s]
            pts = " ".join(f"{m:.2f}" if m is not None else "--" for m in c["mean_progress"])
            print(f"  {s}  auc {c['auc']:.3f} slope {c['slope']:+.2f}  [{pts}]")

    by_auc = sorted(seeds, key=lambda s: aucs[s])
    by_slope = sorted(seeds, key=lambda s: slopes[s])
    show("hardest (lowest auc)", by_auc[:8])
    show("easiest (highest auc)", by_auc[-8:][::-1])
    show("most skill-discriminating (steepest slope)", by_slope[-5:][::-1])
    show("least skill-responsive (flattest/negative slope)", by_slope[:5])


if __name__ == "__main__":
    main()
