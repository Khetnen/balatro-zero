"""BalatroBench five-seed benchmark: our probes against published LLM results.

BalatroBench (balatrobench.com, by the balatrobot authors) runs pure-LLM
tool-calling play on the REAL game at RED deck / WHITE stake over five
seeds, 3 runs per model. Those numbers are an external, absolute standard
-- unlike panel progress, they are set by someone else, so they cannot
drift with our probe or engine changes.

This is the scoreboard for the agent work. Every change gets measured
here. Run it with the same --runs for comparability across checkpoints.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/bench5.py                      # gold
    uv run --no-sync python scripts/bench5.py --probe router
    uv run --no-sync python scripts/bench5.py --probe net --ckpt runs/vX/net.pt

Baseline recorded 2026-07-29 (engine b0b6b9a, probe 829b30f), default
--runs 5 --eps 0.15:
    gold  -> 0/25 wins, mean ante 3.12  (rank 8 of 18 on their board)

Quote numbers only against the same --runs/--eps: a separate sampling
(1 deterministic + 4 at eps=0.15 per seed) gave mean ante 3.44, also 0
wins. Note seed AAAAAAA reached ante 5 under eps randomization but dies
at ante 1 deterministically -- on that seed the scripted economy is
worse than random, which is its own indictment of the econ policy.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

SEEDS = ["AAAAAAA", "BBBBBBB", "CCCCCCC", "DDDDDDD", "EEEEEEE"]

# balatrobench.com leaderboard v1.0.8, read 2026-07-29.
# (model, wins, n, mean_ante) -- same deck/stake/seeds as above.
COMPETITORS = [
    ("gemini-3-pro-preview", 9, 15, 7.13),
    ("gpt-5.2", 3, 16, 6.50),
    ("gemini-3-flash-preview", 5, 15, 6.00),
    ("claude-opus-4.5", 2, 15, 4.93),
    ("claude-sonnet-4.5", 2, 15, 4.20),
    ("grok-4", 0, 15, 3.47),
    ("deepseek-v3.2", 0, 15, 3.20),
    ("grok-4.1-fast", 0, 15, 2.80),
    ("qwen3-max", 0, 15, 2.60),
    ("gpt-oss-20b", 0, 15, 2.27),
    ("gpt-oss-120b", 0, 15, 2.20),
    ("kimi-k2-thinking", 0, 15, 2.13),
    ("kimi-k2.5", 0, 15, 2.00),
    ("claude-haiku-4.5", 0, 15, 1.87),
    ("glm-4.7", 0, 15, 1.40),
    ("minimax-m2.1", 0, 15, 1.40),
    ("mistral-large-2512", 0, 15, 1.13),
]


def _run(job):
    spec, seed, rollout = job
    from balatro_zero.probes import run_probe_game

    return run_probe_game(spec, seed, rollout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="gold",
                    choices=["gold", "router", "chip_greedy", "net"])
    ap.add_argument("--ckpt", default=None, help="required for --probe net")
    ap.add_argument("--runs", type=int, default=5, help="rollouts per seed")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--finisher", action="store_true",
                    help="finish blinds with the beam inside rollouts")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from balatro_zero.probes import ProbeSpec

    params: dict = {"eps": args.eps}
    if args.probe == "net":
        if not args.ckpt:
            raise SystemExit("--probe net requires --ckpt")
        params.update(ckpt=args.ckpt, sims=args.sims, depth=args.depth,
                      blind_finisher=args.finisher)
    label = args.probe if not args.ckpt else f"{args.probe}:{Path(args.ckpt).stem}"
    if args.finisher:
        label += "+beam"
    spec = ProbeSpec(name=label, kind=args.probe, params=params)

    jobs = [(spec, s, r) for s in SEEDS for r in range(args.runs)]
    with get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(_run, jobs)

    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    wins = sum(1 for r in ok if r["won"])
    mean_ante = sum(r["max_ante"] for r in ok) / max(len(ok), 1)

    print(f"\n{'seed':<10} " + " ".join(f"r{i}" for i in range(args.runs)))
    for s in SEEDS:
        cells = []
        for r in range(args.runs):
            m = next((x for x in ok if x["seed"] == s and x["rollout"] == r), None)
            cells.append("ERR" if m is None else
                         ("WIN" if m["won"] else str(m["max_ante"])))
        print(f"{s:<10} " + " ".join(f"{c:>3}" for c in cells))
    if errs:
        print(f"\n{len(errs)} errored: {errs[0]['error']}")

    print(f"\n{'':<3} {'model':<26} {'wins':>7}  mean ante")
    board = [(m, f"{w}/{n}", a, False) for m, w, n, a in COMPETITORS]
    board.append((f"** {label} (ours)", f"{wins}/{len(ok)}", mean_ante, True))
    board.sort(key=lambda t: -t[2])
    for i, (m, w, a, mine) in enumerate(board, 1):
        mark = "->" if mine else "  "
        print(f"{mark} {i:>2} {m:<26} {w:>7}  {a:.2f}")

    out = args.out or f"runs/bench5_{label.replace(':', '_')}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"probe": label, "runs_per_seed": args.runs, "seeds": SEEDS,
                   "wins": wins, "n": len(ok), "mean_ante": mean_ante,
                   "rows": rows}, fh, indent=2, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
