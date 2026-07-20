"""Difficulty-evaluation sweep: probe ladder x seed panel x K rollouts.

Writes one JSONL record per (probe, seed, rollout). Resumable: existing
records are skipped, so the sweep can be re-run after interruption or with
new probes/seeds appended.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/difficulty_eval.py \
        [--n-seeds 100] [--seeds-file F] [--rollouts 8] [--workers 8] \
        [--probes probes.json] [--out runs/difficulty/panel.jsonl]

Default probe ladder spans the skill range we own: chip-greedy floor,
net checkpoints at rising search budgets, scripted router.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import get_context
from pathlib import Path

from balatro_zero.probes import ProbeSpec, probe_worker

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEFAULT_PROBES = [
    ProbeSpec("chip_greedy", "chip_greedy", {"eps": 0.05}),
    ProbeSpec("v6_s16", "net", {"ckpt": "runs/v6/latest.pt", "sims": 16, "depth": 1}),
    ProbeSpec("v10_s16", "net", {"ckpt": "runs/v10/latest.pt", "sims": 16, "depth": 1}),
    ProbeSpec("v10_s64", "net", {"ckpt": "runs/v10/latest.pt", "sims": 64, "depth": 1}),
    ProbeSpec("router_e30", "router", {"eps": 0.30}),   # mid-rung: degraded expert
    ProbeSpec("router_e15", "router", {"eps": 0.15}),   # mid-rung
    ProbeSpec("router", "router", {"eps": 0.05}),
    ProbeSpec("gold", "gold", {"eps": 0.05}),           # top rung: win boundary
]


def load_probes(path: str | None) -> list[ProbeSpec]:
    if path is None:
        return DEFAULT_PROBES
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ProbeSpec(p["name"], p["kind"], p.get("params", {})) for p in raw]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--seeds-file", type=str, default=None,
                    help="one seed per line; overrides --n-seeds generation")
    ap.add_argument("--seed-rng", type=int, default=777,
                    help="rng for panel generation (fixed => stable panel)")
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probes", type=str, default=None)
    ap.add_argument("--out", type=str, default="runs/difficulty/panel.jsonl")
    args = ap.parse_args()

    if args.seeds_file:
        seeds = [s.strip() for s in Path(args.seeds_file).read_text().splitlines() if s.strip()]
    else:
        rng = random.Random(args.seed_rng)
        seeds = ["".join(rng.choices(ALPHABET, k=8)) for _ in range(args.n_seeds)]

    specs = load_probes(args.probes)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str, int]] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["probe"], r["seed"], r["rollout"]))
            except json.JSONDecodeError:
                continue
        print(f"resume: {len(done)} records already present")

    tasks: list[tuple[int, str, int]] = []
    for si, spec in enumerate(specs):
        for seed in seeds:
            for k in range(args.rollouts):
                if (spec.name, seed, k) not in done:
                    tasks.append((si, seed, k))
    print(f"{len(specs)} probes x {len(seeds)} seeds x {args.rollouts} rollouts "
          f"-> {len(tasks)} games to run")
    if not tasks:
        return

    # Interleave probes so cheap and expensive tasks mix evenly per chunk.
    random.Random(0).shuffle(tasks)
    chunks = [tasks[i::args.workers * 4] for i in range(args.workers * 4)]
    chunks = [c for c in chunks if c]

    t0 = time.perf_counter()
    n_done = 0
    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool, open(out_path, "a", encoding="utf-8") as f:
        for results in pool.imap_unordered(
            _chunk_runner, [(specs, c, i) for i, c in enumerate(chunks)]
        ):
            for r in results:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n_done += len(results)
            rate = n_done / (time.perf_counter() - t0)
            print(f"  {n_done}/{len(tasks)} games ({rate:.1f}/s, "
                  f"eta {(len(tasks) - n_done) / max(rate, 1e-9) / 60:.0f}m)", flush=True)

    errs = sum(1 for c in chunks for _ in c) - n_done
    print(f"done: {n_done} records in {(time.perf_counter() - t0) / 60:.1f}m "
          f"-> {out_path}{f' ({errs} lost)' if errs else ''}")


def _chunk_runner(payload):
    specs, chunk, wid = payload
    return probe_worker(specs, chunk, wid)


if __name__ == "__main__":
    main()
