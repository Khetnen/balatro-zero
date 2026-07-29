"""GATE: can the network represent joker board ORDER at all?

Blueprint copies the joker to its RIGHT (card.lua:2321), so on a real
board [Blueprint, Photograph] and [Photograph, Blueprint] are different
game states with different scoring. The SONNET01 run was won on exactly
this: pointing Blueprint at Photograph rather than at Trio.

The v3 network pools joker embeddings with masked mean+max, which is
permutation-invariant -- it maps both orders to the identical vector, so
V is provably constant under reordering no matter how long it trains.
That is an architectural ceiling, not a data problem, and it sits under
the whole v1/v2/v3 plateau lineage.

This is a REPRESENTATION test, not a quality test: it asks whether the
architecture CAN express order, so it is meaningful at random init and
does not need a trained checkpoint. A v3 net must score 0 here and a v4
net must not. Run it before spending on data or training.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/adjacency_probe.py
    uv run --no-sync python scripts/adjacency_probe.py --ckpt runs/x/latest.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

PAIRS = [
    ("j_blueprint", "j_photograph"),   # the SONNET01 winning adjacency
    ("j_blueprint", "j_trio"),
    ("j_brainstorm", "j_baron"),
    ("j_blueprint", "j_hanging_chad"),
]


def _obs_with(jokers: list[str]):
    from balatro_zero.state import (
        N_CONSUMABLE_SLOTS,
        N_JOKER_SLOTS,
        N_MARKET_SLOTS,
        Obs,
        OBS_DIM,
        center_key_id,
    )

    ids = np.zeros(N_JOKER_SLOTS, dtype=np.int64)
    for i, k in enumerate(jokers):
        ids[i] = center_key_id(k)
    return Obs(
        flat=np.zeros(OBS_DIM, dtype=np.float32),
        joker_ids=ids,
        consumable_ids=np.zeros(N_CONSUMABLE_SLOTS, dtype=np.int64),
        market_ids=np.zeros(N_MARKET_SLOTS, dtype=np.int64),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="optional; the test is valid at random init")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from balatro_zero.net import PolicyValueNet, PolicyValueNetV4, evaluate, load_net

    torch.manual_seed(args.seed)
    if args.ckpt:
        nets = [("checkpoint", load_net(args.ckpt))]
    else:
        nets = [("v3 (pooled, baseline)", PolicyValueNet()),
                ("v4 (attention)", PolicyValueNetV4())]

    dev = torch.device("cpu")
    failed = False
    for label, net in nets:
        net.eval()
        print(f"\n{label}")
        deltas = []
        for a, b in PAIRS:
            va = evaluate(net, _obs_with([a, b]), dev)[1][0]
            vb = evaluate(net, _obs_with([b, a]), dev)[1][0]
            d = abs(float(va) - float(vb))
            deltas.append(d)
            print(f"  {a:<14} {b:<16} |dV| = {d:.3e}")
        worst = max(deltas)
        ok = worst > 1e-6
        print(f"  -> max |dV| = {worst:.3e}  "
              f"{'ORDER-SENSITIVE' if ok else 'PERMUTATION-INVARIANT (blind)'}")
        if "v4" in label or args.ckpt:
            failed = failed or not ok

    print()
    if failed:
        raise SystemExit(
            "GATE FAILED: the network is blind to joker order. Adjacency "
            "effects (Blueprint/Brainstorm) cannot be learned; fix the "
            "architecture before spending on data or training."
        )
    print("GATE PASSED: the architecture can represent joker board order.")


if __name__ == "__main__":
    main()
