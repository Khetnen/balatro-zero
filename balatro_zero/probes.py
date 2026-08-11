"""Difficulty-probe ladder: policies at known skill levels, run stochastically.

A probe is a named policy evaluated with K stochastic rollouts per seed, so
each (seed, probe) yields an estimated outcome distribution rather than one
deterministic trajectory (the game is seeded: determinism would make win
rate a 0/1 step). Scripted probes get epsilon-random actions; net probes are
stochastic through Gumbel root noise.

Probe kinds:
  chip_greedy — engine-exact greedy hand play, buys nothing (the no-economy floor)
  router      — full scripted router (flush build + priority economy)
  net         — checkpoint + Gumbel search (sims/depth), root noise on;
                rollouts are DETERMINIZED (honest futures) unless
                params clairvoyant=True. Every net-probe artifact recorded
                before 2026-08-11 used clairvoyant rollouts — do not mix
                the two under one probe name.

Each rollout's rng is derived from (probe, seed, rollout) so results are
reproducible and resumable record-by-record.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from balatro_zero.router import (
    ECON_PHASES,
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


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    kind: str                    # chip_greedy | router | net
    params: dict[str, Any] = field(default_factory=dict)


def rollout_rng_seed(probe_name: str, seed: str, rollout: int) -> int:
    return zlib.crc32(f"{probe_name}|{seed}|{rollout}".encode())


# Per-process net cache (spawn workers load each checkpoint once).
_NET_CACHE: dict[str, Any] = {}


def _get_net(ckpt: str):
    if ckpt not in _NET_CACHE:
        import torch

        from balatro_zero.net import load_net

        torch.set_num_threads(1)
        # load_net sniffs the architecture: the v0-v10 checkpoints are
        # difficulty-ladder rungs and must keep loading after the joker
        # attention encoder landed.
        _NET_CACHE[ckpt] = load_net(ckpt)
    return _NET_CACHE[ckpt]


def _scripted_game(seed: str, rng: np.random.Generator, eps: float,
                   economy: bool) -> dict[str, Any]:
    from jackdaw.env.action_space import ActionType

    gs = new_run(seed)
    moves = 0
    max_ante = ante(gs)
    while not is_terminal(gs) and moves < MAX_MOVES:
        action = None
        legal = None
        if eps > 0 and rng.random() < eps:
            legal = legal_factored(gs)
            if not legal:
                break
            action = legal[int(rng.integers(len(legal)))]
        elif gs.get("phase") in ECON_PHASES:
            if economy:
                action = scripted_econ_action(gs)
            else:
                # No-economy floor: select blinds, skip every purchase.
                legal = legal_factored(gs)
                for want in (ActionType.SelectBlind, ActionType.SkipPack,
                             ActionType.NextRound, ActionType.CashOut):
                    picks = [a for a in legal if a.action_type == want]
                    if picks:
                        action = picks[0]
                        break
        else:
            action = scripted_hand_action(gs)
        if action is None:
            legal = legal if legal is not None else legal_factored(gs)
            if not legal:
                break
            action = legal[0]
        try:
            step_factored(gs, action)
        except Exception:  # noqa: BLE001
            legal = legal_factored(gs)
            if not legal:
                break
            step_factored(gs, legal[0])
        moves += 1
        max_ante = max(max_ante, ante(gs))
    return {
        "won": won(gs), "max_ante": max_ante,
        "progress": round(max(progress(gs), 0.0), 4), "moves": moves,
    }


def _net_game(seed: str, rng: np.random.Generator, ckpt: str,
              sims: int, depth: int,
              blind_finisher: bool = False,
              macro_k: int = 0,
              clairvoyant: bool = False) -> dict[str, Any]:
    import torch

    from balatro_zero.selfplay import SelfPlayConfig, play_game

    net = _get_net(ckpt)
    cfg = SelfPlayConfig(sims=sims, k_max=min(8, max(2, sims)), depth=depth,
                         blind_finisher=blind_finisher, macro_k=macro_k,
                         determinize=not clairvoyant)
    _, st, _ = play_game(
        net, torch.device("cpu"), seed, cfg, rng, root_noise=True
    )
    return {
        "won": st.won, "max_ante": st.max_ante,
        "progress": round(st.progress, 4), "moves": st.moves,
    }


def run_probe_game(spec: ProbeSpec, seed: str, rollout: int) -> dict[str, Any]:
    rng = np.random.default_rng(rollout_rng_seed(spec.name, seed, rollout))
    if spec.kind == "chip_greedy":
        out = _scripted_game(seed, rng, spec.params.get("eps", 0.05), economy=False)
    elif spec.kind == "router":
        out = _scripted_game(seed, rng, spec.params.get("eps", 0.05), economy=True)
    elif spec.kind == "net":
        out = _net_game(
            seed, rng, spec.params["ckpt"],
            int(spec.params.get("sims", 32)), int(spec.params.get("depth", 1)),
            bool(spec.params.get("blind_finisher", False)),
            int(spec.params.get("macro_k", 0)),
            bool(spec.params.get("clairvoyant", False)),
        )
    elif spec.kind == "gold":
        from balatro_zero.goldprobe import gold_game

        g = gold_game(seed, econ_eps=spec.params.get("eps", 0.05), rng=rng)
        out = {k: g[k] for k in ("won", "max_ante", "progress", "moves")}
    else:
        raise ValueError(f"unknown probe kind {spec.kind!r}")
    out.update({"probe": spec.name, "seed": seed, "rollout": rollout})
    return out


def probe_worker(specs: list[ProbeSpec], tasks: list[tuple[int, str, int]],
                 worker_id: int) -> list[dict[str, Any]]:
    """Spawn-safe worker: tasks are (spec_index, seed, rollout)."""
    results = []
    for spec_i, seed, rollout in tasks:
        try:
            results.append(run_probe_game(specs[spec_i], seed, rollout))
        except Exception as e:  # noqa: BLE001 — one bad game must not kill the sweep
            results.append({
                "probe": specs[spec_i].name, "seed": seed, "rollout": rollout,
                "error": f"{type(e).__name__}: {e}",
            })
    return results
