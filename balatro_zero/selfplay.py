"""Self-play: run games with Gumbel search, emit training samples.

A sample is (obs, pi_target, z_win, z_togo) attached after the game ends.
z_togo is PROGRESS-TO-GO: best-progress-ever-reached minus progress at the
position (clipped to [0,1]). Compared to the old shared episode-max scalar,
this conditions the value regression on realized progress, so the head
learns what the *state* (board, money, deck) predicts about the future —
the shop-relevant signal. Search composes it back into an absolute value:
v(s) = 0.5*P(win) + 0.5*(progress(s) + togo(s)).

Deep-state curriculum: games that reach ante >= snapshot_min_ante emit
pickled snapshots; the trainer keeps a pool, and a fraction of subsequent
self-play games START from a sampled snapshot instead of a fresh run —
concentrating experience on the deep states the buffer otherwise almost
never contains (~95% of positions were ante 1-2). Progress-to-go targets
make snapshot-started games train correctly with no special casing.

`worker_run` is the top-level entry point for multiprocessing workers
(Windows spawn-safe): loads a checkpoint, plays its share of games on CPU
with a single torch thread, returns samples + stats + new snapshots.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from balatro_zero.net import PolicyValueNet
from balatro_zero.router import ECON_PHASES, scripted_econ_action
from balatro_zero.search import _apply, gumbel_search
from balatro_zero.state import (
    MAX_ACTIONS,
    Obs,
    ante,
    clone,
    is_terminal,
    legal_factored,
    new_run,
    observe,
    progress,
    step_factored,
    won,
)

MAX_SNAPSHOTS_PER_GAME = 3


@dataclass
class GameStats:
    won: bool
    max_ante: int
    moves: int
    progress: float
    curriculum: bool = False  # game started from a deep-state snapshot
    guided: bool = False      # economy phases played by the scripted router


@dataclass
class SelfPlayConfig:
    sims: int = 16
    k_max: int = 8
    depth: int = 2
    blind_finisher: bool = False    # finish blinds with the beam inside
                                    # rollouts instead of policy-greedy
                                    # card play (see search._batched_rollouts)
    macro_k: int = 0                # offer k whole-blind beam plans as
                                    # ACTIONS at hand nodes (0 = off)
    max_moves: int = 400
    stake: int = 1
    back_key: str = "b_red"
    curriculum_frac: float = 0.35   # fraction of games started from snapshots
    snapshot_min_ante: int = 3      # snapshot when a game first reaches this ante
    seed_pool_frac: float = 0.5     # of curriculum starts: share drawn from the
                                    # static seed pool (externally harvested
                                    # god-run states) vs the live FIFO pool
    guided_frac: float = 0.0        # fraction of games with router-scripted
                                    # economy (DAgger-style: expert shop/pack
                                    # actions become one-hot policy targets;
                                    # hand play stays net+search)


def sample_start_state(
    rng: np.random.Generator,
    cfg: SelfPlayConfig,
    live_pool: list[bytes],
    seed_pool: list[bytes],
) -> dict[str, Any] | None:
    """Curriculum start-state sampling over the live and static seed pools."""
    if not live_pool and not seed_pool:
        return None
    if rng.random() >= cfg.curriculum_frac:
        return None
    use_seed = bool(seed_pool) and (
        not live_pool or rng.random() < cfg.seed_pool_frac
    )
    src = seed_pool if use_seed else live_pool
    return pickle.loads(src[int(rng.integers(len(src)))])


def play_game(
    net: PolicyValueNet,
    device: torch.device,
    seed: str,
    cfg: SelfPlayConfig,
    rng: np.random.Generator,
    *,
    root_noise: bool = True,
    start_state: dict[str, Any] | None = None,
    guided: bool = False,
) -> tuple[list[tuple[Obs, np.ndarray, float, float]], GameStats, list[bytes]]:
    if start_state is not None:
        gs = clone(start_state)
    else:
        gs = new_run(seed, back_key=cfg.back_key, stake=cfg.stake)

    positions: list[tuple[Obs, np.ndarray, float]] = []  # (obs, pi, progress@root)
    snapshots: list[bytes] = []
    max_ante_seen = last_snap_ante = ante(gs)
    max_progress = progress(gs)
    moves = 0

    while not is_terminal(gs) and moves < cfg.max_moves:
        action = None
        if guided and gs.get("phase") in ECON_PHASES:
            # DAgger-style: the router picks the econ action on the agent's
            # own visitation distribution; its choice becomes a one-hot
            # policy target. Hand play stays net+search.
            legal = legal_factored(gs)
            scripted = scripted_econ_action(gs, legal)
            if scripted is not None:
                try:
                    idx = legal.index(scripted)
                except ValueError:
                    idx = -1
                if idx >= 0:
                    pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
                    pi[idx] = 1.0
                    positions.append((observe(gs), pi, progress(gs)))
                    action = scripted
        if action is None:
            res = gumbel_search(
                gs,
                net,
                device,
                n_sims=cfg.sims,
                k_max=cfg.k_max,
                depth=cfg.depth,
                rng=rng,
                root_noise=root_noise,
                blind_finisher=cfg.blind_finisher,
                macro_k=cfg.macro_k,
            )
            if res is None:  # unplayable dead-end state — treat as game over
                break
            positions.append((res.root_obs, res.pi_target, progress(gs)))
            action = res.actions[res.action_idx]
        # _apply, not step_factored: a chosen macro plan is a whole-blind
        # line and must be executed in full.
        _apply(gs, action)
        moves += 1
        a = ante(gs)
        max_ante_seen = max(max_ante_seen, a)
        max_progress = max(max_progress, progress(gs))
        if (
            a >= cfg.snapshot_min_ante
            and a > last_snap_ante
            and len(snapshots) < MAX_SNAPSHOTS_PER_GAME
        ):
            snapshots.append(pickle.dumps(gs, protocol=5))
            last_snap_ante = a

    w = won(gs)
    max_progress = max(max_progress, progress(gs))
    z_win = 1.0 if w else 0.0
    final_best = 1.0 if w else max_progress
    samples = [
        (obs, pi, z_win, float(np.clip(final_best - prog_t, 0.0, 1.0)))
        for obs, pi, prog_t in positions
    ]
    stats = GameStats(
        won=w,
        max_ante=max_ante_seen,
        moves=moves,
        progress=final_best,
        curriculum=start_state is not None,
        guided=guided,
    )
    return samples, stats, snapshots


def _load_pool(path: str | None) -> list[bytes]:
    if not path:
        return []
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError):
        return []


def worker_run(
    ckpt_path: str,
    n_games: int,
    seed_prefix: str,
    worker_id: int,
    cfg: SelfPlayConfig,
    pool_path: str | None = None,
    seed_pool_path: str | None = None,
) -> tuple[list[tuple[Obs, np.ndarray, float, float]], list[GameStats], list[bytes]]:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    net = PolicyValueNet()
    net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    net.eval()

    pool = _load_pool(pool_path)
    seed_pool = _load_pool(seed_pool_path)

    rng = np.random.default_rng(worker_id * 100_003 + 17)
    all_samples: list[tuple[Obs, np.ndarray, float, float]] = []
    stats: list[GameStats] = []
    new_snapshots: list[bytes] = []
    for i in range(n_games):
        seed = f"{seed_prefix}W{worker_id}G{i}"
        start_state = sample_start_state(rng, cfg, pool, seed_pool)
        guided = cfg.guided_frac > 0 and rng.random() < cfg.guided_frac
        try:
            samples, st, snaps = play_game(
                net, device, seed, cfg, rng, start_state=start_state, guided=guided
            )
        except Exception as e:  # noqa: BLE001 — one bad game must not kill the run
            import sys
            import traceback

            print(f"[worker {worker_id}] game {seed} crashed: {e}", file=sys.stderr)
            traceback.print_exc()
            continue
        all_samples.extend(samples)
        stats.append(st)
        new_snapshots.extend(snaps)
    return all_samples, stats, new_snapshots
