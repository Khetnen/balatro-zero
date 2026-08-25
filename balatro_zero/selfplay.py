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

import os
import pickle
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from balatro_zero.net import (
    PolicyValueNet,
    is_factored,
    load_net,
    market_area_lens,
)
from jackdaw.engine.actions import GamePhase

from balatro_zero.router import ECON_PHASES, scripted_econ_action
from balatro_zero.search import _apply, gumbel_search
from balatro_zero.targets import encode_candidates
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

# Guided (DAgger) games script only the shop and pack phases. BLIND_SELECT is
# deliberately excluded: the router's default flags never skip a blind, so
# one-hot SelectBlind targets would smuggle anti-skip supervision into the
# policy head — the bias the skip-neutral v13 reward exists to avoid. The
# skip/select decision stays net+search even in guided games.
GUIDED_PHASES = frozenset(ECON_PHASES) - {GamePhase.BLIND_SELECT}


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
    determinize: bool = True        # rollout clones get a fresh PRNG seed +
                                    # reshuffled undrawn deck (honest futures);
                                    # False = clairvoyant rollouts, the
                                    # pre-2026-08-11 behavior every earlier
                                    # checkpoint was trained/evaluated with
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
) -> tuple[list[tuple[Obs, Any, float, float]], GameStats, list[bytes]]:
    if start_state is not None:
        gs = clone(start_state)
    else:
        gs = new_run(seed, back_key=cfg.back_key, stake=cfg.stake)

    # Factored (V5) nets are supervised by CONTENT: the target is the
    # improved policy over the root's candidate set (targets.CandidateSet),
    # not a positional pi vector — a positional index means nothing to a
    # net that scores actions from type/entity/card factors.
    factored = is_factored(net)
    # Global-entity (V6) nets score entities in the joint 28-slot layout;
    # their targets must be encoded in the same layout, which needs the
    # market's live area lengths at each decision.
    global_ent = getattr(net, "GLOBAL_ENTITY", False)
    positions: list[tuple[Obs, Any, float]] = []  # (obs, target, progress@root)
    snapshots: list[bytes] = []
    max_ante_seen = last_snap_ante = ante(gs)
    max_progress = progress(gs)
    moves = 0

    while not is_terminal(gs) and moves < cfg.max_moves:
        action = None
        if guided and gs.get("phase") in GUIDED_PHASES:
            # DAgger-style: the router picks the econ action on the agent's
            # own visitation distribution; its choice becomes a one-hot
            # policy target. Hand play stays net+search.
            legal = legal_factored(gs, rng)
            scripted = scripted_econ_action(gs, legal)
            if scripted is not None:
                try:
                    idx = legal.index(scripted)
                except ValueError:
                    idx = -1
                if idx >= 0:
                    if factored:
                        # One-hot over the FULL legal set: "this action,
                        # not the others" — a K=1 set would carry no
                        # gradient through the candidate softmax.
                        w = np.zeros(len(legal), dtype=np.float32)
                        w[idx] = 1.0
                        tgt = encode_candidates(
                            legal, w, len(gs.get("hand", [])),
                            market_lens=market_area_lens(gs) if global_ent else None,
                        )
                    else:
                        tgt = np.zeros(MAX_ACTIONS, dtype=np.float32)
                        tgt[idx] = 1.0
                    positions.append((observe(gs), tgt, progress(gs)))
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
                determinize=cfg.determinize,
            )
            if res is None:  # unplayable dead-end state — treat as game over
                break
            if factored:
                tgt = encode_candidates(
                    res.actions,
                    res.pi_target[: len(res.actions)],
                    len(gs.get("hand", [])),
                    market_lens=market_area_lens(gs) if global_ent else None,
                )
            else:
                tgt = res.pi_target
            positions.append((res.root_obs, tgt, progress(gs)))
            action = res.actions[res.action_idx]
        # _apply, not step_factored: a chosen macro plan is a whole-blind
        # line and must be executed in full.
        _apply(gs, action)
        moves += 1
        a = ante(gs)
        max_ante_seen = max(max_ante_seen, a)
        max_progress = max(max_progress, progress(gs))
        if won(gs):
            # The episode ends at the win. The engine itself continues
            # into endless mode, and a lost endless blind CLOBBERS the
            # flag (game.py: gs["won"] = False at any GAME_OVER) — before
            # this break, a game that beat the ante-8 boss and then died
            # in ante 9+ was recorded as a LOSS with z_win = 0. Invisible
            # while nothing ever won; found by the closeout probe
            # (2026-08-18: 13 of its wins were flagged won=False at
            # final_ante 9). Breaking here also keeps post-win states out
            # of the curriculum snapshot pool below.
            break
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


def eval_worker(
    ckpt_path: str,
    seeds: list[str],
    cfg: SelfPlayConfig,
) -> list[GameStats]:
    """Spawn-safe eval worker: greedy games (no root noise) on fixed seeds.

    Eval ran 16 games SERIALLY in the main process while self-play's 48
    got a worker pool — an unlogged ~700s (70%) of every v11 iteration
    once games stopped dying in ante 1. Per-seed rng derivation keeps
    each eval game reproducible regardless of pooling or seed split.
    """
    torch.set_num_threads(1)
    net = load_net(str(ckpt_path))
    stats: list[GameStats] = []
    for seed in seeds:
        rng = np.random.default_rng(zlib.crc32(f"EVAL|{seed}".encode()))
        try:
            _, st, _ = play_game(
                net, torch.device("cpu"), seed, cfg, rng, root_noise=False
            )
        except Exception as e:  # noqa: BLE001 — one bad game must not kill eval
            import sys

            print(f"[eval] game {seed} crashed: {e}", file=sys.stderr)
            continue
        stats.append(st)
    return stats


def _load_pool(path: str | None) -> list[bytes]:
    if not path:
        return []
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError):
        return []


# ---------------------------------------------------------------------------
# Persistent-pool workers (one task per GAME, dynamic load balancing)
#
# The old worker_run gave each worker a FIXED share of the iteration's
# games, so selfplay wall was the slowest worker's share — and game
# lengths vary ~30x (a 9-move ante-1 death vs a 100+-move ante-4 run),
# which at ~3 games per worker made straggler waits a large fraction of
# the phase. One task per game lets any free worker steal the queue, and
# a pool that persists across iterations stops re-paying worker spawn
# (torch import) every iteration. Net and snapshot pools are cached per
# process, keyed by file mtime, so they load once per iteration each.
# ---------------------------------------------------------------------------

_NET_CACHE: dict[tuple[str, int], PolicyValueNet] = {}
_POOL_CACHE: dict[tuple[str, int], list[bytes]] = {}


def worker_init() -> None:
    """Pool initializer: workers do single-thread CPU inference."""
    torch.set_num_threads(1)


def _cached_net(ckpt_path: str) -> PolicyValueNet:
    # load_net sniffs the architecture from the state dict — constructing
    # the base class here crashed outright on any V4/V5 checkpoint.
    key = (ckpt_path, os.stat(ckpt_path).st_mtime_ns)
    net = _NET_CACHE.get(key)
    if net is None:
        # Two slots, FIFO eviction: with eval overlapped into the next
        # iteration's self-play, a worker legitimately serves two nets at
        # once (latest.pt for self-play, ckpt_NNNN.pt for eval). Eviction
        # must be bounded rather than path-keyed — every iteration's eval
        # checkpoint is a NEW path, so per-path staleness would leak one
        # net per eval iteration forever.
        while len(_NET_CACHE) >= 2:
            _NET_CACHE.pop(next(iter(_NET_CACHE)))
        net = load_net(ckpt_path)
        _NET_CACHE[key] = net
    return net


def _cached_pool(path: str | None) -> list[bytes]:
    if not path:
        return []
    try:
        key = (path, os.stat(path).st_mtime_ns)
    except OSError:
        return []
    pool = _POOL_CACHE.get(key)
    if pool is None:
        for k in [k for k in _POOL_CACHE if k[0] == path]:
            del _POOL_CACHE[k]
        pool = _load_pool(path)
        _POOL_CACHE[key] = pool
    return pool


def play_one_game(
    ckpt_path: str,
    seed: str,
    rng_seed: int,
    cfg: SelfPlayConfig,
    pool_path: str | None = None,
    seed_pool_path: str | None = None,
) -> tuple[list[tuple[Obs, Any, float, float]], GameStats | None, list[bytes]]:
    """One self-play game as a pool task; stats None if the game crashed.

    The rng is derived from ``rng_seed`` alone, so a game's trajectory is
    independent of which worker runs it and in what order — scheduling
    stays free to balance load without touching reproducibility.
    """
    net = _cached_net(ckpt_path)
    pool = _cached_pool(pool_path)
    seed_pool = _cached_pool(seed_pool_path)
    rng = np.random.default_rng(rng_seed)
    start_state = sample_start_state(rng, cfg, pool, seed_pool)
    guided = cfg.guided_frac > 0 and rng.random() < cfg.guided_frac
    try:
        return play_game(
            net, torch.device("cpu"), seed, cfg, rng,
            start_state=start_state, guided=guided,
        )
    except Exception as e:  # noqa: BLE001 — one bad game must not kill the run
        import sys
        import traceback

        print(f"[selfplay] game {seed} crashed: {e}", file=sys.stderr)
        traceback.print_exc()
        return [], None, []


def eval_one_game(
    ckpt_path: str,
    seed: str,
    cfg: SelfPlayConfig,
) -> GameStats | None:
    """One greedy eval game as a pool task (rng derivation == eval_worker's,
    so pooled eval numbers stay comparable with every earlier run)."""
    net = _cached_net(ckpt_path)
    rng = np.random.default_rng(zlib.crc32(f"EVAL|{seed}".encode()))
    try:
        _, st, _ = play_game(
            net, torch.device("cpu"), seed, cfg, rng, root_noise=False
        )
    except Exception as e:  # noqa: BLE001 — one bad game must not kill eval
        import sys

        print(f"[eval] game {seed} crashed: {e}", file=sys.stderr)
        return None
    return st
