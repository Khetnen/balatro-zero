"""Expert-iteration training loop.

Each iteration:
  1. Self-play N games with the current net (Gumbel search) -> samples
  2. Add to replay buffer
  3. Train E epochs of minibatches over the buffer
  4. Checkpoint (checkpoints double as the difficulty-reference ensemble)
  5. Evaluate greedily (no root noise) on held-out seeds — in pooled mode
     dispatched ASYNC: the eval games share the worker pool with the NEXT
     iteration's self-play, which runs the very same weights (latest.pt is
     saved from the same state dict as ckpt_it), so eval semantics are
     unchanged but its wall time is hidden. The iteration's metrics row is
     held until its eval resolves, so metrics.jsonl keeps its shape (eval
     of ckpt_it in row it, rows in iteration order) and t_eval_s now means
     "wall actually spent blocked on eval" (~0 when the overlap works). A
     crash loses the held row along with its pending eval — one extra row
     in the splice gap, nothing else.

Run from the balatro-zero directory:
    uv run bzero --iters 50 --games 96 --workers 12
Smoke test:
    uv run bzero --iters 1 --games 2 --sims 8 --depth 0 --workers 0
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from multiprocessing import get_context
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from balatro_zero.net import (
    PolicyValueNet,
    PolicyValueNetV4,
    PolicyValueNetV5,
    PolicyValueNetV6,
    is_factored,
    load_net,
)
from balatro_zero.replay import ReplayBuffer
from balatro_zero.selfplay import (
    GameStats,
    SelfPlayConfig,
    eval_one_game,
    eval_worker,
    play_game,
    play_one_game,
    sample_start_state,
    worker_init,
)
from balatro_zero.targets import collate_candidate_sets, factored_policy_loss

ARCHS = {"v3": PolicyValueNet, "v4": PolicyValueNetV4,
         "v5": PolicyValueNetV5, "v6": PolicyValueNetV6}


def train_epochs(
    net: PolicyValueNet,
    opt: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    progress_loss_weight: float = 0.5,
    demo_buffer: ReplayBuffer | None = None,
    demo_frac: float = 0.0,
) -> dict[str, float]:
    net.train()
    factored = is_factored(net)
    n_batches_per_epoch = max(1, len(buffer) // batch_size)
    losses: dict[str, float] = {"policy": 0.0, "win": 0.0, "progress": 0.0}
    n_steps = 0
    n_demo = 0
    if demo_buffer is not None and len(demo_buffer) > 0 and demo_frac > 0:
        n_demo = max(1, int(batch_size * demo_frac))
    for _ in range(epochs):
        for _ in range(n_batches_per_epoch):
            flat, jid, cid, mid, pi, z_win, z_prog = buffer.sample(batch_size - n_demo, rng)
            if n_demo:
                parts = demo_buffer.sample(n_demo, rng)
                # Policy targets concatenate as arrays (positional) or
                # lists (factored CandidateSets); everything else is arrays.
                pi = (pi + parts[4]) if factored else np.concatenate([pi, parts[4]])
                flat, jid, cid, mid, z_win, z_prog = (
                    np.concatenate([a, b])
                    for a, b in zip((flat, jid, cid, mid, z_win, z_prog),
                                    parts[:4] + parts[5:])
                )
            flat_t = torch.from_numpy(flat).float().to(device)
            jid_t = torch.from_numpy(jid).to(device)
            cid_t = torch.from_numpy(cid).to(device)
            mid_t = torch.from_numpy(mid).to(device)
            z_win_t = torch.from_numpy(z_win).to(device)
            z_prog_t = torch.from_numpy(z_prog).to(device)

            if factored:
                type_lg, ent_lg, card_lg, p_win, prog = net(flat_t, jid_t, cid_t, mid_t)
                fb = collate_candidate_sets(pi, device)
                policy_loss = factored_policy_loss(type_lg, ent_lg, card_lg, fb)
            else:
                logits, p_win, prog = net(flat_t, jid_t, cid_t, mid_t)
                pi_t = torch.from_numpy(pi).to(device)
                log_probs = F.log_softmax(logits, dim=-1)
                policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()
            win_loss = F.binary_cross_entropy(p_win, z_win_t)
            progress_loss = F.binary_cross_entropy(prog, z_prog_t)
            loss = policy_loss + win_loss + progress_loss_weight * progress_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            losses["policy"] += policy_loss.item()
            losses["win"] += win_loss.item()
            losses["progress"] += progress_loss.item()
            n_steps += 1
    net.eval()
    return {k: v / max(n_steps, 1) for k, v in losses.items()}


def run_selfplay(
    ckpt_path: Path,
    net: PolicyValueNet,
    device: torch.device,
    *,
    n_games: int,
    mp_pool: Pool | None,
    seed_prefix: str,
    cfg: SelfPlayConfig,
    rng: np.random.Generator,
    pool_path: Path | None = None,
    seed_pool_path: Path | None = None,
) -> tuple[list, list[GameStats], list[bytes]]:
    if mp_pool is None:
        import pickle as _pickle

        snap_pool: list[bytes] = []
        if pool_path is not None and pool_path.exists():
            snap_pool = _pickle.loads(pool_path.read_bytes())
        static_pool: list[bytes] = []
        if seed_pool_path is not None and seed_pool_path.exists():
            static_pool = _pickle.loads(seed_pool_path.read_bytes())
        samples: list = []
        stats: list[GameStats] = []
        new_snaps: list[bytes] = []
        for i in range(n_games):
            start_state = sample_start_state(rng, cfg, snap_pool, static_pool)
            guided = cfg.guided_frac > 0 and rng.random() < cfg.guided_frac
            s, st, sn = play_game(
                net, device, f"{seed_prefix}G{i}", cfg, rng,
                start_state=start_state, guided=guided,
            )
            samples.extend(s)
            stats.append(st)
            new_snaps.extend(sn)
        return samples, stats, new_snaps

    # One task per game on the persistent pool: any free worker takes the
    # next game, so the phase ends at (total work / workers) + one game's
    # tail instead of waiting on the slowest fixed share. The per-game rng
    # seed makes each game reproducible regardless of scheduling.
    tasks = [
        (str(ckpt_path), f"{seed_prefix}G{g}",
         zlib.crc32(f"{seed_prefix}|G{g}".encode()), cfg,
         str(pool_path) if pool_path is not None else None,
         str(seed_pool_path) if seed_pool_path is not None else None)
        for g in range(n_games)
    ]
    results = mp_pool.starmap(play_one_game, tasks, chunksize=1)
    samples = []
    stats = []
    new_snaps = []
    for s, st, sn in results:
        samples.extend(s)
        if st is not None:
            stats.append(st)
        new_snaps.extend(sn)
    return samples, stats, new_snaps


def evaluate_pooled(
    ckpt_path: Path,
    *,
    n_games: int,
    mp_pool: Pool | None,
    cfg: SelfPlayConfig,
) -> list[GameStats]:
    """Greedy eval on the fixed EVAL seeds, parallelized like self-play.

    Reads the just-saved checkpoint (eval must see POST-training weights,
    which is why the per-iteration ckpt is now saved before eval runs).
    Per-seed rng derivation lives in eval_one_game/eval_worker, so the
    numbers are identical under any scheduling.
    """
    seeds = [f"EVAL{i}" for i in range(n_games)]
    if mp_pool is None:
        return eval_worker(str(ckpt_path), seeds, cfg)
    results = mp_pool.starmap(
        eval_one_game, [(str(ckpt_path), s, cfg) for s in seeds], chunksize=1
    )
    return [st for st in results if st is not None]


def dispatch_eval(
    ckpt_path: Path,
    *,
    n_games: int,
    mp_pool: Pool,
    cfg: SelfPlayConfig,
):
    """Submit the greedy eval to the pool WITHOUT waiting (starmap_async).

    Called right after ckpt_it is saved; the games then share the pool
    with iteration it+1's self-play, which runs the same weights, so the
    eval's wall time is hidden behind work that had to happen anyway.
    Per-seed rng derivation lives in eval_one_game, so the results are
    identical to the synchronous path under any scheduling.
    """
    seeds = [f"EVAL{i}" for i in range(n_games)]
    return mp_pool.starmap_async(
        eval_one_game, [(str(ckpt_path), s, cfg) for s in seeds], chunksize=1
    )


def _summarize(stats: list[GameStats]) -> dict[str, float]:
    # Headline numbers use FRESH UNGUIDED games only — curriculum games start
    # deep and guided games get expert econ help, so mixing either would
    # inflate progress and break comparability with earlier runs.
    fresh = [s for s in stats if not s.curriculum and not s.guided]
    curr = [s for s in stats if s.curriculum]
    guided = [s for s in stats if s.guided]
    if not fresh:
        base = {"win_rate": 0.0, "mean_ante": 0.0, "mean_progress": 0.0, "mean_moves": 0.0}
    else:
        base = {
            "win_rate": sum(s.won for s in fresh) / len(fresh),
            "mean_ante": sum(s.max_ante for s in fresh) / len(fresh),
            "mean_progress": sum(s.progress for s in fresh) / len(fresh),
            "mean_moves": sum(s.moves for s in fresh) / len(fresh),
        }
    base["curriculum_games"] = len(curr)
    base["curriculum_mean_progress"] = (
        sum(s.progress for s in curr) / len(curr) if curr else 0.0
    )
    base["guided_games"] = len(guided)
    base["guided_mean_progress"] = (
        sum(s.progress for s in guided) / len(guided) if guided else 0.0
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Balatro Gumbel expert iteration")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--games", type=int, default=96, help="self-play games per iteration")
    parser.add_argument("--sims", type=int, default=16)
    parser.add_argument("--k", type=int, default=8, help="Gumbel root candidates")
    parser.add_argument("--depth", type=int, default=2, help="greedy rollout depth below root")
    parser.add_argument("--epochs", type=int, default=2, help="training epochs per iteration")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--buffer", type=int, default=60_000)
    parser.add_argument("--workers", type=int, default=0, help="0/1 = inline, else mp pool size")
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=2,
                        help="run the greedy eval every N iterations (eval at n=16 "
                             "is noisy anyway; serial per-iteration eval silently "
                             "cost ~70%% of v11's wall time)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="runs/default")
    parser.add_argument("--curriculum-frac", type=float, default=0.35,
                        help="fraction of self-play games started from deep-state snapshots")
    parser.add_argument("--seed-pool", type=str, default=None,
                        help="static snapshot pool (e.g. harvested god-run states); "
                             "never evicted, sampled for --seed-pool-frac of curriculum starts")
    parser.add_argument("--seed-pool-frac", type=float, default=0.5)
    parser.add_argument("--demos", type=str, default=None,
                        help="demonstration samples (obs, pi, z_win, z_togo) from the "
                             "scripted router; mixed into every training batch")
    parser.add_argument("--demo-frac", type=float, default=0.2,
                        help="fraction of each training batch drawn from --demos")
    parser.add_argument("--guided-frac", type=float, default=0.0,
                        help="fraction of self-play games with router-scripted economy "
                             "(expert econ actions become one-hot policy targets)")
    parser.add_argument("--snapshot-min-ante", type=int, default=3)
    parser.add_argument("--pool-cap", type=int, default=300, help="max snapshots kept")
    parser.add_argument(
        "--arch", choices=sorted(ARCHS), default="v6",
        help="network for FRESH runs (--resume sniffs the checkpoint instead): "
             "v6 = factored + pointer entity/card heads (content-bound), "
             "v5 = factored with positional-input heads, "
             "v3/v4 = positional Discrete(500)",
    )
    parser.add_argument(
        "--clairvoyant", action="store_true",
        help="rollouts replay the run's TRUE RNG (pre-2026-08-11 behavior) instead "
             "of determinized honest futures; only for comparisons against old runs",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="load <out>/latest.pt weights before training (optimizer state starts fresh)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    resume_path = out_dir / "latest.pt"
    start_it = 1
    if args.resume and resume_path.exists():
        # load_net sniffs the architecture, so a resumed run keeps its
        # net regardless of --arch (they may disagree; the checkpoint wins).
        net = load_net(str(resume_path)).to(device)
        net.train()
        print(f"resumed weights from {resume_path} "
              f"({type(net).__name__}, factored={is_factored(net)})")
        existing = sorted(out_dir.glob("ckpt_*.pt"))
        if existing:
            start_it = int(existing[-1].stem.split("_")[-1]) + 1
            print(f"continuing iteration numbering at {start_it}")
    else:
        net = ARCHS[args.arch]().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(capacity=args.buffer)
    rng = np.random.default_rng(1)
    cfg = SelfPlayConfig(
        sims=args.sims, k_max=args.k, depth=args.depth,
        curriculum_frac=args.curriculum_frac, snapshot_min_ante=args.snapshot_min_ante,
        seed_pool_frac=args.seed_pool_frac, guided_frac=args.guided_frac,
        determinize=not args.clairvoyant,
    )
    seed_pool_path = Path(args.seed_pool) if args.seed_pool else None
    if seed_pool_path is not None:
        import pickle as _pickle

        n_static = len(_pickle.loads(seed_pool_path.read_bytes()))
        print(f"seed pool: {n_static} static snapshots from {seed_pool_path}")
    demo_buffer: ReplayBuffer | None = None
    if args.demos:
        import pickle as _pickle

        demo_samples = _pickle.loads(Path(args.demos).read_bytes())
        if is_factored(net) and demo_samples and isinstance(
            demo_samples[0][1], np.ndarray
        ):
            raise SystemExit(
                f"{args.demos} holds POSITIONAL pi targets, which cannot "
                "supervise a factored net (a slot index binds to one state's "
                "enumeration order, and the demo's action list was not "
                "stored). Regenerate demos as CandidateSets, or drop --demos."
            )
        demo_buffer = ReplayBuffer(capacity=len(demo_samples))
        demo_buffer.add(demo_samples)
        print(f"demo buffer: {len(demo_buffer)} expert samples from {args.demos} "
              f"(demo_frac {args.demo_frac})")
    pool_path = out_dir / "curriculum.pkl"
    snap_pool: list[bytes] = []
    if pool_path.exists():
        import pickle as _pickle

        snap_pool = _pickle.loads(pool_path.read_bytes())
        print(f"curriculum pool: {len(snap_pool)} snapshots loaded")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"net: {type(net).__name__} {n_params/1e6:.2f}M params "
          f"(factored={is_factored(net)}) | device: {device} "
          f"| obs_dim: {net.torso[0].in_features}")

    # One spawn pool for the whole run: workers persist across iterations
    # (torch import paid once, not once per iteration per phase) and serve
    # both self-play and eval. Workers are daemonic, so an abnormal exit
    # still reaps them; the normal path terminates the pool after the loop.
    mp_pool = (
        get_context("spawn").Pool(args.workers, initializer=worker_init)
        if args.workers > 1
        else None
    )

    # Eval-overlap state (pooled mode): the record of an eval iteration is
    # held here, unwritten, until its async eval resolves one iteration
    # later — see the module docstring.
    pending_eval = None
    held_record = None

    def _append_record(rec: dict) -> None:
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _resolve_held_eval() -> None:
        """Block on the in-flight eval, fill its held record, append it."""
        nonlocal pending_eval, held_record
        if held_record is None:
            return
        t2 = time.perf_counter()
        try:
            stats = [st for st in pending_eval.get() if st is not None]
        except Exception as e:  # noqa: BLE001 — a dead eval must not kill training
            print(f"[eval] async eval for it {held_record['iter']} failed: {e}")
            stats = []
        waited = time.perf_counter() - t2
        ev = _summarize(stats)
        held_record["eval"] = ev
        held_record["t_eval_s"] = round(waited, 1)
        _append_record(held_record)
        print(
            f"    eval[it {held_record['iter']:3d}]: prog {ev['mean_progress']:.3f} "
            f"ante {ev['mean_ante']:.2f} win {ev['win_rate']:.1%} "
            f"(waited {waited:.0f}s)"
        )
        pending_eval = None
        held_record = None

    for it in range(start_it, start_it + args.iters):
        t0 = time.perf_counter()

        # Workers always read the latest checkpoint (CPU inference in workers).
        ckpt_path = out_dir / "latest.pt"
        torch.save(net.state_dict(), ckpt_path)

        # Self-play and eval run on CPU (single-position evals don't earn
        # their GPU round-trips); training runs on args.device.
        net.to("cpu")
        samples, sp_stats, new_snaps = run_selfplay(
            ckpt_path,
            net,
            torch.device("cpu"),
            n_games=args.games,
            mp_pool=mp_pool,
            seed_prefix=f"SP{it}",
            cfg=cfg,
            rng=rng,
            pool_path=pool_path if snap_pool or pool_path.exists() else None,
            seed_pool_path=seed_pool_path,
        )
        if new_snaps:
            import pickle as _pickle

            snap_pool.extend(new_snaps)
            snap_pool = snap_pool[-args.pool_cap:]
            pool_path.write_bytes(_pickle.dumps(snap_pool, protocol=5))
        net.to(device)
        t_sp = time.perf_counter() - t0

        buffer.add(samples)
        t1 = time.perf_counter()
        losses = train_epochs(
            net, opt, buffer,
            epochs=args.epochs, batch_size=args.batch, device=device, rng=rng,
            demo_buffer=demo_buffer, demo_frac=args.demo_frac,
        )
        t_train = time.perf_counter() - t1

        # Checkpoint BEFORE eval: eval workers load it from disk, and they
        # must see POST-training weights (latest.pt holds pre-selfplay ones).
        it_ckpt = out_dir / f"ckpt_{it:04d}.pt"
        torch.save(net.state_dict(), it_ckpt)

        # The previous eval iteration's games have had this whole iteration
        # (self-play + training) to finish on the pool; collect that row
        # before this iteration's eval is dispatched.
        _resolve_held_eval()

        eval_due = it % args.eval_every == 0 or it == start_it + args.iters - 1
        sp = _summarize(sp_stats)
        record = {
            "iter": it,
            "buffer": len(buffer),
            "selfplay": sp,
            "eval": None,
            "losses": losses,
            "t_selfplay_s": round(t_sp, 1),
            "t_train_s": round(t_train, 1),
            "t_eval_s": 0.0,
        }
        if eval_due and mp_pool is not None:
            pending_eval = dispatch_eval(
                it_ckpt, n_games=args.eval_games, mp_pool=mp_pool, cfg=cfg
            )
            held_record = record  # appended by _resolve_held_eval
            ev_str = "eval: dispatched"
        elif eval_due:
            t2 = time.perf_counter()
            ev = _summarize(evaluate_pooled(
                it_ckpt, n_games=args.eval_games, mp_pool=None, cfg=cfg
            ))
            t_eval = time.perf_counter() - t2
            record["eval"] = ev
            record["t_eval_s"] = round(t_eval, 1)
            _append_record(record)
            ev_str = (
                f"eval: prog {ev['mean_progress']:.3f} ante {ev['mean_ante']:.2f} "
                f"win {ev['win_rate']:.1%} ({t_eval:.0f}s)"
            )
        else:
            _append_record(record)
            ev_str = "eval: —"
        print(
            f"[it {it:3d}] selfplay: prog {sp['mean_progress']:.3f} ante {sp['mean_ante']:.2f} "
            f"win {sp['win_rate']:.1%} ({args.games}g, {t_sp:.0f}s) "
            f"| {ev_str} "
            f"| loss p {losses['policy']:.3f} w {losses['win']:.3f} pr {losses['progress']:.3f} "
            f"| buf {len(buffer)} | pool {len(snap_pool)} (+{len(new_snaps)}, {sp['curriculum_games']:.0f} curr-g) "
            f"| guided {sp['guided_games']:.0f}g prog {sp['guided_mean_progress']:.2f} "
            f"| train {t_train:.0f}s"
        )

    # The final iteration's eval has no successor self-play to hide behind;
    # collect it synchronously before tearing the pool down.
    _resolve_held_eval()

    if mp_pool is not None:
        mp_pool.terminate()
        mp_pool.join()


if __name__ == "__main__":
    main()
