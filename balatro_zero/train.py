"""Expert-iteration training loop.

Each iteration:
  1. Self-play N games with the current net (Gumbel search) -> samples
  2. Add to replay buffer
  3. Train E epochs of minibatches over the buffer
  4. Evaluate greedily (no root noise) on held-out seeds
  5. Checkpoint (checkpoints double as the difficulty-reference ensemble)

Run from the balatro-zero directory:
    uv run bzero --iters 50 --games 96 --workers 12
Smoke test:
    uv run bzero --iters 1 --games 2 --sims 8 --depth 0 --workers 0
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from balatro_zero.net import PolicyValueNet
from balatro_zero.replay import ReplayBuffer
from balatro_zero.selfplay import (
    GameStats,
    SelfPlayConfig,
    play_game,
    sample_start_state,
    worker_run,
)


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
                flat, jid, cid, mid, pi, z_win, z_prog = (
                    np.concatenate([a, b])
                    for a, b in zip((flat, jid, cid, mid, pi, z_win, z_prog), parts)
                )
            flat_t = torch.from_numpy(flat).float().to(device)
            jid_t = torch.from_numpy(jid).to(device)
            cid_t = torch.from_numpy(cid).to(device)
            mid_t = torch.from_numpy(mid).to(device)
            pi_t = torch.from_numpy(pi).to(device)
            z_win_t = torch.from_numpy(z_win).to(device)
            z_prog_t = torch.from_numpy(z_prog).to(device)

            logits, p_win, prog = net(flat_t, jid_t, cid_t, mid_t)
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
    workers: int,
    seed_prefix: str,
    cfg: SelfPlayConfig,
    rng: np.random.Generator,
    pool_path: Path | None = None,
    seed_pool_path: Path | None = None,
) -> tuple[list, list[GameStats], list[bytes]]:
    if workers <= 1:
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

    per_worker = [n_games // workers + (1 if i < n_games % workers else 0) for i in range(workers)]
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = pool.starmap(
            worker_run,
            [
                (str(ckpt_path), pw, seed_prefix, wid, cfg,
                 str(pool_path) if pool_path is not None else None,
                 str(seed_pool_path) if seed_pool_path is not None else None)
                for wid, pw in enumerate(per_worker)
                if pw > 0
            ],
        )
    samples = []
    stats = []
    new_snaps = []
    for s, st, sn in results:
        samples.extend(s)
        stats.extend(st)
        new_snaps.extend(sn)
    return samples, stats, new_snaps


def evaluate_greedy(
    net: PolicyValueNet,
    device: torch.device,
    *,
    n_games: int,
    cfg: SelfPlayConfig,
) -> list[GameStats]:
    rng = np.random.default_rng(0)
    stats = []
    for i in range(n_games):
        _, st, _ = play_game(net, device, f"EVAL{i}", cfg, rng, root_noise=False)
        stats.append(st)
    return stats


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
        "--resume", action="store_true",
        help="load <out>/latest.pt weights before training (optimizer state starts fresh)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    net = PolicyValueNet().to(device)
    resume_path = out_dir / "latest.pt"
    start_it = 1
    if args.resume and resume_path.exists():
        net.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))
        print(f"resumed weights from {resume_path}")
        existing = sorted(out_dir.glob("ckpt_*.pt"))
        if existing:
            start_it = int(existing[-1].stem.split("_")[-1]) + 1
            print(f"continuing iteration numbering at {start_it}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(capacity=args.buffer)
    rng = np.random.default_rng(1)
    cfg = SelfPlayConfig(
        sims=args.sims, k_max=args.k, depth=args.depth,
        curriculum_frac=args.curriculum_frac, snapshot_min_ante=args.snapshot_min_ante,
        seed_pool_frac=args.seed_pool_frac, guided_frac=args.guided_frac,
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
    print(f"net: {n_params/1e6:.2f}M params | device: {device} | obs_dim: {net.torso[0].in_features}")

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
            workers=args.workers,
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

        net.to("cpu")
        eval_stats = evaluate_greedy(net, torch.device("cpu"), n_games=args.eval_games, cfg=cfg)
        net.to(device)

        sp = _summarize(sp_stats)
        ev = _summarize(eval_stats)
        record = {
            "iter": it,
            "buffer": len(buffer),
            "selfplay": sp,
            "eval": ev,
            "losses": losses,
            "t_selfplay_s": round(t_sp, 1),
            "t_train_s": round(t_train, 1),
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        torch.save(net.state_dict(), out_dir / f"ckpt_{it:04d}.pt")

        print(
            f"[it {it:3d}] selfplay: prog {sp['mean_progress']:.3f} ante {sp['mean_ante']:.2f} "
            f"win {sp['win_rate']:.1%} ({args.games}g, {t_sp:.0f}s) "
            f"| eval: prog {ev['mean_progress']:.3f} ante {ev['mean_ante']:.2f} win {ev['win_rate']:.1%} "
            f"| loss p {losses['policy']:.3f} w {losses['win']:.3f} pr {losses['progress']:.3f} "
            f"| buf {len(buffer)} | pool {len(snap_pool)} (+{len(new_snaps)}, {sp['curriculum_games']:.0f} curr-g) "
            f"| guided {sp['guided_games']:.0f}g prog {sp['guided_mean_progress']:.2f} "
            f"| train {t_train:.0f}s"
        )


if __name__ == "__main__":
    main()
