"""Behaviour-cloning pretrain of the policy and value heads on LLM games.

Initialisation, not a training method. The point is to give Gumbel search
a prior worth expanding: with a random policy the root candidates are
noise, so the search spends its budget discovering that most shop
actions are bad. A frontier LLM already knows that, and BalatroBench
recorded 241 of its games on the real engine.

WHICH GAMES FOR WHICH HEAD:
  * policy  -- strong models only. BC caps you at the demonstrator, so
    cloning a model that dies at ante 2 teaches dying at ante 2.
  * value   -- everything, wins AND losses. A value head trained only on
    wins has no idea what losing looks like and cannot rank states.

Value targets come from the RUN OUTCOME, not from a bootstrap: win is
0/1 and progress is blinds-beaten/24, matching state.progress(). That is
Monte-Carlo regression on real trajectories, which is exactly what the
heads are asked to predict at inference.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/bc_pretrain.py --epochs 8
    uv run --no-sync python scripts/bc_pretrain.py --out runs/bc/net.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

# The five models that ever won on balatrobench.com. Cloning below this
# line teaches the failure modes of models that reach ante 2.
STRONG = {
    "gemini-3-pro-preview", "gpt-5.2", "gemini-3-flash-preview",
    "claude-opus-4.5", "claude-sonnet-4.5",
}


def load(path: Path, drift_ok: bool):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not drift_ok and r.get("after_money_drift"):
                continue
            rows.append(r)
    return rows


def to_tensors(rows, device):
    o = [r["obs"] for r in rows]
    return (
        torch.tensor(np.array([x["flat"] for x in o], dtype=np.float32), device=device),
        torch.tensor(np.array([x["joker_ids"] for x in o], dtype=np.int64), device=device),
        torch.tensor(np.array([x["consumable_ids"] for x in o], dtype=np.int64), device=device),
        torch.tensor(np.array([x["market_ids"] for x in o], dtype=np.int64), device=device),
        torch.tensor(np.array([r["action_idx"] for r in rows], dtype=np.int64), device=device),
        torch.tensor(np.array([1.0 if r["run_won"] else 0.0 for r in rows],
                              dtype=np.float32), device=device),
        torch.tensor(np.array([r["outcome_progress"] for r in rows],
                              dtype=np.float32), device=device),
        torch.tensor(np.array([r["model"] in STRONG for r in rows],
                              dtype=bool), device=device),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="runs/bb_pairs.jsonl")
    ap.add_argument("--out", default="runs/bc/net.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--include-drift", action="store_true",
                    help="keep pairs recorded after money drift began")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from balatro_zero.net import PolicyValueNetV4

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load(Path(args.pairs), args.include_drift)
    if not rows:
        raise SystemExit(f"no usable pairs in {args.pairs}")

    # Split by SEED, not by row: consecutive steps of one game are highly
    # correlated, so a random row split leaks the answer across it and
    # the validation number means nothing.
    rng = np.random.default_rng(args.seed)
    seeds = sorted({r["seed"] for r in rows})
    rng.shuffle(seeds)
    n_val = max(1, int(len(seeds) * args.val_frac))
    val_seeds = set(seeds[:n_val])
    tr = [r for r in rows if r["seed"] not in val_seeds]
    va = [r for r in rows if r["seed"] in val_seeds]

    n_pol = sum(1 for r in tr if r["action_idx"] >= 0 and r["model"] in STRONG)
    print(f"{len(rows)} pairs | train {len(tr)} / val {len(va)} "
          f"(held-out seeds: {sorted(val_seeds)})")
    print(f"policy-usable train pairs (strong + resolved index): {n_pol}")
    print(f"device: {dev}")

    net = PolicyValueNetV4().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    TR = to_tensors(tr, dev)
    VA = to_tensors(va, dev)

    def run_epoch(T, train: bool):
        flat, jid, cid, mid, aidx, wonv, prog, strong = T
        n = flat.shape[0]
        order = torch.randperm(n, device=dev) if train else torch.arange(n, device=dev)
        tot = {"pol": 0.0, "win": 0.0, "prog": 0.0, "acc": 0.0, "npol": 0, "n": 0}
        net.train(train)
        for s in range(0, n, args.batch):
            b = order[s: s + args.batch]
            with torch.set_grad_enabled(train):
                logits, pw, pg = net(flat[b], jid[b], cid[b], mid[b])
                # policy: strong demonstrators with a resolved index only
                m = strong[b] & (aidx[b] >= 0)
                loss_pol = torch.zeros((), device=dev)
                if m.any():
                    loss_pol = ce(logits[m], aidx[b][m])
                    tot["acc"] += (logits[m].argmax(-1) == aidx[b][m]).sum().item()
                    tot["npol"] += int(m.sum())
                loss = loss_pol + bce(pw, wonv[b]) + mse(pg, prog[b])
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
            k = len(b)
            tot["pol"] += float(loss_pol) * k
            tot["win"] += float(bce(pw, wonv[b])) * k
            tot["prog"] += float(mse(pg, prog[b])) * k
            tot["n"] += k
        return tot

    print(f"\n{'ep':>3} {'pol':>8} {'win':>8} {'prog':>8} {'acc':>7} | "
          f"{'val pol':>8} {'val prog':>8} {'val acc':>7}")
    # Keep the best-by-validation weights, not the last. With ~2.4k
    # policy-usable pairs drawn from four seeds this overfits within a
    # couple of epochs: train accuracy keeps climbing while held-out
    # policy loss turns around. The last checkpoint is the worst one.
    best = (float("inf"), None, 0)
    for ep in range(1, args.epochs + 1):
        t = run_epoch(TR, True)
        v = run_epoch(VA, False)
        vpol = v["pol"] / v["n"]
        if vpol < best[0]:
            best = (vpol, {k: x.detach().cpu().clone()
                           for k, x in net.state_dict().items()}, ep)
        print(f"{ep:>3} {t['pol']/t['n']:>8.4f} {t['win']/t['n']:>8.4f} "
              f"{t['prog']/t['n']:>8.4f} "
              f"{t['acc']/max(t['npol'],1):>7.3f} | "
              f"{v['pol']/v['n']:>8.4f} {v['prog']/v['n']:>8.4f} "
              f"{v['acc']/max(v['npol'],1):>7.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if best[1] is not None:
        net.load_state_dict(best[1])
        print(f"\nbest epoch {best[2]} (val policy loss {best[0]:.4f}); "
              "later epochs discarded as overfit")
    torch.save(net.state_dict(), out)
    print(f"saved {out}")
    print(f"\nheld-out seed: {sorted(val_seeds)}")
    print("!! bench5 runs AAAAAAA..EEEEEEE, which are the ONLY seeds this")
    print("!! data covers, so a bench5 score for this checkpoint is")
    print("!! contaminated on every seed except the held-out one. Read the")
    print(f"!! {sorted(val_seeds)[0]} column, not the total.")
    print(f"\n  uv run --no-sync python scripts/bench5.py --probe net --ckpt {out}")


if __name__ == "__main__":
    main()
