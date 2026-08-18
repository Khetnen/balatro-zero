"""Behaviour-cloning pretrain of the V7 policy and value heads on LLM games.

Initialisation, not a training method. The point is to give Gumbel search
a prior worth expanding: with a random policy the root candidates are
noise, so the search spends its budget discovering that most shop
actions are bad. A frontier LLM already knows that — and since 2026-08-15
we harvest our own games natively (llm_run.py --unaided --pairs), so the
obs in each pair is the CURRENT 1064-dim v14 layout captured in our
engine at the decision moment, with zero replay desync.

WHICH GAMES FOR WHICH HEAD (--policy-data):
  * policy  -- default `wins`: only pairs from runs that WON. The user's
    2026-08-15 directive: RL tails already graze late antes; the scarce
    skill is CONVERTING — so the policy prior clones only play that
    actually converted. `strong`/`all` remain as A/B arms; losses ride
    along in every harvest file regardless, so the A/B is free.
  * value   -- everything, wins AND losses. A value head trained only on
    wins has no idea what losing looks like and cannot rank states.

Value targets come from the RUN OUTCOME, not a bootstrap: win is 0/1 and
outcome_progress is blinds-beaten/24. NOTE the progress scale predates
the v13 frontier formula — as an init that mismatch is acceptable (both
are monotone [0,1]); RL fine-tuning re-fits the head to the live signal.

ENTITY TARGETS are the V7-specific part. V6/V7 score ONE GLOBAL slot
space [jokers 0-11 | consumables 12-15 | market 16-27] whose market
offsets depend on the live shop layout. Pairs recorded since the
entity_global field carry the exact slot; older pairs are backfilled
from the obs itself (market_ids classified by center-key prefix + the
v14 per-slot is_pack_card flags reconstruct the market area lens).
Unrecoverable targets drop the entity factor for that pair only.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/bc_pretrain.py \
        --pairs "runs/llm_pilot/pairs_*.jsonl" "runs/llm_measure/pairs_*.jsonl" \
        --epochs 8 --out runs/bc/v7_wins.pt
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Demonstrators whose play is worth cloning under --policy-data strong:
# the five that ever won on balatrobench.com plus our own shootout/pilot
# winners (2026-08-15: deepseek-v4-pro and both current geminis).
STRONG = {
    "gemini-3-pro-preview", "gpt-5.2", "gemini-3-flash-preview",
    "claude-opus-4.5", "claude-sonnet-4.5",
    "deepseek-v4-pro-0813", "gemini-3.1-pro-preview", "gemini-3.7-flash",
}


def _model_name(r: dict) -> str:
    return str(r.get("model", "")).rsplit("/", 1)[-1]


def load(paths: list[str], drift_ok: bool) -> list[dict]:
    rows = []
    files: list[str] = []
    for p in paths:
        hits = sorted(glob.glob(p))
        if not hits and Path(p).exists():
            hits = [p]
        files.extend(hits)
    if not files:
        raise SystemExit(f"no pairs files match {paths}")
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if not drift_ok and r.get("after_money_drift"):
                    continue
                rows.append(r)
    print(f"loaded {len(rows)} pairs from {len(files)} files")
    return rows


# --- entity_global backfill ------------------------------------------------


def _id_to_key() -> dict[int, str]:
    from jackdaw.env.observation import _CENTER_KEY_TO_ID

    return {v: k for k, v in _CENTER_KEY_TO_ID.items()}


def _market_lens_from_obs(r: dict, id2key: dict[int, str]) -> tuple[int, int, int] | None:
    """Reconstruct (n shop cards, n vouchers, n boosters) from the obs.

    market_ids is packed in observe()'s order (cards | vouchers |
    boosters | pack). Voucher keys are v_*, booster packs p_*; the v14
    per-slot market rows carry an is_pack_card flag that bounds the shop
    region. Pre-v14 obs (no market rows) return None.
    """
    from balatro_zero.state import (
        MARKET_FEAT_DIM,
        MARKET_FEAT_OFFSET,
        N_MARKET_SLOTS,
        OBS_DIM,
    )

    flat = r["obs"]["flat"]
    if len(flat) < OBS_DIM:
        return None
    ids = r["obs"]["market_ids"]
    n_items = 0
    for k in range(min(len(ids), N_MARKET_SLOTS)):
        if int(ids[k]) != 0:
            n_items = k + 1
    shop_n = 0
    for k in range(n_items):
        if flat[MARKET_FEAT_OFFSET + k * MARKET_FEAT_DIM] > 0.5:
            break
        shop_n += 1
    n_v = n_p = 0
    for k in range(shop_n):
        key = id2key.get(int(ids[k]), "")
        if key.startswith("v_"):
            n_v += 1
        elif key.startswith("p_"):
            n_p += 1
    return (shop_n - n_v - n_p, n_v, n_p)


def entity_global(r: dict, id2key: dict[int, str]) -> int:
    """The pair's global entity slot, stored or reconstructed; -1 = none."""
    if "entity_global" in r:
        return int(r["entity_global"])
    e = r.get("entity_target")
    if e is None:
        return -1
    e = int(e)
    from jackdaw.env.action_space import ActionType

    from balatro_zero.net import (
        ENT_OFF_CONS,
        ENT_OFF_JOKER,
        ENT_OFF_MARKET,
        GLOBAL_ENTITY_SLOTS,
    )
    from balatro_zero.state import N_CONSUMABLE_SLOTS, N_JOKER_SLOTS

    t = int(r["action_type"])
    if t == int(ActionType.SellJoker):
        g = ENT_OFF_JOKER + e if e < N_JOKER_SLOTS else -1
    elif t in (int(ActionType.SellConsumable), int(ActionType.UseConsumable)):
        g = ENT_OFF_CONS + e if e < N_CONSUMABLE_SLOTS else -1
    elif t == int(ActionType.BuyCard):
        g = ENT_OFF_MARKET + e
    elif t in (int(ActionType.RedeemVoucher), int(ActionType.OpenBooster),
               int(ActionType.PickPackCard)):
        lens = _market_lens_from_obs(r, id2key)
        if lens is None:
            return -1
        off = ENT_OFF_MARKET
        if t == int(ActionType.RedeemVoucher):
            off += lens[0]
        elif t == int(ActionType.OpenBooster):
            off += lens[0] + lens[1]
        else:
            off += lens[0] + lens[1] + lens[2]
        g = off + e
    else:
        return -1
    return g if 0 <= g < GLOBAL_ENTITY_SLOTS else -1


# --- tensors ---------------------------------------------------------------


def to_tensors(rows, policy_mask, device, obs_dim: int):
    """Factored V7 targets: action type, GLOBAL entity slot, card set."""
    from balatro_zero.net import N_HAND_SLOTS

    id2key = _id_to_key()
    n = len(rows)
    flat = np.zeros((n, obs_dim), dtype=np.float32)
    n_padded = 0
    for i, r in enumerate(rows):
        f = np.asarray(r["obs"]["flat"], dtype=np.float32)
        if f.shape[0] < obs_dim:
            flat[i, :f.shape[0]] = f     # append-only layout: zero-pad old obs
            n_padded += 1
        else:
            flat[i] = f[:obs_dim]
    if n_padded:
        print(f"note: {n_padded} pairs carry a pre-v14 obs, zero-padded "
              f"(appended block reads as absent)")
    card = np.zeros((n, N_HAND_SLOTS), dtype=np.float32)
    has = np.zeros(n, dtype=bool)
    eg = np.full(n, -1, dtype=np.int64)
    for i, r in enumerate(rows):
        for j in r.get("card_target") or []:
            if 0 <= int(j) < N_HAND_SLOTS:
                card[i, int(j)] = 1.0
                has[i] = True
        eg[i] = entity_global(r, id2key)
    o = [r["obs"] for r in rows]
    return (
        torch.tensor(flat, device=device),
        torch.tensor(np.array([x["joker_ids"] for x in o], dtype=np.int64), device=device),
        torch.tensor(np.array([x["consumable_ids"] for x in o], dtype=np.int64), device=device),
        torch.tensor(np.array([x["market_ids"] for x in o], dtype=np.int64), device=device),
        torch.tensor(np.array([r["action_type"] for r in rows], dtype=np.int64), device=device),
        torch.tensor(eg, device=device),
        torch.tensor(card, device=device),
        torch.tensor(has, device=device),
        torch.tensor(np.array([1.0 if r["run_won"] else 0.0 for r in rows],
                              dtype=np.float32), device=device),
        torch.tensor(np.array([r["outcome_progress"] for r in rows],
                              dtype=np.float32), device=device),
        torch.tensor(np.array([policy_mask(r) for r in rows], dtype=bool),
                     device=device),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+",
                    default=["runs/llm_pilot/pairs_*.jsonl",
                             "runs/llm_measure/pairs_*.jsonl"],
                    help="pairs jsonl paths/globs (all files merged)")
    ap.add_argument("--out", default="runs/bc/v7.pt")
    ap.add_argument("--policy-data", default="wins",
                    choices=("wins", "strong", "all"),
                    help="which pairs supervise the POLICY heads "
                         "(value always trains on everything)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--include-drift", action="store_true",
                    help="keep pairs recorded after money drift began")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from balatro_zero.net import GLOBAL_ENTITY_SLOTS, PolicyValueNetV7
    from balatro_zero.state import OBS_DIM

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load(args.pairs, args.include_drift)
    if not rows:
        raise SystemExit("no usable pairs")

    policy_mask = {
        "wins": lambda r: bool(r["run_won"]),
        "strong": lambda r: _model_name(r) in STRONG,
        "all": lambda r: True,
    }[args.policy_data]

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

    n_pol = sum(1 for r in tr if policy_mask(r))
    print(f"{len(rows)} pairs | train {len(tr)} / val {len(va)} "
          f"(held-out seeds: {sorted(val_seeds)})")
    print(f"policy-usable train pairs ({args.policy_data}): {n_pol}")
    print(f"device: {dev}")

    net = PolicyValueNetV7().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss()

    TR = to_tensors(tr, policy_mask, dev, OBS_DIM)
    VA = to_tensors(va, policy_mask, dev, OBS_DIM)
    n_ent = int((TR[5] >= 0).sum())
    print(f"entity-supervised train pairs: {n_ent} "
          f"(global slots 0..{GLOBAL_ENTITY_SLOTS - 1}; -1 pairs skip the factor)")

    def run_epoch(T, train: bool):
        (flat, jid, cid, mid, atype, aent, acard, ahas,
         wonv, prog, polm) = T
        n = flat.shape[0]
        order = torch.randperm(n, device=dev) if train else torch.arange(n, device=dev)
        tot = {"pol": 0.0, "win": 0.0, "prog": 0.0, "acc": 0.0, "npol": 0, "n": 0}
        net.train(train)
        for s in range(0, n, args.batch):
            b = order[s: s + args.batch]
            with torch.set_grad_enabled(train):
                tl, el, cl, pw, pg = net(flat[b], jid[b], cid[b], mid[b])
                m = polm[b]
                loss_pol = torch.zeros((), device=dev)
                if m.any():
                    # action TYPE -- always supervised
                    loss_pol = ce(tl[m], atype[b][m])
                    tot["acc"] += (tl[m].argmax(-1) == atype[b][m]).sum().item()
                    tot["npol"] += int(m.sum())
                    # GLOBAL ENTITY slot -- only where one was resolved
                    me = m & (aent[b] >= 0)
                    if me.any():
                        loss_pol = loss_pol + ce(el[me], aent[b][me])
                    # CARD SET -- Bernoulli membership over hand slots
                    mc = m & ahas[b]
                    if mc.any():
                        loss_pol = loss_pol + bce_logits(cl[mc], acard[b][mc])
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
    # Keep the best-by-validation weights, not the last: with few seeds
    # this overfits within a couple of epochs.
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
    print(f"\nheld-out seeds: {sorted(val_seeds)}")
    print("!! every seed in this corpus is BC-contaminated for eval:")
    print("!! evaluate BC checkpoints on FRESH panel seeds only, never on")
    print("!! the harvest seeds or bench5.")


if __name__ == "__main__":
    main()
