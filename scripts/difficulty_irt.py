"""Graded-response (IRT) fits of per-seed difficulty curves from a panel.

Model: one latent skill->outcome curve per seed, from which BOTH requested
outputs derive (win rate and expected ante).  For seed s at probe skill theta:

    P(max_ante >= k | theta, s) = sigmoid( a_s * (theta - b_s - d_k) )

with k = 2..9 (9 = beat ante 8 = win), shared ante-spacing cutpoints d_k
(d_2 < ... < d_9 = 0 across all seeds), and per-seed parameters:

    b_s  difficulty  — the skill theta at which the seed is a 50% win
                       (anchor d_9 = 0 makes this exact)
    a_s  discrimination — how sharply outcomes improve with skill

Win-rate data alone cannot identify per-seed curves on the current ladder
(all wins sit at the single gold theta); the graded model borrows strength
from the full ante ladder to pin curve shape, with gold wins anchoring the
top.  Derived curves:

    win curve   W_s(theta)     = sigmoid(a_s (theta - b_s))
    ante curve  E[ante|theta]  = 1 + sum_{k=2..9} P(max_ante >= k)

Records with max_ante >= 9 count as wins (beat ante 8) regardless of the
`won` flag (a handful of endless-mode ante-9 losses still cleared ante 8).

Fitting: pooled MLE for cutpoints + pooled (a, b) init, then alternating
per-seed (a_s, b_s) MLE / global cutpoint refit.  Mild ridge pulls a_s
toward the pooled value so degenerate seeds (e.g. every probe dies in
ante 1) stay finite; seeds with zero observed wins get `extrapolated: true`
— their b_s rests on the shared-spacing assumption, not on observed wins.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/difficulty_irt.py \
        [runs/difficulty/panel.jsonl] [runs/difficulty/irt.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import spearmanr

K_MAX = 9          # top category: reached ante 9 == beat ante 8 == win
N_CUT = K_MAX - 1  # cutpoints k = 2..9
RIDGE_LOG_A = 1.0  # (log a_s - log a_pool)^2 weight
RIDGE_B = 0.05     # (b_s - b_pool)^2 weight
N_ROUNDS = 3


def cuts_from_gaps(g: np.ndarray) -> np.ndarray:
    """g (N_CUT-1 gap params) -> d_2..d_9 with d_9 = 0, strictly increasing."""
    d = np.zeros(N_CUT)
    d[:-1] = -np.cumsum(np.exp(g)[::-1])[::-1]
    return d


def nll(theta, y, cnt, log_a, b, d, s_idx=None):
    """Negative log-likelihood of aggregated cells (theta, y, count).

    log_a/b are scalars (pooled) or per-seed arrays indexed by s_idx.
    """
    a = np.exp(log_a if s_idx is None else log_a[s_idx])
    bb = b if s_idx is None else b[s_idx]
    z = a[..., None] * (theta[:, None] - bb[..., None] - d[None, :]) \
        if np.ndim(a) else a * (theta[:, None] - bb - d[None, :])
    # P(y >= k) for k=2..9; pad with P(>=1)=1 and P(>=10)=0
    ge = np.concatenate([np.ones((len(theta), 1)), expit(z),
                         np.zeros((len(theta), 1))], axis=1)
    p = ge[np.arange(len(theta)), y - 1] - ge[np.arange(len(theta)), y]
    return -float(np.sum(cnt * np.log(np.clip(p, 1e-12, None))))


def fit_seed(theta, y, cnt, d, log_a0, b0):
    def obj(p):
        return (nll(theta, y, cnt, p[0], p[1], d)
                + RIDGE_LOG_A * (p[0] - log_a0) ** 2
                + RIDGE_B * (p[1] - b0) ** 2)
    r = minimize(obj, np.array([log_a0, b0]), method="Nelder-Mead",
                 options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 400})
    return r.x


def main() -> None:
    panel = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/difficulty/panel.jsonl")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/difficulty/irt.json")

    by_probe = defaultdict(list)
    cells = defaultdict(int)  # (seed, theta_probe, y) -> count
    for line in panel.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "error" in r:
            continue
        by_probe[r["probe"]].append(r["progress"])
        y = min(r["max_ante"], K_MAX)
        cells[(r["seed"], r["probe"], y)] += 1
    theta_probe = {p: float(np.mean(v)) for p, v in by_probe.items()}

    seeds = sorted({s for s, _, _ in cells})
    sid = {s: i for i, s in enumerate(seeds)}
    S = np.array([sid[s] for s, _, _ in cells])
    T = np.array([theta_probe[p] for _, p, _ in cells])
    Y = np.array([y for _, _, y in cells])
    C = np.array(list(cells.values()), dtype=float)
    n_rec = int(C.sum())
    print(f"panel: {n_rec} records, {len(seeds)} seeds, "
          f"{len(theta_probe)} probes, {len(C)} cells")

    # ---- pooled init: (log_a, b, gaps) over all data ----
    def pooled_obj(p):
        return nll(T, Y, C, p[0], p[1], cuts_from_gaps(p[2:]))
    p0 = np.concatenate([[np.log(5.0), 0.6], np.full(N_CUT - 1, np.log(0.15))])
    rp = minimize(pooled_obj, p0, method="Nelder-Mead",
                  options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-6})
    log_a_pool, b_pool, gaps = rp.x[0], rp.x[1], rp.x[2:]
    d = cuts_from_gaps(gaps)
    print(f"pooled: a={np.exp(log_a_pool):.2f} b={b_pool:.3f} "
          f"nll/rec={rp.fun / n_rec:.4f}")
    print("cutpoints d_k (k=2..9, skill offsets vs win):",
          np.round(d, 3).tolist())

    # ---- alternate per-seed fits and cutpoint refits ----
    log_a = np.full(len(seeds), log_a_pool)
    b = np.full(len(seeds), b_pool)
    for rnd in range(N_ROUNDS):
        for s in seeds:
            i = sid[s]
            m = S == i
            log_a[i], b[i] = fit_seed(T[m], Y[m], C[m], d, log_a_pool, b_pool)
        total = nll(T, Y, C, log_a, b, d, S)
        print(f"round {rnd + 1}: after seed fits  nll/rec={total / n_rec:.4f}")
        if rnd < N_ROUNDS - 1:
            rg = minimize(lambda g: nll(T, Y, C, log_a, b, cuts_from_gaps(g), S),
                          gaps, method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-5})
            gaps = rg.x
            d = cuts_from_gaps(gaps)
            print(f"           after cutpoint refit nll/rec={rg.fun / n_rec:.4f}")

    a = np.exp(log_a)

    # ---- calibration: predicted vs observed wins at gold theta ----
    th_gold = theta_probe["gold"]
    obs_wins = defaultdict(int)
    n_gold = defaultdict(int)
    for (s, p, y), c in cells.items():
        if p == "gold":
            n_gold[s] += int(c)
            if y >= K_MAX:
                obs_wins[s] += int(c)
    pred_gold = expit(a * (th_gold - b))
    exp_wins = float(sum(pred_gold[sid[s]] * n_gold[s] for s in seeds))
    tot_obs = sum(obs_wins.values())
    print(f"\ncalibration at gold theta={th_gold:.3f}: "
          f"predicted {exp_wins:.1f} wins vs observed {tot_obs}")

    win_any = {s: obs_wins.get(s, 0) > 0 for s in seeds}
    per_seed_obs = np.array([obs_wins.get(s, 0) / max(n_gold.get(s, 1), 1)
                             for s in seeds])
    r_wins = spearmanr(pred_gold, per_seed_obs)
    print(f"per-seed spearman(pred win@gold, obs win@gold) = "
          f"{r_wins.statistic:.3f}")

    # Win-anchored difficulty: the graded b is dominated by the ante ladder
    # and misses seeds whose winnability outruns their survivability (e.g.
    # 8F8H89NR: 9/16 gold wins, mid-pack ante profile).  Anchor a second
    # scalar on each seed's own gold wins, borrowing only the slope a_s:
    #   b_win = theta_gold - logit(p_hat)/a_s,  p_hat Jeffreys (w+.5)/(n+1).
    # For 0-win seeds this is a LOWER BOUND (p_hat from 0.5/(n+1)).
    b_win = np.empty(len(seeds))
    for s in seeds:
        i = sid[s]
        n = max(n_gold.get(s, 0), 1)
        p_hat = (obs_wins.get(s, 0) + 0.5) / (n + 1)
        b_win[i] = th_gold - np.log(p_hat / (1 - p_hat)) / a[i]

    # skill needed for 50% win vs the ladder
    print(f"\nb (skill for 50% win): median {np.median(b):.3f}  "
          f"IQR [{np.percentile(b, 25):.3f}, {np.percentile(b, 75):.3f}]  "
          f"range [{b.min():.3f}, {b.max():.3f}]")
    print(f"gold theta = {th_gold:.3f}; seeds with b <= gold: "
          f"{int(np.sum(b <= th_gold))}")
    print(f"a (discrimination): median {np.median(a):.2f}  "
          f"IQR [{np.percentile(a, 25):.2f}, {np.percentile(a, 75):.2f}]")

    # agreement with the nonparametric scalar
    curves_path = panel.parent / "curves.json"
    if curves_path.exists():
        cj = json.load(open(curves_path, encoding="utf-8"))["curves"]
        auc = np.array([cj[s]["auc"] for s in seeds])
        r_auc = spearmanr(b, auc)
        print(f"spearman(b, auc) = {r_auc.statistic:.3f} "
              f"(expect strongly negative)")

    wi = [sid[s] for s in seeds if win_any[s]]
    print("\nmost winnable by b_win (win-anchored, uncensored seeds only):")
    for i in sorted(wi, key=lambda i: b_win[i])[:8]:
        s = seeds[i]
        print(f"  {s}  b_win={b_win[i]:.3f}  b={b[i]:.3f}  "
              f"gold wins {obs_wins.get(s, 0)}/{n_gold.get(s, 0)}")
    r_bw = spearmanr(b[wi], b_win[wi])
    print(f"spearman(b, b_win) among winnable = {r_bw.statistic:.3f}")

    order = np.argsort(b)
    print("\neasiest 10 (lowest b = least skill to win):")
    for i in order[:10]:
        s = seeds[i]
        print(f"  {s}  b={b[i]:.3f} a={a[i]:.2f}  "
              f"gold wins {obs_wins.get(s, 0)}/{n_gold.get(s, 0)}")
    print("hardest 5 (highest b):")
    for i in order[-5:][::-1]:
        s = seeds[i]
        print(f"  {s}  b={b[i]:.3f} a={a[i]:.2f}")

    result = {
        "model": "P(ante>=k|theta,seed) = sigmoid(a_s*(theta - b_s - d_k)); "
                 "d_9=0 so b_s = theta at 50% win; win = reached ante 9",
        "theta_probe": theta_probe,
        "cutpoints": {str(k): round(float(v), 4)
                      for k, v in zip(range(2, K_MAX + 1), d)},
        "pooled": {"a": round(float(np.exp(log_a_pool)), 3),
                   "b": round(float(b_pool), 4)},
        "ridge": {"log_a": RIDGE_LOG_A, "b": RIDGE_B},
        "seeds": {
            s: {
                "a": round(float(a[sid[s]]), 3),
                "b": round(float(b[sid[s]]), 4),
                "b_win": round(float(b_win[sid[s]]), 4),
                "b_win_censored": not win_any[s],
                "pred_win_at_gold": round(float(pred_gold[sid[s]]), 5),
                "obs_win_at_gold": f"{obs_wins.get(s, 0)}/{n_gold.get(s, 0)}",
                "extrapolated": not win_any[s],
            } for s in seeds
        },
    }
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
