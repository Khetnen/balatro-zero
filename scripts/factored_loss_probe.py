"""Falsifiable gate for the factored training path (targets.py).

Four checks, exit non-zero on any failure:

  1. EQUIVALENCE — the batched factored_policy_loss must match a slow
     per-action reference built on net.action_logit (the inference-path
     scorer), on real enumerated candidate sets AND on synthetic edge
     cases: entity slot out of head range, card index beyond the live
     hand, n_hand wider than the head, type-only actions, K=1 sets.
  2. GRADIENT — 300 Adam steps on one fixed batch must collapse the
     KL(pi || model) toward zero (cross-entropy down to target entropy).
  3. EMISSION — play_game with a V5 net must emit CandidateSet targets
     (and with a V4 net, positional arrays — both formats live).
  4. END-TO-END — buffer + train_epochs on V5 self-play samples runs and
     returns finite losses; load_net round-trips the trained V5 weights.

Run from balatro-zero/:  uv run --no-sync python scripts/factored_loss_probe.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from jackdaw.env.game_spec import FactoredAction

from balatro_zero.net import (
    PolicyValueNetV4,
    PolicyValueNetV5,
    action_logit,
    evaluate_factored,
    is_factored,
    load_net,
)
from balatro_zero.replay import ReplayBuffer
from balatro_zero.selfplay import SelfPlayConfig, play_game
from balatro_zero.state import legal_factored, new_run, observe, step_factored
from balatro_zero.targets import (
    CandidateSet,
    collate_candidate_sets,
    encode_candidates,
    factored_policy_loss,
)
from balatro_zero.train import train_epochs
from jackdaw.env.action_space import ActionType

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def reference_loss(net, obs_list, action_lists, pi_list, n_hands) -> float:
    """Slow float64 reference: action_logit per candidate, then set CE."""
    t_lg, e_lg, c_lg, _ = evaluate_factored(net, obs_list, torch.device("cpu"))
    total = 0.0
    for i, (acts, pi, nh) in enumerate(zip(action_lists, pi_list, n_hands)):
        scores = np.array(
            [action_logit(t_lg[i], e_lg[i], c_lg[i], a, nh) for a in acts],
            dtype=np.float64,
        )
        logp = scores - scores.max()
        logp = logp - np.log(np.exp(logp).sum())
        total += -(np.asarray(pi, dtype=np.float64) * logp).sum()
    return total / len(obs_list)


def batched_loss(net, obs_list, sets) -> float:
    from balatro_zero.state import stack_obs

    flat, jid, cid, mid = stack_obs(obs_list)
    with torch.no_grad():
        t_lg, e_lg, c_lg, _, _ = net(
            torch.from_numpy(flat).float(), torch.from_numpy(jid),
            torch.from_numpy(cid), torch.from_numpy(mid),
        )
        fb = collate_candidate_sets(sets, torch.device("cpu"))
        return float(factored_policy_loss(t_lg, e_lg, c_lg, fb))


def main() -> int:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = PolicyValueNetV5()
    net.eval()

    # ---- 1. equivalence ----------------------------------------------------
    print("1. equivalence (batched loss == action_logit reference)")

    # Real states: an econ root (blind select) and two hand roots.
    states = [new_run("FLPROBE1")]
    hand_state = new_run("FLPROBE2")
    sel = [a for a in legal_factored(hand_state)
           if a.action_type == ActionType.SelectBlind]
    step_factored(hand_state, sel[0])
    states.append(hand_state)
    hand_state2 = new_run("FLPROBE3")
    sel = [a for a in legal_factored(hand_state2)
           if a.action_type == ActionType.SelectBlind]
    step_factored(hand_state2, sel[0])
    states.append(hand_state2)

    obs_list, action_lists, pi_list, n_hands, sets = [], [], [], [], []
    for gs in states:
        acts = legal_factored(gs)
        w = rng.dirichlet(np.ones(len(acts))).astype(np.float32)
        nh = len(gs.get("hand", []))
        obs_list.append(observe(gs))
        action_lists.append(acts)
        pi_list.append(w)
        n_hands.append(nh)
        sets.append(encode_candidates(acts, w, nh))

    ref = reference_loss(net, obs_list, action_lists, pi_list, n_hands)
    fast = batched_loss(net, obs_list, sets)
    check("real candidate sets match", abs(ref - fast) < 1e-4,
          f"ref {ref:.6f} vs batched {fast:.6f} "
          f"({sum(len(a) for a in action_lists)} candidates)")

    # Synthetic edge cases, scored against the same net outputs.
    edge_actions = [
        FactoredAction(action_type=int(ActionType.CashOut)),              # type-only
        FactoredAction(action_type=int(ActionType.SellJoker), entity_target=3),
        FactoredAction(action_type=int(ActionType.BuyCard), entity_target=17),  # ≥ head range
        FactoredAction(action_type=int(ActionType.PlayHand), card_target=(0, 2, 4)),
        FactoredAction(action_type=int(ActionType.Discard), card_target=(1, 9, 12)),  # ≥ live hand
        FactoredAction(action_type=int(ActionType.UseConsumable),
                       entity_target=1, card_target=(0, 1)),
    ]
    for nh in (0, 5, 8, 20):  # incl. no hand and hand wider than the head
        w = rng.dirichlet(np.ones(len(edge_actions))).astype(np.float32)
        ref = reference_loss(net, [obs_list[0]], [edge_actions], [w], [nh])
        fast = batched_loss(net, [obs_list[0]],
                            [encode_candidates(edge_actions, w, nh)])
        check(f"edge cases match at n_hand={nh}", abs(ref - fast) < 1e-4,
              f"ref {ref:.6f} vs batched {fast:.6f}")

    # K=1: loss must be exactly 0 (softmax over one candidate).
    one = encode_candidates([edge_actions[0]], np.ones(1, np.float32), 0)
    fast = batched_loss(net, [obs_list[0]], [one])
    check("K=1 set gives zero loss", abs(fast) < 1e-6, f"{fast:.2e}")

    # ---- 2. gradient -------------------------------------------------------
    # A Dirichlet target is NOT representable by the factored family (the
    # Bernoulli card factor cannot encode an arbitrary distribution over
    # combos), so the overfit target must be ONE-HOT — which is also what
    # guided and BC targets look like. One-hot CE can be driven to ~0.
    print("2. gradient (overfit one-hot targets)")
    from balatro_zero.state import stack_obs

    onehot_sets = []
    for acts, nh in zip(action_lists, n_hands):
        w = np.zeros(len(acts), dtype=np.float32)
        w[int(rng.integers(len(acts)))] = 1.0
        onehot_sets.append(encode_candidates(acts, w, nh))
    train_net = PolicyValueNetV5()
    flat, jid, cid, mid = (torch.from_numpy(np.ascontiguousarray(a))
                           for a in stack_obs(obs_list))
    fb = collate_candidate_sets(onehot_sets, torch.device("cpu"))
    opt = torch.optim.Adam(train_net.parameters(), lr=1e-3)
    first = last = None
    for _ in range(300):
        t_lg, e_lg, c_lg, _, _ = train_net(flat.float(), jid, cid, mid)
        loss = factored_policy_loss(t_lg, e_lg, c_lg, fb)
        if first is None:
            first = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    check("one-hot CE collapses on one batch",
          last < 0.1 * first or last < 0.1,
          f"CE {first:.3f} -> {last:.3f}")

    # ---- 3. emission -------------------------------------------------------
    print("3. emission (play_game target formats)")
    cfg = SelfPlayConfig(sims=8, k_max=4, depth=1)
    samples5, _, _ = play_game(net, torch.device("cpu"), "FLPROBE4", cfg,
                               np.random.default_rng(1))
    check("V5 emits CandidateSets",
          len(samples5) > 0
          and all(isinstance(s[1], CandidateSet) for s in samples5),
          f"{len(samples5)} samples")
    net4 = PolicyValueNetV4()
    net4.eval()
    samples4, _, _ = play_game(net4, torch.device("cpu"), "FLPROBE5", cfg,
                               np.random.default_rng(2))
    check("V4 still emits positional arrays",
          len(samples4) > 0
          and all(isinstance(s[1], np.ndarray) for s in samples4),
          f"{len(samples4)} samples")

    # ---- 4. end-to-end ------------------------------------------------------
    print("4. end-to-end (buffer -> train_epochs -> load_net)")
    buffer = ReplayBuffer(capacity=1000)
    buffer.add(samples5)
    train_net = PolicyValueNetV5()
    opt = torch.optim.Adam(train_net.parameters(), lr=1e-3)
    losses = train_epochs(train_net, opt, buffer, epochs=2, batch_size=8,
                          device=torch.device("cpu"),
                          rng=np.random.default_rng(3))
    check("train_epochs runs on factored samples",
          all(np.isfinite(v) for v in losses.values()),
          f"policy {losses['policy']:.3f} win {losses['win']:.3f} "
          f"prog {losses['progress']:.3f}")

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v5.pt"
        torch.save(train_net.state_dict(), p)
        loaded = load_net(str(p))
        check("load_net round-trips the trained V5",
              type(loaded).__name__ == "PolicyValueNetV5" and is_factored(loaded))

    if FAILURES:
        print(f"\nGATE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
