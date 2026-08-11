"""Gumbel sequential-halving root search (simplified Gumbel AlphaZero).

Per decision:
  1. Enumerate legal actions; get root policy logits + value from the net.
  2. Gumbel-top-k picks k root candidates (this IS the exploration — no
     separate Dirichlet noise needed).
  3. Sequential halving splits the simulation budget across candidates;
     each simulation clones the state, applies the candidate, follows
     policy-greedy steps up to `depth`, then evaluates the leaf.
  4. Action choice: argmax over survivors of logits + gumbel + sigma(q).
  5. Policy target: softmax(logits + sigma(completed-Q)) over all legal
     actions, unvisited actions backed off to the root value ("completed
     Q-values" from the Gumbel MuZero paper) — an improved policy even at
     tiny simulation budgets.

Simplifications vs the paper: no tree reuse below the root, no
non-root Gumbel.

Rollout futures are DETERMINIZED by default: each simulation's clone
gets a fresh PRNG seed and a reshuffled undrawn deck (state.determinize)
before the candidate is applied, so Q-values are Monte Carlo estimates
over honest futures rather than readings of this run's true one.
determinize=False restores clairvoyant rollouts — the gold probe's
design point, and how every checkpoint before 2026-08-11 was trained
and evaluated (their recorded numbers are clairvoyant numbers).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Any

import numpy as np
import torch

from balatro_zero.goldprobe import plan_blind
from balatro_zero.net import (
    PolicyValueNet,
    action_logit,
    evaluate,
    evaluate_factored,
    global_entity_slot,
    is_factored,
    market_area_lens,
)
from jackdaw.engine.actions import GamePhase

from balatro_zero.state import (
    MAX_ACTIONS,
    Obs,
    blinds_beaten,
    clone,
    determinize as _determinize,
    is_terminal,
    legal_factored,
    observe,
    progress,
    step_factored,
    won,
)


def terminal_value(gs: dict[str, Any]) -> float:
    """Value of a finished game, consistent with the net's two heads."""
    w = 1.0 if won(gs) else 0.0
    return 0.5 * w + 0.5 * progress(gs)


@dataclass
class SearchResult:
    action_idx: int          # index into `actions`
    actions: list[Any]       # enumerated FactoredActions at the root
    pi_target: np.ndarray    # improved policy over MAX_ACTIONS slots
    root_obs: Obs
    root_value: float


def _sigma(q: np.ndarray, max_visits: float, c_visit: float, c_scale: float) -> np.ndarray:
    return (c_visit + max_visits) * c_scale * q


def _norm_q(q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Min-max normalize q against the reference set (Gumbel MuZero style).

    Balatro's progress-based values differ across actions by only ~1e-3
    (chip deltas compressed by /24); without per-root normalization sigma
    is flat and search degenerates to the prior.
    """
    lo = float(reference.min())
    hi = float(reference.max())
    # Below ~1e-5 the spread is numerical noise, not signal — amplifying it
    # would train the policy on confidently-random targets.
    if hi - lo < 1e-5:
        return np.zeros_like(q)
    return (q - lo) / (hi - lo)


def _apply(sim, action) -> None:
    """Apply a primitive action or an entire macro plan."""
    if isinstance(action, MacroPlan):
        for a in action.seq:
            if is_terminal(sim):
                return
            try:
                step_factored(sim, a)
            except Exception:  # noqa: BLE001
                return
        return
    step_factored(sim, action)


def _batched_rollouts(
    gs: dict[str, Any],
    actions: list[Any],
    sim_specs: list[int],
    net: PolicyValueNet,
    device: torch.device,
    *,
    econ_root: bool,
    depth: int,
    blind_finisher: bool = False,
    determinize_futures: bool = False,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Run one simulation per entry of sim_specs, batching net evaluations.

    All rollouts advance in lockstep: each outer pass fast-forwards every
    live simulation through terminal checks and single-action states, then
    evaluates ALL pending policy choices in one batched forward instead of
    one call per simulation. Semantics match the old sequential simulate()
    exactly (same greedy choices, same depth accounting); only the forward
    batching differs.

    Value semantics: terminal -> terminal_value; leaf -> net value plus
    0.5 * realized progress (an untrained value net is numerically
    insensitive to per-play chip deltas — measured bit-flat across plays —
    so ground-truth progress keeps Q discriminating from iteration 0;
    training targets stay pure Monte Carlo).
    """
    root_beaten = blinds_beaten(gs)
    max_steps = (max(depth, 14) if econ_root else depth)

    states = []
    for a_idx in sim_specs:
        sim = clone(gs)
        # Determinize BEFORE applying the candidate: its consequences (the
        # cards a discard draws into, what a pack contains) must be sampled,
        # not read off the run's true future. Successive draws from the same
        # generator give each simulation an independent future.
        if determinize_futures:
            _determinize(sim, rng)
        _apply(sim, actions[a_idx])
        states.append(sim)

    n = len(states)
    values: list[float | None] = [None] * n
    depths = [0] * n
    live = set(range(n))  # still rolling out (not terminal, not at leaf)
    pending_acts: dict[int, list[Any]] = {}

    while True:
        need_policy: list[int] = []
        for i in list(live):
            s = states[i]
            while True:
                if is_terminal(s):
                    values[i] = terminal_value(s)
                    live.discard(i)
                    break
                if depths[i] >= max_steps or (
                    econ_root and blinds_beaten(s) != root_beaten
                ):
                    live.discard(i)  # leaf — valued after the loop
                    break
                if blind_finisher and s.get("phase") == GamePhase.SELECTING_HAND:
                    # Finish the blind with the BEAM rather than walking
                    # it one policy-greedy card at a time. A rollout
                    # exists to say what a candidate is worth, and an
                    # undertrained policy playing cards singly answers
                    # that badly: on the bench-five seeds the gold beam
                    # reaches mean ante 3.12 where the same net under a
                    # depth-1 policy rollout reaches 1.72, and the whole
                    # gap is tactical play. One plan counts as one depth
                    # step, which also collapses the ~8-decisions-per-
                    # blind horizon that makes credit assignment hard.
                    #
                    # blend=0: the beam runs on the engine alone. A
                    # net-valued beam would cost ~65 candidates x 12
                    # plies of forward passes per blind per simulation.
                    # V is applied once, at the leaf, where it is
                    # affordable.
                    plan = plan_blind(s, blend=0.0)
                    if plan:
                        for a in plan:
                            if is_terminal(s):
                                break
                            try:
                                step_factored(s, a)
                            except Exception:  # noqa: BLE001
                                break
                        depths[i] += 1
                        continue
                acts = legal_factored(s)
                if not acts:
                    live.discard(i)  # dead-end — valued as leaf
                    break
                if len(acts) == 1:
                    step_factored(s, acts[0])
                    depths[i] += 1
                    continue
                pending_acts[i] = acts
                need_policy.append(i)
                break
        if not need_policy:
            break
        obs_l = [observe(states[i]) for i in need_policy]
        st_l = [states[i] for i in need_policy]
        act_l = [pending_acts[i] for i in need_policy]
        lg_l, _ = _priors(net, obs_l, st_l, act_l, device)
        for row, i in enumerate(need_policy):
            acts = pending_acts.pop(i)
            step_factored(states[i], acts[int(np.argmax(lg_l[row]))])
            depths[i] += 1

    leaves = [i for i in range(n) if values[i] is None]
    if leaves:
        _, v = _priors(net, [observe(states[i]) for i in leaves],
                       [states[i] for i in leaves],
                       [[] for _ in leaves], device)
        for row, i in enumerate(leaves):
            values[i] = float(v[row]) + 0.5 * progress(states[i])
    return values  # type: ignore[return-value]


def _priors(net, obs_list, states, action_lists, device):
    """Prior logits per candidate, for factored or positional nets.

    A factored net scores an action from its CONTENT -- type, entity
    slot, and card-set membership -- so any action is scorable, including
    card combinations the enumerator never proposed and whole-blind
    plans, which have no position in a list of primitives. A positional
    net can only index the list it was given.
    """
    if not is_factored(net):
        lg, v = evaluate(net, obs_list, device)
        return [lg[i, : len(action_lists[i])].astype(np.float64)
                for i in range(len(action_lists))], v
    tl, el, cl, v = evaluate_factored(net, obs_list, device)
    global_ent = getattr(net, "GLOBAL_ENTITY", False)
    out = []
    for i, acts in enumerate(action_lists):
        n_hand = len(states[i].get("hand", []))
        lens = market_area_lens(states[i]) if global_ent else None
        row = []
        for a in acts:
            head = _head_action(a)
            if global_ent:
                s = global_entity_slot(head, lens)
                slot = s if s is not None else -1  # -1: no entity factor
            else:
                slot = None  # legacy within-area convention
            row.append(action_logit(tl[i], el[i], cl[i], head, n_hand,
                                    ent_slot=slot))
        out.append(np.asarray(row, dtype=np.float64))
    return out, v


def _head_action(a):
    """A candidate's scorable action: a plan is scored by its opening."""
    return a.seq[0] if isinstance(a, MacroPlan) else a


@dataclass
class MacroPlan:
    """A whole-blind line offered to search as a single action.

    Expanding plans instead of individual card plays is what puts the
    beam's tactical strength into what gets PLAYED rather than only into
    how candidates are valued -- the rollout-only version measured 1.64
    against 1.72, i.e. nothing. The net still chooses among lines by
    value, so hand-level consequences keep flowing into V; the beam
    proposes, the net disposes.
    """
    seq: list


def gumbel_search(
    gs: dict[str, Any],
    net: PolicyValueNet,
    device: torch.device,
    *,
    n_sims: int = 16,
    k_max: int = 8,
    depth: int = 2,
    rng: np.random.Generator,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    root_noise: bool = True,
    blind_finisher: bool = False,
    macro_k: int = 0,
    determinize: bool = True,
) -> SearchResult | None:
    actions: list[Any] = legal_factored(gs)
    if macro_k > 1 and gs.get("phase") == GamePhase.SELECTING_HAND:
        plans = [p for p in plan_blind(gs, k=macro_k) if p]
        # Drop the primitive whose move a plan already opens with:
        # duplicates score identically, so leaving both in halves the
        # effective candidate count and lets search take the bare move
        # when the whole line was the point.
        heads = {(p[0].action_type, p[0].entity_target, p[0].card_target)
                 for p in plans}
        actions = [a for a in actions
                   if (a.action_type, a.entity_target, a.card_target) not in heads]
        actions = actions + [MacroPlan(p) for p in plans]
    n = len(actions)
    if n == 0:
        # Reachable dead-end (e.g. hand and deck exhausted mid-blind with
        # hands remaining) — caller treats it as game over.
        return None
    root_obs = observe(gs)
    lg_list, v_root_arr = _priors(net, [root_obs], [gs], [actions], device)
    logits = lg_list[0]
    # The progress head predicts progress-TO-GO, and child q values include
    # +0.5*realized progress — add the root's realized progress so v_root is
    # on the same absolute scale for the completed-Q backoff/normalization.
    v_root = float(v_root_arr[0]) + 0.5 * progress(gs)

    pi_target = np.zeros(MAX_ACTIONS, dtype=np.float32)
    if n == 1:
        pi_target[0] = 1.0
        return SearchResult(0, actions, pi_target, root_obs, v_root)

    g = rng.gumbel(size=n) if root_noise else np.zeros(n)
    k = int(min(k_max, n, n_sims))
    candidates = list(np.argsort(logits + g)[::-1][:k])

    q_sum = np.zeros(n)
    q_cnt = np.zeros(n)

    # Economy decisions (shop / blind select / packs) get blind-horizon
    # rollouts: play greedily until the round counter advances, so a
    # purchase's payoff during the NEXT blind lands in realized progress —
    # fixed shallow depth left buy-vs-skip Q identical, capping every run
    # at the no-economy ceiling (~0.10). Hand states keep cheap fixed depth:
    # their immediate-progress signal is already sharp, and whole-blind
    # rollouts there only add variance.
    econ_root = str(gs.get("phase")) in ("shop", "blind_select", "pack_opening")

    rounds = max(1, ceil(log2(k)))
    sims_left = n_sims
    for r in range(rounds):
        m = len(candidates)
        per = max(1, sims_left // (m * (rounds - r)))
        specs = [a for a in candidates for _ in range(per)]
        vals = _batched_rollouts(
            gs, actions, specs, net, device, econ_root=econ_root, depth=depth,
            blind_finisher=blind_finisher,
            determinize_futures=determinize, rng=rng,
        )
        for a, v in zip(specs, vals):
            q_sum[a] += v
            q_cnt[a] += 1
        sims_left -= per * m
        if m > 2 and r < rounds - 1:
            visited = np.array(candidates)
            q_hat = q_sum[visited] / np.maximum(q_cnt[visited], 1)
            q_n = _norm_q(q_hat, q_hat)
            score = logits[visited] + g[visited] + _sigma(q_n, q_cnt.max(), c_visit, c_scale)
            keep = ceil(m / 2)
            candidates = [int(visited[i]) for i in np.argsort(score)[::-1][:keep]]
        if sims_left <= 0:
            break

    visited = np.array(candidates)
    q_hat = q_sum[visited] / np.maximum(q_cnt[visited], 1)
    q_n = _norm_q(q_hat, q_hat)
    score = logits[visited] + g[visited] + _sigma(q_n, q_cnt.max(), c_visit, c_scale)
    action_idx = int(visited[int(np.argmax(score))])

    # Completed Q-values -> improved policy target (no gumbel in the target).
    # Unvisited actions back off to min(v_root, mean visited Q): with a fresh
    # net v_root is optimistic noise, and backing off to v_root alone let the
    # target dump its mass on whatever search did NOT visit (observed as
    # near-uniform pi targets, entropy ~3.3, in early v3 training).
    v_backoff = min(v_root, float(q_hat.mean()))
    q_completed = np.where(q_cnt > 0, q_sum / np.maximum(q_cnt, 1), v_backoff)
    q_completed = _norm_q(q_completed, q_completed)
    target_logits = logits + _sigma(q_completed, q_cnt.max(), c_visit, c_scale)
    target_logits -= target_logits.max()
    pi = np.exp(target_logits)
    pi /= pi.sum()
    pi_target[:n] = pi.astype(np.float32)

    return SearchResult(action_idx, actions, pi_target, root_obs, v_root)
