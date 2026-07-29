"""Replay BalatroBench runs on our sim: differential check + BC pairs.

ONE pipeline, two products, which is why this is not two scripts:

  * differential -- their runs are real-game trajectories recorded through
    balatrobot, so replaying the action sequence and comparing state is a
    free, headless, parallel differential channel with no live game, no
    Steam and no session degradation.
  * behaviour cloning -- replaying THEIR actions on OUR sim regenerates
    OUR observation at every step, so each step yields (our_obs, their
    action) in our native format. We never need their state for this.

The two are inseparable: if the sim desyncs from the real game mid-run,
every pair after that point is (wrong_state, their_action) -- corrupted
training data. So the divergence point IS the usable prefix length, and
the sync rate IS the real dataset size.

Alignment: gamestates.jsonl[i] is the state AFTER successful action i
(the first record of a winning run is ROUND_EVAL because the first
action was a blind-clearing play). Responses with no valid tool call are
skipped and do not consume a state -- they are the benchmark's own
"invalid/missing tool call" metric.

Their bot never emits SelectBlind or CashOut (its harness does those
automatically), so the replay supplies them.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/bb_replay.py --limit 5
    uv run --no-sync python scripts/bb_replay.py --model gemini-3-pro-preview
    uv run --no-sync python scripts/bb_replay.py --all --out runs/bb_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

ROOT = Path("runs/balatrobench/runs/runs/v1.0.8/default")

# Money is a SOFT divergence: it does not stop the replay.
#
# balatrobot is known to under-collect at cash-out -- our own lockstep
# work catalogued the eval-stall signature as "live under-collects by
# exactly one blind reward ($3/$4/$5)" plus a cash_out race that pays
# current_round.dollars before the bottom eval row commits. Every money
# divergence seen here has our side HIGHER, which is that artifact's
# direction, and content (jokers, consumables, hand levels) stays in
# agreement across it.
#
# Continuing is safe precisely BECAUSE we are richer: we replay their
# choices rather than making our own, so their purchases remain
# affordable for us. Content divergence still hard-stops -- that would
# mean the trajectories genuinely parted.
SOFT_FIELDS = ("money",)

# How far ahead to look when re-aligning to their state stream.
RESYNC_WINDOW = 4

def compare_state(gs, t, key_of, ante) -> dict:
    """Structured diff of our state against their recorded one.

    Ordered joker/consumable KEYS matter more than money: money is
    usually the downstream symptom of having acquired something
    different several steps earlier, so a money-only differ reports the
    wrong step and sends you hunting a pricing bug that isn't there.
    Joker order is compared too -- it is a game mechanic (Blueprint
    copies rightward), not presentation.
    """
    diff = {}
    for k, ours in (("ante_num", ante(gs)),
                    ("round_num", gs.get("round", 0)),
                    ("money", gs.get("dollars", 0))):
        if t.get(k) is not None and ours != t[k]:
            diff[k] = (ours, t[k])

    for field, mine in (("jokers", gs.get("jokers", [])),
                        ("consumables", gs.get("consumables", []))):
        theirs = [c.get("key") for c in (t.get(field) or {}).get("cards", [])]
        ours = [key_of(c) for c in mine]
        if ours != theirs:
            diff[field] = (ours, theirs)

    # Compare (chips, mult) rather than the level integer: it is what
    # actually scores, and it catches a planet applied to the wrong hand
    # even when both sides agree on how many levels were gained.
    hl = gs.get("hand_levels")
    bad = {}
    if hl is not None and hasattr(hl, "get"):
        for name, h in (t.get("hands") or {}).items():
            if h.get("chips") is None or h.get("mult") is None:
                continue
            try:
                ours = tuple(hl.get(name))
            except Exception:  # noqa: BLE001 -- hand type we do not model
                continue
            if ours != (h["chips"], h["mult"]):
                bad[name] = (ours, (h["chips"], h["mult"]))
    if bad:
        diff["hand_levels"] = bad
    return diff


def extract_actions(run: Path) -> list[tuple[str, dict]]:
    """Ordered (verb, args) for every response carrying a valid tool call."""
    out = []
    with open(run / "responses.jsonl", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                tc = rec["response"]["body"]["choices"][0]["message"].get("tool_calls")
                fn = (tc or [])[0]["function"]
                out.append((fn["name"], json.loads(fn["arguments"])))
            except Exception:  # noqa: BLE001 -- their invalid/missing-call metric
                continue
    return out


def _auto_advance(gs, step_factored, legal_factored, ActionType, GamePhase, log):
    """Take the transitions their harness makes without a tool call."""
    for _ in range(8):
        phase = gs.get("phase")
        want = None
        if phase == GamePhase.BLIND_SELECT:
            want = ActionType.SelectBlind
        elif phase == GamePhase.ROUND_EVAL:
            want = ActionType.CashOut
        if want is None:
            return
        picks = [a for a in legal_factored(gs) if a.action_type == int(want)]
        if not picks:
            return
        step_factored(gs, picks[0])
        log.append(want.name)


def map_action(verb: str, args: dict, gs, ActionType, FactoredAction):
    """Their tool call -> our FactoredAction list (rearrange needs several)."""
    F = FactoredAction
    if verb == "play":
        return [F(action_type=int(ActionType.PlayHand),
                  card_target=tuple(sorted(args.get("cards", []))))]
    if verb == "discard":
        return [F(action_type=int(ActionType.Discard),
                  card_target=tuple(sorted(args.get("cards", []))))]
    if verb == "next_round":
        return [F(action_type=int(ActionType.NextRound))]
    if verb == "reroll":
        return [F(action_type=int(ActionType.Reroll))]
    if verb == "buy":
        if "pack" in args:
            return [F(action_type=int(ActionType.OpenBooster),
                      entity_target=int(args["pack"]))]
        if "voucher" in args:
            return [F(action_type=int(ActionType.RedeemVoucher),
                      entity_target=int(args["voucher"]))]
        if "card" in args:
            return [F(action_type=int(ActionType.BuyCard),
                      entity_target=int(args["card"]))]
        return []
    if verb == "pack":
        if args.get("skip"):
            return [F(action_type=int(ActionType.SkipPack))]
        if "card" in args:
            tgt = tuple(sorted(args.get("targets", []))) or None
            return [F(action_type=int(ActionType.PickPackCard),
                      entity_target=int(args["card"]), card_target=tgt)]
        return []
    if verb == "sell":
        if "joker" in args:
            return [F(action_type=int(ActionType.SellJoker),
                      entity_target=int(args["joker"]))]
        if "consumable" in args:
            return [F(action_type=int(ActionType.SellConsumable),
                      entity_target=int(args["consumable"]))]
        return []
    if verb == "use":
        tgt = tuple(sorted(args.get("cards", []))) or None
        return [F(action_type=int(ActionType.UseConsumable),
                  entity_target=int(args["consumable"]), card_target=tgt)]
    if verb == "rearrange":
        order = args.get("jokers")
        if not order:
            return []          # hand/consumable reorders are cosmetic here
        acts = []
        cur = list(range(len(order)))
        for pos, want in enumerate(order):
            if want not in cur:
                return []
            at = cur.index(want)
            while at > pos:
                acts.append(F(action_type=int(ActionType.SwapJokersLeft),
                              entity_target=at))
                cur[at - 1], cur[at] = cur[at], cur[at - 1]
                at -= 1
        return acts
    return []


def replay(run: Path, collect_pairs: bool) -> dict:
    from jackdaw.engine.actions import GamePhase
    from jackdaw.env.action_space import ActionType
    from jackdaw.env.game_spec import FactoredAction

    from balatro_zero.router import flags_override, key_of
    from balatro_zero.state import (
        ante, is_terminal, legal_factored, new_run, observe, step_factored, won,
    )

    task = json.loads((run / "task.json").read_text(encoding="utf-8"))
    seed = task["seed"]
    theirs = [json.loads(l) for l in open(run / "gamestates.jsonl", encoding="utf-8")]
    actions = extract_actions(run)

    result = {
        "run": run.name, "model": task["model"]["name"], "seed": seed,
        "deck": task["deck"], "stake": task["stake"],
        "n_actions": len(actions), "n_their_states": len(theirs),
        "synced_steps": 0, "diverged_at": None, "divergence": None,
        "money_drift_at": None, "money_drift": None, "resyncs": 0,
        "illegal_at": None, "illegal": None, "our_final_ante": None,
        "our_won": None, "pairs": 0,
    }
    pairs: list[dict] = []

    with flags_override(peek=False, skip_tags=False):
        gs = new_run(seed)
        auto: list[str] = []
        tptr = 0
        for i, (verb, args) in enumerate(actions):
            if is_terminal(gs) or won(gs):
                break
            _auto_advance(gs, step_factored, legal_factored, ActionType,
                          GamePhase, auto)
            if is_terminal(gs) or won(gs):
                break
            acts = map_action(verb, args, gs, ActionType, FactoredAction)
            if not acts:
                continue
            if collect_pairs:
                obs = observe(gs)
                pairs.append({
                    "seed": seed, "model": task["model"]["name"], "step": i,
                    "verb": verb, "args": {k: v for k, v in args.items()
                                           if k != "reasoning"},
                    "obs": {
                        "flat": obs.flat.tolist(),
                        "joker_ids": obs.joker_ids.tolist(),
                        "consumable_ids": obs.consumable_ids.tolist(),
                        "market_ids": obs.market_ids.tolist(),
                    },
                })
            try:
                for a in acts:
                    step_factored(gs, a)
            except Exception as e:  # noqa: BLE001
                result["illegal_at"] = i
                result["illegal"] = f"{verb} {args!r}: {type(e).__name__}: {e}"
                if collect_pairs and pairs:
                    pairs.pop()      # the pair we just staged is unusable
                break

            # Self-healing alignment. gamestates[i] is USUALLY the state
            # after action i, but ~1% of their tool calls are valid yet
            # not executable in the live state (their own "failed" metric),
            # which burns a response without producing a state record and
            # shifts every later index by one. A fixed index then reports
            # a divergence whose "difference" is simply their NEXT action
            # already applied -- one such case decoded exactly to the
            # rearrange [3,0,2,1,4] issued on the following step.
            #
            # So scan a small window and take the first state that
            # matches on content, advancing the pointer past whatever was
            # skipped. Only if nothing in the window matches has the
            # replay genuinely parted from the recording.
            best = None
            for k in range(tptr, min(tptr + RESYNC_WINDOW, len(theirs))):
                diff = compare_state(gs, theirs[k], key_of, ante)
                hard = {f: v for f, v in diff.items() if f not in SOFT_FIELDS}
                if not hard:
                    best = (k, diff)
                    break
                if best is None:
                    best = (k, diff)
            if best is not None:
                k, diff = best
                soft = {f: v for f, v in diff.items() if f in SOFT_FIELDS}
                hard = {f: v for f, v in diff.items() if f not in SOFT_FIELDS}
                if soft and result["money_drift_at"] is None:
                    result["money_drift_at"] = i
                    result["money_drift"] = {"after": f"{verb} {args!r}"[:90],
                                             "fields": soft}
                if hard:
                    result["diverged_at"] = i
                    result["divergence"] = {"after": f"{verb} {args!r}"[:90],
                                            "fields": hard}
                    break
                if k > tptr:
                    result["resyncs"] += 1
                tptr = k + 1
            result["synced_steps"] = i + 1

        result["our_final_ante"] = ante(gs)
        result["our_won"] = bool(won(gs))
    result["pairs"] = len(pairs)
    return result, pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=None, help="write BC pairs jsonl here")
    args = ap.parse_args()

    if not ROOT.exists():
        raise SystemExit(f"dataset not found at {ROOT}")
    runs = sorted(ROOT.glob(f"*/{args.model or '*'}/*/"))
    runs = [r for r in runs if (r / "responses.jsonl").exists()]
    if not args.all:
        runs = runs[: args.limit]
    print(f"replaying {len(runs)} runs\n")

    rows, all_pairs = [], []
    for r in runs:
        res, pairs = replay(r, collect_pairs=bool(args.out))
        rows.append(res)
        all_pairs.extend(pairs)
        status = ("CLEAN" if res["diverged_at"] is None and res["illegal_at"] is None
                  else f"DIVERGED@{res['diverged_at']}" if res["diverged_at"] is not None
                  else f"ILLEGAL@{res['illegal_at']}")
        print(f"  {res['model'][:22]:<23} {res['seed']} "
              f"{res['synced_steps']:>4}/{res['n_actions']:<4} {status}")
        if res["illegal"]:
            print(f"      {res['illegal'][:150]}")
        if res["divergence"]:
            print(f"      after {res['divergence']['after'][:70]} "
                  f"-> {res['divergence']['fields']}")

    n = len(rows)
    clean = sum(1 for r in rows if r["diverged_at"] is None and r["illegal_at"] is None)
    tot_act = sum(r["n_actions"] for r in rows)
    tot_sync = sum(r["synced_steps"] for r in rows)
    print(f"\n{clean}/{n} fully replayed | "
          f"{tot_sync}/{tot_act} actions synced ({100 * tot_sync / max(tot_act, 1):.1f}%)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for p in all_pairs:
                fh.write(json.dumps(p) + "\n")
        print(f"wrote {len(all_pairs)} BC pairs -> {args.out}")
    Path("runs").mkdir(exist_ok=True)
    with open("runs/bb_replay_report.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print("report -> runs/bb_replay_report.json")


if __name__ == "__main__":
    main()
