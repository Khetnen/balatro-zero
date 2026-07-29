"""LLM Balatro strategist. Works with any OpenAI-compatible endpoint.

TWO MODES, and the difference decides what the games are worth:

  default   the beam proposes a move at each hand stop and the model may
            `pass`/`auto`. Fine for playing, useless as cloning data --
            the demonstrations are of a BEAM-ASSISTED policy, and a
            trained agent will not have the beam.

  --unaided the beam is never consulted and never shown. The model names
            its own cards at every hand stop, so the trajectory
            demonstrates UNIFIED play: in-round and out-of-round
            decisions from one policy. That is the ability the winning
            BalatroBench models have and our agents do not.

With --pairs a run also writes behaviour-cloning pairs in the schema
bc_pretrain consumes. These are native: the observation is captured in
OUR engine at the moment of decision, so there is no replay and no
desync -- the BalatroBench pipeline lost 35% of its actions to that and
was confined to five seeds. Here any seed is fair game, and seed
diversity is the binding constraint on what BC can learn.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/llm_run.py SEED [--model qwen2.5:7b]
        [--url http://127.0.0.1:11434/v1/chat/completions] [--temp 0.3]
        [--unaided] [--pairs runs/llm_pairs.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

sys.path.insert(0, "scripts")
from interactive_run import (  # noqa: E402
    CTL0,
    advance,
    apply_act,
    describe,
    summary,
)

from balatro_zero.state import (  # noqa: E402
    ante,
    blinds_beaten,
    is_terminal,
    new_run,
    progress,
    won,
)

SYSTEM = """You are an expert Balatro player making the strategy decisions \
for a run (Red Deck, White stake). Goal: WIN (beat the ante-8 boss). A strong \
engine ("the beam") handles raw chip extraction; you steer it.

Two kinds of stops. At an ECON stop (shop/blind select/packs), answer with an \
option number or a unique option-text substring. At a HAND stop you see your \
drawn cards and the beam's intended action; answer with ONE command:
  pass                       accept the beam's shown action
  auto                       let the beam finish this blind
  play <cards>               e.g.  play Kh Qh 7s   (10s or Ts; Kh#2 for dups)
  discard <cards>
  use <consumable> [on <cards>]     e.g.  use c_sixth_sense on 6d
  sell <joker|consumable>    legal mid-blind
  copy <card> onto <card>    Death: copies first card onto second (free swaps)
  order jokers <k1, k2, ..>  reorder board (trigger order; also at shops)
  veto <hand types>          forbid the beam these hands (e.g. veto Flush)
  require <hand types>       restrict the beam to these hands
  clear constraints
Constraints persist across blinds until cleared -- use them for jokers like \
Obelisk (veto your most-played hand) or Card Sharp (require one type).
Duplicate consumables/pack cards take a #N suffix (use c_strength#2 on Td).
The PASS option shows a whole-blind projection: [proj: CLEARS ...] means the \
beam alone wins the blind; [proj: DIES at X/Y] means steer NOW (pins, tarots, \
constraints) or the blind is lost -- do the pace math before passing.

Key setup facts: board capacity 5 jokers (+1 per Negative), 2 consumable \
slots, interest +$1 per $5 held (max $5). Suit-converter and card-targeting \
tarots CAN be used at hand stops via `use ... on <cards>`. Planets from packs \
apply instantly. Skip tags are shown on blind-select options.
Answer with ONLY the command (or number), optionally followed by a dash and \
max 8 words of reasoning."""

UNAIDED_SYSTEM = """You are an expert Balatro player playing a full run \
(Red Deck, White stake). Goal: WIN (beat the ante-8 boss). You make EVERY \
decision yourself -- nothing chooses cards for you.

At an ECON stop (shop / blind select / packs), answer with an option number \
or a unique option-text substring.

At a HAND stop you are shown your drawn cards, the blind target and your \
score so far. There is no "pass" and no suggested move. Answer with ONE \
command, naming the cards yourself:
  play <cards>               e.g.  play Kh Qh Jh Th 9h  (10s or Ts; Kh#2 dups)
  discard <cards>            e.g.  discard 3c 4d
  use <consumable> [on <cards>]     e.g.  use c_death on 7c Kh
  sell <joker|consumable>    legal mid-blind
  copy <card> onto <card>    Death: copies first card onto second
  order jokers <k1, k2, ..>  trigger order matters -- Blueprint copies the
                             joker to its RIGHT

Play the pace, not just the hand: you have a limited number of hands, so \
weigh what a play scores against what you still need. Think about which cards \
your jokers actually reward -- retriggers, held-in-hand effects, suit and \
rank conditions, whether a joker wants a specific hand type or wants you to \
AVOID your most-played one, and whether a consumable is worth firing now on \
the cards you were just dealt.

Key facts: 5 joker slots (+1 per Negative), 2 consumable slots, interest +$1 \
per $5 held (max $5). Planets from packs apply instantly. Skip tags show on \
blind-select options.
Answer with ONLY the command (or number), optionally followed by a dash and \
max 8 words of reasoning."""


def ask_llm(url: str, model: str, temp: float, state_txt: str,
            options: list[str], recent: list[str], kind: str,
            system: str = SYSTEM) -> str:
    ctx = ("Recent decisions: " + "; ".join(recent[-6:]) + "\n\n") if recent else ""
    user = (f"{ctx}{state_txt}\n\n[{kind.upper()} stop] options:\n"
            + "\n".join(f"  [{i}] {o}" for i, o in enumerate(options))
            + "\n\nYour command?")
    body = json.dumps({
        "model": model, "temperature": temp,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"].strip()


def _rescue(gs, ctl, kind: str) -> str:
    """A legal move for when the model's reply cannot be parsed.

    Unaided mode has no `pass`, so fall back to the beam's best step --
    the run survives, and the caller tags that decision `fallback` so it
    is excluded from cloning data. A fallback is not a demonstration.
    """
    if kind != "hand":
        return "0"
    if not ctl.get("unaided"):
        return "pass"
    from balatro_zero.goldprobe import plan_blind

    seq = plan_blind(gs)
    if not seq:
        return "0"
    d = describe(gs, seq[0])
    if d.startswith("PLAY "):
        return "play " + d[len("PLAY "):].split(" (")[0]
    if d.startswith("DISCARD "):
        return "discard " + d[len("DISCARD "):]
    return "0"


def write_pairs(path: str, ctl, seed: str, model: str, gs) -> int:
    """Append behaviour-cloning pairs in bc_pretrain's schema."""
    won_run = bool(won(gs))
    prog = 1.0 if won_run else max(blinds_beaten(gs), 0) / 24.0
    kept = 0
    with open(path, "a", encoding="utf-8") as fh:
        for step, r in enumerate(ctl.get("log", [])):
            if "obs" not in r or "action_type" not in r:
                continue
            if r.get("source") in ("beam-pass", "beam-auto", "fallback"):
                continue                      # not the model's own decision
            fh.write(json.dumps({
                "seed": seed, "model": model, "step": step,
                "verb": (r["entry"].split() or ["?"])[0].lower(), "args": {},
                "action_type": r["action_type"],
                "entity_target": r["entity_target"],
                "card_target": r["card_target"],
                "action_idx": -1,             # factored head needs no index
                "n_legal": 0,
                "run_won": won_run,
                "outcome_progress": prog,
                "after_money_drift": False,
                "obs": r["obs"],
            }) + "\n")
            kept += 1
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--url", default="http://127.0.0.1:11434/v1/chat/completions")
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--unaided", action="store_true",
                    help="hide the beam; the model chooses every card itself")
    ap.add_argument("--pairs", default=None,
                    help="append (obs, action) cloning pairs to this jsonl")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    gs = new_run(args.seed)
    ctl = dict(CTL0)
    ctl["unaided"] = args.unaided
    ctl["capture_pairs"] = bool(args.pairs)
    system = UNAIDED_SYSTEM if args.unaided else SYSTEM

    recent: list[str] = []
    n = 0
    while n < args.max_decisions:
        kind, opts = advance(gs, ctl)
        if won(gs) or is_terminal(gs) or not opts:
            break
        state_txt = summary(gs, ctl)
        descs = [o["desc"] for o in opts]
        applied = None
        for attempt in range(2):
            try:
                reply = ask_llm(args.url, args.model, args.temp, state_txt,
                                descs, recent, kind, system)
            except Exception as e:  # noqa: BLE001
                print(f"LLM error: {e}", flush=True)
                reply = None
                break
            act = reply.split("-")[0].strip() if "-" in reply[:40] else reply
            act = act.splitlines()[0].strip().strip("`\"'")
            applied = apply_act(gs, ctl, kind, opts, act, None)
            if not args.quiet:
                print(f"[{n:3d}] LLM({attempt}) {kind}: {act!r} -> "
                      f"{applied if applied is not None else 'REJECTED'}"
                      f"  | {reply[:70]}", flush=True)
            if applied is not None:
                break
        if applied is None:
            fallback = _rescue(gs, ctl, kind)
            applied = apply_act(gs, ctl, kind, opts, fallback, None)
            if not args.quiet:
                print(f"[{n:3d}] fallback {fallback!r} -> {applied}", flush=True)
            if applied is None:
                print("fallback failed; stopping", flush=True)
                break
            if ctl.get("log"):
                ctl["log"][-1]["source"] = "fallback"
        if applied:
            recent.append(applied)
        n += 1

    if args.pairs:
        kept = write_pairs(args.pairs, ctl, args.seed, args.model, gs)
        print(f"wrote {kept} BC pairs -> {args.pairs}", flush=True)

    print("\n" + summary(gs, ctl))
    tag = "WON" if won(gs) else ("GAME OVER" if is_terminal(gs) else "STOPPED")
    print(f"*** {tag}: ante {ante(gs)} prog {progress(gs):.3f} "
          f"after {n} decisions ***")


if __name__ == "__main__":
    main()
