"""Local-LLM Balatro strategist: a chat model makes every strategy decision
(econ stops AND per-hand watch stops); the beam handles residual chip
extraction. Works with any OpenAI-compatible endpoint (Ollama, LM Studio,
vLLM).

Usage (from balatro-zero/):
    uv run --no-sync python scripts/llm_run.py SEED [--model qwen2.5:7b]
        [--url http://127.0.0.1:11434/v1/chat/completions] [--temp 0.3]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

sys.path.insert(0, "scripts")
from interactive_run import CTL0, advance, apply_act, summary  # noqa: E402

from balatro_zero.state import ante, is_terminal, new_run, progress, won  # noqa: E402

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
  copy <card> onto <card>    Death: copies first card onto second (free swaps)
  order jokers <k1, k2, ..>  reorder board (trigger order; also at shops)
  veto <hand types>          forbid the beam these hands (e.g. veto Flush)
  require <hand types>       restrict the beam to these hands
  clear constraints
Constraints persist across blinds until cleared — use them for jokers like \
Obelisk (veto your most-played hand) or Card Sharp (require one type).
Duplicate consumables/pack cards take a #N suffix (use c_strength#2 on Td).
The PASS option shows a whole-blind projection: [proj: CLEARS ...] means the \
beam alone wins the blind; [proj: DIES at X/Y] means steer NOW (pins, tarots, \
constraints) or the blind is lost — do the pace math before passing.

Key setup facts: board capacity 5 jokers (+1 per Negative), 2 consumable \
slots, interest +$1 per $5 held (max $5). Suit-converter and card-targeting \
tarots CAN now be used at hand stops via `use ... on <cards>`. Planets from \
packs apply instantly. Skip tags are shown on blind-select options.
Answer with ONLY the command (or number), optionally followed by a dash and \
max 8 words of reasoning."""


def ask_llm(url: str, model: str, temp: float, state_txt: str, options: list[str],
            recent: list[str], kind: str) -> str:
    ctx = ("Recent decisions: " + "; ".join(recent[-6:]) + "\n\n") if recent else ""
    user = (f"{ctx}{state_txt}\n\n[{kind.upper()} stop] options:\n"
            + "\n".join(f"  [{i}] {o}" for i, o in enumerate(options))
            + "\n\nYour command?")
    body = json.dumps({
        "model": model, "temperature": temp,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--url", default="http://127.0.0.1:11434/v1/chat/completions")
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--max-decisions", type=int, default=400)
    args = ap.parse_args()

    gs = new_run(args.seed)
    ctl = dict(CTL0)
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
                                descs, recent, kind)
            except Exception as e:  # noqa: BLE001
                print(f"LLM error: {e}", flush=True)
                reply = None
                break
            act = reply.split("-")[0].strip() if "-" in reply[:40] else reply
            act = act.splitlines()[0].strip().strip("`\"'")
            applied = apply_act(gs, ctl, kind, opts, act, None)
            print(f"[{n:3d}] LLM({attempt}) {kind}: {act!r} -> "
                  f"{applied if applied is not None else 'REJECTED'}  | {reply[:70]}",
                  flush=True)
            if applied is not None:
                break
        if applied is None:
            fallback = "pass" if kind == "hand" else "0"
            applied = apply_act(gs, ctl, kind, opts, fallback, None)
            print(f"[{n:3d}] fallback {fallback} -> {applied}", flush=True)
            if applied is None:
                print("fallback failed; stopping", flush=True)
                break
        if applied:
            recent.append(applied)
        n += 1

    print("\n" + summary(gs, ctl))
    tag = "WON" if won(gs) else ("GAME OVER" if is_terminal(gs) else "STOPPED")
    print(f"*** {tag}: ante {ante(gs)} prog {progress(gs):.3f} after {n} decisions ***")


if __name__ == "__main__":
    main()
