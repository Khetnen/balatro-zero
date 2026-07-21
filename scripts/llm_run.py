"""Local-LLM Balatro strategist: a chat model makes every econ decision;
the beam plays hands. Works with any OpenAI-compatible endpoint
(Ollama, LM Studio, vLLM).

Usage (from balatro-zero/):
    uv run --no-sync python scripts/llm_run.py SEED [--model qwen2.5:7b]
        [--url http://127.0.0.1:11434/v1/chat/completions] [--temp 0.3]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

sys.path.insert(0, "scripts")
from interactive_run import advance, describe, summary  # noqa: E402

from balatro_zero.state import ante, is_terminal, new_run, progress, won, step_factored  # noqa: E402
from jackdaw.env.action_space import ActionType  # noqa: E402

SYSTEM = """You are an expert Balatro player making the economy/strategy decisions \
for a run (Red Deck, White stake). Goal: WIN (beat the ante-8 boss). Hand play is \
automated by a strong engine; you only pick shop/blind/pack decisions.

Key rules of this setup:
- Board capacity 5 jokers, 2 consumable slots. Interest +$1 per $5 held (max $5).
- Suit-converter and card-targeting tarots (Star/Moon/Sun/World/Strength/Lovers...) \
are NEVER used by the engine - do not buy or pick them (Chariot/Justice/Death/\
Hanged Man are auto-used; planets from packs apply instantly).
- You win on scaling: get xmult jokers and hand levels by ante 7; flat +mult \
commons alone die around ante 4-5. Cheap bodies early beat an empty board.
- Skip tags: tag_buffoon (free mega joker pack) and tag_investment ($25 after \
boss) are good when your board is strong; other skips usually waste money.
- bl_tooth costs $1/card played (~$20/blind); bl_eye forbids repeating hand types.
Answer with ONLY the option number, optionally followed by a dash and max 8 words."""


def ask_llm(url: str, model: str, temp: float, state_txt: str, options: list[str],
            recent: list[str]) -> str:
    ctx = ("Recent decisions: " + "; ".join(recent[-6:]) + "\n\n") if recent else ""
    user = (f"{ctx}{state_txt}\n\noptions:\n"
            + "\n".join(f"  [{i}] {o}" for i, o in enumerate(options))
            + "\n\nWhich option? Answer with the number.")
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
    ap.add_argument("--max-decisions", type=int, default=250)
    args = ap.parse_args()

    gs = new_run(args.seed)
    recent: list[str] = []
    n = 0
    while n < args.max_decisions:
        opts = advance(gs)
        if won(gs) or is_terminal(gs) or not opts:
            break
        state_txt = summary(gs)
        descs = [describe(gs, a) for a in opts]
        choice = None
        for attempt in range(2):
            try:
                reply = ask_llm(args.url, args.model, args.temp, state_txt, descs, recent)
            except Exception as e:  # noqa: BLE001
                print(f"LLM error: {e}", flush=True)
                break
            m = re.search(r"\d+", reply)
            if m and 0 <= int(m.group()) < len(opts):
                choice = int(m.group())
                print(f"[{n:3d}] LLM({attempt}) -> [{choice}] {descs[choice]}  | {reply[:60]}",
                      flush=True)
                break
        if choice is None:
            for want in (ActionType.SelectBlind, ActionType.NextRound, ActionType.SkipPack):
                idx = next((i for i, a in enumerate(opts) if a.action_type == want), None)
                if idx is not None:
                    choice = idx
                    break
            choice = 0 if choice is None else choice
            print(f"[{n:3d}] fallback -> [{choice}] {descs[choice]}", flush=True)
        recent.append(descs[choice])
        try:
            step_factored(gs, opts[choice])
        except Exception as e:  # noqa: BLE001
            print(f"step failed: {e}", flush=True)
            break
        n += 1

    print("\n" + summary(gs))
    tag = "WON" if won(gs) else ("GAME OVER" if is_terminal(gs) else "STOPPED")
    print(f"*** {tag}: ante {ante(gs)} prog {progress(gs):.3f} after {n} decisions ***")


if __name__ == "__main__":
    main()
