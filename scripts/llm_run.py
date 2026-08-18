"""LLM Balatro strategist. Works with any OpenAI-compatible endpoint.

TWO MODES, and the difference decides what the games are worth:

  default   the beam proposes a move at each hand stop and the model may
            `pass`/`auto`. Fine for playing, useless as cloning data --
            the demonstrations are of a BEAM-ASSISTED policy, and a
            trained agent will not have the beam. Free-text protocol
            (legacy prompt).

  --unaided the beam is never consulted and never shown. The model names
            its own cards at every hand stop, so the trajectory
            demonstrates UNIFIED play: in-round and out-of-round
            decisions from one policy. That is the ability the winning
            BalatroBench models have and our agents do not.

            Unaided uses the HARVEST PROMPT (scripts/prompts/): a
            BalatroBench-style three-part message -- invariant guide
            (game manual + doctrine + full key reference, sent with
            cache_control so it is cached, the single biggest cost
            lever), rendered state, and a memory part carrying the
            model's own reasoning for recent decisions -- plus real
            FUNCTION CALLING (tool_choice: required). With free text
            impossible, the fallback path only remains for schema-level
            failures, and those stay excluded from the BC export.

With --pairs a run also writes behaviour-cloning pairs in the schema
bc_pretrain consumes. These are native: the observation is captured in
OUR engine at the moment of decision, so there is no replay and no
desync -- the BalatroBench pipeline lost 35% of its actions to that and
was confined to five seeds. Here any seed is fair game, and seed
diversity is the binding constraint on what BC can learn.

Defaults target OpenRouter, which is what BalatroBench itself ran on --
their stored responses carry OpenRouter generation IDs (`gen-...`), the
`vendor/model` id form, and a `provider` routing field. One key reaches
every vendor, and the endpoint is OpenAI-compatible, so this driver
needs no vendor-specific client. Set OPENROUTER_API_KEY.

PIN THE PROVIDER for a harvest. OpenRouter routes the same model id to
different upstreams; across BalatroBench's runs one model was served by
Amazon Bedrock, Google, SiliconFlow, AtlasCloud and DeepInfra in turn.
Upstreams differ in quantisation and sampling defaults, so some of the
variance in their leaderboard is provider variance, not model variance.
--provider pins it; the provider that served each decision is recorded
either way.

Usage (from balatro-zero/):
    export OPENROUTER_API_KEY=sk-or-...
    uv run --no-sync python scripts/llm_run.py SEED --unaided
        --model anthropic/claude-sonnet-5 --provider Anthropic
        --pairs runs/llm_pairs.jsonl

    # plumbing test, no network, no cost:
    uv run --no-sync python scripts/llm_run.py SEED --unaided --mock
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "scripts")
from interactive_run import (  # noqa: E402
    CTL0,
    advance,
    apply_act,
    describe,
    hand_labels,
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

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

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


# ---------------------------------------------------------------------------
# Harvest prompt (unaided): three-part message + function calling
# ---------------------------------------------------------------------------


def load_harvest_prompt() -> tuple[str, list]:
    guide = (PROMPTS_DIR / "harvest_guide.md").read_text(encoding="utf-8")
    tools = json.loads(
        (PROMPTS_DIR / "harvest_tools.json").read_text(encoding="utf-8"))
    return guide, tools


def build_state_part(gs, ctl, kind: str, opts: list[dict]) -> str:
    """Message part 1: the rendered stop. Numbered options only at ECON
    stops -- a HAND stop is free-form by design (enumerating legal
    combos would be thousands of lines and anchor the model)."""
    lines = ["# Current state", "", summary(gs, ctl), ""]
    if kind == "econ":
        lines.append("[ECON stop] Numbered options:")
        for i, o in enumerate(opts):
            if o["kind"] != "freeform":
                lines.append(f"  [{i}] {o['desc']}")
        lines.append("")
        lines.append("Act with `choose` (or pick_pack_card for a targeted "
                     "pack pick, use_consumable, sell, order_jokers).")
    else:
        lines.append("[HAND stop] Act with one tool: play / discard / "
                     "use_consumable / sell / copy_card / order_jokers.")
    return "\n".join(lines)


def build_memory_part(history: list[dict], gs) -> str:
    """Message part 2: the model's own recent decisions WITH the
    reasoning it gave for them. BalatroBench carried a memory part;
    carrying the reasoning is what makes it a working memory rather
    than a bare move log."""
    lines = ["# Memory"]
    if not history:
        lines += ["", "Run start -- no decisions yet. Establish a strategic "
                  "direction and build toward a specific poker hand type."]
        return "\n".join(lines)
    lines += ["", f"Decisions so far: {len(history)} | blinds beaten: "
              f"{blinds_beaten(gs)} | current ante: {ante(gs)}", "",
              "Recent decisions (oldest first; your own reasoning quoted):"]
    for h in history[-12:]:
        r = (h.get("reason") or "").strip().replace("\n", " ")
        if len(r) > 220:
            r = r[:217] + "..."
        lines.append(f"- [{h['n']}] ante {h['ante']} {h['kind']}: "
                     f"{h['entry']}" + (f" -- {r}" if r else ""))
    return "\n".join(lines)


def tool_to_act(name: str, args: dict) -> str | None:
    """Map a validated tool call onto the harness act-string grammar
    (see interactive_run.apply_act). Returns None on malformed args."""
    try:
        if name == "play":
            return "play " + " ".join(args["cards"])
        if name == "discard":
            return "discard " + " ".join(args["cards"])
        if name == "use_consumable":
            s = "use " + args["consumable"]
            if args.get("cards"):
                s += " on " + " ".join(args["cards"])
            return s
        if name == "sell":
            return "sell " + args["item"]
        if name == "copy_card":
            return f"copy {args['source']} onto {args['target']}"
        if name == "order_jokers":
            return "order jokers " + ", ".join(args["order"])
        if name == "choose":
            return str(int(args["option"]))
        if name == "pick_pack_card":
            if args.get("targets"):
                return f"pick {args['card']} on " + " ".join(args["targets"])
            return args["card"]  # untargeted: resolves via option substring
    except (KeyError, TypeError, ValueError):
        return None
    return None


def ask_llm_tools(url: str, model: str, temp: float, guide: str, tools: list,
                  state_txt: str, memory_txt: str,
                  api_key: str | None, provider: str | None,
                  error_note: str | None = None,
                  reasoning_effort: str | None = None,
                  tool_choice: str = "required") -> tuple[str, dict, dict]:
    """One function-calling request. Returns (tool_name, args, meta).

    The three text parts are one user message; cache_control sits on the
    invariant guide ONLY, so every later part can change per decision
    without invalidating the cached prefix. The tools array must stay
    byte-identical across calls for the same reason. An error_note from
    a rejected previous attempt is APPENDED as a fourth part -- after
    the cached prefix, so retries still hit the cache.
    """
    content = [
        {"type": "text", "text": guide,
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": state_txt},
        {"type": "text", "text": memory_txt},
    ]
    if error_note:
        content.append({"type": "text", "text": error_note})
    payload: dict = {
        "model": model, "temperature": temp,
        "messages": [{"role": "user", "content": content}],
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
        "usage": {"include": True},
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    if provider:
        # OpenRouter routing control: one upstream, no silent failover.
        payload["provider"] = {"order": [provider], "allow_fallbacks": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # OpenRouter attribution headers; harmless elsewhere.
        headers["X-Title"] = "balatro-zero harvest"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers)
    last_err: Exception | None = None
    for attempt in range(3):          # transient 429/5xx must not cost a game
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 529) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            try:
                body = e.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                body = ""
            raise RuntimeError(f"HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    else:  # pragma: no cover
        raise RuntimeError(f"request failed: {last_err}")

    meta = {"provider": out.get("provider"), "served_model": out.get("model"),
            "usage": out.get("usage")}
    msg = out["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        # tool_choice=required makes this rare; surface what came back.
        raise ValueError(
            f"no tool call in reply (content: {str(msg.get('content'))[:120]!r})")
    fn = tcs[0]["function"]
    args = json.loads(fn.get("arguments") or "{}")
    return fn["name"], args, meta


def mock_reply(kind: str, gs, opts: list[dict]) -> tuple[str, dict, dict]:
    """Offline stand-in for ask_llm_tools: exercises the full mapping /
    apply / record / pairs path with zero network and zero cost. Not a
    policy -- plays the first cards it sees and walks forward through
    blinds and shops."""
    meta = {"provider": "mock", "served_model": "mock",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    if kind == "hand":
        labels = [l.split("[")[0] for l in hand_labels(gs)][:5]
        return "play", {"cards": labels, "reasoning": "mock play"}, meta
    descs = [o["desc"] for o in opts]
    for want in ("SelectBlind", "NextRound"):
        if want in descs:
            return "choose", {"option": descs.index(want),
                              "reasoning": f"mock {want}"}, meta
    return "choose", {"option": 0, "reasoning": "mock first option"}, meta


class UsageMeter:
    """Accumulates the numbers the measured game is FOR: tokens per
    decision, cache hit rate, cost."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompt = 0
        self.cached = 0
        self.completion = 0
        self.cost = 0.0

    def add(self, usage: dict | None) -> str:
        if not usage:
            return ""
        self.calls += 1
        p = usage.get("prompt_tokens", 0) or 0
        c = usage.get("completion_tokens", 0) or 0
        det = usage.get("prompt_tokens_details") or {}
        cached = det.get("cached_tokens", 0) or 0
        cost = usage.get("cost", 0) or 0
        self.prompt += p
        self.cached += cached
        self.completion += c
        self.cost += cost
        return f"tok {p}({cached}$)+{c}"

    def report(self) -> str:
        if not self.calls:
            return "usage: no billed calls"
        hit = (self.cached / self.prompt * 100) if self.prompt else 0.0
        return (f"usage: {self.calls} calls | prompt {self.prompt} "
                f"(cached {self.cached}, {hit:.0f}%) | completion "
                f"{self.completion} | avg {self.prompt // self.calls}+"
                f"{self.completion // self.calls}/call | cost ${self.cost:.4f}")


# ---------------------------------------------------------------------------
# Legacy free-text path (aided mode)
# ---------------------------------------------------------------------------


def ask_llm(url: str, model: str, temp: float, state_txt: str,
            options: list[str], recent: list[str], kind: str,
            system: str = SYSTEM, api_key: str | None = None,
            provider: str | None = None) -> tuple[str, dict]:
    """Free-text ask used by the beam-assisted mode. Unchanged legacy
    protocol; the unaided harvest path uses ask_llm_tools instead."""
    ctx = ("Recent decisions: " + "; ".join(recent[-6:]) + "\n\n") if recent else ""
    user = (f"{ctx}{state_txt}\n\n[{kind.upper()} stop] options:\n"
            + "\n".join(f"  [{i}] {o}" for i, o in enumerate(options))
            + "\n\nYour command?")
    payload: dict = {
        "model": model, "temperature": temp,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    if provider:
        payload["provider"] = {"order": [provider], "allow_fallbacks": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-Title"] = "balatro-zero harvest"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    meta = {"provider": out.get("provider"), "served_model": out.get("model"),
            "usage": out.get("usage")}
    return out["choices"][0]["message"]["content"].strip(), meta


def _rescue(gs, ctl, kind: str) -> str:
    """A legal move for when the model's reply cannot be used.

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
                "entity_global": r.get("entity_global", -1),
                "card_target": r["card_target"],
                "action_idx": -1,             # factored head needs no index
                "n_legal": 0,
                "run_won": won_run,
                "outcome_progress": prog,
                "after_money_drift": False,
                "provider": r.get("provider"),
                "served_model": r.get("served_model"),
                "reasoning": r.get("reasoning"),
                "obs": r["obs"],
            }) + "\n")
            kept += 1
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed")
    ap.add_argument("--model", default="anthropic/claude-sonnet-5",
                    help="OpenRouter vendor/model id, e.g. "
                         "anthropic/claude-opus-5, google/gemini-3-pro-preview")
    ap.add_argument("--url",
                    default="https://openrouter.ai/api/v1/chat/completions")
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY",
                    help="env var holding the API key (blank for local Ollama)")
    ap.add_argument("--provider", default=None,
                    help="pin the OpenRouter upstream (e.g. Anthropic, Google) "
                         "so a harvest is not split across upstreams")
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--reasoning", default=None,
                    choices=("low", "medium", "high"),
                    help="OpenRouter reasoning effort (unaided path only); "
                         "default off. NOTE: Anthropic models cannot think "
                         "under tool_choice=required -- pair this with "
                         "--tool-choice auto for them.")
    ap.add_argument("--tool-choice", default="required",
                    choices=("required", "auto"),
                    help="required guarantees a tool call every reply (no "
                         "parse failures, but disables Anthropic extended "
                         "thinking); auto permits text replies (retried once "
                         "with a nudge, then fallback)")
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--unaided", action="store_true",
                    help="hide the beam; the model chooses every card itself "
                         "(harvest prompt + function calling)")
    ap.add_argument("--mock", action="store_true",
                    help="no network: canned tool calls exercise the whole "
                         "unaided pipeline for free")
    ap.add_argument("--pairs", default=None,
                    help="append (obs, action) cloning pairs to this jsonl")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get(args.key_env) if args.key_env else None
    if (args.key_env and not api_key and "openrouter" in args.url
            and not args.mock):
        raise SystemExit(
            f"{args.key_env} is not set. Get a key from openrouter.ai and "
            f"export it, or pass --key-env '' for a local endpoint."
        )
    if args.mock and not args.unaided:
        raise SystemExit("--mock only exercises the unaided pipeline")

    gs = new_run(args.seed)
    ctl = dict(CTL0)
    ctl["unaided"] = args.unaided
    ctl["capture_pairs"] = bool(args.pairs)

    guide, tools = load_harvest_prompt() if args.unaided else ("", [])
    meter = UsageMeter()
    history: list[dict] = []   # unaided memory (entries carry reasoning)
    recent: list[str] = []     # aided memory (legacy one-liners)
    n = 0
    fallbacks = 0
    while n < args.max_decisions:
        kind, opts = advance(gs, ctl)
        if won(gs) or is_terminal(gs) or not opts:
            break
        state_txt = summary(gs, ctl)
        applied = None
        meta: dict = {}
        reason = ""
        if args.unaided:
            state_part = build_state_part(gs, ctl, kind, opts)
            memory_part = build_memory_part(history, gs)
            error_note = None
            for attempt in range(2):
                try:
                    if args.mock:
                        name, targs, meta = mock_reply(kind, gs, opts)
                    else:
                        name, targs, meta = ask_llm_tools(
                            args.url, args.model, args.temp, guide, tools,
                            state_part, memory_part, api_key, args.provider,
                            error_note, args.reasoning, args.tool_choice)
                except ValueError as e:
                    # Text reply instead of a tool call (possible under
                    # tool_choice=auto): nudge once, then fall back.
                    print(f"LLM no-tool reply: {e}", flush=True)
                    error_note = ("Your previous reply did not include a "
                                  "tool call. You MUST respond by calling "
                                  "exactly one tool.")
                    continue
                except Exception as e:  # noqa: BLE001
                    print(f"LLM error: {e}", flush=True)
                    break
                reason = str((targs or {}).get("reasoning", ""))
                act = tool_to_act(name, targs)
                tok = meter.add(meta.get("usage"))
                if act is None:
                    error_note = (
                        f"Your call to `{name}` had missing or malformed "
                        f"arguments ({json.dumps(targs)[:200]}). Call one "
                        f"tool with arguments exactly matching its schema.")
                    if not args.quiet:
                        print(f"[{n:3d}] LLM({attempt}) {kind}: BAD ARGS "
                              f"{name} {tok}", flush=True)
                    continue
                applied = apply_act(gs, ctl, kind, opts, act, None)
                if not args.quiet:
                    print(f"[{n:3d}] LLM({attempt}) {kind}: {act!r} -> "
                          f"{applied if applied is not None else 'REJECTED'}"
                          f"  {tok} | {reason[:60]}", flush=True)
                if applied is not None:
                    break
                error_note = (
                    f"Your previous action `{act}` was REJECTED as illegal "
                    f"or unmatched at this {kind} stop. Re-read the state "
                    f"and options and choose a valid action.")
        else:
            descs = [o["desc"] for o in opts]
            for attempt in range(2):
                try:
                    reply, meta = ask_llm(args.url, args.model, args.temp,
                                          state_txt, descs, recent, kind,
                                          SYSTEM, api_key, args.provider)
                except Exception as e:  # noqa: BLE001
                    print(f"LLM error: {e}", flush=True)
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
            fallbacks += 1
            if not args.quiet:
                print(f"[{n:3d}] fallback {fallback!r} -> {applied}", flush=True)
            if applied is None:
                print("fallback failed; stopping", flush=True)
                break
            if ctl.get("log"):
                ctl["log"][-1]["source"] = "fallback"
        if applied and ctl.get("log"):
            # Stamp what actually served this decision -- OpenRouter can
            # route the same model to a different upstream mid-run -- and
            # the model's own reasoning, which feeds the memory part.
            if meta.get("provider"):
                ctl["log"][-1].setdefault("provider", meta["provider"])
                ctl["log"][-1].setdefault("served_model", meta.get("served_model"))
            if reason:
                ctl["log"][-1].setdefault("reasoning", reason)
        if applied:
            recent.append(applied)
            if applied != "":     # constraint edits apply nothing
                history.append({"n": n, "ante": ante(gs), "kind": kind,
                                "entry": applied, "reason": reason})
        n += 1

    if args.pairs:
        kept = write_pairs(args.pairs, ctl, args.seed, args.model, gs)
        print(f"wrote {kept} BC pairs -> {args.pairs}", flush=True)

    print("\n" + summary(gs, ctl))
    tag = "WON" if won(gs) else ("GAME OVER" if is_terminal(gs) else "STOPPED")
    print(f"*** {tag}: ante {ante(gs)} prog {progress(gs):.3f} "
          f"after {n} decisions ({fallbacks} fallbacks) ***")
    if args.unaided:
        print(meter.report(), flush=True)


if __name__ == "__main__":
    main()
