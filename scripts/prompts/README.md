# Harvest prompt v2 (draft)

The BalatroBench-style prompt for the unaided LLM harvest (`llm_run.py --unaided
--pairs`). Replaces the ~1,200-char `UNAIDED_SYSTEM` command reference. Design
per the 2026-08-01 plan of record: their structure (manual + strategy + state +
memory), function calling instead of free-text parsing, prompt caching on the
invariant part, state duplication trimmed.

## Files

- `harvest_guide.md` — message part 0, INVARIANT across every call of every
  game. Role + interface contract + game manual + strategy doctrine + complete
  key reference (all 150 jokers, 22 tarots, 12 planets, 18 spectrals, 32
  vouchers, 24 tags, 30 blinds — keys and numbers generated from
  `jackdaw/engine/data/*.json`, effect text spot-checked against
  `balatro-source/localization/en-us.lua`).
- `harvest_tools.json` — the 8 function-calling tools (OpenAI shape, works
  through OpenRouter). Every tool requires a `reasoning` string (BalatroBench's
  pattern, kept deliberately: each BC pair carries its own justification).

## Message assembly (per decision)

One user message, three content parts — same shape BalatroBench used, which is
also the correct shape for caching:

```
part 0: harvest_guide.md                      <- cache_control: {"type": "ephemeral"}
part 1: rendered state (summary() + stop kind + numbered options at econ stops)
part 2: memory (recent decisions incl. the model's own reasoning strings;
        one-line ante trajectory)
```

- `cache_control` goes on part 0 ONLY (BalatroBench never set it and re-sent
  ~10.5k tokens per call; this is the single biggest cost lever, ~$5.60 ->
  under $2 per game). Via OpenRouter this passes through to Anthropic models;
  non-Anthropic providers ignore it harmlessly or cache implicitly.
- The `tools` array must be IDENTICAL on every call — tools are part of the
  cached prefix. Do not swap tool subsets per stop kind; the guide tells the
  model which tools are valid where, and an invalid call gets one retry with
  the error in the retry message.
- `tool_choice: "required"` (OpenRouter maps this per provider) — every reply
  must be a tool call. With free text impossible, the fallback path only
  remains for schema-level failures, and those stay excluded from BC export.

## Tool call -> harness act string (wiring map)

| tool | act string for `apply_act` |
|---|---|
| play {cards} | `play <cards joined>` |
| discard {cards} | `discard <cards joined>` |
| use_consumable {consumable, cards?} | `use <key>` / `use <key> on <cards>` |
| sell {item} | `sell <key>` |
| copy_card {source, target} | `copy <source> onto <target>` |
| order_jokers {order} | `order jokers <keys joined with ", ">` |
| choose {option} | `<option index as string>` |
| pick_pack_card {card, targets?} | `pick <key> on <cards>` / `<matching option>` |

## Wiring status (2026-08-15)

DONE — `llm_run.py --unaided` now runs this prompt end-to-end: three-part
message with `cache_control` on part 0, constant tool set,
`tool_choice: required`, one retry with the rejection appended AFTER the
cached prefix, transient-HTTP retries, a UsageMeter (tokens / cache hits /
cost per game), and a part-2 memory carrying each decision's own `reasoning`.
BC pairs now also carry `reasoning`. `--mock` exercises the whole pipeline
offline (verified: tool call -> act string -> engine -> pairs with 1064-dim
v14 obs).

Also DONE alongside: `summary()` shows all 12 hand values, and `play`
preserves card order — the engine scores played cards in SELECTION order
(`game.py _handle_play_hand`), which `resolve_hand_act` used to throw away
with `sorted()`. The guide and the `play` tool now teach ordering (xMult
cards last). Hand-rearrange is deliberately NOT a tool: play order covers
scoring order, and held-in-hand effects multiply, so hand order is inert.
`order_jokers` covers the board.

REMAINING:

1. Keep `--provider` pinned for the harvest (provider variance is real; see
   balatrobench notes); served provider is recorded per decision.
2. One measured game BEFORE the harvest: actual tokens/decision, cache hit
   rate, invalid-tool-call rate (the UsageMeter + fallback count print at
   game end). That measurement is the go/no-go for the ~300-seed run.
   Blocked only on OPENROUTER_API_KEY.
