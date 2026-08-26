# balatro-zero

Gumbel expert-iteration (AlphaZero-style) agent for Balatro, built on the
[jackdaw](https://github.com/TylerFlar/jackdaw-balatro) simulator
(validated against the real game by lockstep differential replay; installed
automatically by `uv sync`). Self-play
with search-improved policy targets, plus supervised training on LLM
demonstration wins.

## Why this design

- **Search + learned prior, not pure PPO**:
  [taggarttufte/balatro-rl](https://github.com/taggarttufte/balatro-rl)'s
  shaped-reward PPO plateaued at 2.35% win rate and its scaling postmortem
  concluded the bottleneck is exploration/search, not capacity.
- **Gumbel sequential halving** (Danihelka et al.) gives sound policy
  improvement at tiny simulation budgets (16-32 sims/move), which is what a
  Python simulator can afford.
- **Two value heads** (P(win), normalized furthest-ante): early in training
  P(win) is ~0 everywhere; the ante head supplies the learning signal — a
  curriculum in value space rather than in the reward. The progress
  target uses a frontier formula (gate: `scripts/frontier_progress_probe.py`)
  that only rewards progress past the run's best ante.
- **LLM demonstrations**: ~47k self-play games across 15 training
  generations produced zero wins. Mixing in 189 replay-verified LLM
  demonstration wins (harvested with `scripts/llm_run.py`) and using their
  states as a win-pool curriculum produced the first wins. The economy play
  required to win does not appear in self-play data.
- **Pickle-based cloning** (~0.9ms/clone, 2x faster than deepcopy) is the
  search's dominant cost.
- **A checkpoint ladder, not a single agent**: the end use is
  seed-difficulty extraction, where checkpoints at increasing strength act
  as the reference policies (rollout win-rate/score distributions per
  seed).

## Status (2026-08)

- On a fixed-seed evaluation panel (`scripts/difficulty_eval.py`, mean
  furthest-ante over K rollouts/seed): scripted chip-greedy baseline 1.65;
  pure self-play nets plateau at 2.4-2.65 across many variants; the current
  demo-trained net with router-guided economy reaches 2.92; the scripted
  expert router scores 2.79.
- From mid-run states of winning demonstration games, the current nets
  convert ante-8 closeouts at up to ~37% (early generations: ~1%). Full-run
  win rate is still ~0; the remaining gap is early-game (ante 1-4) build
  quality.
- Negative result: adding boss identity and skip tags to the observation
  (`--arch v7`) fixed real blindness but did not move the panel numbers or
  the training curves.

## Usage

```sh
# smoke test (inline, tiny)
uv run bzero --iters 1 --games 2 --sims 8 --depth 0 --workers 0 --eval-games 2

# from-scratch training run
uv run bzero --iters 200 --games 48 --workers 16 --sims 32 --depth 1 \
    --epochs 4 --eval-games 16 --out runs/v0

# the full current recipe: LLM demos + win-pool curriculum + guided economy
uv run bzero --iters 400 --games 48 --workers 16 --sims 32 --depth 1 \
    --epochs 4 --eval-games 16 --arch v7 \
    --demos runs/demo_replay/demos_wins.pkl --demo-frac 0.2 \
    --seed-pool runs/demo_replay/win_pool_a23456.pkl \
    --guided-frac 0.25 --out runs/v19
```

The demo/seed-pool pickles are produced by the harvest pipeline
(`scripts/llm_run.py` -> `scripts/demo_replay.py`). The harvested corpus —
349 LLM-played games, 189 replay-verified wins — is published as the
[llm-corpus-v1 release](https://github.com/Khetnen/balatro-zero/releases/tag/llm-corpus-v1)
(CC0); unzip it into the repo and run `demo_replay.py` per its README to
rebuild the pickles, or omit the demo flags to train without them. Metrics stream to `<out>/metrics.jsonl`;
checkpoints to `<out>/ckpt_NNNN.pt`; `--resume` continues from
`<out>/latest.pt`. Watch `eval mean_ante` — it should climb past 2-3 within
the first dozens of iterations; `win_rate` stays ~0 until much later (a win
requires beating ante 8).

## Repository layout

`balatro_zero/` — the trainer: `state.py` (engine-facing game state +
observation encoding), `net.py` (policy/value nets, `--arch v3`-`v7`; v6+
use content-bound pointer heads), `search.py` (Gumbel sequential halving),
`selfplay.py`, `targets.py` (factored policy/value targets), `router.py`
(scripted expert; used for guided games and as a baseline), `goldprobe.py`
(clairvoyant upper-anchor probe), `train.py`.

`scripts/` highlights:

- **LLM harvest**: `llm_run.py` (function-calling strategist for any
  OpenAI-compatible endpoint; system prompt in `prompts/harvest_guide.md`),
  `demo_replay.py` (bit-exact replay of harvested games -> training samples
  + curriculum snapshots), `bc_pretrain.py` (behaviour-cloning pretrain).
- **Probes**: each mechanism has a gate script that must pass before the
  mechanism is trusted — `binding_probe.py`, `determinize_probe.py`,
  `observability_probe.py`, `factored_loss_probe.py`, `closeout_probe.py`.
  `gold_rebaseline.py` checks the gold probe against the current engine;
  run it before long sweeps.
- **Difficulty extraction**: `difficulty_eval.py` (probe ladder x seed
  panel), `difficulty_curves.py`, `difficulty_irt.py` (graded-response
  fits).
- **Misc**: `interactive_run.py` (human- or LLM-driven runs with beam
  assistance), `bb_replay.py` / `bench5.py` (cross-checks against the
  BalatroBench dataset), `seed_scout.py`, `route_harvest.py`.

## Known simplifications (deliberate, revisit later)

- Search rollouts are **determinized**: each simulation clone gets a fresh
  PRNG seed and a reshuffled undrawn deck (`state.determinize`), so search
  cannot see the run's actual future draws. `--clairvoyant` makes rollouts
  replay the run's true RNG — only for comparisons against runs from
  before determinization was the default (2026-08-11). The gold probe stays
  clairvoyant by design (it is the difficulty ladder's upper anchor). The
  beam paths (`macro_k`, `blind_finisher`) plan against the clone's sampled
  order — no longer the true future, but still clairvoyant *within* their
  sample.
- No tree reuse below the root; rollouts are policy-greedy with depth cap.
- Worker inference is per-position CPU; batched GPU inference across
  parallel games is the main remaining throughput optimization.
- Flat Discrete(500) action slots (jackdaw's convention): slot semantics
  are state-dependent; card-combo enumeration subsampled at 200.

## License

MIT — see [LICENSE](LICENSE).
