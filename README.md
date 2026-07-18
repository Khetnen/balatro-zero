# balatro-zero

Gumbel expert-iteration (AlphaZero-style) agent for Balatro, built on the
[jackdaw](../jackdaw-balatro) simulator (local clone, `local-fixes` branch —
live-validated engine).

## Why this design

- **Search + learned prior, not pure PPO**: taggarttufte/balatro-rl's
  shaped-reward PPO plateaued at 2.35% win rate and its scaling postmortem
  concluded the bottleneck is exploration/search, not capacity.
- **Gumbel sequential halving** (Danihelka et al.) gives sound policy
  improvement at tiny simulation budgets (16-32 sims/move), which is what a
  Python simulator can afford.
- **Two value heads** (P(win), normalized furthest-ante): early in training
  P(win) is ~0 everywhere; the ante head supplies the learning signal — a
  value-space curriculum instead of misleading reward shaping.
- **Pickle-based cloning** (~0.9ms/clone, 2x faster than deepcopy) is the
  search's dominant cost; measured throughput math in the project memory.
- **Checkpoints are the product**: for seed-difficulty extraction, the
  ensemble of checkpoints at increasing strength serves as the
  reference-policy ladder (rollout win-rate/score distributions per seed).

## Usage

```sh
# smoke test (inline, tiny)
uv run bzero --iters 1 --games 2 --sims 8 --depth 0 --workers 0 --eval-games 2

# real training run (12 worker processes, GPU training steps)
uv run bzero --iters 200 --games 96 --workers 12 --sims 16 --depth 2 --out runs/v0
```

Metrics stream to `<out>/metrics.jsonl`; checkpoints to `<out>/ckpt_NNNN.pt`.
Watch `eval mean_ante` — it should climb past 2-3 within the first dozens of
iterations; `win_rate` stays ~0 until much later (a win requires beating
ante 8).

## Known simplifications (deliberate, revisit later)

- Search is **clairvoyant** on the run's true seed (no determinization of
  hidden future RNG). Fine for a difficulty-reference policy; a "blind"
  determinized variant is future work.
- No tree reuse below the root; rollouts are policy-greedy with depth cap.
- Worker inference is per-position CPU; batched GPU inference across
  parallel games is the big future throughput win.
- Flat Discrete(500) action slots (jackdaw's convention): slot semantics
  are state-dependent; card-combo enumeration subsampled at 200.
