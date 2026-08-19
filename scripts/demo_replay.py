"""Replay harvested LLM games from their stdout logs; emit curriculum
snapshots and factored demonstration samples.

WHY REPLAY: the pairs export (llm_run.write_pairs) stores the observation
and the model's action at each decision, but neither the LEGAL SET there
(which a factored CandidateSet target needs: "this action, NOT the other
legal ones") nor pickled engine states (which curriculum starts need).
Both exist only transiently during play, so the games are re-run from
their logs: every act line -- model decisions, rejected attempts, and
beam fallbacks alike -- is re-applied through the same
interactive_run.apply_act path the harvest used. Rejected attempts are
re-applied too: a few of them mutate state before failing (death-macro
pre-swaps, partial joker reorders), so skipping them could silently fork
the trajectory.

VERIFICATION, three layers, and a game only contributes artifacts if all
pass:
  1. every act's accept/reject AND its applied-entry string match the log;
  2. the recomputed observation matches the stored pairs file at every
     model decision (when pairs_<seed>.jsonl exists) -- a per-decision
     state fingerprint, not just an endpoint check;
  3. the final outcome (WON / ante) reproduces.
A verified replay doubles as a validity certificate for the corpus: the
recorded games really are legal trajectories of the committed engine.

ONE GAME PER _combo_rng RESET: the harvest ran one game per process, and
option enumeration consumes the module-global subsample stream, so
replaying many games in one process would advance it mid-game and could
flip an index-resolved option. Reset it to its import-time state before
each game.

Outputs (--out DIR):
  snapshots.pkl   dict seed -> {ante: pickled gs} -- state at the FIRST
                  decision stop of each ante 2..8 (play_game start_state
                  compatible; pool files are derived downstream)
  snap_meta.jsonl one row per snapshot (seed, ante, phase, dollars, ...)
  demos_wins.pkl  list[(Obs, CandidateSet, z_win, z_togo)] in train.py's
                  --demos format: one-hot over the decision's legal set
                  (the demonstrated action appended when combo subsampling
                  dropped it), z from the run outcome exactly as
                  selfplay.play_game computes them
  report.json     per-game verdicts + corpus totals

Usage (from balatro-zero/):
    uv run --no-sync python scripts/demo_replay.py \
        --logs "runs/llm_harvest/fl_*.log" "runs/llm_harvest/ds_*.log" \
        --out runs/demo_replay --wins-only
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import pickle
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from interactive_run import CTL0, advance, apply_act  # noqa: E402

import balatro_zero.state as bz_state  # noqa: E402
from balatro_zero.net import market_area_lens  # noqa: E402
from balatro_zero.state import (  # noqa: E402
    Obs,
    ante,
    is_terminal,
    legal_factored,
    new_run,
    progress,
    won,
)
from balatro_zero.targets import encode_candidates  # noqa: E402
from jackdaw.env.game_spec import FactoredAction  # noqa: E402

# [  5] LLM(0) econ: '2' -> PICK j_delayed_grat  tok 12336(11715$)+194 | why...
LLM_RE = re.compile(r"^\[\s*(\d+)\] LLM\(\d+\) (hand|econ): ('.*?'|\".*?\") -> (.*)$")
# [ 68] fallback '0' -> Reroll
FB_RE = re.compile(r"^\[\s*(\d+)\] fallback ('.*?'|\".*?\") -> (.*)$")
WON_RE = re.compile(r"^\*\*\* WON: ante (\d+)")
OVER_RE = re.compile(r"^\*\*\* GAME OVER: ante (\d+)")


def _strip_llm_tail(rest: str) -> str:
    """The applied entry, shorn of the usage/reasoning tail.

    The line format is  f"{applied}  {tok} | {reason}"  with
    tok = "tok 12336(11715$)+194" (one token after 'tok '), or three
    spaces + "| reason" when the meter had nothing. Entry strings never
    contain " | " or a trailing " tok <word>", so peeling from the right
    is unambiguous.
    """
    s = rest.split(" | ")[0].rstrip()
    return re.sub(r"\s+tok \S+$", "", s).rstrip()


def parse_log(path: Path) -> dict:
    """Ordered act events + the recorded outcome, from one game log.

    A crashed-and-retried game appends a second block to the same file;
    a step counter reset to 0 after progress starts a fresh event list,
    so only the LAST complete game block is replayed.
    """
    events: list[dict] = []
    outcome = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LLM_RE.match(line)
        fb = None if m else FB_RE.match(line)
        if m or fb:
            g = (m or fb).groups()
            n = int(g[0])
            if n == 0 and events and events[-1]["n"] > 0:
                events = []          # retry block: start over
                outcome = None
            applied = _strip_llm_tail(g[-1]) if m else g[-1].rstrip()
            rejected = applied == ("REJECTED" if m else "None")
            events.append({
                "n": n,
                "fallback": fb is not None,
                "act": ast.literal_eval(g[2] if m else g[1]),
                "ok": not rejected,
                "applied": None if rejected else applied,
            })
            continue
        w = WON_RE.match(line)
        o = None if w else OVER_RE.match(line)
        if w or o:
            outcome = {"won": w is not None, "ante": int((w or o).group(1))}
    return {"events": events, "outcome": outcome}


def load_pairs_fingerprint(pairs_path: Path) -> list[dict] | None:
    """The stored (flat obs, action fields) per model decision, in order."""
    if not pairs_path.exists():
        return None
    rows = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "flat": np.asarray(r["obs"]["flat"], dtype=np.float32),
                "action_type": r["action_type"],
                "entity_global": r.get("entity_global", -1),
            })
    return rows


def replay_game(seed: str, events: list[dict], expect_won: bool) -> dict:
    """Re-apply one game's act stream; return artifacts + verdict."""
    # Import-time stream state == a fresh harvest process (llm_run ran
    # one game per invocation).
    bz_state._combo_rng = np.random.default_rng(0)
    gs = new_run(seed)
    ctl = dict(CTL0)
    ctl["unaided"] = True
    # capture_pairs makes _record snapshot the observation at the exact
    # point the harvest did — AFTER apply_act's internal pre-mutations
    # (the copy/death macro free-swaps hand cards before recording, so a
    # pre-apply observe() disagrees with the stored pairs there).
    ctl["capture_pairs"] = True
    demo_rng = np.random.default_rng(zlib.crc32(f"DEMO|{seed}".encode()))

    snaps: dict[int, bytes] = {}
    snap_meta: list[dict] = []
    demos: list[tuple] = []          # (Obs, CandidateSet, progress@decision)
    max_progress = progress(gs)
    i = 0
    mismatch = None
    while i < len(events):
        kind, opts = advance(gs, ctl)
        if won(gs) or is_terminal(gs) or not opts:
            break
        a = ante(gs)
        if 2 <= a <= 8 and a not in snaps:
            snaps[a] = pickle.dumps(gs, protocol=5)
            snap_meta.append({
                "seed": seed, "ante": a, "stop": kind,
                "phase": str(gs.get("phase")), "dollars": gs.get("dollars", 0),
                "decision_idx": i, "progress": round(progress(gs), 4),
            })
        ev = events[i]
        i += 1
        # Demo-eligible: the model's own decision, applied, and not a
        # copy/death macro — that one free-swaps the hand before the
        # recorded action, so a pre-apply legal set is misaligned with
        # the action's card indices (the row still counts for the pairs
        # fingerprint, which uses _record's own post-swap obs).
        cap = (ev["ok"] and not ev["fallback"]
               and not ev["act"].lower().startswith("copy "))
        if cap:
            legal = legal_factored(gs, demo_rng)
            n_hand = len(gs.get("hand", []))
            mlens = market_area_lens(gs)
            prog_here = progress(gs)
        applied = apply_act(gs, ctl, kind, opts, ev["act"], None)
        ok = applied is not None
        if ok and ev["fallback"] and ctl.get("log"):
            # llm_run stamped fallback provenance after the fact; the
            # pairs export filters on it, so the replay must too.
            ctl["log"][-1]["source"] = "fallback"
        if ok != ev["ok"] or (ok and applied != ev["applied"]):
            mismatch = {"decision": ev["n"], "act": ev["act"],
                        "expected": ev["applied"], "got": applied}
            break
        max_progress = max(max_progress, progress(gs))
        if ok and cap:
            rec = ctl["log"][-1]
            if "action_type" in rec:      # ORDER etc. carry no single action
                obs = Obs(
                    flat=np.asarray(rec["obs"]["flat"], dtype=np.float32),
                    joker_ids=np.asarray(rec["obs"]["joker_ids"], dtype=np.int64),
                    consumable_ids=np.asarray(rec["obs"]["consumable_ids"],
                                              dtype=np.int64),
                    market_ids=np.asarray(rec["obs"]["market_ids"], dtype=np.int64),
                )
                ct = tuple(rec["card_target"]) if rec["card_target"] else None
                fa = FactoredAction(action_type=rec["action_type"],
                                    card_target=ct,
                                    entity_target=rec["entity_target"])
                cands = list(legal)
                try:
                    idx = cands.index(fa)
                except ValueError:
                    # Subsampling dropped it, or the play's card order
                    # differs from the enumerator's sorted combos -- the
                    # action is legal (it just ran), so append it.
                    cands.append(fa)
                    idx = len(cands) - 1
                w = np.zeros(len(cands), dtype=np.float32)
                w[idx] = 1.0
                demos.append((obs,
                              encode_candidates(cands, w, n_hand,
                                                market_lens=mlens),
                              prog_here))

    if mismatch is None and i >= len(events) and not (won(gs) or is_terminal(gs)):
        advance(gs, ctl)             # roll to the end, as llm_run's loop top did

    game_won = bool(won(gs))
    verified = (mismatch is None and i >= len(events)
                and game_won == expect_won)
    # z targets exactly as selfplay.play_game: win => togo counts to 1.0.
    max_progress = max(max_progress, progress(gs))
    final_best = 1.0 if game_won else max_progress
    z_win = 1.0 if game_won else 0.0
    samples = [
        (obs, cs, z_win, float(np.clip(final_best - p, 0.0, 1.0)))
        for obs, cs, p in demos
    ]
    # The pairs-file fingerprint, filtered EXACTLY as write_pairs did:
    # records carrying obs + action fields whose source is not
    # beam/fallback. This includes engine-rejected attempts (recorded
    # before step_factored raised) and copy-macro rows -- both of which
    # the harvest's export also kept.
    fp_obs = [
        np.asarray(r["obs"]["flat"], dtype=np.float32)
        for r in ctl.get("log", [])
        if "obs" in r and "action_type" in r
        and r.get("source") not in ("beam-pass", "beam-auto", "fallback")
    ]
    return {
        "verified": verified, "won": game_won, "final_ante": ante(gs),
        "n_events": len(events), "n_applied": i, "mismatch": mismatch,
        "samples": samples, "snaps": snaps, "snap_meta": snap_meta,
        "fp_obs": fp_obs,
    }


def check_pairs(fp_obs: list[np.ndarray], fingerprint: list[dict] | None) -> dict:
    """Bit-compare replayed observations against the harvest's pairs."""
    if fingerprint is None:
        return {"checked": False}
    if len(fp_obs) != len(fingerprint):
        return {"checked": True, "ok": False,
                "why": f"count {len(fp_obs)} vs pairs {len(fingerprint)}"}
    worst = 0.0
    for flat, fp in zip(fp_obs, fingerprint):
        stored = fp["flat"]
        if flat.shape != stored.shape:
            return {"checked": True, "ok": False, "why": "obs dim mismatch"}
        worst = max(worst, float(np.abs(flat - stored).max()))
    return {"checked": True, "ok": worst <= 1e-6, "max_abs_diff": worst}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="log globs, e.g. runs/llm_harvest/fl_*.log")
    ap.add_argument("--out", default="runs/demo_replay")
    ap.add_argument("--wins-only", action="store_true",
                    help="replay only games whose log records a WIN")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N games (0 = all); for smoke tests")
    args = ap.parse_args()

    paths = sorted({Path(p) for pat in args.logs for p in glob.glob(pat)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_snaps: dict[str, dict[int, bytes]] = {}
    all_samples: list[tuple] = []
    report: list[dict] = []
    meta_rows: list[dict] = []
    n_done = 0
    t0 = time.perf_counter()
    for path in paths:
        m = re.match(r"^[a-z]+_([A-Z0-9]+)\.log$", path.name)
        if not m:
            continue
        seed = m.group(1)
        parsed = parse_log(path)
        outcome = parsed["outcome"]
        if outcome is None:
            report.append({"seed": seed, "status": "no-outcome-line"})
            continue
        if args.wins_only and not outcome["won"]:
            continue
        try:
            res = replay_game(seed, parsed["events"], outcome["won"])
        except Exception as e:  # noqa: BLE001 -- one bad game must not kill the sweep
            report.append({"seed": seed, "status": "crash", "error": repr(e)})
            print(f"[{seed}] CRASH {e!r}", flush=True)
            continue
        pairs_path = path.with_name(f"pairs_{path.stem}.jsonl")
        pc = check_pairs(res["fp_obs"], load_pairs_fingerprint(pairs_path))
        good = res["verified"] and (not pc.get("checked") or pc.get("ok"))
        row = {
            "seed": seed, "status": "ok" if good else "FAILED",
            "won": res["won"], "final_ante": res["final_ante"],
            "n_events": res["n_events"], "n_applied": res["n_applied"],
            "n_demos": len(res["samples"]),
            "snap_antes": sorted(res["snaps"]),
            "pairs_check": pc, "mismatch": res["mismatch"],
        }
        report.append(row)
        if good:
            all_snaps[seed] = res["snaps"]
            all_samples.extend(res["samples"])
            meta_rows.extend(res["snap_meta"])
        n_done += 1
        print(f"[{seed}] {row['status']} won={res['won']} "
              f"demos={len(res['samples'])} snaps={row['snap_antes']} "
              f"pairs={pc}", flush=True)
        if args.limit and n_done >= args.limit:
            break

    (out / "snapshots.pkl").write_bytes(pickle.dumps(all_snaps, protocol=5))
    (out / "demos_wins.pkl").write_bytes(pickle.dumps(all_samples, protocol=5))
    with open(out / "snap_meta.jsonl", "w", encoding="utf-8") as f:
        for r in meta_rows:
            f.write(json.dumps(r) + "\n")
    ok = sum(1 for r in report if r.get("status") == "ok")
    bad = [r["seed"] for r in report if r.get("status") not in ("ok", None)
           and r.get("status") != "ok"]
    summary = {
        "games_replayed": n_done, "verified": ok, "failed": bad,
        "demo_samples": len(all_samples),
        "snapshots": sum(len(s) for s in all_snaps.values()),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (out / "report.json").write_text(
        json.dumps({"summary": summary, "games": report}, indent=1),
        encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
