"""Falsifiable gates for the v14 observability block (state.py `_v14_block`).

One gate per bundled defect fix, each with a zero-movement CONTROL in the
style of adjacency_probe/binding_probe — valid at random init:

  * boss identity / skip-tag offers: the info must reach the obs (and the
    net); the legacy prefix must be PROVABLY unchanged (that invariance is
    the pre-v14 blindness), and the trained v13 (V6, 614-dim) checkpoint
    must return bit-identical outputs, since the appended block is
    invisible to any pre-v14 net by construction.
  * undrawn-deck histogram: removing one card moves exactly its count
    dim; determinize() (deck reshuffle) must not move the block at all —
    the histogram is multiset-only, i.e. honest public information.
  * hand rows 9-16 and pack-slot content: BINDING gates. The pointer
    query (card_query / ent_query) is frozen to a constant vector so the
    head is isolated from the torso pathway; then a content swap between
    two slots must EXCHANGE their logits exactly for V7, while the V6
    architecture (card_tail bias; embed-only market content — both pack
    playing cards embed c_base) must be EXACTLY invariant.

Run from balatro-zero/:  uv run --no-sync python scripts/observability_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, ".")

from jackdaw.env.action_space import ActionType

from balatro_zero.net import (
    ENT_OFF_MARKET,
    PolicyValueNetV6,
    PolicyValueNetV7,
    evaluate_factored,
    load_net,
)
from balatro_zero.state import (
    EXTRA_HAND_OFFSET,
    HAND_FLAT_DIM,
    HAND_FLAT_ROWS,
    MARKET_FEAT_DIM,
    MARKET_FEAT_OFFSET,
    N_BLIND_KEYS,
    N_DECK_ENH,
    N_DECK_HIST,
    N_MARKET_SLOTS,
    NUM_TAGS,
    OBS_DIM,
    OBS_DIM_LEGACY,
    V14_BOSS_OFFSET,
    V14_DECK_OFFSET,
    V14_TAG_OFFSET,
    _RANK_IDX,
    _SUIT_IDX,
    NUM_RANKS,
    clone,
    determinize,
    legal_factored,
    new_run,
    observe,
    obs_vector,
    step_factored,
)

FAILURES: list[str] = []
DEV = torch.device("cpu")
V13_CKPT = Path("runs/v13/latest.pt")


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def freeze_query(net: nn.Module, name: str) -> None:
    """Constant pointer query: isolates the head from the torso pathway,
    making content-swap exchange algebra exact instead of approximate."""
    q = getattr(net, name)
    with torch.no_grad():
        q.weight.zero_()
        gen = torch.Generator().manual_seed(123)
        q.bias.copy_(torch.randn(q.bias.shape, generator=gen))


def outputs(net, gs) -> tuple[np.ndarray, ...]:
    t, e, c, v = evaluate_factored(net, observe(gs), DEV)
    return t[0], e[0], c[0], v


def diff_span(fa: np.ndarray, fb: np.ndarray) -> tuple[int, int] | None:
    idx = np.nonzero(fa != fb)[0]
    if len(idx) == 0:
        return None
    return int(idx.min()), int(idx.max())


def exchange_ok(a_i, a_j, b_i, b_j, tol=1e-5) -> bool:
    """Swapped slots exchange content contributions (binding_probe algebra)."""
    return (abs((a_i - b_j) - (b_i - a_j)) < tol  # positional terms cancel
            and abs((a_i + a_j) - (b_i + b_j)) < tol
            and abs(a_i - b_i) > 1e-6)            # and content actually moved


def main() -> int:
    torch.manual_seed(0)
    gs0 = new_run("OBSPROBE")
    rr = gs0["round_resets"]

    print("0. public state present at blind select")
    boss = rr.get("blind_choices", {}).get("Boss", "")
    tags = rr.get("blind_tags", {})
    check("ante's boss rolled", boss.startswith("bl_"), boss)
    check("skip tags rolled", bool(tags.get("Small")) and bool(tags.get("Big")),
          str(tags))

    print("1. structure")
    f0 = obs_vector(gs0)
    check("obs length == OBS_DIM", len(f0) == OBS_DIM, f"{len(f0)} vs {OBS_DIM}")
    expected = (OBS_DIM_LEGACY + N_BLIND_KEYS + 2 * NUM_TAGS
                + N_DECK_HIST + N_DECK_ENH + 8 * HAND_FLAT_DIM
                + N_MARKET_SLOTS * MARKET_FEAT_DIM)
    check("block layout adds up", OBS_DIM == expected)
    check("boss one-hot set", f0[V14_BOSS_OFFSET:V14_TAG_OFFSET].sum() == 1.0)

    v13 = load_net(str(V13_CKPT)) if V13_CKPT.exists() else None
    if v13 is None:
        print(f"  (no {V13_CKPT}; trained-control checks skipped)")

    torch.manual_seed(1)
    v7 = PolicyValueNetV7()
    v7.eval()

    print("2. boss identity reaches the net (was identically zero: no bl_*")
    print("   key exists in centers.json, so v[26] was dead in every run)")
    a, b = clone(gs0), clone(gs0)
    a["round_resets"]["blind_choices"]["Boss"] = "bl_needle"
    b["round_resets"]["blind_choices"]["Boss"] = "bl_psychic"
    fa, fb = obs_vector(a), obs_vector(b)
    check("legacy prefix blind to the boss (the defect, proven)",
          bool((fa[:OBS_DIM_LEGACY] == fb[:OBS_DIM_LEGACY]).all()))
    span = diff_span(fa, fb)
    check("diff confined to the boss block",
          span is not None and span[0] >= V14_BOSS_OFFSET and span[1] < V14_TAG_OFFSET,
          str(span))
    ta, tb = outputs(v7, a)[0], outputs(v7, b)[0]
    check("V7 type logits move with the boss",
          float(np.abs(ta - tb).max()) > 1e-6)
    if v13 is not None:
        oa, ob = outputs(v13, a), outputs(v13, b)
        check("v13 control: bit-identical (boss invisible to pre-v14 nets)",
              all((x == y).all() for x, y in zip(oa, ob)))

    print("3. skip-tag offers reach the net")
    a, b = clone(gs0), clone(gs0)
    a["round_resets"]["blind_tags"]["Small"] = "tag_negative"
    b["round_resets"]["blind_tags"]["Small"] = "tag_economy"
    fa, fb = obs_vector(a), obs_vector(b)
    check("legacy prefix blind to the offer (the defect, proven)",
          bool((fa[:OBS_DIM_LEGACY] == fb[:OBS_DIM_LEGACY]).all()))
    span = diff_span(fa, fb)
    check("diff confined to the tag block",
          span is not None and span[0] >= V14_TAG_OFFSET and span[1] < V14_DECK_OFFSET,
          str(span))
    ta, tb = outputs(v7, a)[0], outputs(v7, b)[0]
    check("V7 type logits move with the offer",
          float(np.abs(ta - tb).max()) > 1e-6)
    if v13 is not None:
        oa, ob = outputs(v13, a), outputs(v13, b)
        check("v13 control: bit-identical (offer invisible to pre-v14 nets)",
              all((x == y).all() for x, y in zip(oa, ob)))

    print("4. undrawn-deck histogram")
    c = clone(gs0)
    popped = c["deck"].pop()
    si = _SUIT_IDX[popped.base.suit.value]
    ri = _RANK_IDX[popped.base.rank.value]
    dim = V14_DECK_OFFSET + si * NUM_RANKS + ri
    fc = obs_vector(c)
    block = slice(V14_DECK_OFFSET, V14_DECK_OFFSET + N_DECK_HIST + N_DECK_ENH)
    idx = np.nonzero(f0[block] != fc[block])[0]
    check("removing one card moves exactly its count dim",
          len(idx) == 1 and idx[0] + V14_DECK_OFFSET == dim
          and abs(f0[dim] - fc[dim] - 0.25) < 1e-6,
          f"dims moved: {idx + V14_DECK_OFFSET}, expected {dim}")
    d = clone(gs0)
    determinize(d, np.random.default_rng(5))
    fd = obs_vector(d)
    check("determinize (deck reshuffle) leaves the v14 block fixed",
          bool((f0[OBS_DIM_LEGACY:] == fd[OBS_DIM_LEGACY:]).all()))

    print("5. hand rows 9-16 (binding: frozen card_query)")
    gh = clone(gs0)
    sel = [x for x in legal_factored(gh)
           if int(x.action_type) == int(ActionType.SelectBlind)]
    step_factored(gh, sel[0])
    while True:  # deal two extra, distinct cards into slots 8/9
        while len(gh["hand"]) < 10:
            gh["hand"].append(gh["deck"].pop())
        c8, c9 = gh["hand"][8], gh["hand"][9]
        if (c8.base.rank.value, c8.base.suit.value) != (c9.base.rank.value,
                                                        c9.base.suit.value):
            break
        gh["hand"].pop()
    gswap = clone(gh)
    gswap["hand"][8], gswap["hand"][9] = gswap["hand"][9], gswap["hand"][8]
    fa, fb = obs_vector(gh), obs_vector(gswap)
    check("legacy prefix blind to cards 9/10 (the defect, proven)",
          bool((fa[:OBS_DIM_LEGACY] == fb[:OBS_DIM_LEGACY]).all()))
    row = slice(EXTRA_HAND_OFFSET, EXTRA_HAND_OFFSET + 2 * HAND_FLAT_DIM)
    check("rows 9/10 populated and distinct",
          fa[row].any() and not (fa[row] == fb[row]).all())
    if v13 is not None:
        oa, ob = outputs(v13, gh), outputs(v13, gswap)
        check("v13 control: bit-identical (cards 9/10 invisible)",
              all((x == y).all() for x, y in zip(oa, ob)))
    torch.manual_seed(1)
    v7f = PolicyValueNetV7()
    v7f.eval()
    freeze_query(v7f, "card_query")
    ca, cb = outputs(v7f, gh)[2], outputs(v7f, gswap)[2]
    check("V7: slots 8/9 exchange content contributions",
          exchange_ok(ca[8], ca[9], cb[8], cb[9]),
          f"A=({ca[8]:.5f},{ca[9]:.5f}) B=({cb[8]:.5f},{cb[9]:.5f})")
    check("V7: other card slots unaffected",
          float(np.abs(np.delete(ca, [8, 9]) - np.delete(cb, [8, 9])).max()) < 1e-6)
    torch.manual_seed(1)
    v6f = PolicyValueNetV6()  # new-width V6: same obs, tail-bias card head
    v6f.eval()
    freeze_query(v6f, "card_query")
    ca, cb = outputs(v6f, gh)[2], outputs(v6f, gswap)[2]
    check("V6 control: card logits exactly invariant (tail is content-free)",
          float(np.abs(ca - cb).max()) == 0.0)

    print("6. pack-slot content binding (frozen ent_query)")
    gp = clone(gs0)
    ca_, cb_ = gp["deck"].pop(), gp["deck"].pop()
    while ca_.base.rank.value == cb_.base.rank.value:
        cb_ = gp["deck"].pop()
    gp["pack_cards"] = [ca_, cb_]
    gq = clone(gp)
    gq["pack_cards"] = [gq["pack_cards"][1], gq["pack_cards"][0]]
    oa_ids, ob_ids = observe(gp).market_ids, observe(gq).market_ids
    check("both pack cards embed c_base (the V6 residual, proven)",
          bool((oa_ids == ob_ids).all()) and oa_ids[0] != 0)
    fa, fb = obs_vector(gp), obs_vector(gq)
    r0 = slice(MARKET_FEAT_OFFSET, MARKET_FEAT_OFFSET + MARKET_FEAT_DIM)
    r1 = slice(MARKET_FEAT_OFFSET + MARKET_FEAT_DIM,
               MARKET_FEAT_OFFSET + 2 * MARKET_FEAT_DIM)
    mask = np.ones(MARKET_FEAT_DIM, dtype=bool)
    mask[1 + 12] = False  # position_in_hand is per-SLOT, stays put on a swap
    check("market rows 0/1 carry the cards and swap with them",
          bool((fa[r0][mask] == fb[r1][mask]).all()
               and (fa[r1][mask] == fb[r0][mask]).all()
               and not (fa[r0][mask] == fa[r1][mask]).all()))
    torch.manual_seed(1)
    v7f = PolicyValueNetV7()
    v7f.eval()
    freeze_query(v7f, "ent_query")
    i, j = ENT_OFF_MARKET + 0, ENT_OFF_MARKET + 1
    ea, eb = outputs(v7f, gp)[1], outputs(v7f, gq)[1]
    check("V7: pack slots exchange content contributions",
          exchange_ok(ea[i], ea[j], eb[i], eb[j]),
          f"A=({ea[i]:.5f},{ea[j]:.5f}) B=({eb[i]:.5f},{eb[j]:.5f})")
    check("V7: other entity slots unaffected",
          float(np.abs(np.delete(ea, [i, j]) - np.delete(eb, [i, j])).max()) < 1e-6)
    torch.manual_seed(1)
    v6f = PolicyValueNetV6()
    v6f.eval()
    freeze_query(v6f, "ent_query")
    ea, eb = outputs(v6f, gp)[1], outputs(v6f, gq)[1]
    check("V6 control: entity logits exactly invariant (embed-only content)",
          float(np.abs(ea - eb).max()) == 0.0)

    print("7. ladder compatibility")
    if v13 is not None:
        t, e, cc, v = outputs(v13, gs0)
        check("pre-v14 checkpoint runs on the new obs (prefix slice)",
              np.isfinite(t).all() and np.isfinite(e).all()
              and np.isfinite(cc).all() and np.isfinite(v).all())

    if FAILURES:
        print(f"\nGATE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
