"""Scout a seed's purchasable future for the LLM build planner.

Chip-cheat walkthrough of antes 1-4: record every shop's cards/vouchers,
peek every pack's contents (buy with cheated dollars, read, skip). The
walkthrough's stream state drifts from a real run after the first
divergent purchase, so treat contents beyond the first shops as
approximate — good enough for build planning ("this seed offers Hologram
early"), not for exact routing.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/seed_scout.py SEED [SEED ...] > scouts.txt
"""
from __future__ import annotations

import sys

from jackdaw.bridge.backend import RPCError, SimBackend


def keys_costs(items) -> list[str]:
    return [
        f"{getattr(c, 'center_key', '?')}(${getattr(c, 'cost', '?')})" for c in items
    ]


def scout(seed: str, max_ante: int = 4) -> str:
    sim = SimBackend()
    sim.handle("start", {"deck": "RED", "stake": "WHITE", "seed": seed})
    gs = sim._gs
    lines = [f"=== {seed} ==="]
    shop_n = 0
    for ante_i in range(1, max_ante + 1):
        for blind in ("Small", "Big", "Boss"):
            try:
                sim.handle("select", {})
                sim.handle("set", {"chips": 100000})
                sim.handle("play", {"cards": [0, 1, 2, 3, 4]})
                sim.handle("cash_out", {})
            except RPCError:
                lines.append(f"  [scout stuck at ante {ante_i} {blind}]")
                return "\n".join(lines)
            shop_n += 1
            entry = [f"shop{shop_n} (a{ante_i} after {blind}):"]
            cards = keys_costs(gs.get("shop_cards", []))
            vouchers = keys_costs(gs.get("shop_vouchers", []))
            if cards:
                entry.append("cards " + " ".join(cards))
            if vouchers:
                entry.append("voucher " + " ".join(vouchers))
            packs = []
            n_boost = len(gs.get("shop_boosters", []))
            for _ in range(n_boost):
                if not gs.get("shop_boosters"):
                    break
                bkey = getattr(gs["shop_boosters"][0], "center_key", "?")
                sim.handle("set", {"dollars": 100})
                try:
                    sim.handle("buy", {"pack": 0})
                except RPCError:
                    break
                contents = [getattr(c, "center_key", "?") for c in gs.get("pack_cards", [])]
                packs.append(f"{bkey}[{','.join(contents)}]")
                for method, params in (("skip", {}), ("pack", {"card": 0}), ("pack", {"cards": [0]})):
                    try:
                        sim.handle(method, params)
                        break
                    except RPCError:
                        continue
                else:
                    lines.append("  " + " | ".join(entry + packs) + " [stuck in pack]")
                    return "\n".join(lines)
            if packs:
                entry.append("packs " + " ".join(packs))
            lines.append("  " + " | ".join(entry))
            try:
                sim.handle("next_round", {})
            except RPCError:
                return "\n".join(lines)
    return "\n".join(lines)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        try:
            print(scout(s), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"=== {s} ===\n  [scout failed: {type(e).__name__}: {e}]", flush=True)
