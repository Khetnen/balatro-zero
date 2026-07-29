"""Generate joker descriptor features for the network's embedding table.

WHY: net.py identifies jokers by a bare learned ID embedding, so nothing
transfers between similar jokers -- each must be learned from its own
exposures. At ~20 joker offers per game and ~150 jokers under rarity
weighting, a given RARE appears about once every 18 games, so covering
them by data alone costs on the order of 1,000 games. A descriptor the
embedding can be initialised from (or concatenated with) lets an unseen
rare inherit from mechanically similar commons and cuts that sharply.

Two authoritative sources, no scraping:
  * the game's own display text, balatro-source/localization/en-us.lua,
    loaded with LuaJIT rather than regex (nested table literal). The
    colour markup is itself the signal: {X:mult,...} tags an X-mult
    joker, {C:chips} a chip joker, {C:attention} the trigger noun.
  * jackdaw's centers, which already carry rarity, cost, the game's own
    `effect` category (Blueprint's is "Copycat") and the compat flags.

Usage (from balatro-zero/):
    uv run --no-sync python scripts/gen_joker_features.py
Writes balatro_zero/data/joker_features.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

LUA_DUMP = r"""
local path = arg[1]
local L = assert(loadfile(path))()
local jokers = L.descriptions.Joker
local keys = {}
for k in pairs(jokers) do keys[#keys + 1] = k end
table.sort(keys)
for _, k in ipairs(keys) do
    local e = jokers[k]
    local parts = {}
    for _, line in ipairs(e.text or {}) do parts[#parts + 1] = line end
    io.write(k, "\t", e.name or "", "\t", table.concat(parts, " / "), "\n")
end
"""

# (feature name, regex) matched against text + effect + config, because no
# single source is complete: the display text carries the mechanics, the
# `effect` categorical covers jokers whose text is parameterised away
# (j_misprint's text is empty), and the config carries the literal hand
# type for the "+Mult if hand contains #2#" family, where the type is a
# placeholder rather than a word.
#
# Colour conventions are from an actual token census over all 150 jokers,
# NOT from guessing: flat mult is {C:mult} (28 uses) AND {C:red} (22) --
# missing the second one silently emptied the descriptor for the entire
# core common family (Joker/Jolly/Zany/Mad/Crazy/Droll/Half). Chips is
# {C:chips} (29) and {C:blue} (3). X-mult is {X:mult} (45) and {X:red}
# (5), both caught by the bare \{X: prefix.
#
# Ordered; this ordering IS the feature vector layout, so append only.
TEXT_FEATURES: list[tuple[str, str]] = [
    ("xmult",              r"\{X:"),
    ("plus_mult",          r"\{C:mult\}|\{C:red|t_mult|Type Mult|Random Mult"
                           r"|Hand Size Mult|played mult"),
    ("plus_chips",         r"\{C:chips\}|\{C:blue|t_chips|Chips"),
    ("money",              r"\{C:money\}|\$"),
    ("probabilistic",      r"\{C:green\}|chance"),
    ("scales",             r"Currently|increases|gains|for each|per "),
    ("creates_consumable", r"\{C:tarot\}|\{C:planet\}|\{C:spectral\}|create"),
    ("retrigger",          r"[Rr]etrigger"),
    ("copies",             r"[Cc]opies|[Cc]opy"),
    ("destroys",           r"[Dd]estroy"),
    ("on_scored",          r"when scored"),
    ("held_in_hand",       r"held in hand"),
    ("on_discard",         r"discard"),
    ("end_of_round",       r"end of round"),
    ("shop_related",       r"[Ss]hop|[Vv]oucher|[Rr]eroll|[Bb]ooster"),
    ("blind_related",      r"[Bb]lind"),
    ("suit_specific",      r"\{C:hearts\}|\{C:diamonds\}|\{C:spades\}|\{C:clubs\}|suit"),
    # \b after the bare letters matters: without it the J of
    # "{C:attention}Joker" reads as a Jack and every copy joker
    # (Blueprint, Brainstorm) is mislabelled rank-specific.
    ("rank_specific",      r"\{C:attention\}(?:Ace|King|Queen|Jack|Face|[0-9]|[AKQJ]\b)"),
    ("hand_type_specific", r"Pair|Straight|Flush|Full House|of a Kind|High Card"
                           r"|'type'|\"type\""),
    ("first_or_final",     r"[Ff]irst|[Ff]inal|[Ll]ast"),
    ("hand_size",          r"hand size"),
    ("consumable_slots",   r"consumable|[Jj]oker [Ss]lot"),
    # Rule modifiers change what the game COUNTS rather than adding to the
    # score: Pareidolia (all cards are face cards), Splash (every played
    # card scores), Four Fingers, Shortcut, Smeared. They are exactly the
    # family engine bug #27 was about -- hand detection ran without them
    # -- so a policy that cannot see them plans hands the engine will not
    # score the way it expects.
    ("rule_modifier",      r"considered|counts in scoring|only requires"
                           r"|with gaps|same suit|All cards"),
    ("adds_playing_card",  r"playing card|add a random|[Ss]eal"),
]


def find_luajit() -> str:
    for cand in (
        os.environ.get("LUAJIT"),
        shutil.which("luajit"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\LuaJIT\bin\luajit.exe"),
    ):
        if cand and Path(cand).exists():
            return cand
        if cand and shutil.which(cand):
            return cand
    raise SystemExit("luajit not found; set $LUAJIT")


def dump_text(loc: Path) -> dict[str, tuple[str, str]]:
    with tempfile.NamedTemporaryFile("w", suffix=".lua", delete=False) as fh:
        fh.write(LUA_DUMP)
        script = fh.name
    try:
        out = subprocess.run(
            [find_luajit(), script, str(loc)],
            capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout
    finally:
        os.unlink(script)
    rows = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            rows[parts[0]] = (parts[1], parts[2])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="../balatro-source/localization/en-us.lua")
    ap.add_argument("--out", default="balatro_zero/data/joker_features.json")
    args = ap.parse_args()

    loc = Path(args.source)
    if not loc.exists():
        raise SystemExit(f"localization not found: {loc}")
    texts = dump_text(loc)

    from jackdaw.engine.card import _resolve_center

    feature_names = (
        ["rarity_common", "rarity_uncommon", "rarity_rare", "rarity_legendary",
         "cost_norm", "blueprint_compat", "eternal_compat", "perishable_compat"]
        + [n for n, _ in TEXT_FEATURES]
    )

    entries: dict[str, dict] = {}
    missing_center = []
    for key, (name, text) in sorted(texts.items()):
        try:
            c = _resolve_center(key)
        except Exception:  # noqa: BLE001
            missing_center.append(key)
            continue
        rarity = int(c.get("rarity", 1) or 1)
        vec = [
            1.0 if rarity == 1 else 0.0,
            1.0 if rarity == 2 else 0.0,
            1.0 if rarity == 3 else 0.0,
            1.0 if rarity == 4 else 0.0,
            min(float(c.get("cost", 0) or 0) / 20.0, 1.0),
            1.0 if c.get("blueprint_compat") else 0.0,
            1.0 if c.get("eternal_compat") else 0.0,
            1.0 if c.get("perishable_compat") else 0.0,
        ]
        blob = f"{text} || {c.get('effect') or ''} || {c.get('config')!r}"
        vec += [1.0 if re.search(pat, blob) else 0.0 for _, pat in TEXT_FEATURES]
        entries[key] = {
            "name": name,
            "text": text,
            "rarity": rarity,
            "cost": c.get("cost"),
            "effect": c.get("effect"),
            "features": vec,
        }

    payload = {
        "feature_names": feature_names,
        "dim": len(feature_names),
        "n": len(entries),
        "source": "balatro-source/localization/en-us.lua + jackdaw centers",
        "jokers": entries,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"{len(entries)} jokers, {len(feature_names)}-dim descriptor -> {out}")
    if missing_center:
        print(f"no center for {len(missing_center)}: {missing_center[:8]}")

    # QA: a joker whose descriptor is entirely structural (rarity/cost/
    # compat) carries no mechanical signal and is exactly the silent
    # failure that hid the {C:red} miss. Fail loudly rather than ship it.
    n_struct = 8
    dead = [k for k, e in entries.items() if not any(e["features"][n_struct:])]
    if dead:
        raise SystemExit(
            f"{len(dead)} jokers have no mechanical features -- the pattern "
            f"table is missing a convention: {dead}"
        )
    print("QA: every joker has at least one mechanical feature")
    # sanity: the features must actually separate mechanically distinct jokers
    for k in ("j_blueprint", "j_baron", "j_photograph", "j_obelisk", "j_half"):
        if k in entries:
            e = entries[k]
            on = [n for n, v in zip(feature_names, e["features"]) if v and not n.startswith("rarity")]
            print(f"  {k:<16} {e['effect'] or '-':<12} {on}")


if __name__ == "__main__":
    main()
