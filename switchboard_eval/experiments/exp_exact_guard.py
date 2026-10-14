#!/usr/bin/env python3
"""Experiment: guard the exact layer against speech it doesn't own (findings of exact_audit.py).

Variables, alone and together, against v5:
  wordgap  template spaces need whitespace or a word edge, so "play {query}" no longer matches "playing by"
           (switchboard fix/exact-guard; v5's matcher is restored by setting templates.WORD_GAP back to r"\\s*")
  ml2      ExactLayer min_literal 2: one-word-plus-slot templates ("play {query}", "call {contact}",
           "timer [for] {duration}", "{schedkind} later") train the classifier but no longer claim the exact layer
Readouts: the bar per split and device (active route; LLM cache misses asked of the LLM server and cached), every changed
decision against v5 with its gold, and changed decisions on the 508 archived real Mac transcripts (LLM off, no gold).

Usage (from intent-eval, in .venv-bench, with the switchboard checkout on fix/exact-guard):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_exact_guard <switchboard>/scripts
"""
from switchboard_eval.paths import ROOT
import json
import os
import sys

from switchboard_eval.experiments.exp_mac_draft import build, describe


V5_GAP = r"\s*"
VARIANTS = {"v5": (False, 1), "wordgap": (True, 1), "ml2": (False, 2), "both": (True, 2)}


def router_for(ch, harness, device, schema, wordgap, min_literal, llm=True):
    """A router with the chosen matcher and exact-layer literal minimum."""
    from ovos_switchboard.core import templates
    templates.WORD_GAP = router_for.fixed_gap if wordgap else V5_GAP
    try:
        router = build(ch, harness, device, schema, llm=llm)
        layer = router.exact.exact
        layer.templates = [(label, templates.template_regex(line)) for label, line in layer._entries  # pylint: disable=W0212
                           if templates.literal_words(line) >= min_literal]
    finally:
        templates.WORD_GAP = router_for.fixed_gap
    return router


def bar(ch, cp, harness, schema):
    """Bar cells and changed decisions per split and device."""
    for split in ("dev", "test"):
        gold, slices, stock = cp.setup(split)
        _, _, _, ctx = cp.load_inputs(split)
        for device in ("mac", "hub"):
            preds, cells = {}, []
            for name, (wordgap, ml) in VARIANTS.items():
                router = router_for(ch, harness, device, schema, wordgap, ml)
                preds[name] = {rid: router.route(r["utterance"], {"gap_s": ctx.get(rid, cp.COLD)["gap_s"],
                                                                  "echo_run": ctx.get(rid, cp.COLD)["echo_run"]})
                               for rid, r in gold.items()}
                m = cp.evaluate(gold, slices, stock, preds[name])
                cells.append(f"{name} {m['magic']:5.1f}/{m[device]:5.1f}/{m['fa']:4.1f}/{m['hx']}")
            print(f"bar {split:4} {device}  magic/own/FA/hx  " + "   ".join(cells))
            for rid, r in gold.items():
                moved = {n: describe(preds[n][rid]) for n in VARIANTS if
                         preds[n][rid].get("intent") != preds["v5"][rid].get("intent")}
                if moved:
                    print(f"  {rid} {r['utterance']!r} gold {r['intent']} ({r.get('category')})\n"
                          f"      v5: {describe(preds['v5'][rid])}\n" +
                          "".join(f"      {n}: {d}\n" for n, d in moved.items()), end="")


def real(ch, harness, schema):
    """Changed decisions on the archived real Mac transcripts, LLM off."""
    with open(os.path.join(harness, "data", "mac", "context.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    routers = {n: router_for(ch, harness, "mac", schema, w, ml, llm=False) for n, (w, ml) in VARIANTS.items()}
    print(f"\nreal Mac transcripts: {len(rows)}")
    for r in rows:
        ctx = {"gap_s": r["gap_s"] if r["gap_s"] is not None else 1e9, "echo_run": r["echo_run"]}
        ds = {n: routers[n].route(r["text"], ctx) for n in VARIANTS}
        key = {n: (d.get("intent"), d.get("stage")) for n, d in ds.items()}
        moved = {n: describe(ds[n]) for n in VARIANTS if key[n] != key["v5"]}
        if moved:
            print(f"  {r['text']!r}\n      v5: {describe(ds['v5'])}\n" +
                  "".join(f"      {n}: {d}\n" for n, d in moved.items()), end="")


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp
    from ovos_switchboard.core import templates
    from ovos_switchboard.core.schema import Schema

    if templates.WORD_GAP == V5_GAP:
        raise SystemExit("needs the switchboard checkout on fix/exact-guard")
    router_for.fixed_gap = templates.WORD_GAP
    harness = ROOT
    os.chdir(harness)
    schema = Schema.load(os.path.join(harness, "schema/intents.yaml"))
    if "--bar-only" not in sys.argv:
        real(ch, harness, schema)
    bar(ch, cp, harness, schema)


if __name__ == "__main__":
    main(sys.argv[1])
