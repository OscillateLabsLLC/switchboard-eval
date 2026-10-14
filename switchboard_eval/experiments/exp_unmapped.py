#!/usr/bin/env python3
"""Experiment: let the classifier dispatch confident picks of intents missing from schema/intents.yaml.

Single variable: ClassifierOptions.dispatch_unmapped (False = the measured v5 behavior, True = act on unmapped
top classes above the threshold instead of passing them to the LLM). Uses the packaged Switchboard router built
exactly as scripts/check_harness.py builds it, with cached LLM answers only (dispatching earlier only removes LLM
calls, so no new model calls are needed).

Gold labels are canonical intents, and an unmapped label has no canonical by definition, so the bar can never
credit an unmapped dispatch. Every changed decision is therefore listed for a by-hand verdict.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_unmapped <switchboard>/scripts
"""
from switchboard_eval.paths import ROOT
import dataclasses
import json
import os
import sys


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    sys.path.insert(0, ROOT)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp

    harness = ROOT
    os.chdir(harness)
    flips = []
    for split in ("dev", "test"):
        gold, slices, stock = cp.setup(split)
        _, _, _, ctx = cp.load_inputs(split)
        for device in ("mac", "hub"):
            for route in ("always", "active"):
                router = ch.build_router(harness, device, route)
                preds = {}
                for flag in (False, True):
                    router.classifier.options = dataclasses.replace(router.classifier.options, dispatch_unmapped=flag)
                    preds[flag] = {}
                    for rid, r in gold.items():
                        c = ctx.get(rid, cp.COLD)
                        preds[flag][rid] = router.route(r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]})
                base, new = (cp.evaluate(gold, slices, stock, preds[f]) for f in (False, True))
                own = "hub" if device == "hub" else "mac"
                print(f"{split:4} {device} {route:6}  magic {base['magic']:5.1f} -> {new['magic']:5.1f}   "
                      f"own {base[own]:5.1f} -> {new[own]:5.1f}   FA {base['fa']:5.1f} -> {new['fa']:5.1f}   "
                      f"hx {base['hx']} -> {new['hx']}")
                for rid, r in gold.items():
                    b, n = preds[False][rid].get("intent"), preds[True][rid].get("intent")
                    if b != n:
                        flips.append({"split": split, "device": device, "route": route, "id": rid,
                                      "utterance": r["utterance"], "category": r.get("category"),
                                      "source": r.get("source"), "gold": r["intent"], "before": b, "after": n,
                                      "before_stage": preds[False][rid].get("stage")})
    out = os.path.join(harness, "preds", "exp_unmapped_flips.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for fl in flips:
            f.write(json.dumps(fl) + "\n")
    print(f"\n{len(flips)} changed decisions -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
