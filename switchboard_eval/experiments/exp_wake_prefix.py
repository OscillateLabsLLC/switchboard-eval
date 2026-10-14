#!/usr/bin/env python3
"""Experiment: strip leading wake phrases before exact matching, and the trailing-speech prefix stage.

Variables, each alone and then together, against the v5 router (both off):
  wake    strip a leading wake phrase before the exact (and prefix) match: Mac "hey mycroft", Hub "hey neon"
  prefix  PrefixStage after the classifier: slot-free templates of >= 3 literal words matched as a prefix
Uses the packaged Switchboard router built as scripts/check_harness.py builds it, cached LLM answers only. A
variant can only REMOVE LLM calls (it decides earlier), so no new model calls are needed.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_wake_prefix <switchboard>/scripts
"""
from switchboard_eval.paths import ROOT
import json
import os
import sys


WAKE = {"mac": ("hey_mycroft",), "hub": ("hey_neon",)}
PREFIX_MIN_LITERAL = 3


def configure(router, device, wake, prefix):
    from ovos_switchboard.core.stages import PrefixStage
    from ovos_switchboard.core.wake import wake_phrase_pattern
    pattern = wake_phrase_pattern(WAKE[device]) if wake else None
    router.exact.wake = pattern
    router.prefix = None
    if prefix:
        router.exact.exact.enable_prefix(PREFIX_MIN_LITERAL)
        router.prefix = PrefixStage(router.exact.exact, router.resolver, pattern)


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    sys.path.insert(0, ROOT)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp

    harness = ROOT
    os.chdir(harness)
    variants = {"v5": (False, False), "wake": (True, False), "prefix": (False, True), "both": (True, True)}
    flips = []
    for split in ("dev", "test"):
        gold, slices, stock = cp.setup(split)
        _, _, _, ctx = cp.load_inputs(split)
        for device in ("mac", "hub"):
            for route in ("always", "active"):
                router = ch.build_router(harness, device, route)
                preds = {}
                for name, (wake, prefix) in variants.items():
                    configure(router, device, wake, prefix)
                    preds[name] = {}
                    for rid, r in gold.items():
                        c = ctx.get(rid, cp.COLD)
                        preds[name][rid] = router.route(r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]})
                own = "hub" if device == "hub" else "mac"
                cells = []
                for name in variants:
                    m = cp.evaluate(gold, slices, stock, preds[name])
                    cells.append(f"{name} {m['magic']:5.1f}/{m[own]:5.1f}/{m['fa']:4.1f}/{m['hx']}")
                print(f"{split:4} {device} {route:6}  magic/own/FA/hx  " + "   ".join(cells))
                for name in ("wake", "prefix", "both"):
                    for rid, r in gold.items():
                        b, n = preds["v5"][rid].get("intent"), preds[name][rid].get("intent")
                        if b != n:
                            flips.append({"variant": name, "split": split, "device": device, "route": route,
                                          "id": rid, "utterance": r["utterance"], "category": r.get("category"),
                                          "gold": r["intent"], "before": b, "after": n,
                                          "after_stage": preds[name][rid].get("stage")})
    out = os.path.join(harness, "preds", "exp_wake_prefix_flips.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for fl in flips:
            f.write(json.dumps(fl) + "\n")
    print(f"\n{len(flips)} changed decisions -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
