#!/usr/bin/env python3
"""Experiment: exact-layer collision priority (Daniel McKnight's idea, 2026-10-14).

When templates of 2+ intents match, the shipped exact layer passes to the classifier. ExactOptions.collision_priority
"core" / "user" lets the one colliding intent whose skills are all core (or all not core) win instead (core =
.openvoiceos / .neongeckocom suffixes and "ocp"). Measured: every device x split, LLM off and with the 14B (cache
first), shipped default "none" vs "core" vs "user" vs skill_priority [Home Assistant] and [easter eggs, Home
Assistant]; the bar (serves-aware scoring) and every changed row.

Usage (from intent-eval, in .venv-bench, switchboard checkout with collision_priority):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_collision_priority <switchboard>/scripts
"""
from switchboard_eval.paths import ROOT
import os
import sys

from switchboard_eval.harness import score
from switchboard_eval.experiments.exp_float32 import router_for
from switchboard_eval.harness.serves import equivalences

HA = "skill-homeassistant.oscillatelabsllc"
# also an ordered skill_priority list: "if on this list, route there"
PRIORITIES = {"none": {}, "core": {"collision_priority": "core"}, "user": {"collision_priority": "user"},
              "list": {"skill_priority": (HA,)}, "eggs+": {"skill_priority": ("skill-easter-eggs.openvoiceos", HA)}}


def run(cp, router, split, priority):
    from ovos_switchboard.core.options import ExactOptions
    router.exact.exact.options = ExactOptions(**PRIORITIES[priority])
    gold, _, _ = cp.setup(split)
    _, _, _, ctx = cp.load_inputs(split)
    return {rid: router.route(r["utterance"], {k: ctx.get(rid, cp.COLD)[k] for k in ("gap_s", "echo_run")})
            for rid, r in gold.items()}


def bar_line(cp, split, device, preds):
    gold, s, stock = cp.setup(split)
    served = equivalences(device)
    pct = lambda rows, ok: 100 * sum(ok(preds[r["id"]], r) for r in rows) / len(rows)
    right = lambda p, r: score.correct(r, p, served)
    return (f"magic {pct(s['magic'], right):5.1f}  own {pct(s[device], right):5.1f}  "
            f"FA {pct(s['neg'], lambda p, r: p.get('intent') is not None):4.1f}")


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp

    harness = ROOT
    os.chdir(harness)
    for llm in (False, True):
        for device in ("mac", "hub"):
            router = router_for(ch, harness, device, "active")
            if not llm:
                router.llm = None
            for split in ("dev", "test"):
                gold, _, _ = cp.setup(split)
                preds = {p: run(cp, router, split, p) for p in PRIORITIES}
                for p in PRIORITIES:
                    print(f"{'14B' if llm else 'noLLM'} {device} {split} {p:5} {bar_line(cp, split, device, preds[p])}")
                for p in list(PRIORITIES)[1:]:
                    for rid, g in gold.items():
                        a, b = preds["none"][rid].get("intent"), preds[p][rid].get("intent")
                        if a != b:
                            print(f"    {p}: {g['utterance']!r} gold {g['intent']}: {a} -> {b}")
                sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1])
