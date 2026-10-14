#!/usr/bin/env python3
"""Experiment: a lower catch-all yield (ExactOptions.catch_all_yield) against the launcher catch-all.

exp_scale_errors.py: the largest wrong-answer source at every scale tier is ONE catch-all template, the OVOS
application launcher's "(launch|start|open|run|fire up) {application}", catching other skills' phrases (113 wrong at
146 intents, 304 at 465). A catch-all yields only when the classifier ranks another intent first at >= 0.8, and
that gets rarer as classes dilute confidence. exp_guard_rules.py (2026-10-13) chose 0.8 from {0.8, 0.9, 0.95};
nothing lower was tried. Here: 0.8 (shipped) down to 0.01 (yield whenever another intent ranks first).

Readouts, NO LLM (the deterministic path; a yielded row the classifier passes is a pass):
  bar       dev and test, each device's router on its own traffic (magic / own / FA / high-risk executions)
  negatives non-command speech the exact stage dispatches (exact_audit corpora), per device
  holdout   exp_lang_scale's template hold-out at 146 (en-us) and 465 (en-us-T3) intents: right / wrong / pass

Selection rule, fixed before results: among values no worse than 0.8 on dev magic, dev own-device accuracy and dev
FA on BOTH devices, with zero high-risk executions, the one with the lowest T3 hold-out wrong %; ties -> fewer
non-command dispatches, then the higher value (closer to shipped). Test is reported for 0.8 and the chosen value.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_yield_sweep <switchboard>/scripts
"""
from switchboard_eval.paths import ROOT
import json
import os
import sys

from switchboard_eval.experiments import exp_lang_scale as scale
from switchboard_eval.experiments.exact_audit import negative_corpora


YIELDS = (0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.01)
SHIPPED = 0.8
HOLDOUT_MANIFESTS = ("en-us.json", "en-us-T3.json")


def set_yield(router, tau):
    from ovos_switchboard.core.options import ExactOptions
    router.exact.options = ExactOptions(catch_all_yield=tau)


def bar(cp, routers, split):
    """{tau: {device: metrics}} for one split."""
    gold, slices, stock = cp.setup(split)
    _, _, _, ctx = cp.load_inputs(split)
    out = {}
    for tau in YIELDS:
        out[tau] = {}
        for device, router in routers.items():
            set_yield(router, tau)
            preds = {rid: router.route(r["utterance"], {k: ctx.get(rid, cp.COLD)[k] for k in ("gap_s", "echo_run")})
                     for rid, r in gold.items()}
            out[tau][device] = cp.evaluate(gold, slices, stock, preds)
    return out


def negatives(harness, routers):
    """{tau: {device: count of non-command utterances the exact stage dispatches}}."""
    corpora = {d: {u.lower().strip() for c in negative_corpora(harness, d).values() for u in c if u.strip()}
               for d in routers}
    out = {}
    for tau in YIELDS:
        out[tau] = {}
        for device, router in routers.items():
            set_yield(router, tau)
            out[tau][device] = sum(1 for u in corpora[device] if router.exact.decide(u).intent)
    return out


def holdout(harness, schema):
    """{manifest: {tau: holdout metrics}}: one build per manifest, the yield swept on it."""
    out = {}
    for name in HOLDOUT_MANIFESTS:
        with open(os.path.join(harness, "skills", "manifests", name), encoding="utf-8") as f:
            doc = json.load(f)
        train_doc, held = scale.split(doc)
        _, exact, classifier, _ = scale.build(train_doc, "en-us", schema)
        out[name] = {}
        for tau in YIELDS:
            scale.CATCH_ALL_YIELD = tau
            out[name][tau] = score_holdout(held, exact, classifier, schema)
    return out


def score_holdout(held, exact, classifier, schema):
    right = wrong = passed = 0
    for label, line in held:
        gold = schema.canon_or_unmapped(label)
        for e in list(dict.fromkeys(scale.expansions(line)))[:scale.EXPANSIONS_PER_LINE]:
            utt = scale.SLOT_TOKEN.sub(scale.SLOT_FILLER, e).lower()
            got = scale.decide_exact(exact, classifier, utt)
            if not got:
                top, conf = classifier.rank(utt)[0]
                got = top if conf >= scale.THRESHOLD else None
            passed += got is None
            right += got == gold
            wrong += got is not None and got != gold
    n = right + wrong + passed
    return {"right": round(100 * right / n, 1), "wrong": round(100 * wrong / n, 1), "pass": round(100 * passed / n, 1)}


def choose(dev, negs, hold):
    base = dev[SHIPPED]
    ok = [t for t in YIELDS if all(dev[t][d]["magic"] >= base[d]["magic"] and dev[t][d][d] >= base[d][d]
                                   and dev[t][d]["fa"] <= base[d]["fa"] and dev[t][d]["hx"] == 0 for d in dev[t])]
    return min(ok, key=lambda t: (hold["en-us-T3.json"][t]["wrong"], sum(negs[t].values()), -t))


def fmt(m, device):
    return f"{m['magic']:5.1f}/{m[device]:5.1f}/{m['fa']:4.1f}/{m['hx']}"


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp
    from ovos_switchboard.core.schema import Schema

    harness = ROOT
    os.chdir(harness)
    schema = Schema.load(os.path.join(harness, "schema/intents.yaml"))
    routers = {}
    for device in ("mac", "hub"):
        routers[device] = ch.build_router(harness, device, "active")
        routers[device].llm = None
    negs = negatives(harness, routers)
    dev = bar(cp, routers, "dev")
    hold = holdout(harness, schema)
    print("yield  DEV mac magic/own/FA/hx   DEV hub magic/own/FA/hx   non-cmd mac/hub   "
          "holdout 146 r/w/p      holdout 465 r/w/p")
    for t in YIELDS:
        h1, h3 = hold["en-us.json"][t], hold["en-us-T3.json"][t]
        print(f"{t:<6} {fmt(dev[t]['mac'], 'mac'):24} {fmt(dev[t]['hub'], 'hub'):24} "
              f"{negs[t]['mac']:3d}/{negs[t]['hub']:<3d}          "
              f"{h1['right']:4.1f}/{h1['wrong']:4.1f}/{h1['pass']:4.1f}       "
              f"{h3['right']:4.1f}/{h3['wrong']:4.1f}/{h3['pass']:4.1f}")
    chosen = choose(dev, negs, hold)
    print(f"\nchosen on dev + hold-out: {chosen}")
    test = bar(cp, routers, "test")
    for t in dict.fromkeys((SHIPPED, chosen)):
        print(f"TEST {t:<6} mac {fmt(test[t]['mac'], 'mac')}   hub {fmt(test[t]['hub'], 'hub')}")
    with open(os.path.join(harness, "preds", "exp_yield_sweep.json"), "w", encoding="utf-8") as f:
        json.dump({"dev": dev, "test": test, "negatives": negs, "holdout": hold, "chosen": chosen}, f, indent=1,
                  default=str)


if __name__ == "__main__":
    main(sys.argv[1])
