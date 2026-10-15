#!/usr/bin/env python3
"""Experiment: ExactOptions.collision_resolution "specific", the most specific matching template wins a collision.

When templates of 2+ intents match (and no priority list decides), the shipped exact layer hands the phrase to the
classifier; "specific" lets the intent whose matching template has the most literal words win, if exactly one does.
Found by reading the 44 OVOS golden utterances (en-US) Switchboard left unanswered without an LLM, so en-US gold is
the set that motivated it; the verdict comes from data it wasn't designed on.

Decided before the results:
  adopt as an option if (1) no bar criterion flips PASS -> FAIL on dev or test, either device, LLM off or on,
  (2) no significant loss on real speaker or desktop requests (paired McNemar p < 0.05 against the default),
  (3) clean gold (ca-ES, gl-ES, fr-FR) is not worse; it can become the default only if it also helps there.

Part 1, this harness: both devices, active route, LLM off and on (cache first), default vs specific.
Part 2, OVOS golden utterances (data/ovos_gold/): en-US on the desktop's router; ca-ES, gl-ES and fr-FR on routers
built from their language manifests (skills/manifests/) as exp_galician builds gl-es; LLM off; skill-intent level.
Usage (from the repo root): PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_specific \\
    <switchboard>/scripts
"""
import json
import os
import sys

from switchboard_eval.experiments.exp_float32 import router_for
from switchboard_eval.experiments.exp_ovos_gold import gold_rows
from switchboard_eval.harness import score
from switchboard_eval.harness.serves import equivalences
from switchboard_eval.paths import ROOT

VARIANTS = {"classifier": {"collision_resolution": "classifier"}, "specific": {"collision_resolution": "specific"}}
GOLD_LANGS = {"ca-ES": "ca-es", "gl-ES": "gl-es", "fr-FR": "fr-fr"}


def set_variant(router, variant):
    from ovos_switchboard.core.options import ExactOptions
    router.exact.exact.options = ExactOptions(**VARIANTS[variant])


def harness_part(ch, cp):
    for llm_on in (False, True):
        for device in ("mac", "hub"):
            router = router_for(ch, ROOT, device, "active")
            if not llm_on:
                router.llm = None
            for split in ("dev", "test"):
                gold, slices, _ = cp.setup(split)
                _, _, _, ctx = cp.load_inputs(split)
                preds = {}
                for variant in VARIANTS:
                    set_variant(router, variant)
                    preds[variant] = {rid: router.route(r["utterance"], {k: ctx.get(rid, cp.COLD)[k]
                                                                       for k in ("gap_s", "echo_run")})
                                      for rid, r in gold.items()}
                tag = f"{'14B' if llm_on else 'noLLM'} {device} {split}"
                served = equivalences(device)
                for variant, p in preds.items():
                    pct = lambda rows, ok: 100 * sum(ok(p[r["id"]], r) for r in rows) / len(rows)
                    right = lambda q, r: score.correct(r, q, served)
                    print(f"{tag} {variant:10} magic {pct(slices['magic'], right):5.1f}  own {pct(slices[device], right):5.1f}"
                          f"  FA {pct(slices['neg'], lambda q, r: q.get('intent') is not None):4.1f}", flush=True)
                own = {r["id"]: r for r in slices[device]}
                wins, losses, pval = score.mcnemar(own, preds["specific"], preds["classifier"], served, served)
                print(f"    real {device} specific vs default: {wins}+ / {losses}- p={pval:.3g}")
                for rid, g in gold.items():
                    a, b = preds["classifier"][rid].get("intent"), preds["specific"][rid].get("intent")
                    if a != b:
                        print(f"    {g['utterance']!r} gold {g['intent']}: {a} -> {b}")


def language_router(lang_file):
    from ovos_switchboard.core.device import Device, harvest_dialog_negatives
    from ovos_switchboard.core.router import Router
    from ovos_switchboard.core.schema import Schema
    from ovos_switchboard.core.tools import build_tools
    from switchboard_eval.experiments.exp_galician import dialog_lines
    with open(os.path.join(ROOT, "skills", "manifests", f"{lang_file}.json"), encoding="utf-8") as f:
        doc = json.load(f)
    device = Device.from_definitions(doc["definitions"], lang_file)
    labels = {f"{s}:{d['intent_name']}" for s, defs in doc["definitions"].items() for d in defs}
    defs = doc["definitions"]
    negatives = harvest_dialog_negatives(dialog_lines(lang_file), defs, lang_file)
    schema = Schema.load(os.path.join(ROOT, "schema", "intents.yaml"))
    return Router(device, schema, build_tools(defs, lang_file), synthetic=negatives), labels


def gold_score(router, rows):
    cold = {"gap_s": None, "echo_run": 0}
    out = {}
    for variant in VARIANTS:
        set_variant(router, variant)
        preds = [router.route(r["utterance"], cold) for r in rows]
        labels = [p.get("label") if p.get("intent") else None for p in preds]
        right = sum(lab == r["label"] for lab, r in zip(labels, rows))
        none = sum(lab is None for lab in labels)
        out[variant] = (right, len(rows) - right - none, none)
    return out


def gold_part(ch):
    router = router_for(ch, ROOT, "mac", "active")
    router.llm = None
    sets = [("en-US (motivating)", router, gold_rows("mac"))]
    for lang, lang_file in GOLD_LANGS.items():
        lrouter, labels = language_router(lang_file)
        with open(os.path.join(ROOT, "data", "ovos_gold", f"{lang}.jsonl"), encoding="utf-8") as f:
            rows = [r for r in map(json.loads, f) if r["label"] in labels]
        sets.append((f"{lang} (clean)", lrouter, rows))
    for name, r, rows in sets:
        res = gold_score(r, rows)
        print(f"gold {name:20} n={len(rows):4}  " + "  |  ".join(
            f"{v}: {100 * a / len(rows):5.1f}% right {100 * b / len(rows):4.1f}% wrong {100 * c / len(rows):4.1f}% none"
            for v, (a, b, c) in res.items()), flush=True)


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    import check_harness as ch  # pylint: disable=import-error  # Switchboard's scripts
    from switchboard_eval.harness import context_policy as cp
    os.chdir(ROOT)
    gold_part(ch)
    harness_part(ch, cp)


if __name__ == "__main__":
    main(sys.argv[1])
