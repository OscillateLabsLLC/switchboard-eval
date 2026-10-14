#!/usr/bin/env python3
"""Diagnostic: where do exp_lang_scale's held-out wrong answers land? The same holdout, tallied by (stage, predicted
intent), plus which gold intents the application launcher's catch-all swallows.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_scale_errors en-us-T3.json
"""
from switchboard_eval.paths import ROOT
import collections
import json
import os
import sys

sys.path.insert(0, ROOT)
from switchboard_eval.experiments import exp_lang_scale as x  # noqa: E402
from ovos_switchboard.core.schema import Schema  # noqa: E402

harness = ROOT
schema = Schema.load(os.path.join(harness, "schema/intents.yaml"))
doc = json.load(open(os.path.join(harness, "skills", "manifests", sys.argv[1])))
train_doc, held = x.split(doc)
device, exact, classifier, _ = x.build(train_doc, "en-us", schema)
wrong = collections.Counter()
victims = collections.Counter()
samples = {}
total = 0
for label, line in held:
    gold = schema.canon_or_unmapped(label)
    for e in list(dict.fromkeys(x.expansions(line)))[:x.EXPANSIONS_PER_LINE]:
        utt = x.SLOT_TOKEN.sub(x.SLOT_FILLER, e).lower()
        got, stage = x.decide_exact(exact, classifier, utt), "exact"
        if not got:
            top, conf = classifier.rank(utt)[0]
            got, stage = (top if conf >= x.THRESHOLD else None), "classifier"
        total += 1
        if got and got != gold:
            wrong[(stage, got)] += 1
            if got.endswith(':launch'):
                victims[gold] += 1
                samples.setdefault(gold, utt)
print(sys.argv[1], "utterances", total, "wrong", sum(wrong.values()))
for (stage, got), n in wrong.most_common(12):
    print(f"  {n:4d} {stage:10s} {got}")
print('launcher victims by gold intent:')
for g, n in victims.most_common(15):
    print(f'  {n:4d} {g}  e.g. {samples[g]!r}')
