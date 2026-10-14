#!/usr/bin/env python3
"""Experiment: Switchboard's deterministic layers per language and at scale (no LLM, no labels needed).

Per language, from skills/manifests/<lang>.json (build_lang_manifest.py: the same installed skills for every language):
  build     device, exact layer and classifier build time, peak memory (one language per process)
  audit     every template expansion through the full exact layer: reaches its own intent / collision (exact_audit)
  holdout   every 5th template line of each intent (intents with >= 5 lines) is held out; the stack is built from the
            rest; up to 3 expansions per held-out line (slots filled with "Lugo") go through the shipped exact step
            (catch-all templates yield to a confident classifier) -> classifier.
            Scored at skill-intent level: right / wrong / pass (classifier below its 0.5 threshold).
  latency   exact + classifier per held-out utterance, p50 / p95
Synthetic rows (English-only) and dialog negatives are left out for every language, so languages are comparable; the
classifier still gets one English description per canonical intent (training_data), identical for all languages.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_lang_scale <lang> [--manifest path]   (appends to preds/exp_lang_scale.jsonl)
"""
from switchboard_eval.paths import ROOT
import json
import os
import resource
import statistics
import sys
import time

from ovos_switchboard.core.classifier import Classifier
from ovos_switchboard.core.device import Device
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.schema import Schema
from ovos_switchboard.core.templates import SLOT_TOKEN, expansions


HOLDOUT_EVERY = 5
EXPANSIONS_PER_LINE = 3
SLOT_FILLER = "lugo"
THRESHOLD = 0.5
CATCH_ALL_YIELD = 0.8  # ExactOptions.catch_all_yield


def split(doc):
    """(train_doc, held_out [(label, line)]): every HOLDOUT_EVERY-th template line of intents with enough lines."""
    train, held = {}, []
    for skill_id, defs in doc["definitions"].items():
        train[skill_id] = []
        for d in defs:
            lines = d["definition"]["samples"]
            keep = lines
            if len(lines) >= HOLDOUT_EVERY:
                keep = [l for i, l in enumerate(lines) if i % HOLDOUT_EVERY != HOLDOUT_EVERY - 1]
                held += [(f"{skill_id}:{d['intent_name']}", l) for i, l in enumerate(lines)
                         if i % HOLDOUT_EVERY == HOLDOUT_EVERY - 1]
            train[skill_id].append({**d, "definition": {**d["definition"], "samples": keep}})
    return {"definitions": train}, held


def build(doc, lang, schema):
    """(device, exact layer, classifier, timings)."""
    t0 = time.perf_counter()
    device = Device.from_definitions(doc["definitions"], lang)
    t1 = time.perf_counter()
    exact = ExactLayer(device, schema)
    t2 = time.perf_counter()
    classifier = Classifier.for_device(device, schema)
    t3 = time.perf_counter()
    return device, exact, classifier, {"device_s": round(t1 - t0, 2), "exact_s": round(t2 - t1, 2),
                                       "classifier_s": round(t3 - t2, 2), "examples": len(device.examples)}


def audit(device, exact):
    """Counts of template expansions reaching their own intent vs colliding."""
    own = collide = other = 0
    for label, line in device.template_entries():
        owner = exact.schema.canon_or_unmapped(label)
        for e in dict.fromkeys(expansions(line)):
            got = exact.route(SLOT_TOKEN.sub(SLOT_FILLER, e).lower())
            own += got == owner
            collide += got is None
            other += got not in (owner, None)
    return {"own": own, "collision": collide, "other": other}


def decide_exact(exact, classifier, utterance):
    """The exact step as shipped (ExactStage): a catch-all match yields to a confident classifier pick of another
    intent (ExactOptions.catch_all_yield)."""
    owner, catch_all = exact.match(utterance)
    if owner and catch_all:
        top, conf = classifier.rank(utterance)[0]
        if top != owner and conf >= CATCH_ALL_YIELD:
            return None
    return owner


def holdout(doc, lang, schema):
    """Accuracy and latency on held-out template lines."""
    train_doc, held = split(doc)
    device, exact, classifier, _ = build(train_doc, lang, schema)
    right = wrong = passed = by_exact = 0
    times = []
    for label, line in held:
        gold = schema.canon_or_unmapped(label)
        for e in list(dict.fromkeys(expansions(line)))[:EXPANSIONS_PER_LINE]:
            utt = SLOT_TOKEN.sub(SLOT_FILLER, e).lower()
            t0 = time.perf_counter()
            got = decide_exact(exact, classifier, utt)
            if got:
                by_exact += 1
            else:
                top, conf = classifier.rank(utt)[0]
                got = top if conf >= THRESHOLD else None
            times.append((time.perf_counter() - t0) * 1000)
            if got is None:
                passed += 1
            elif got == gold:
                right += 1
            else:
                wrong += 1
    n = right + wrong + passed
    q = statistics.quantiles(times, n=20) if len(times) > 20 else [0] * 19
    return {"held_lines": len(held), "utterances": n, "right_pct": round(100 * right / n, 1),
            "wrong_pct": round(100 * wrong / n, 1), "pass_pct": round(100 * passed / n, 1),
            "by_exact_pct": round(100 * by_exact / n, 1),
            "p50_ms": round(statistics.median(times), 2), "p95_ms": round(q[18], 2)}


def main(argv):
    lang = argv[0].lower()
    harness = ROOT
    path = argv[argv.index("--manifest") + 1] if "--manifest" in argv else os.path.join(harness, "skills", "manifests",
                                                                                         f"{lang}.json")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    schema = Schema.load(os.path.join(harness, "schema/intents.yaml"))
    device, exact, _, timings = build(doc, lang, schema)
    result = {"lang": lang, "manifest": os.path.basename(path),
              "intents": sum(len(v) for v in doc["definitions"].values()),
              "template_lines": sum(len(d["definition"]["samples"]) for v in doc["definitions"].values() for d in v),
              **timings, "audit": audit(device, exact), "holdout": holdout(doc, lang, schema),
              "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)}
    print(json.dumps(result))
    with open(os.path.join(harness, "preds", "exp_lang_scale.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
