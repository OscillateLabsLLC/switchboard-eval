#!/usr/bin/env python3
"""Measurement: Switchboard's memory by build phase, for one device build (what a live runtime holds).

exp_lang_scale.py's peak RSS covers two full builds plus training in one process, and the classifier's weights turned
out to be ~14 MB (exp_classifier_memory.py), so the peak is not the model. This builds the stack once, in the
runtime's order, and reports current RSS (ps) after each phase and the process peak, then drops the training data and
measures what stays resident.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_memory_phases <manifest.json> <lang>   (appends preds/exp_memory_phases.jsonl)
"""
from switchboard_eval.paths import ROOT
import gc
import json
import os
import resource
import subprocess
import sys
import time


def rss_mb():
    """Current resident memory of this process, MB."""
    return round(int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 1024, 1)


def peak_mb():
    """Peak resident memory so far, MB (macOS reports bytes)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)


def main(path, lang):
    phases = [("start", rss_mb(), peak_mb())]

    def mark(name):
        gc.collect()
        phases.append((name, rss_mb(), peak_mb()))

    from ovos_switchboard.core.classifier import Classifier, training_data
    from ovos_switchboard.core.device import Device
    from ovos_switchboard.core.exact import ExactLayer
    from ovos_switchboard.core.labels import LabelSelector
    from ovos_switchboard.core.schema import Schema
    from ovos_switchboard.core.slots import SlotExtractor
    from ovos_switchboard.core.stages import LabelResolver
    mark("imports (sklearn, numpy, switchboard)")
    harness = ROOT
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    schema = Schema.load(os.path.join(harness, "schema/intents.yaml"))
    mark("manifest + schema loaded")
    t0 = time.perf_counter()
    device = Device.from_definitions(doc["definitions"], lang)
    mark(f"device: {len(device.examples)} training phrases")
    resolver = LabelResolver(LabelSelector(device, schema), SlotExtractor(device, schema))
    mark("label resolver (selector + slot extractor)")
    exact = ExactLayer(device, schema)
    mark(f"exact layer: {len(exact.templates)} templates")
    rows = training_data(device, schema)
    mark(f"training rows: {len(rows)}")
    model = Classifier.train(rows)
    build_s = round(time.perf_counter() - t0, 1)
    mark("classifier trained")
    del rows
    mark("training rows released (steady state)")
    held = (resolver, exact, model)  # keep the stack alive through the last measurement
    result = {"manifest": os.path.basename(path), "lang": lang, "build_s": build_s,
              "phases": [{"phase": n, "rss_mb": r, "peak_mb": p} for n, r, p in phases],
              "classes": len(held[2].classes)}
    for p in result["phases"]:
        print(f"{p['phase']:45} rss {p['rss_mb']:8.1f} MB   peak {p['peak_mb']:8.1f} MB")
    with open(os.path.join(harness, "preds", "exp_memory_phases.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
