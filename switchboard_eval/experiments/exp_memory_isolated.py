#!/usr/bin/env python3
"""Measurement: classifier training in a child process (switchboard adapters/isolated_training.py) vs in process.

exp_memory_phases.py: in-process training leaves ~0.9 GB (146 intents) / ~3.5 GB (465) resident in the host, while
a cache-loaded model costs ~207 MB (exp_memory_train.py). Decision (2026-10-14): train in a child process. Prediction:
the parent's RSS after training ~ the cache-loaded figure; the child's peak ~ the old in-process peak (the peak is a
separate problem). One mode per fresh process so the two can't contaminate each other.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_memory_isolated {inprocess|isolated} <manifest> <lang>
"""
from switchboard_eval.paths import ROOT
import gc
import json
import os
import resource
import sys

from switchboard_eval.experiments.exp_memory_phases import rss_mb


def child_peak_mb():
    """Largest RSS any finished child reached (macOS reports bytes, Linux KiB)."""
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(peak / (2**20 if sys.platform == "darwin" else 2**10))


def main(mode, manifest, lang):
    from ovos_switchboard.core.classifier import Classifier, training_data
    from ovos_switchboard.core.device import Device
    from ovos_switchboard.core.schema import Schema
    harness = ROOT
    with open(manifest, encoding="utf-8") as f:
        device = Device.from_definitions(json.load(f)["definitions"], lang)
    rows = training_data(device, Schema.load(os.path.join(harness, "schema/intents.yaml")))
    gc.collect()
    before = rss_mb()
    if mode == "isolated":
        from ovos_switchboard.adapters.isolated_training import train_isolated
        model = train_isolated(rows)
    else:
        model = Classifier.train(rows)
    gc.collect()
    print(json.dumps({"mode": mode, "manifest": os.path.basename(manifest), "rows": len(rows),
                      "classes": len(model.classes), "rss_before_mb": before, "rss_after_mb": rss_mb(),
                      "child_peak_mb": child_peak_mb() if mode == "isolated" else None}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
