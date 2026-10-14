#!/usr/bin/env python3
"""Measurement: which half of classifier training holds the memory, and what a cache-loaded model costs.

exp_memory_phases.py: training adds ~760 MB at 146 intents and ~3.3 GB at 465, never released, while the weights are
~14 MB. Two hypotheses: (a) training leaves objects alive, (b) the allocator keeps training's high-water mark after
its temporaries are freed. (a) predicts a large footprint for a model loaded from its saved state in a fresh
process; (b) predicts a small one. Modes:
  train <manifest> <lang> <state.npz>   RSS after features, after the regression fit; saves the state (.npz + .json)
  load <state.npz>                      fresh process: RSS after loading the state (what a model-cache hit holds)

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_memory_train train skills/manifests/en-us-T3.json en-us /tmp/state.npz
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_memory_train load /tmp/state.npz
"""
from switchboard_eval.paths import ROOT
import gc
import json
import os
import sys

import numpy as np

from switchboard_eval.experiments.exp_memory_phases import peak_mb, rss_mb


def train(path, lang, out):
    from ovos_switchboard.core.classifier import CharEncoder, Classifier, LogRegHead, training_data
    from ovos_switchboard.core.device import Device
    from ovos_switchboard.core.schema import Schema
    harness = ROOT
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    rows = training_data(Device.from_definitions(doc["definitions"], lang),
                         Schema.load(os.path.join(harness, "schema/intents.yaml")))
    texts, labels = [t for t, _ in rows], [label for _, label in rows]
    gc.collect()
    print(f"rows {len(rows)}: rss {rss_mb()} MB")
    encoder = CharEncoder().fit(texts)
    features = encoder.encode(texts)
    gc.collect()
    print(f"features {features.shape}, nnz {features.nnz}: rss {rss_mb()} MB, peak {peak_mb()} MB")
    head = LogRegHead().fit(features, labels)
    gc.collect()
    print(f"regression fitted, {len(head.classes)} classes: rss {rss_mb()} MB, peak {peak_mb()} MB")
    arrays, meta = Classifier(encoder, head).state()
    np.savez(out, **arrays)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"state saved: {os.path.getsize(out) / 2**20:.1f} MB on disk")


def load(path):
    from ovos_switchboard.core.classifier import Classifier
    print(f"fresh process, imports: rss {rss_mb()} MB")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with open(path + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    model = Classifier.from_state(arrays, meta)
    gc.collect()
    print(f"model loaded from state ({len(model.classes)} classes): rss {rss_mb()} MB, peak {peak_mb()} MB")
    top = model.rank("what time is it")[0]
    print(f"rank works ({top[0]} {top[1]:.2f}): rss {rss_mb()} MB")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        load(sys.argv[2])
