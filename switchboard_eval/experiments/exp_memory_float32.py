#!/usr/bin/env python3
"""Measurement: does float32 cut the classifier's training PEAK? (2026-10-14)

Prediction before running: little. The fit's big allocation is L-BFGS's correction history (10 pairs of
parameter-sized vectors), held by SciPy's L-BFGS-B in float64 whatever the input dtype; float32 features shrink
only the feature matrix and the loss/gradient temporaries. One dtype per fresh process; peak = ru_maxrss of this
process (training in process, nothing else running).

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_memory_float32 {float64|float32} <manifest> <lang>
"""
from switchboard_eval.paths import ROOT
import json
import os
import sys
import time

import numpy as np

from switchboard_eval.experiments.exp_memory_phases import peak_mb


def main(dtype, manifest, lang):
    from ovos_switchboard.core import classifier as clf
    from ovos_switchboard.core.device import Device
    from ovos_switchboard.core.schema import Schema
    if dtype == "float32":
        original = clf._vectorizers

        def float32_vectorizers():
            vectorizers = original()
            for v in vectorizers:
                v.set_params(dtype=np.float32)
            return vectorizers
        clf._vectorizers = float32_vectorizers
    harness = ROOT
    with open(manifest, encoding="utf-8") as f:
        device = Device.from_definitions(json.load(f)["definitions"], lang)
    rows = clf.training_data(device, Schema.load(os.path.join(harness, "schema/intents.yaml")))
    t0 = time.perf_counter()
    model = clf.Classifier.train(rows)
    print(json.dumps({"dtype": dtype, "manifest": os.path.basename(manifest), "classes": len(model.classes),
                      "train_s": round(time.perf_counter() - t0, 1), "peak_mb": peak_mb(),
                      "coef_dtype": str(model.head.m.coef_.dtype), "n_iter": int(model.head.m.n_iter_[0])}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
