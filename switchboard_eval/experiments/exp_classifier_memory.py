#!/usr/bin/env python3
"""Experiment: where the classifier's memory goes, and what tightening it costs in accuracy.

Shipped: char_wb 2-5-grams (min_df=1) + word 1-2-grams TF-IDF, logistic regression with a dense float64
classes x features weight matrix. Variants (vectorizer changes retrain; storage changes reuse the shipped model):
  shipped   as is
  min_df2   n-grams seen in fewer than 2 training rows are dropped (both vectorizers)
  cap100k   at most 100k features per vectorizer (most frequent kept)
  float32   shipped model, weights stored as float32
  prune     shipped model, weights with |w| < 1e-3 zeroed and stored sparse (float32)
Per variant: features, classes, weight bytes, model bytes (state() arrays), train seconds, and the template hold-out
score of exp_lang_scale.py (exact step as shipped -> classifier). Memory is measured on the objects themselves, so
the number is the model's own footprint, not a process peak.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_classifier_memory <manifest.json> <lang>
"""
from switchboard_eval.paths import ROOT
import json
import os
import sys
import time

import numpy as np
from scipy import sparse

from switchboard_eval.experiments import exp_lang_scale as lang_scale
from ovos_switchboard.core import classifier as clf_mod
from ovos_switchboard.core.classifier import Classifier, training_data
from ovos_switchboard.core.device import Device
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.schema import Schema
from ovos_switchboard.core.templates import SLOT_TOKEN, expansions


PRUNE_BELOW = 1e-3


def vectorizers(min_df=1, max_features=None):
    """The shipped vectorizers with min_df / max_features changed."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=min_df,
                            max_features=max_features),
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[a-z0-9']+",
                            min_df=min_df, max_features=max_features)]


def train(rows, **vec):
    """(classifier, seconds) with the given vectorizer settings."""
    shipped = clf_mod._vectorizers  # pylint: disable=protected-access
    clf_mod._vectorizers = lambda: vectorizers(**vec)  # pylint: disable=protected-access
    try:
        t0 = time.perf_counter()
        model = Classifier.train(rows)
        return model, round(time.perf_counter() - t0, 1)
    finally:
        clf_mod._vectorizers = shipped  # pylint: disable=protected-access


def sizes(model):
    """(features, classes, weight bytes) of a classifier."""
    coef = model.head.m.coef_
    nbytes = coef.data.nbytes + coef.indices.nbytes + coef.indptr.nbytes if sparse.issparse(coef) else coef.nbytes
    return coef.shape[1], coef.shape[0], nbytes


def holdout_score(doc, lang, schema, model_for):
    """exp_lang_scale's hold-out score with the classifier from model_for(train_device)."""
    train_doc, held = lang_scale.split(doc)
    device = Device.from_definitions(train_doc["definitions"], lang)
    exact = ExactLayer(device, schema)
    model = model_for(device)
    right = wrong = passed = 0
    for label, line in held:
        gold = schema.canon_or_unmapped(label)
        for e in list(dict.fromkeys(expansions(line)))[:lang_scale.EXPANSIONS_PER_LINE]:
            utt = SLOT_TOKEN.sub(lang_scale.SLOT_FILLER, e).lower()
            got = lang_scale.decide_exact(exact, model, utt)
            if not got:
                top, conf = model.rank(utt)[0]
                got = top if conf >= lang_scale.THRESHOLD else None
            right += got == gold
            wrong += got not in (gold, None)
            passed += got is None
    n = right + wrong + passed
    return {"right_pct": round(100 * right / n, 1), "wrong_pct": round(100 * wrong / n, 1),
            "pass_pct": round(100 * passed / n, 1)}


def to_float32(model):
    """The model with float32 weights."""
    model.head.m.coef_ = model.head.m.coef_.astype(np.float32)
    model.head.m.intercept_ = model.head.m.intercept_.astype(np.float32)
    return model


def pruned(model):
    """The model with tiny weights zeroed and stored sparse (float32)."""
    coef = model.head.m.coef_.astype(np.float32)
    coef[np.abs(coef) < PRUNE_BELOW] = 0
    model.head.m.coef_ = sparse.csr_matrix(coef)
    return model


VARIANTS = {"shipped": ({}, None), "min_df2": ({"min_df": 2}, None), "cap100k": ({"max_features": 100_000}, None),
            "float32": ({}, to_float32), "prune": ({}, pruned)}


def main(argv):
    path, lang = argv[0], argv[1].lower()
    harness = ROOT
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    schema = Schema.load(os.path.join(harness, "schema/intents.yaml"))
    full = training_data(Device.from_definitions(doc["definitions"], lang), schema)
    for name, (vec, post) in VARIANTS.items():
        model, secs = train(full, **vec)
        model = post(model) if post else model
        features, classes, weights = sizes(model)

        def model_for(device, vec=vec, post=post):
            m, _ = train(training_data(device, schema), **vec)
            return post(m) if post else m

        result = {"manifest": os.path.basename(path), "lang": lang, "variant": name, "rows": len(full),
                  "features": features, "classes": classes, "weights_mb": round(weights / 2**20, 1),
                  "train_s": secs, "holdout": holdout_score(doc, lang, schema, model_for)}
        print(json.dumps(result), flush=True)
        with open(os.path.join(harness, "preds", "exp_classifier_memory.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
