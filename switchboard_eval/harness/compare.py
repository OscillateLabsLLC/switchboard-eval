#!/usr/bin/env python3
"""One markdown table comparing many prediction files against a reference.

Usage: compare.py data/dataset.jsonl reference.jsonl preds1.jsonl [preds2.jsonl ...] [--split dev|test]

Columns: headline / unseen accuracy, raw false accept, high-risk wrong fires,
the first threshold where false accept <= FA_TARGET (coverage and precision
there), latency, and exact McNemar p vs the reference. Latency is measured on
whatever machine produced the file; it is not the mini PC unless run there.
"""
from switchboard_eval.paths import ROOT
import os, sys

sys.path.insert(0, ROOT)
from switchboard_eval.harness.score import load, load_gold, mcnemar, pop_split, score

FA_TARGET = 20.0


def operating_point(rep):
    for s in rep.get("threshold_sweep") or []:
        if s["false_accept_rate"] is not None and s["false_accept_rate"] <= FA_TARGET:
            return f"{s['threshold']:.2f}: {s['coverage']}% cov / {s['precision_when_acting']}% prec"
    return "never"


def row(name, rep, p=None):
    tc = rep.get("template_coverage", {})
    lat = rep["latency_ms"]
    return (f"| {name} | {rep['headline_accuracy']} | {tc.get('unseen', {}).get('acc')} "
            f"| {rep['false_accept_rate']} | {len(rep['high_risk_wrong_executions'])} "
            f"| {len(rep['high_risk_wrong_fires'])} "
            f"| {operating_point(rep)} | {lat['p50']} / {lat['p95']} "
            f"| {'—' if p is None else p} |")


def main(argv):
    split, argv = pop_split(argv)
    if len(argv) < 3:
        sys.exit(__doc__)
    gold, ref_path = load_gold(argv[0], split), argv[1]
    if split:
        print(f"split: {split} ({len(gold)} rows)\n")
    ref = load(ref_path)
    print(f"| system | headline % | unseen % | false accept % | high-risk wrong exec (blocker) "
          f"| high-risk wrong confirms | first thr with FA <= {FA_TARGET:.0f}% "
          f"| latency p50 / p95 ms | McNemar p vs ref |")
    print("|---|---|---|---|---|---|---|---|---|")
    print(row(os.path.basename(ref_path) + " (ref)", score(gold, ref)))
    for path in argv[2:]:
        pred = load(path)
        a_only, b_only, p = mcnemar(gold, ref, pred)
        print(row(os.path.basename(path).removesuffix(".jsonl"), score(gold, pred),
                  f"{p:.3g} ({b_only}+/{a_only}-)"))


if __name__ == "__main__":
    main(sys.argv[1:])
